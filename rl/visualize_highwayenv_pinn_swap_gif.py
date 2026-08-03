"""Render synchronized numerical-teacher and context-PINN HighwayEnv GIFs."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from rl.env.highwayenv_social_env import make_social_highwayenv_env, resolve_traffic_config
from rl.risk.field_backend import make_field_backend
from rl.visualize_highwayenv_sb3_suite import (
    PlannerSpec,
    RenderStep,
    _find_wrapper_instance,
    _load_sb3_model,
    _overlay_payload_from_field,
    _save_multiplanner_frame,
    _swap_ego_to_idm,
)


def _policy_action(model, observation):
    obs = np.asarray(observation, dtype=np.float32)
    expected = tuple(model.observation_space.shape)
    if obs.shape != expected:
        obs = obs.reshape(expected)
    action, _ = model.predict(obs, deterministic=True)
    action_array = np.asarray(action)
    if hasattr(model.action_space, "n"):
        return int(action_array.reshape(-1)[0])
    return action_array.astype(np.float32).reshape(model.action_space.shape)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-id",
        choices=("highway-v0", "merge-v0", "roundabout-v0", "intersection-v0"),
        default="merge-v0",
    )
    parser.add_argument("--planner", choices=("idm", "rl"), default="idm")
    parser.add_argument("--algo", choices=("ppo", "dqn"), default="ppo")
    parser.add_argument(
        "--action-mode",
        choices=("default", "discrete_meta", "discrete_kinematic", "continuous"),
        default="default",
        help="Must match the action interface used to train the policy.",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--pinn-checkpoint", required=True)
    parser.add_argument("--pinn-device", default="cpu")
    parser.add_argument("--traffic-preset", default="medium")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--snapshot-index",
        type=int,
        default=-1,
        help="GIF frame used for static snapshots; negative selects the midpoint.",
    )
    parser.add_argument("--drift-warmup-s", type=float, default=2.0)
    parser.add_argument(
        "--teacher-backend",
        choices=("numerical", "prospective"),
        default="prospective",
    )
    parser.add_argument("--append-risk-obs", type=_str2bool, default=False)
    parser.add_argument(
        "--stop-on-offroad",
        type=_str2bool,
        default=True,
        help="Stop before saving an off-road frame; useful for clean qualitative field figures.",
    )
    parser.add_argument(
        "--output", default="evaluation/pinn_highwayenv_gifs/merge_teacher_vs_pinn.gif"
    )
    args = parser.parse_args()
    if args.planner == "rl" and not args.checkpoint:
        parser.error("--checkpoint is required for --planner rl")

    traffic = resolve_traffic_config(preset=args.traffic_preset)
    env = make_social_highwayenv_env(
        env_id=args.env_id,
        interface="stock",
        render_mode="rgb_array",
        traffic=traffic,
        use_drift=True,
        action_mode=args.action_mode,
        append_risk_obs=bool(args.append_risk_obs),
        drift_warmup_s=args.drift_warmup_s,
        record_risk_metrics=False,
        field_backend=args.teacher_backend,
        pinn_checkpoint=(
            args.pinn_checkpoint if args.teacher_backend == "prospective" else None
        ),
    )
    observation, _ = env.reset(seed=args.seed)
    if args.planner == "idm":
        _swap_ego_to_idm(env)
        observation = env.unwrapped.observation_type.observe()
    model = (
        _load_sb3_model(args.algo, args.checkpoint)
        if args.planner == "rl"
        else None
    )
    wrapper = _find_wrapper_instance(env, "get_drift_grid")
    if wrapper is None:
        raise RuntimeError("HighwayEnv DRIFT wrapper is unavailable")
    cfg = wrapper.get_drift_config()
    surrogate = make_field_backend(
        "pinn",
        sim_cfg=cfg,
        pinn_checkpoint=args.pinn_checkpoint,
        pinn_device=args.pinn_device,
        pinn_time_mode="error",
    )
    road_mask = wrapper.get_road_mask()
    if road_mask is not None:
        surrogate.set_road_mask(road_mask)
    raw = env.unwrapped
    dt = 1.0 / float(raw.config["policy_frequency"])
    ego, vehicles = wrapper._collect_drift_state()
    surrogate.warmup(
        vehicles,
        ego,
        dt=dt,
        duration=args.drift_warmup_s,
        substeps=1,
        source_fn=wrapper._source_fn,
        full_field=False,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered: list[Image.Image] = []
    rendered_steps: list[list[RenderStep]] = []
    episode_return = 0.0
    with tempfile.TemporaryDirectory(prefix="safer_pinn_swap_") as temporary:
        temp_dir = Path(temporary)
        for step in range(int(args.frames)):
            action = 1 if args.planner == "idm" else _policy_action(model, observation)
            observation, reward, terminated, truncated, _info = env.step(action)
            episode_return += float(reward)
            if args.stop_on_offroad and not bool(getattr(raw.vehicle, "on_road", True)):
                print(
                    f"Stopped at step {step}: ego left the road before frame capture.",
                    flush=True,
                )
                break
            ego, vehicles = wrapper._collect_drift_state()
            surrogate.step(
                vehicles,
                ego,
                dt=dt,
                substeps=1,
                source_fn=wrapper._source_fn,
                full_field=True,
            )
            frame = env.render()
            if frame is None:
                continue
            teacher = np.asarray(wrapper.get_masked_risk_field(), dtype=np.float32)
            prediction = np.asarray(surrogate.risk_field, dtype=np.float32)
            if road_mask is not None:
                prediction = np.where(np.asarray(road_mask) > 0.05, prediction, np.nan)
            teacher_rgba, teacher_extent = _overlay_payload_from_field(env, teacher)
            pinn_rgba, pinn_extent = _overlay_payload_from_field(env, prediction)
            speed = float(np.hypot(*raw.vehicle.velocity))

            def panel(label, rgba, extent):
                return RenderStep(
                    planner=label,
                    step=step,
                    action=action,
                    reward=float(reward),
                    total_return=episode_return,
                    speed=speed,
                    ttc=float("nan"),
                    min_spacing=float("nan"),
                    social_score=float("nan"),
                    corridor_risk=float("nan"),
                    crashed=bool(raw.vehicle.crashed),
                    frame=np.asarray(frame),
                    overlays={"field": (rgba, extent, label)},
                )

            png = temp_dir / f"frame_{step:04d}.png"
            current_steps = [
                panel(
                    "Prospective numerical teacher"
                    if args.teacher_backend == "prospective"
                    else "Legacy numerical PDE teacher",
                    teacher_rgba,
                    teacher_extent,
                ),
                panel("Context PINN surrogate", pinn_rgba, pinn_extent),
            ]
            _save_multiplanner_frame(
                str(png),
                current_steps,
                env_id=args.env_id,
                traffic_label=args.traffic_preset,
                step=step,
                overlay_mode="field",
            )
            with Image.open(png) as image:
                rendered.append(image.convert("RGB").copy())
            rendered_steps.append(current_steps)
            if terminated or truncated:
                break
    env.close()
    if not rendered:
        raise RuntimeError("No visualization frames were produced")
    duration = max(20, int(round(1000.0 / max(args.fps, 0.1))))
    rendered[0].save(
        output,
        save_all=True,
        append_images=rendered[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )
    snapshot_index = (
        len(rendered) // 2
        if int(args.snapshot_index) < 0
        else min(int(args.snapshot_index), len(rendered) - 1)
    )
    rendered[snapshot_index].save(output.with_name(f"{output.stem}_snapshot.png"))
    _save_multiplanner_frame(
        str(output.with_name(f"{output.stem}_snapshot_column.png")),
        rendered_steps[snapshot_index],
        env_id=args.env_id,
        traffic_label=args.traffic_preset,
        step=rendered_steps[snapshot_index][0].step,
        overlay_mode="field",
        layout="vertical",
    )
    print(f"Saved {len(rendered)} synchronized frames to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
