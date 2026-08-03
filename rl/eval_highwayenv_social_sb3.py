from __future__ import annotations

import argparse
from collections import deque
import importlib
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
HIGHWAYENV_ROOT = REPO_ROOT / "HighwayEnv-master"
for _path in (str(HIGHWAYENV_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rl.utils.typing_compat import ensure_typing_extensions_compat

ensure_typing_extensions_compat()
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
import gymnasium as gym

from rl.env.highwayenv_social_env import (
    load_reward_config,
    make_social_highwayenv_env,
    resolve_traffic_config,
)


@dataclass
class EpisodeMetrics:
    env_id: str
    seed: int
    episode_return: float
    episode_length: int
    crashed: bool
    truncated: bool
    progress: float
    mean_speed: float
    mean_abs_accel: float
    mean_abs_steer: float
    mean_jerk_abs: float
    action_change_rate: float
    lane_change_rate: float
    ttc_min: float
    thw_min: float
    drac_max: float
    imposed_rear_decel_max: float
    bad_cut_in_any: bool
    missed_opportunity_rate: float
    corridor_risk_mean: float
    risk_flux_mean: float
    mean_action_selection_ms: float
    p95_action_selection_ms: float
    mean_field_backend_ms: float
    p95_field_backend_ms: float
    reward_terms_mean: dict[str, float]


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _checkpoint_file(path: str) -> str:
    return str(path) if str(path).endswith(".zip") else f"{path}.zip"


def _model_observation(model, observation) -> np.ndarray:
    value = np.asarray(observation, dtype=np.float32)
    expected = tuple(model.observation_space.shape)
    if value.shape == expected:
        return value
    if value.size != int(np.prod(expected)):
        raise ValueError(
            f"Checkpoint expects observation shape {expected}; environment returned {value.shape}"
        )
    return value.reshape(expected)


def _ppo_action_space_from_state(state: dict[str, torch.Tensor]):
    action_dim = int(state["action_net.weight"].shape[0])
    if "log_std" in state:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
    return gym.spaces.Discrete(action_dim)


def load_sb3_model(algo: str, checkpoint: str):
    # Checkpoints produced under NumPy 2.x may contain this private module
    # path. NumPy 1.26 exposes the same objects through the legacy path.
    try:
        importlib.import_module("numpy._core.numeric")
    except ModuleNotFoundError:
        sys.modules.setdefault(
            "numpy._core.numeric", importlib.import_module("numpy.core.numeric")
        )
    algo = str(algo).strip().lower()
    path = _checkpoint_file(checkpoint)
    custom_objects = None
    try:
        with zipfile.ZipFile(path) as archive:
            state = torch.load(
                io.BytesIO(archive.read("policy.pth")),
                map_location="cpu",
                weights_only=False,
        )
        if algo == "ppo":
            obs_dim = int(state["mlp_extractor.policy_net.0.weight"].shape[1])
            action_space = _ppo_action_space_from_state(state)
        elif algo == "dqn":
            obs_dim = int(state["q_net.q_net.0.weight"].shape[1])
            q_weights = [
                value
                for key, value in state.items()
                if key.startswith("q_net.q_net.") and key.endswith(".weight")
            ]
            action_space = gym.spaces.Discrete(int(q_weights[-1].shape[0]))
        else:
            obs_dim = 0
            action_space = None
        if obs_dim and action_space is not None:
            custom_objects = {
                "observation_space": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32,
                ),
                "action_space": action_space,
                "_last_obs": None,
                "_last_episode_starts": None,
                "ep_info_buffer": deque(maxlen=100),
                "ep_success_buffer": deque(maxlen=100),
            }
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile):
        custom_objects = None
    if algo == "ppo":
        return PPO.load(path, custom_objects=custom_objects, device="cpu")
    if algo == "dqn":
        return DQN.load(path, custom_objects=custom_objects, device="cpu")
    if algo == "sac":
        return SAC.load(path, custom_objects=custom_objects, device="cpu")
    if algo == "td3":
        return TD3.load(path, custom_objects=custom_objects, device="cpu")
    if algo == "ddpg":
        return DDPG.load(path, custom_objects=custom_objects, device="cpu")
    raise ValueError(f"Unsupported algo '{algo}'")


def evaluate_episode(
    model,
    *,
    env_id: str,
    interface: str,
    traffic,
    reward_config,
    ablation: str,
    seed: int,
    use_drift: bool,
    action_mode: str,
    append_risk_obs: bool = False,
    field_backend: str = "numerical",
    pinn_checkpoint: str | None = None,
    pinn_device: str = "cpu",
    pinn_time_mode: str = "error",
    max_steps: int = 600,
) -> EpisodeMetrics:
    env = make_social_highwayenv_env(
        env_id=env_id,
        interface=interface,
        traffic=traffic,
        reward_config=reward_config,
        ablation=ablation,
        use_drift=use_drift,
        action_mode=action_mode,
        append_risk_obs=append_risk_obs,
        record_risk_metrics=False,
        field_backend=field_backend,
        pinn_checkpoint=pinn_checkpoint,
        pinn_device=pinn_device,
        pinn_time_mode=pinn_time_mode,
    )
    obs, _info = env.reset(seed=seed)
    returns = 0.0
    steps = []
    action_selection_samples_ms: list[float] = []
    field_backend_samples_ms: list[float] = []
    terminated = False
    truncated = False
    horizon_reached = False
    for decision_index in range(max(1, int(max_steps))):
        action_t0 = time.perf_counter()
        action, _ = model.predict(
            _model_observation(model, obs),
            deterministic=True,
        )
        action_selection_samples_ms.append(1000.0 * (time.perf_counter() - action_t0))
        obs, reward, terminated, truncated, info = env.step(action)
        returns += float(reward)
        step = dict(info.get("social_step", {}))
        if step:
            steps.append(step)
        if "field_backend_step_ms" in info:
            field_backend_samples_ms.append(float(info["field_backend_step_ms"]))
        if terminated or truncated:
            break
    else:
        horizon_reached = True
        truncated = True
    env.close()

    if not steps:
        steps = [dict(info.get("social_episode_summary", {}))]
    reward_keys = sorted(k for k in steps[0].keys() if k.startswith("r_") or k.startswith("c_") or k in {"stock_reward", "reward_total"})

    return EpisodeMetrics(
        env_id=env_id,
        seed=int(seed),
        episode_return=float(returns),
        episode_length=int(len(steps)),
        crashed=bool(max(row.get("collision", 0.0) for row in steps)),
        truncated=bool(truncated or horizon_reached),
        progress=float(np.sum([row.get("progress_dx", 0.0) for row in steps])),
        mean_speed=float(np.mean([row.get("speed", 0.0) for row in steps])),
        mean_abs_accel=float(np.mean([abs(row.get("accel", 0.0)) for row in steps])),
        mean_abs_steer=float(np.mean([abs(row.get("steer", 0.0)) for row in steps])),
        mean_jerk_abs=float(np.mean([abs(row.get("jerk", 0.0)) for row in steps])),
        action_change_rate=float(np.mean([row.get("action_delta", 0.0) for row in steps])),
        lane_change_rate=float(np.mean([row.get("lane_change_cmd", 0.0) for row in steps])),
        ttc_min=float(np.min([row.get("ttc_min", 60.0) for row in steps])),
        thw_min=float(np.min([row.get("thw_same", 60.0) for row in steps])),
        drac_max=float(np.max([row.get("drac_same", 0.0) for row in steps])),
        imposed_rear_decel_max=float(np.max([row.get("imposed_rear_decel", 0.0) for row in steps])),
        bad_cut_in_any=bool(max(row.get("bad_cut_in", 0.0) for row in steps)),
        missed_opportunity_rate=float(np.mean([row.get("missed_opportunity", 0.0) for row in steps])),
        corridor_risk_mean=float(np.mean([row.get("r_corr", 0.0) for row in steps])),
        risk_flux_mean=float(np.mean([row.get("risk_flux_backward", 0.0) for row in steps])),
        mean_action_selection_ms=float(np.mean(action_selection_samples_ms)) if action_selection_samples_ms else 0.0,
        p95_action_selection_ms=float(np.percentile(action_selection_samples_ms, 95)) if action_selection_samples_ms else 0.0,
        mean_field_backend_ms=float(np.mean(field_backend_samples_ms)) if field_backend_samples_ms else 0.0,
        p95_field_backend_ms=float(np.percentile(field_backend_samples_ms, 95)) if field_backend_samples_ms else 0.0,
        reward_terms_mean={
            key: float(np.mean([row.get(key, 0.0) for row in steps]))
            for key in reward_keys
        },
    )


def summarize_eval(episodes: list[EpisodeMetrics]) -> dict[str, object]:
    if not episodes:
        return {"episodes": 0}
    reward_keys = sorted(episodes[0].reward_terms_mean.keys())
    return {
        "episodes": len(episodes),
        "return_mean": float(np.mean([ep.episode_return for ep in episodes])),
        "return_std": float(np.std([ep.episode_return for ep in episodes])),
        "episode_length_mean": float(np.mean([ep.episode_length for ep in episodes])),
        "collision_rate": float(np.mean([float(ep.crashed) for ep in episodes])),
        "truncated_rate": float(np.mean([float(ep.truncated) for ep in episodes])),
        "progress_mean": float(np.mean([ep.progress for ep in episodes])),
        "mean_speed": float(np.mean([ep.mean_speed for ep in episodes])),
        "mean_abs_accel": float(np.mean([ep.mean_abs_accel for ep in episodes])),
        "mean_abs_steer": float(np.mean([ep.mean_abs_steer for ep in episodes])),
        "mean_jerk_abs": float(np.mean([ep.mean_jerk_abs for ep in episodes])),
        "action_change_rate": float(np.mean([ep.action_change_rate for ep in episodes])),
        "lane_change_rate": float(np.mean([ep.lane_change_rate for ep in episodes])),
        "ttc_min_mean": float(np.mean([ep.ttc_min for ep in episodes])),
        "thw_min_mean": float(np.mean([ep.thw_min for ep in episodes])),
        "drac_max_mean": float(np.mean([ep.drac_max for ep in episodes])),
        "imposed_rear_decel_max_mean": float(np.mean([ep.imposed_rear_decel_max for ep in episodes])),
        "bad_cut_in_rate": float(np.mean([float(ep.bad_cut_in_any) for ep in episodes])),
        "missed_opportunity_rate": float(np.mean([ep.missed_opportunity_rate for ep in episodes])),
        "corridor_risk_mean": float(np.mean([ep.corridor_risk_mean for ep in episodes])),
        "risk_flux_mean": float(np.mean([ep.risk_flux_mean for ep in episodes])),
        "mean_action_selection_ms": float(np.mean([ep.mean_action_selection_ms for ep in episodes])),
        "p95_action_selection_ms": float(np.mean([ep.p95_action_selection_ms for ep in episodes])),
        "mean_field_backend_ms": float(np.mean([ep.mean_field_backend_ms for ep in episodes])),
        "p95_field_backend_ms": float(np.mean([ep.p95_field_backend_ms for ep in episodes])),
        "reward_terms_mean": {
            key: float(np.mean([ep.reward_terms_mean.get(key, 0.0) for ep in episodes]))
            for key in reward_keys
        },
        "episodes_detail": [ep.__dict__ for ep in episodes],
    }


def evaluate_model(
    model,
    *,
    env_ids: list[str],
    interface: str,
    traffic,
    reward_config,
    ablation: str,
    episodes: int,
    use_drift: bool,
    action_mode: str = "default",
    append_risk_obs: bool = False,
    field_backend: str = "numerical",
    pinn_checkpoint: str | None = None,
    pinn_device: str = "cpu",
    pinn_time_mode: str = "error",
    max_steps: int = 600,
) -> dict[str, dict[str, object]]:
    results = {}
    for env_id in env_ids:
        rows = [
            evaluate_episode(
                model,
                env_id=env_id,
                interface=interface,
                traffic=traffic,
                reward_config=reward_config,
                ablation=ablation,
                seed=seed,
                use_drift=use_drift,
                action_mode=action_mode,
                append_risk_obs=append_risk_obs,
                field_backend=field_backend,
                pinn_checkpoint=pinn_checkpoint,
                pinn_device=pinn_device,
                pinn_time_mode=pinn_time_mode,
                max_steps=max_steps,
            )
            for seed in range(int(episodes))
        ]
        results[env_id] = summarize_eval(rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SB3 agents on social-reward HighwayEnv benchmarks.")
    parser.add_argument("--algo", choices=["ppo", "dqn", "sac", "td3", "ddpg"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", nargs="+", default=["highway-v0", "merge-v0"])
    parser.add_argument("--interface", choices=["stock", "decision"], default="stock")
    parser.add_argument(
        "--action-mode",
        choices=["default", "discrete_meta", "discrete_kinematic", "continuous"],
        default="default",
        help="Must match the action protocol used during training.",
    )
    parser.add_argument("--reward-config", default="rl/config/social_reward_v1.json")
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=600,
        help="Explicit evaluation horizon; required for scenarios without native time truncation.",
    )
    parser.add_argument("--use-drift", type=_str2bool, default=True)
    parser.add_argument(
        "--append-risk-obs",
        type=_str2bool,
        default=False,
        help="Append the 8-D field descriptor used for a true policy-backend swap.",
    )
    parser.add_argument(
        "--field-backend",
        choices=["numerical", "prospective", "pinn"],
        default="numerical",
    )
    parser.add_argument("--pinn-checkpoint", default="")
    parser.add_argument("--pinn-device", default="cpu")
    parser.add_argument("--pinn-time-mode", choices=["error", "clip"], default="error")
    parser.add_argument("--traffic-preset", default="medium")
    parser.add_argument("--vehicles-count", type=int, default=None)
    parser.add_argument("--vehicles-density", type=float, default=None)
    parser.add_argument("--ego-spacing", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--sv-speed-min", type=float, default=None)
    parser.add_argument("--sv-speed-max", type=float, default=None)
    parser.add_argument("--sv-speed-noise", type=float, default=None)
    parser.add_argument("--lane-speed-bias", default="", help="Comma-separated lane speed biases.")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    lane_bias = tuple(float(x) for x in args.lane_speed_bias.split(",") if str(x).strip())
    traffic = resolve_traffic_config(
        preset=args.traffic_preset,
        vehicles_count=args.vehicles_count,
        vehicles_density=args.vehicles_density,
        ego_spacing=args.ego_spacing,
        duration=args.duration,
        sv_speed_min=args.sv_speed_min,
        sv_speed_max=args.sv_speed_max,
        sv_speed_noise=args.sv_speed_noise,
        lane_speed_bias=lane_bias,
    )
    reward_config = load_reward_config(args.reward_config)
    model = load_sb3_model(args.algo, args.checkpoint)
    summary = {
        "algo": args.algo,
        "checkpoint": args.checkpoint,
        "interface": args.interface,
        "action_mode": args.action_mode,
        "ablation": args.ablation,
        "use_drift": bool(args.use_drift),
        "append_risk_obs": bool(args.append_risk_obs),
        "field_backend": args.field_backend if args.use_drift else "disabled",
        "pinn_checkpoint": str(args.pinn_checkpoint),
        "traffic": traffic.to_dict(),
        "results": evaluate_model(
            model,
            env_ids=list(args.env_id),
            interface=args.interface,
            traffic=traffic,
            reward_config=reward_config,
            ablation=args.ablation,
            episodes=args.episodes,
            use_drift=bool(args.use_drift),
            action_mode=args.action_mode,
            append_risk_obs=bool(args.append_risk_obs),
            field_backend=args.field_backend,
            pinn_checkpoint=str(args.pinn_checkpoint).strip() or None,
            pinn_device=args.pinn_device,
            pinn_time_mode=args.pinn_time_mode,
            max_steps=int(args.max_episode_steps),
        ),
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
