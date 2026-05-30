"""
Watch a trained MetaDrive SB3 agent drive — live 3D view or top-down view.

Usage
-----
    # 3D panda3d window (interactive camera, watch from chase view)
    python rl/watch_metadrive_agent.py \
        --checkpoint rl/checkpoints/metadrive/smoke_50k_n8_d03/final.zip \
        --episodes 3 --density 0.3

    # Top-down view in a separate pygame window
    python rl/watch_metadrive_agent.py \
        --checkpoint rl/checkpoints/.../final.zip \
        --view top_down --episodes 3

    # IDM baseline (no checkpoint needed)
    python rl/watch_metadrive_agent.py --planner idm --view 3d --episodes 3

Controls (3D view):
    [B]   toggle top-down camera
    [R]   reset episode
    [Esc] quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
METADRIVE_ROOT = REPO_ROOT / "metadrive"
if METADRIVE_ROOT.is_dir() and str(METADRIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(METADRIVE_ROOT))

import numpy as np

try:
    from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
except ImportError:
    DDPG = DQN = PPO = SAC = TD3 = None

from rl.config.metadrive_config import METADRIVE_PROTOCOLS, MetaDriveRLConfig, get_metadrive_protocol
from rl.env.metadrive_envs import make_metadrive_eval_env
from rl.eval_metadrive import _continuous_to_discrete, _space_signature

TRAFFIC_MODE_CHOICES = ("protocol", "trigger", "respawn", "basic", "hybrid")
ALGO_CHOICES = ("auto", "ppo", "dqn", "sac", "td3", "ddpg")
RISK_OVERLAY_CHOICES = ("none", "drift")
_ALGOS = ("dqn", "sac", "td3", "ddpg", "ppo")


def _native_idm_policy_class():
    from metadrive.policy.idm_policy import IDMPolicy

    return IDMPolicy


def _infer_algo_from_checkpoint(checkpoint: str) -> str:
    """Infer SB3 algorithm from a checkpoint path/run name."""
    text = str(checkpoint).replace("\\", "/").lower()
    for algo in _ALGOS:
        tokens = (f"_{algo}_", f"-{algo}-", f"/{algo}_", f"_{algo}/", f"{algo}_")
        if any(token in text for token in tokens):
            return algo
    return "ppo"


def _load_policy(checkpoint: Optional[str], algo: str):
    if checkpoint is None:
        return None
    if PPO is None:
        raise RuntimeError("stable-baselines3 not installed")
    inferred_algo = _infer_algo_from_checkpoint(checkpoint)
    resolved_algo = inferred_algo if algo == "auto" else str(algo)
    if algo != "auto" and inferred_algo != "ppo" and inferred_algo != resolved_algo:
        print(
            f"Warning: checkpoint path looks like {inferred_algo.upper()}, "
            f"but --algo {resolved_algo} was requested; using {inferred_algo.upper()}."
        )
        resolved_algo = inferred_algo
    loader = {"ppo": PPO, "dqn": DQN, "sac": SAC, "td3": TD3, "ddpg": DDPG}[resolved_algo]
    print(f"Loading {resolved_algo.upper()} checkpoint: {checkpoint}")
    try:
        return loader.load(checkpoint, device="cpu")
    except TypeError as exc:
        hint = ""
        if algo != "auto":
            inferred = _infer_algo_from_checkpoint(checkpoint)
            if inferred != algo:
                hint = (
                    f"\nThe checkpoint path looks like {inferred.upper()}, "
                    f"but --algo {algo} was requested. Use --algo {inferred} "
                    "or omit --algo to auto-detect."
                )
        raise RuntimeError(f"Failed to load checkpoint as {resolved_algo.upper()}.{hint}") from exc


def _select_action(env, planner: str, model, obs, protocol):
    if planner == "idm":
        raw = env.unwrapped
        if hasattr(env.action_space, "shape"):
            return np.zeros(env.action_space.shape, dtype=np.float32)
        policy = getattr(raw, "_watch_idm_policy", None)
        if policy is None or getattr(policy, "control_object", None) is not raw.agent:
            from metadrive.policy.idm_policy import IDMPolicy
            policy = IDMPolicy(raw.agent, random_seed=0)
            raw._watch_idm_policy = policy
        action = np.asarray(policy.act(agent_id=None), dtype=np.float32)
        if protocol.discrete_action:
            return _continuous_to_discrete(action, protocol)
        return action
    if planner == "random":
        return env.action_space.sample()
    # RL
    action, _ = model.predict(obs, deterministic=True)
    if protocol.discrete_action:
        return int(np.asarray(action).item())
    return np.asarray(action, dtype=np.float32)


def _risk_grid_centers(mdcfg: MetaDriveRLConfig, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ego-body-frame risk-grid centers aligned to the recorded grid shape."""
    risk = np.asarray(grid)
    if risk.ndim != 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    ny, nx = int(risk.shape[0]), int(risk.shape[1])
    xs = np.linspace(float(mdcfg.GRID_X_MIN), float(mdcfg.GRID_X_MAX), nx)
    ys = np.linspace(float(mdcfg.GRID_Y_MIN), float(mdcfg.GRID_Y_MAX), ny)
    return xs, ys


def _body_to_world(ego_pos: np.ndarray, heading: float, x_body: float, y_body: float) -> tuple[float, float]:
    c, s = float(np.cos(heading)), float(np.sin(heading))
    return (
        float(ego_pos[0]) + c * float(x_body) - s * float(y_body),
        float(ego_pos[1]) + s * float(x_body) + c * float(y_body),
    )


def _frame_to_screen(renderer, world_xy: tuple[float, float]) -> tuple[float, float]:
    field_w, field_h = renderer.screen_canvas.get_size()
    track = renderer.current_track_agent
    if renderer.center_on_map:
        canvas_w, canvas_h = renderer._frame_canvas.get_size()
        camera_pix = (canvas_w / 2.0, canvas_h / 2.0)
    else:
        camera_pos = renderer.position or track.position
        camera_pix = renderer._frame_canvas.pos2pix(*camera_pos)
    off_x = float(camera_pix[0]) - float(field_w) / 2.0
    off_y = float(camera_pix[1]) - float(field_h) / 2.0
    pix = renderer._frame_canvas.pos2pix(*world_xy)
    return float(pix[0]) - off_x, float(pix[1]) - off_y


def _drift_heatmap_rgba(grid: np.ndarray, *, alpha: int, grid_stride: int) -> np.ndarray | None:
    """Convert a DRIFT grid into a smoothed RGBA heatmap surface payload."""
    risk = np.asarray(grid, dtype=np.float32)
    if risk.ndim != 2 or risk.size == 0:
        return None

    stride = max(1, int(grid_stride))
    risk = risk[::stride, ::stride]
    finite_positive = risk[np.isfinite(risk) & (risk > 0.0)]
    if finite_positive.size < 4:
        return None

    from scipy.ndimage import gaussian_filter
    from matplotlib import colormaps

    risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)
    smooth = gaussian_filter(risk, sigma=1.1)
    positive = smooth[smooth > 0.0]
    if positive.size < 4:
        return None

    vmax = max(1e-3, float(np.percentile(positive, 99.5)))
    visible_floor = float(np.percentile(positive, 60.0))
    scaled = np.clip(smooth / vmax, 0.0, 1.0)
    field_alpha = np.clip(
        (smooth - visible_floor) / max(vmax - visible_floor, 1e-6),
        0.0,
        1.0,
    ) * (float(np.clip(alpha, 0, 255)) / 255.0)
    rgba = colormaps["inferno"](scaled)
    rgba[..., 3] = np.where(smooth > 0.0, field_alpha, 0.0)

    # cfg.y grows upward while image rows grow downward.
    rgba = np.flipud(rgba)
    return np.ascontiguousarray(np.clip(rgba * 255.0, 0.0, 255.0).astype(np.uint8))


def _draw_live_drift_overlay(env, info: dict, mdcfg: MetaDriveRLConfig, *,
                             alpha: int, grid_stride: int) -> bool:
    """Overlay the recorded ego-relative DRIFT field as a live BEV heatmap."""
    grid = info.get("risk_grid")
    current = env
    while grid is None and current is not None:
        get_grid = getattr(current, "get_risk_grid", None)
        if callable(get_grid):
            grid = get_grid()
            break
        current = getattr(current, "env", None)
    renderer = getattr(env.unwrapped, "top_down_renderer", None)
    if grid is None or renderer is None or getattr(renderer, "target_agent_heading_up", False):
        return False
    risk = np.asarray(grid, dtype=np.float32)
    rgba = _drift_heatmap_rgba(risk, alpha=alpha, grid_stride=grid_stride)
    if rgba is None:
        return False

    try:
        from metadrive.utils import import_pygame
    except ImportError:
        from metadrive.utils.utils import import_pygame
    pygame = import_pygame()

    xs, ys = _risk_grid_centers(mdcfg, risk)
    if xs.size == 0 or ys.size == 0:
        return False
    ego = env.unwrapped.agent
    ego_pos = np.asarray(ego.position, dtype=float)
    heading = float(ego.heading_theta)

    field_center_body = (0.5 * (float(xs[0]) + float(xs[-1])), 0.5 * (float(ys[0]) + float(ys[-1])))
    field_center_screen = _frame_to_screen(
        renderer,
        _body_to_world(ego_pos, heading, field_center_body[0], field_center_body[1]),
    )
    scale = float(getattr(renderer._frame_canvas, "scaling", 1.0))
    field_w_px = max(2, int(round(abs(float(xs[-1]) - float(xs[0])) * scale)))
    field_h_px = max(2, int(round(abs(float(ys[-1]) - float(ys[0])) * scale)))

    raw_heatmap = pygame.image.fromstring(
        rgba.tobytes(),
        (int(rgba.shape[1]), int(rgba.shape[0])),
        "RGBA",
    ).convert_alpha()
    heatmap = pygame.transform.smoothscale(raw_heatmap, (field_w_px, field_h_px))
    heatmap = pygame.transform.rotozoom(heatmap, float(np.degrees(heading)), 1.0)
    renderer.screen_canvas.blit(
        heatmap,
        (
            int(round(field_center_screen[0] - heatmap.get_width() / 2.0)),
            int(round(field_center_screen[1] - heatmap.get_height() / 2.0)),
        ),
    )
    renderer.blit()
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="SB3 .zip checkpoint (omit for --planner idm or random)")
    p.add_argument("--algo", choices=ALGO_CHOICES, default="auto",
                   help="SB3 algorithm loader. Default auto infers from checkpoint path.")
    p.add_argument("--planner", choices=("rl", "idm", "random"), default="rl")
    p.add_argument("--protocol", choices=tuple(sorted(METADRIVE_PROTOCOLS)),
                   default="matched_social_risk",
                   help="Protocol used to build the watch environment")
    p.add_argument("--view", choices=("3d", "top_down", "none"), default="3d",
                   help="3d = panda3d chase camera window; top_down = pygame bird's-eye")
    p.add_argument("--risk-overlay", choices=RISK_OVERLAY_CHOICES, default="none",
                   help="For --view top_down, draw the live numerical DRIFT risk heatmap on the BEV.")
    p.add_argument("--risk-overlay-alpha", type=int, default=125,
                   help="Opacity 0-255 for the live DRIFT risk overlay.")
    p.add_argument("--risk-overlay-grid-stride", type=int, default=1,
                   help="Downsample the recorded risk grid by N before heatmap smoothing.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--traffic-mode", choices=TRAFFIC_MODE_CHOICES, default="protocol",
                   help=(
                       "Traffic mode override for viewing. Use respawn/basic when "
                       "trigger traffic appears static; keep protocol for matched eval."
                   ))
    p.add_argument("--use-risk", action="store_true", default=None,
                   help="Deprecated override for all risk hooks")
    p.add_argument("--no-risk", action="store_false", dest="use_risk")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--realtime", action="store_true", default=True,
                   help="Sleep to MetaDrive's natural 10 Hz between steps")
    p.add_argument("--no-realtime", action="store_false", dest="realtime")
    p.add_argument("--debug-actions", action="store_true",
                   help="Print selected action and applied control commands")
    p.add_argument("--debug-obs-tail", action="store_true",
                   help="Print the final 8 observation entries, useful for risk-observation policies")
    args = p.parse_args()

    if args.planner == "rl" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required when --planner rl")

    mdcfg = MetaDriveRLConfig()
    if args.risk_overlay == "drift":
        mdcfg.RECORD_GRID_EVERY = 1
    proto = get_metadrive_protocol(args.protocol)
    effective_traffic_mode = proto.traffic_mode if args.traffic_mode == "protocol" else args.traffic_mode
    # Use_render=True opens a panda3d window in 3d mode.
    use_render = (args.view == "3d")
    # Configure the env scenario range so every per-episode seed is valid.
    # MetaDrive asserts seed ∈ [start_seed, start_seed + num_scenarios).
    range_start = int(args.seed)
    range_len = max(int(args.episodes), 10)
    env = make_metadrive_eval_env(
        config=mdcfg,
        protocol=proto,
        use_risk=args.use_risk,
        append_risk_obs=proto.append_risk_obs if args.use_risk is None else None,
        shape_risk_reward=proto.shape_risk_reward if args.use_risk is None else None,
        compute_risk_metrics=True,
        traffic_density=float(args.density),
        traffic_mode=None if args.traffic_mode == "protocol" else args.traffic_mode,
        record_grid=(args.view == "top_down" and args.risk_overlay == "drift"),
        use_render=use_render,
        start_seed=range_start,
        num_scenarios=range_len,
        agent_policy=_native_idm_policy_class() if args.planner == "idm" else None,
        discrete_action=False if args.planner == "idm" else None,
    )
    model = _load_policy(args.checkpoint, args.algo) if args.planner == "rl" else None
    if model is not None:
        model_obs_sig = _space_signature(model.observation_space)
        env_obs_sig = _space_signature(env.observation_space)
        model_act_sig = _space_signature(model.action_space)
        env_act_sig = _space_signature(env.action_space)
        if model_obs_sig != env_obs_sig or model_act_sig != env_act_sig:
            env.close()
            raise SystemExit(
                "Checkpoint/protocol mismatch:\n"
                f"  model obs={model_obs_sig}, env obs={env_obs_sig}\n"
                f"  model act={model_act_sig}, env act={env_act_sig}\n"
                "Choose the protocol used during training."
            )

    # If top_down view, render to a separate pygame window using MetaDrive's
    # built-in top-down renderer.
    top_down_window = (args.view == "top_down")

    print(
        f"\nPlanner: {args.planner.upper()}  | Protocol: {proto.name}  "
        f"| View: {args.view}  | Density: {args.density}  "
        f"| Traffic: {effective_traffic_mode}"
    )
    if args.traffic_mode != "protocol":
        print("Traffic mode is a CLI override; label this as a visualization/stress-test run unless the checkpoint was trained with the same mode.")
    if args.risk_overlay != "none" and not top_down_window:
        print("Risk overlay is only drawn in --view top_down; this run keeps the selected view unchanged.")
    print(f"Press Ctrl-C to quit early.\n")

    try:
        for ep in range(int(args.episodes)):
            seed = int(args.seed) + ep
            obs, info = env.reset(seed=seed)
            print(f"\n=== Episode {ep+1}/{args.episodes} (seed={seed}) ===")
            ep_reward = 0.0
            ep_risk = 0.0
            ep_comfort = 0.0
            peak_r_ego = 0.0
            for step in range(int(args.max_steps)):
                action = _select_action(env, args.planner, model, obs, proto)
                obs_tail = np.asarray(obs).flatten()[-8:] if args.debug_obs_tail else None
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                r_ego = float(info.get("r_ego", 0.0))
                comfort_cost = float(info.get("comfort_cost", 0.0))
                ep_risk += r_ego * 0.1
                ep_comfort += comfort_cost
                peak_r_ego = max(peak_r_ego, r_ego)

                if use_render:
                    # MetaDrive's onscreen render is driven by use_render=True
                    # at construction time; the panda3d window updates on
                    # every env.step(). Calling env.unwrapped.render() (with
                    # no kwarg) lets us inject an overlay text without
                    # tripping gymnasium's Wrapper.render() signature check.
                    env.unwrapped.render(
                        text={"step": step,
                              "r_ego": f"{r_ego:.3f}",
                              "comfort": f"{comfort_cost:.3f}",
                              "reward": f"{reward:.2f}",
                              "speed_kmh": f"{env.unwrapped.agent.speed_km_h:.1f}"}
                    )
                if top_down_window:
                    env.unwrapped.render(
                        mode="top_down",
                        window=True,
                        text={"step": step,
                              "r_ego": f"{r_ego:.3f}",
                              "comfort": f"{comfort_cost:.3f}",
                              "reward": f"{reward:.2f}"},
                    )
                    if args.risk_overlay == "drift":
                        _draw_live_drift_overlay(
                            env,
                            info,
                            mdcfg,
                            alpha=int(np.clip(args.risk_overlay_alpha, 0, 255)),
                            grid_stride=max(1, int(args.risk_overlay_grid_stride)),
                        )

                should_print = step % 25 == 0 or (
                    (args.debug_actions or args.debug_obs_tail) and step < 10
                )
                if should_print:
                    msg = (f"  t={step:4d} reward={reward:+.3f} r_ego={r_ego:.3f} "
                           f"comfort={comfort_cost:.3f} "
                           f"speed={env.unwrapped.agent.speed_km_h:.1f} km/h")
                    if args.debug_actions:
                        msg += (
                            f" action={np.asarray(action).tolist()} "
                            f"steer={float(info.get('control_steer', 0.0)):+.3f} "
                            f"throttle={float(info.get('control_throttle', 0.0)):+.3f} "
                            f"base={float(info.get('base_reward', reward)):+.3f}"
                        )
                    if obs_tail is not None:
                        msg += f" obs_tail={np.round(obs_tail, 3).tolist()}"
                    print(msg)

                if args.realtime:
                    time.sleep(0.10)  # natural 10 Hz

                if terminated or truncated:
                    end = "success" if info.get("arrive_dest") else \
                          "crash"   if info.get("crash_vehicle") else \
                          "OOR"     if info.get("out_of_road") else \
                          "timeout"
                    print(f"  -> {end} at t={step}: "
                          f"ep_reward={ep_reward:.2f} ep_risk_exposure={ep_risk:.2f} "
                          f"ep_comfort_cost={ep_comfort:.2f} "
                          f"peak_r_ego={peak_r_ego:.3f} "
                          f"route_completion={info.get('route_completion', 0.0):.2%}")
                    break
            else:
                print(f"  -> hit max_steps. ep_reward={ep_reward:.2f} ep_comfort_cost={ep_comfort:.2f}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
