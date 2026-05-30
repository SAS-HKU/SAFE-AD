"""
MetaDrive offline top-down visualizer
=====================================
Renders one or more planners side-by-side for a single eval episode,
overlaying the DRIFT risk field on a top-down view of the MetaDrive scene.

This script does NOT replay trajectories from disk — that would require a
custom recording format. Instead it re-runs the eval in `record_grid=True`
mode, capturing ego pose, surrounding vehicles, and the ego-relative risk
grid every `step_stride` steps, then renders the captured frames offline
to a single PNG (or one PNG per timestep with --per-step).

Usage
-----
    python rl/visualize_metadrive_comparison.py \
        --planners "risk_ppo@matched_social_risk:checkpoints/.../final.zip,idm@matched_stock" \
        --seed 17 --density 0.1 --max-steps 200 \
        --out rl/logs/metadrive/viz/seed17.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
METADRIVE_ROOT = REPO_ROOT / "metadrive"
if METADRIVE_ROOT.is_dir() and str(METADRIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(METADRIVE_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from rl.config.metadrive_config import MetaDriveRLConfig
from rl.env.metadrive_envs import make_metadrive_eval_env
from rl.risk.metadrive_scene_adapter import enumerate_traffic_vehicles
from rl.eval_metadrive import _parse_planner, _load_policy, _act, PlannerSpec

TRAFFIC_MODE_CHOICES = ("protocol", "trigger", "respawn", "basic", "hybrid")


def _native_idm_policy_class():
    from metadrive.policy.idm_policy import IDMPolicy

    return IDMPolicy


def _collect_frames(planner: PlannerSpec, *, seed: int, density: float,
                    mdcfg: MetaDriveRLConfig, max_steps: int,
                    traffic_mode: str | None = None) -> dict:
    proto = planner.protocol
    env = make_metadrive_eval_env(
        config=mdcfg,
        protocol=proto,
        append_risk_obs=proto.append_risk_obs,
        shape_risk_reward=proto.shape_risk_reward,
        compute_risk_metrics=True,
        traffic_density=float(density),
        traffic_mode=traffic_mode,
        record_grid=True,
        start_seed=int(seed),
        num_scenarios=10,
        agent_policy=_native_idm_policy_class() if planner.kind == "idm" else None,
        discrete_action=False if planner.kind == "idm" else None,
    )
    model = _load_policy(planner)
    frames: list[dict] = []
    try:
        obs, info = env.reset(seed=int(seed))
        raw = env.unwrapped
        for step_idx in range(int(max_steps)):
            ego = raw.agent
            ego_pos = np.asarray(ego.position, dtype=float)
            ego_heading = float(ego.heading_theta)
            traffic = enumerate_traffic_vehicles(env)
            traffic_state = [
                dict(
                    position=np.asarray(v.position, dtype=float).tolist(),
                    heading=float(v.heading_theta),
                    length=float(getattr(v, "LENGTH", 4.5)),
                    width=float(getattr(v, "WIDTH", 2.0)),
                )
                for v in traffic
            ]
            frame = dict(
                step=int(step_idx),
                ego_pos=ego_pos.tolist(),
                ego_heading=ego_heading,
                ego_length=float(getattr(ego, "LENGTH", 4.5)),
                ego_width=float(getattr(ego, "WIDTH", 2.0)),
                traffic=traffic_state,
                grid=info.get("risk_grid"),
                r_ego=float(info.get("r_ego", 0.0)),
                comfort_cost=float(info.get("comfort_cost", 0.0)),
                action_delta=float(info.get("action_delta", 0.0)),
            )
            frames.append(frame)

            action = _act(env, planner, model, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
        if model is not None:
            del model
    grid_axes_x, grid_axes_y = None, None
    return dict(planner=planner.name, frames=frames)


def _draw_panel(ax, frame: dict, mdcfg: MetaDriveRLConfig, *,
                title: str, view_radius: float = 70.0) -> None:
    """Draw one panel: ego-centered, +x forward, +y left. Risk overlay in
    ego-body frame using the wrapper's grid (cfg.x, cfg.y).
    """
    from config import Config as cfg
    ax.set_aspect("equal", adjustable="box")
    ego_x, ego_y = float(frame["ego_pos"][0]), float(frame["ego_pos"][1])
    heading = float(frame["ego_heading"])
    cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))

    # Risk grid overlay (in ego-body frame coords)
    grid = frame.get("grid")
    if grid is not None:
        risk = np.asarray(grid, dtype=np.float32)
        if risk.ndim == 2 and risk.size > 0:
            X_body = cfg.X
            Y_body = cfg.Y
            risk_smooth = gaussian_filter(risk, sigma=1.1)
            vmax = max(0.5, float(np.nanpercentile(risk_smooth, 95)))
            ax.contourf(
                X_body, Y_body, risk_smooth,
                levels=20, cmap="jet", alpha=0.45, vmin=0.0, vmax=vmax,
            )

    # Ego (body frame origin)
    _draw_vehicle_body_frame(ax, x=0.0, y=0.0, heading=0.0,
                              length=float(frame["ego_length"]),
                              width=float(frame["ego_width"]),
                              color="#1f8a4d")

    # Surrounding vehicles transformed into ego body frame
    for veh in frame.get("traffic", []):
        wx = float(veh["position"][0]) - ego_x
        wy = float(veh["position"][1]) - ego_y
        bx = cos_h * wx + sin_h * wy
        by = -sin_h * wx + cos_h * wy
        if abs(bx) > view_radius or abs(by) > view_radius:
            continue
        bheading = float(veh["heading"]) - heading
        _draw_vehicle_body_frame(ax, x=bx, y=by, heading=bheading,
                                  length=float(veh["length"]),
                                  width=float(veh["width"]),
                                  color="#1f4dff")

    ax.set_xlim(-30, 90)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("x [m] (ego body frame)")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"{title} | t={frame['step']} | r_ego={frame['r_ego']:.2f} "
        f"| c={frame.get('comfort_cost', 0.0):.2f}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)


def _draw_vehicle_body_frame(ax, *, x: float, y: float, heading: float,
                              length: float, width: float, color: str) -> None:
    """Draw a rotated rectangle representing one vehicle in ego body frame."""
    corners = np.array([
        [+length / 2, +width / 2],
        [+length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2, +width / 2],
    ], dtype=np.float32)
    c, s = float(np.cos(heading)), float(np.sin(heading))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rotated = corners @ R.T + np.array([x, y], dtype=np.float32)
    ax.fill(rotated[:, 0], rotated[:, 1], color=color, alpha=0.8)
    ax.plot(rotated[[0, 1, 2, 3, 0], 0], rotated[[0, 1, 2, 3, 0], 1],
            color="black", linewidth=0.5)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planners", type=str, required=True,
                   help="Comma-separated planner specs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--density", type=float, default=0.1)
    p.add_argument("--traffic-mode", choices=TRAFFIC_MODE_CHOICES, default="protocol",
                   help="Traffic mode override for top-down visualization")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--step-stride", type=int, default=20,
                   help="Render one frame every N captured steps")
    p.add_argument("--out", type=str, default="rl/logs/metadrive/viz/compare.png")
    args = p.parse_args()

    planners = [_parse_planner(s) for s in args.planners.split(",") if s.strip()]
    mdcfg = MetaDriveRLConfig()
    mdcfg.RECORD_GRID_EVERY = 1

    captures = []
    for planner in planners:
        effective_traffic_mode = planner.protocol.traffic_mode if args.traffic_mode == "protocol" else args.traffic_mode
        print(
            f"Capturing {planner.name} "
            f"(seed={args.seed}, density={args.density}, traffic={effective_traffic_mode})..."
        )
        cap = _collect_frames(
            planner, seed=args.seed, density=args.density,
            mdcfg=mdcfg, max_steps=args.max_steps,
            traffic_mode=None if args.traffic_mode == "protocol" else args.traffic_mode,
        )
        captures.append(cap)

    n_planners = len(captures)
    frame_indices = list(range(0, len(captures[0]["frames"]), int(args.step_stride)))
    if not frame_indices:
        print("No frames captured; aborting.")
        return 1

    n_rows = len(frame_indices)
    fig, axes = plt.subplots(n_rows, n_planners,
                              figsize=(6.0 * n_planners, 3.0 * n_rows),
                              squeeze=False)
    for row, t_idx in enumerate(frame_indices):
        for col, cap in enumerate(captures):
            if t_idx >= len(cap["frames"]):
                axes[row, col].axis("off")
                continue
            frame = cap["frames"][t_idx]
            _draw_panel(axes[row, col], frame, mdcfg, title=cap["planner"])

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
