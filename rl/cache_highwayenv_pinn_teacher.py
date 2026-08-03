"""Cache numerical DRIFT fields from multiple HighwayEnv scenarios and seeds.

Each episode is treated as one recording.  Global numerical fields are sampled
onto a fixed ego-local grid, which gives the surrogate a coordinate frame that
is shared by straight roads, merges, intersections, and roundabouts.  The
recording-disjoint caches can be passed directly to ``train_context_pinn``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from rl.env.highwayenv_social_env import make_social_highwayenv_env, resolve_traffic_config
from rl.risk.pinn_snapshot_cache import write_snapshot_cache
from rl.risk.scene_conditioning import compute_dist_nearest_field, summarize_selected_agents
from rl.visualize_highwayenv_sb3_suite import _swap_ego_to_idm


def _seeds(value: str) -> list[int]:
    value = str(value).strip()
    if ":" in value:
        start, stop = (int(part) for part in value.split(":", 1))
        return list(range(start, stop))
    return [int(part) for part in value.split(",") if part.strip()]


def _find_drift_wrapper(env):
    current = env
    while current is not None:
        if hasattr(current, "get_drift_config") and hasattr(current, "drift"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("DriftOverlayWrapper is absent")


def _local_grid(args) -> SimpleNamespace:
    x = np.linspace(args.x_min, args.x_max, args.nx, dtype=np.float32)
    y = np.linspace(args.y_min, args.y_max, args.ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    return SimpleNamespace(
        x=x,
        y=y,
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        dx=float(x[1] - x[0]),
        dy=float(y[1] - y[0]),
    )


def _local_to_world(local, ego: dict) -> tuple[np.ndarray, np.ndarray]:
    heading = float(ego.get("heading", 0.0))
    c = float(np.cos(heading))
    s = float(np.sin(heading))
    Xw = float(ego["x"]) + c * local.X - s * local.Y
    Yw = float(ego["y"]) + s * local.X + c * local.Y
    return Xw.astype(np.float32), Yw.astype(np.float32)


def _world_to_local_vehicle(vehicle: dict, ego: dict) -> dict:
    heading = float(ego.get("heading", 0.0))
    c = float(np.cos(heading))
    s = float(np.sin(heading))
    dx = float(vehicle["x"]) - float(ego["x"])
    dy = float(vehicle["y"]) - float(ego["y"])
    vx = float(vehicle.get("vx", 0.0))
    vy = float(vehicle.get("vy", 0.0))
    result = dict(vehicle)
    result.update(
        x=c * dx + s * dy,
        y=-s * dx + c * dy,
        vx=c * vx + s * vy,
        vy=-s * vx + c * vy,
        heading=float(vehicle.get("heading", 0.0)) - heading,
    )
    return result


def _sample_grid(field, cfg, Xw, Yw, *, fill_value=0.0) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (np.asarray(cfg.y), np.asarray(cfg.x)),
        np.asarray(field, dtype=np.float32),
        bounds_error=False,
        fill_value=float(fill_value),
    )
    points = np.column_stack([Yw.reshape(-1), Xw.reshape(-1)])
    return interpolator(points).reshape(Xw.shape).astype(np.float32)


def _capture_snapshot(
    wrapper,
    local,
    *,
    time_s: float,
    frame_id: int,
    previous_global_R: np.ndarray | None = None,
    previous_global_R_t: np.ndarray | None = None,
    ego_yaw_rate: float = 0.0,
) -> dict:
    cfg = wrapper.get_drift_config()
    ego, vehicles = wrapper._collect_drift_state()
    backend = wrapper.drift
    if backend.last_Q is None:
        raise RuntimeError("Numerical coefficients are unavailable after warm-up")
    Xw, Yw = _local_to_world(local, ego)

    R = _sample_grid(backend.risk_field, cfg, Xw, Yw)
    R_prev = (
        R.copy()
        if previous_global_R is None
        else _sample_grid(previous_global_R, cfg, Xw, Yw)
    )
    delegate = getattr(backend, "_delegate", None)
    solver = getattr(delegate, "solver", None)
    current_global_R_t = np.asarray(
        getattr(solver, "R_t", np.zeros_like(backend.risk_field)), dtype=np.float32
    )
    R_t = _sample_grid(current_global_R_t, cfg, Xw, Yw)
    R_t_prev = (
        R_t.copy()
        if previous_global_R_t is None
        else _sample_grid(previous_global_R_t, cfg, Xw, Yw)
    )
    Q = _sample_grid(backend.last_Q, cfg, Xw, Yw)
    vx_world = _sample_grid(backend.last_vx, cfg, Xw, Yw)
    vy_world = _sample_grid(backend.last_vy, cfg, Xw, Yw)
    D = _sample_grid(backend.last_D, cfg, Xw, Yw)
    heading = float(ego.get("heading", 0.0))
    c = float(np.cos(heading))
    s = float(np.sin(heading))
    relative_vx = vx_world - float(ego.get("vx", 0.0))
    relative_vy = vy_world - float(ego.get("vy", 0.0))
    vx = (c * relative_vx + s * relative_vy + float(ego_yaw_rate) * local.Y).astype(
        np.float32
    )
    vy = (-s * relative_vx + c * relative_vy - float(ego_yaw_rate) * local.X).astype(
        np.float32
    )

    _q_total, _q_vehicle, _q_occ, occ_world = wrapper._source_fn(
        vehicles, ego, cfg.X, cfg.Y
    )
    occ_mask = _sample_grid(occ_world, cfg, Xw, Yw)
    global_road_mask = wrapper.get_road_mask()
    road_mask = (
        np.ones_like(R, dtype=np.float32)
        if global_road_mask is None
        else _sample_grid(global_road_mask, cfg, Xw, Yw)
    )

    selected = summarize_selected_agents(
        ego=ego,
        vehicles=vehicles,
        perception_range=80.0,
        selection_mode="soft_topk",
        top_k=5,
        threshold_ratio=0.15,
    )
    local_agents = [
        _world_to_local_vehicle(vehicle, ego)
        for vehicle in selected["selected_agents"]
    ]
    dist_nearest = compute_dist_nearest_field(
        local.X, local.Y, local_agents, fill_value=80.0
    )
    ego_local = _world_to_local_vehicle(ego, ego)
    return {
        "R": R,
        "R_prev": R_prev,
        "R_t": R_t,
        "R_t_prev": R_t_prev,
        "Q": Q,
        "vx": vx,
        "vy": vy,
        "D": D,
        "occ_mask": occ_mask,
        "road_mask": road_mask,
        "dist_nearest": dist_nearest,
        "t": float(time_s),
        "dt": 0.0,
        "frame_id": int(frame_id),
        "N_agents": float(selected["N_agents_selected"]),
        "truck_presence": float(selected["truck_presence"]),
        "occlusion_score": float(selected["occlusion_score"]),
        "selection_mass": float(selected["mass_retained"]),
        "ego_x": 0.0,
        "ego_y": 0.0,
        "ego_ax": float(ego_local.get("a", 0.0)),
        "ego_v_lat": float(ego_local.get("vy", 0.0)),
        "ego_heading": 0.0,
        "ego_vx": float(ego_local.get("vx", 0.0)),
        "ego_vy": float(ego_local.get("vy", 0.0)),
        "ego_trackId": 0,
        "ego_world_x": float(ego["x"]),
        "ego_world_y": float(ego["y"]),
        "ego_world_heading": float(ego.get("heading", 0.0)),
        "ego_yaw_rate": float(ego_yaw_rate),
    }


def _signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _remove_cache(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    if resolved == root.resolve():
        raise RuntimeError("Refusing to remove the cache root")
    if resolved.exists():
        shutil.rmtree(resolved)


def cache_episode(args, env_id: str, seed: int, local, traffic) -> Path:
    slug = env_id.replace("-", "_")
    dataset = f"highwayenv_{slug}"
    recording_id = f"seed{seed:04d}"
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"{dataset}_{recording_id}"
    spec = {
        "cache_kind": "highwayenv_numerical_teacher",
        "dataset": dataset,
        "recording_id": recording_id,
        "environment": env_id,
        "seed": int(seed),
        "planner": "idm",
        "traffic": traffic.to_dict(),
        "max_steps": int(args.max_steps),
        "drift_warmup_s": float(args.drift_warmup_s),
        "local_grid": {
            "x_min": float(args.x_min),
            "x_max": float(args.x_max),
            "y_min": float(args.y_min),
            "y_max": float(args.y_max),
            "nx": int(args.nx),
            "ny": int(args.ny),
        },
        "coordinate_mode": "ego_local",
        "transport_frame": "ego_relative_rotating_frame",
        "temporal_alignment": "previous_world_field_sampled_in_current_ego_frame",
    }
    signature = _signature(spec)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature and manifest.get("complete"):
            print(f"[cache] restored {output}")
            return output
        raise RuntimeError(f"Incompatible cache at {output}; use --rebuild")

    env = make_social_highwayenv_env(
        env_id=env_id,
        interface="stock",
        render_mode=None,
        traffic=traffic,
        use_drift=True,
        drift_warmup_s=float(args.drift_warmup_s),
        record_risk_metrics=False,
        field_backend="numerical",
    )
    try:
        env.reset(seed=int(seed))
        _swap_ego_to_idm(env)
        wrapper = _find_drift_wrapper(env)
        raw = env.unwrapped
        dt = 1.0 / float(raw.config["policy_frequency"])
        snapshots = [
            _capture_snapshot(
                wrapper,
                local,
                time_s=float(args.drift_warmup_s),
                frame_id=0,
            )
        ]
        snapshots[0]["dt"] = dt
        for step in range(1, int(args.max_steps) + 1):
            previous_global_R = np.asarray(wrapper.drift.risk_field, dtype=np.float32).copy()
            delegate = getattr(wrapper.drift, "_delegate", None)
            solver = getattr(delegate, "solver", None)
            previous_global_R_t = np.asarray(
                getattr(solver, "R_t", np.zeros_like(previous_global_R)), dtype=np.float32
            ).copy()
            previous_ego, _previous_vehicles = wrapper._collect_drift_state()
            _obs, _reward, terminated, truncated, _info = env.step(1)
            current_ego, _current_vehicles = wrapper._collect_drift_state()
            heading_delta = np.arctan2(
                np.sin(float(current_ego.get("heading", 0.0)) - float(previous_ego.get("heading", 0.0))),
                np.cos(float(current_ego.get("heading", 0.0)) - float(previous_ego.get("heading", 0.0))),
            )
            snapshot = _capture_snapshot(
                wrapper,
                local,
                time_s=float(args.drift_warmup_s) + step * dt,
                frame_id=step,
                previous_global_R=previous_global_R,
                previous_global_R_t=previous_global_R_t,
                ego_yaw_rate=float(heading_delta / max(dt, 1e-6)),
            )
            snapshot["dt"] = dt
            snapshots.append(snapshot)
            if terminated or truncated:
                break
    finally:
        env.close()

    building = root / f".{output.name}.building"
    _remove_cache(building, root)
    write_snapshot_cache(
        snapshots=snapshots,
        x_grid=local.x,
        y_grid=local.y,
        output_dir=building,
        metadata={**spec, "signature": signature},
    )
    if output.exists():
        if not args.rebuild:
            _remove_cache(building, root)
            raise FileExistsError(output)
        _remove_cache(output, root)
    building.replace(output)
    print(f"[cache] wrote {output} ({len(snapshots)} snapshots)")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-ids",
        nargs="+",
        default=["highway-v0", "merge-v0", "roundabout-v0", "intersection-v0"],
    )
    parser.add_argument("--seeds", type=_seeds, default=_seeds("0:5"))
    parser.add_argument("--traffic-preset", default="medium")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--drift-warmup-s", type=float, default=2.0)
    parser.add_argument("--cache-dir", default="evaluation/pinn_highway_teacher_cache")
    parser.add_argument("--x-min", type=float, default=-40.0)
    parser.add_argument("--x-max", type=float, default=100.0)
    parser.add_argument("--y-min", type=float, default=-20.0)
    parser.add_argument("--y-max", type=float, default=20.0)
    parser.add_argument("--nx", type=int, default=141)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    local = _local_grid(args)
    traffic = resolve_traffic_config(preset=args.traffic_preset)
    paths = []
    for env_id in args.env_ids:
        for seed in args.seeds:
            paths.append(cache_episode(args, env_id, seed, local, traffic))
    print(json.dumps({"cached_recordings": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
