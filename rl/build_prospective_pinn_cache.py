"""Build restorable prospective-v2 teacher caches from recorded coefficients.

The input caches retain the scene-conditioned source, transport, diffusion,
and ego pose for each sensing frame.  This command reprojects those causal
coefficients to a common ego-local grid and evaluates the finite-horizon
prospective solver.  Calibration and held-out recordings are transformed by
the same fixed operator, while their split remains recording-disjoint.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from rl.risk.pinn_snapshot_cache import CachedRecording, write_snapshot_cache
from rl.risk.prospective_solver import ProspectiveRiskSolver, ProspectiveSolverConfig
from rl.risk.recurrent_pinn_operator import warp_ego_local_field


def _paths(patterns: list[str]) -> list[Path]:
    result: list[Path] = []
    for pattern in patterns:
        matches = [Path(value) for value in sorted(glob.glob(pattern))]
        if not matches:
            raise FileNotFoundError(f"No input cache matches {pattern!r}")
        result.extend(matches)
    return list(dict.fromkeys(result))


def _signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _safe_remove(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    if resolved == root.resolve():
        raise RuntimeError("Refusing to remove cache root")
    if resolved.exists():
        shutil.rmtree(resolved)


def _sample(field, x_grid, y_grid, X, Y, *, fill=0.0) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (np.asarray(y_grid), np.asarray(x_grid)),
        np.asarray(field, dtype=np.float32),
        method="linear",
        bounds_error=False,
        fill_value=float(fill),
    )
    points = np.column_stack([np.asarray(Y).reshape(-1), np.asarray(X).reshape(-1)])
    return interpolator(points).reshape(np.asarray(X).shape).astype(np.float32)


def _pose(snapshot: dict) -> tuple[float, float, float]:
    return (
        float(snapshot.get("ego_world_x", snapshot.get("ego_x", 0.0))),
        float(snapshot.get("ego_world_y", snapshot.get("ego_y", 0.0))),
        float(snapshot.get("ego_world_heading", snapshot.get("ego_heading", 0.0))),
    )


def _to_local(
    snapshot: dict,
    recording: CachedRecording,
    *,
    local_x: np.ndarray,
    local_y: np.ndarray,
    source_coordinate_mode: str,
) -> dict:
    Xl, Yl = np.meshgrid(local_x, local_y)
    pose = _pose(snapshot)
    if source_coordinate_mode == "ego_local":
        Xq, Yq = Xl, Yl
    else:
        c, s = float(np.cos(pose[2])), float(np.sin(pose[2]))
        Xq = pose[0] + c * Xl - s * Yl
        Yq = pose[1] + s * Xl + c * Yl

    fields: dict[str, np.ndarray] = {}
    for key in (
        "Q",
        "D",
        "occ_mask",
        "road_mask",
        "dist_nearest",
        "Q_vehicle",
        "Q_occlusion",
        "Q_merge",
        "Q_behavior_refinement",
    ):
        if key not in snapshot:
            continue
        default = 80.0 if key == "dist_nearest" else 0.0
        fields[key] = _sample(
            snapshot[key],
            recording.x_grid,
            recording.y_grid,
            Xq,
            Yq,
            fill=default,
        )
    if "road_mask" not in fields:
        fields["road_mask"] = np.ones_like(Xl, dtype=np.float32)
    if "occ_mask" not in fields:
        fields["occ_mask"] = np.zeros_like(Xl, dtype=np.float32)
    if "dist_nearest" not in fields:
        fields["dist_nearest"] = np.full_like(Xl, 80.0, dtype=np.float32)

    vx_sampled = _sample(
        snapshot["vx"], recording.x_grid, recording.y_grid, Xq, Yq
    )
    vy_sampled = _sample(
        snapshot["vy"], recording.x_grid, recording.y_grid, Xq, Yq
    )
    if source_coordinate_mode == "ego_local":
        fields["vx"] = vx_sampled
        fields["vy"] = vy_sampled
    else:
        c, s = float(np.cos(pose[2])), float(np.sin(pose[2]))
        rel_vx = vx_sampled - float(snapshot.get("ego_vx", 0.0))
        rel_vy = vy_sampled - float(snapshot.get("ego_vy", 0.0))
        yaw_rate = float(snapshot.get("ego_yaw_rate", 0.0))
        fields["vx"] = (c * rel_vx + s * rel_vy + yaw_rate * Yl).astype(np.float32)
        fields["vy"] = (-s * rel_vx + c * rel_vy - yaw_rate * Xl).astype(np.float32)
    return fields


def build_recording(args, input_path: Path, output_root: Path) -> Path:
    recording = CachedRecording(input_path)
    source_mode = str(recording.manifest.get("coordinate_mode", "native"))
    if source_mode not in {"native", "ego_local"}:
        raise ValueError(f"Unsupported source coordinate mode {source_mode!r}: {input_path}")
    local_x = np.linspace(args.x_min, args.x_max, args.nx, dtype=np.float32)
    local_y = np.linspace(args.y_min, args.y_max, args.ny, dtype=np.float32)
    solver_config = ProspectiveSolverConfig(
        horizon_s=args.horizon_s,
        integration_step_s=args.integration_step_s,
        decay_rate=args.decay_rate,
        transport_scale=args.transport_scale,
        max_diffusion_sigma_cells=args.max_diffusion_sigma_cells,
    ).validated()
    spec = {
        "cache_kind": "prospective_numerical_teacher",
        "teacher_version": ProspectiveRiskSolver.version,
        "source_cache": str(input_path.resolve()),
        "source_cache_signature": str(recording.manifest.get("signature", "")),
        "source_coordinate_mode": source_mode,
        "coordinate_mode": "ego_local",
        "solver": solver_config.to_dict(),
        "local_grid": {
            "x_min": float(args.x_min),
            "x_max": float(args.x_max),
            "y_min": float(args.y_min),
            "y_max": float(args.y_max),
            "nx": int(args.nx),
            "ny": int(args.ny),
        },
    }
    signature = _signature(spec)
    output = output_root / input_path.name
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") and manifest.get("signature") == signature:
            print(f"[prospective-cache] restored {output}")
            return output
        raise RuntimeError(f"Incompatible cache at {output}; pass --rebuild")

    solver = ProspectiveRiskSolver(
        x_grid=local_x,
        y_grid=local_y,
        config=solver_config,
    )
    snapshots: list[dict] = []
    previous_risk = np.zeros(solver.shape, dtype=np.float32)
    previous_rate = np.zeros(solver.shape, dtype=np.float32)
    previous_pose: tuple[float, float, float] | None = None
    for index in range(len(recording)):
        source = recording[index]
        local = _to_local(
            source,
            recording,
            local_x=local_x,
            local_y=local_y,
            source_coordinate_mode=source_mode,
        )
        current_pose = _pose(source)
        if index and previous_pose is not None:
            aligned_previous = warp_ego_local_field(
                previous_risk,
                x_grid=local_x,
                y_grid=local_y,
                previous_pose=previous_pose,
                current_pose=current_pose,
            )
            aligned_rate = warp_ego_local_field(
                previous_rate,
                x_grid=local_x,
                y_grid=local_y,
                previous_pose=previous_pose,
                current_pose=current_pose,
            )
        else:
            aligned_previous = np.zeros(solver.shape, dtype=np.float32)
            aligned_rate = np.zeros(solver.shape, dtype=np.float32)
        risk = solver.solve(
            local["Q"],
            local["vx"],
            local["vy"],
            local["D"],
            road_mask=local["road_mask"],
        )
        dt = max(float(source.get("dt", 0.1)), 1e-4)
        risk_rate = ((risk - aligned_previous) / dt).astype(np.float32)
        scalar = {
            key: value
            for key, value in source.items()
            if np.isscalar(value) and key not in {"recording_id", "dataset", "sequence_end"}
        }
        scalar.update(
            ego_x=0.0,
            ego_y=0.0,
            ego_heading=0.0,
        )
        snapshot = {
            **scalar,
            **local,
            "R": risk,
            "R_prev": aligned_previous,
            "R_t": risk_rate,
            "R_t_prev": aligned_rate,
            "Q_terminal": solver.last_terminal_source.copy(),
        }
        snapshots.append(snapshot)
        previous_risk = risk
        previous_rate = risk_rate
        previous_pose = current_pose
        if (index + 1) % 100 == 0 or index + 1 == len(recording):
            print(
                f"[prospective-cache] {input_path.name}: {index + 1}/{len(recording)}",
                flush=True,
            )

    output_root.mkdir(parents=True, exist_ok=True)
    building = output_root / f".{output.name}.building"
    _safe_remove(building, output_root)
    write_snapshot_cache(
        snapshots=snapshots,
        x_grid=local_x,
        y_grid=local_y,
        output_dir=building,
        metadata={
            **{
                key: value
                for key, value in recording.manifest.items()
                if key
                not in {
                    "cache_version",
                    "complete",
                    "n_snapshots",
                    "grid_shape",
                    "grid_fields",
                    "optional_grid_fields",
                    "scalar_fields",
                    "signature",
                }
            },
            **spec,
            "signature": signature,
            "complete": True,
        },
    )
    if output.exists():
        if not args.rebuild:
            _safe_remove(building, output_root)
            raise FileExistsError(output)
        _safe_remove(output, output_root)
    building.replace(output)
    print(f"[prospective-cache] wrote {output} ({len(snapshots)} snapshots)")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache-globs", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--horizon-s", type=float, default=3.0)
    parser.add_argument("--integration-step-s", type=float, default=0.25)
    parser.add_argument("--decay-rate", type=float, default=0.25)
    parser.add_argument("--transport-scale", type=float, default=1.0)
    parser.add_argument("--max-diffusion-sigma-cells", type=float, default=4.0)
    parser.add_argument("--x-min", type=float, default=-40.0)
    parser.add_argument("--x-max", type=float, default=100.0)
    parser.add_argument("--y-min", type=float, default=-20.0)
    parser.add_argument("--y-max", type=float, default=20.0)
    parser.add_argument("--nx", type=int, default=141)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    outputs = [
        build_recording(args, path, args.output_root)
        for path in _paths(args.input_cache_globs)
    ]
    print(json.dumps({"prospective_caches": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
