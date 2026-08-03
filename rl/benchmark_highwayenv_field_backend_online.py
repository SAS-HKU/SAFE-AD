"""Measure complete numerical and PINN field backends in HighwayEnv.

This benchmark uses a lane-valid IDM ego and excludes rendering, model loading,
and policy inference.  Unlike the cached field-core benchmark, it includes
scene coefficient construction, ego-local projection, query context, field
solve/inference, and interpolation setup performed during each environment
step.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from highway_env.vehicle.behavior import IDMVehicle

from rl.benchmark_prospective_field_runtime import _gpu_load
from rl.env.highwayenv_social_env import (
    make_social_highwayenv_env,
    resolve_traffic_config,
)


def _seeds(value: str) -> list[int]:
    value = value.strip()
    if ":" in value:
        start, stop = (int(item) for item in value.split(":", 1))
        return list(range(start, stop))
    return [int(item) for item in value.split(",") if item.strip()]


def _swap_ego_to_idm(env) -> None:
    raw = env.unwrapped
    old_vehicle = raw.vehicle
    idm_vehicle = IDMVehicle.create_from(old_vehicle)
    index = raw.road.vehicles.index(old_vehicle)
    raw.road.vehicles[index] = idm_vehicle
    raw.controlled_vehicles[0] = idm_vehicle
    raw.vehicle = idm_vehicle
    raw.action_type.controlled_vehicle = idm_vehicle


def _summarize(rows: list[dict]) -> list[dict]:
    timing_keys = (
        "coefficient_ms",
        "context_ms",
        "transfer_in_ms",
        "kernel_ms",
        "transfer_out_ms",
        "postprocess_ms",
        "inference_ms",
        "total_ms",
        "environment_step_ms",
    )
    groups = sorted(
        {(row["backend"], row["device"]) for row in rows},
        key=lambda item: (
            0 if item[0] == "prospective" else 1,
            0 if item[1] == "cpu" else 1,
        ),
    )
    output = []
    for backend, device in groups:
        selected = [
            row for row in rows
            if row["backend"] == backend and row["device"] == device
        ]
        summary = {
            "backend": backend,
            "device": device,
            "n_steps": len(selected),
            "n_offroad": int(sum(not bool(row["on_road"]) for row in selected)),
        }
        for key in timing_keys:
            values = np.asarray([row[key] for row in selected], dtype=float)
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_p95"] = float(np.percentile(values, 95))
        output.append(summary)
    reference = next(row for row in output if row["backend"] == "prospective")
    for row in output:
        row["field_speedup_vs_pde"] = float(
            reference["total_ms_mean"] / max(row["total_ms_mean"], 1e-12)
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="merge-v0")
    parser.add_argument("--traffic-preset", default="medium")
    parser.add_argument("--seeds", type=_seeds, default=_seeds("100:103"))
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--discard-first", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--checkpoint",
        default="rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/pinn_prospective_runtime_online_v3",
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    traffic = resolve_traffic_config(preset=args.traffic_preset)
    configurations = [("prospective", "cpu")]
    configurations.extend(
        ("pinn", device)
        for device in args.devices
        if not device.startswith("cuda") or torch.cuda.is_available()
    )

    gpu_before = _gpu_load()
    rows = []
    for backend, device in configurations:
        for seed in args.seeds:
            env = make_social_highwayenv_env(
                env_id=args.env_id,
                interface="stock",
                render_mode=None,
                traffic=traffic,
                use_drift=True,
                action_mode="default",
                append_risk_obs=False,
                record_risk_metrics=False,
                field_backend=backend,
                pinn_checkpoint=args.checkpoint,
                pinn_device=device,
                drift_warmup_s=0.1,
            )
            try:
                env.reset(seed=seed)
                _swap_ego_to_idm(env)
                for step in range(int(args.steps)):
                    started = time.perf_counter()
                    _obs, _reward, terminated, truncated, info = env.step(1)
                    environment_step_ms = 1000.0 * (time.perf_counter() - started)
                    timing = dict(info.get("field_backend_timing_ms", {}))
                    if step >= int(args.discard_first) and timing:
                        row = {
                            "backend": backend,
                            "device": device,
                            "seed": int(seed),
                            "step": int(step),
                            "on_road": bool(getattr(env.unwrapped.vehicle, "on_road", True)),
                            "environment_step_ms": float(environment_step_ms),
                        }
                        for key in (
                            "coefficient_ms",
                            "context_ms",
                            "transfer_in_ms",
                            "kernel_ms",
                            "transfer_out_ms",
                            "postprocess_ms",
                            "inference_ms",
                            "total_ms",
                        ):
                            row[key] = float(timing.get(key, 0.0))
                        rows.append(row)
                    if terminated or truncated:
                        break
            finally:
                env.close()
            print(f"[{backend}/{device}] seed={seed} complete", flush=True)
    if not rows:
        raise RuntimeError("No online timing samples were collected")

    summary = _summarize(rows)
    gpu_after = _gpu_load()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "online_runtime_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "online_runtime_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    payload = {
        "scope": (
            "complete online field backend; excludes rendering, model loading, "
            "and policy inference"
        ),
        "environment": args.env_id,
        "traffic_preset": args.traffic_preset,
        "controller": "HighwayEnv IDMVehicle",
        "seeds": list(args.seeds),
        "gpu_load_before": gpu_before,
        "gpu_load_after": gpu_after,
        "summary": summary,
    }
    (output / "online_runtime_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
