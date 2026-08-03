"""Benchmark numerical and PINN field cores on identical held-out snapshots.

The benchmark deliberately separates neural-kernel time from the complete
field-core query.  Cached snapshots begin after scene coefficients have been
projected to the ego-local grid, so these numbers measure propagation/query
cost rather than simulator, perception, rendering, or policy latency.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import platform
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl.risk.pinn_snapshot_cache import CachedSnapshotCollection
from rl.risk.prospective_solver import ProspectiveRiskSolver, ProspectiveSolverConfig
from rl.risk.recurrent_pinn_operator import (
    build_operator_input,
    checkpoint_domain_scales,
    load_recurrent_pinn_checkpoint,
    select_checkpoint_inputs,
)


def _paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(value) for value in sorted(glob.glob(pattern))]
        if not matches:
            raise FileNotFoundError(f"No cache matches {pattern!r}")
        paths.extend(matches)
    return list(dict.fromkeys(paths))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(started: float) -> float:
    return 1000.0 * (time.perf_counter() - started)


def _gpu_load() -> dict[str, float | str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=5).splitlines()[0]
        name, utilization, used, total = (part.strip() for part in line.split(","))
        return {
            "name": name,
            "utilization_percent": float(utilization),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
        }
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        return {
            "name": None,
            "utilization_percent": None,
            "memory_used_mib": None,
            "memory_total_mib": None,
        }


def _selected_examples(collection: CachedSnapshotCollection, frames_per_recording: int):
    examples = []
    for recording in collection.recordings:
        count = min(max(1, int(frames_per_recording)), len(recording))
        for index in np.linspace(0, len(recording) - 1, count, dtype=int):
            examples.append((recording, int(index), recording[int(index)]))
    return examples


def _percentiles(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _benchmark_numerical(examples, solver_spec: dict, *, warmup: int) -> list[dict]:
    recording = examples[0][0]
    solver = ProspectiveRiskSolver(
        x_grid=recording.x_grid,
        y_grid=recording.y_grid,
        config=ProspectiveSolverConfig(**solver_spec),
    )
    snapshot = examples[0][2]
    for _ in range(max(1, int(warmup))):
        solver.solve(
            snapshot["Q"], snapshot["vx"], snapshot["vy"], snapshot["D"],
            road_mask=snapshot["road_mask"],
        )
    rows = []
    for source, index, snapshot in examples:
        solver.solve(
            snapshot["Q"], snapshot["vx"], snapshot["vy"], snapshot["D"],
            road_mask=snapshot["road_mask"],
        )
        kernel_ms = float(solver.last_timing_ms)
        rows.append(
            {
                "backend": "Prospective PDE",
                "device": "cpu",
                "dataset": source.dataset,
                "recording_id": source.recording_id,
                "frame_index": int(index),
                "context_ms": 0.0,
                "transfer_in_ms": 0.0,
                "kernel_ms": kernel_ms,
                "transfer_out_ms": 0.0,
                "postprocess_ms": 0.0,
                "complete_core_ms": kernel_ms,
            }
        )
    return rows


def _benchmark_pinn(
    examples,
    checkpoint_path: str,
    *,
    device_name: str,
    warmup: int,
) -> tuple[list[dict], float]:
    device = torch.device(device_name)
    started = time.perf_counter()
    model, checkpoint = load_recurrent_pinn_checkpoint(
        checkpoint_path, device=str(device)
    )
    recording = examples[0][0]
    warm = torch.zeros(
        (1, int(checkpoint["input_channels"]), len(recording.y_grid), len(recording.x_grid)),
        dtype=torch.float32,
        device=device,
    )
    if device.type == "cpu":
        model = torch.jit.freeze(torch.jit.trace(model, warm, check_trace=False).eval())
    with torch.inference_mode():
        for _ in range(max(1, int(warmup))):
            model(warm)
    _synchronize(device)
    startup_ms = _elapsed_ms(started)

    rows = []
    for source, index, snapshot in examples:
        domain = (
            "highwayenv"
            if str(source.dataset).startswith("highwayenv_")
            else "naturalistic"
        )
        scales = checkpoint_domain_scales(checkpoint, domain)
        total_started = time.perf_counter()

        context_started = time.perf_counter()
        operator_input = build_operator_input(
            snapshot,
            x_grid=source.x_grid,
            y_grid=source.y_grid,
            scales=scales,
        )
        inputs = select_checkpoint_inputs(
            operator_input, checkpoint, domain=domain
        )
        context_ms = _elapsed_ms(context_started)

        transfer_started = time.perf_counter()
        tensor = torch.from_numpy(inputs[None]).to(device)
        _synchronize(device)
        transfer_in_ms = _elapsed_ms(transfer_started)

        kernel_started = time.perf_counter()
        with torch.inference_mode():
            risk_norm, _rate_norm = model(tensor)
        _synchronize(device)
        kernel_ms = _elapsed_ms(kernel_started)

        transfer_started = time.perf_counter()
        risk = risk_norm[0, 0].detach().cpu().numpy()
        _synchronize(device)
        transfer_out_ms = _elapsed_ms(transfer_started)

        post_started = time.perf_counter()
        _risk = (
            np.asarray(risk, dtype=np.float32)
            * float(scales.risk)
            * np.asarray(snapshot["road_mask"], dtype=np.float32)
        )
        postprocess_ms = _elapsed_ms(post_started)
        rows.append(
            {
                "backend": "Context PINN",
                "device": str(device),
                "dataset": source.dataset,
                "recording_id": source.recording_id,
                "frame_index": int(index),
                "context_ms": context_ms,
                "transfer_in_ms": transfer_in_ms,
                "kernel_ms": kernel_ms,
                "transfer_out_ms": transfer_out_ms,
                "postprocess_ms": postprocess_ms,
                "complete_core_ms": _elapsed_ms(total_started),
            }
        )
    return rows, startup_ms


def _summarize(rows: list[dict]) -> list[dict]:
    stages = (
        "context_ms",
        "transfer_in_ms",
        "kernel_ms",
        "transfer_out_ms",
        "postprocess_ms",
        "complete_core_ms",
    )
    groups = sorted(
        {(row["backend"], row["device"]) for row in rows},
        key=lambda item: (
            0 if item[0] == "Prospective PDE" else 1,
            0 if item[1] == "cpu" else 1,
        ),
    )
    summary = []
    for backend, device in groups:
        selected = [
            row for row in rows
            if row["backend"] == backend and row["device"] == device
        ]
        entry = {
            "backend": backend,
            "device": device,
            "n_frames": len(selected),
        }
        for stage in stages:
            stats = _percentiles([row[stage] for row in selected])
            entry.update({f"{stage}_{key}": value for key, value in stats.items()})
        summary.append(entry)
    numerical = next(
        item for item in summary if item["backend"] == "Prospective PDE"
    )
    reference = numerical["complete_core_ms_mean_ms"]
    for entry in summary:
        entry["speedup_vs_pde_mean"] = float(
            reference / max(entry["complete_core_ms_mean_ms"], 1e-12)
        )
    return summary


def _write_latex(path: Path, summary: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired field-core latency after ego-local scene conditioning. "
        r"Values are mean with the 95th percentile in parentheses; batch size is one.}",
        r"\label{tab:pinn-runtime-decomposition}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Backend & Device & Context & Kernel & Transfer & Complete \\",
        r"\midrule",
    ]
    for row in summary:
        transfer_mean = (
            row["transfer_in_ms_mean_ms"] + row["transfer_out_ms_mean_ms"]
        )
        transfer_p95 = (
            row["transfer_in_ms_p95_ms"] + row["transfer_out_ms_p95_ms"]
        )
        lines.append(
            f"{row['backend']} & {row['device'].upper()} & "
            f"{row['context_ms_mean_ms']:.2f} ({row['context_ms_p95_ms']:.2f}) & "
            f"{row['kernel_ms_mean_ms']:.2f} ({row['kernel_ms_p95_ms']:.2f}) & "
            f"{transfer_mean:.2f} ({transfer_p95:.2f}) & "
            f"{row['complete_core_ms_mean_ms']:.2f} "
            f"({row['complete_core_ms_p95_ms']:.2f}) \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, summary: list[dict]) -> None:
    labels = [
        f"{row['backend']}\n{row['device'].upper()}" for row in summary
    ]
    stages = (
        ("context_ms_mean_ms", "Context", "#4C78A8"),
        ("kernel_ms_mean_ms", "Solve / forward", "#F58518"),
        ("transfer_in_ms_mean_ms", "Host-to-device", "#54A24B"),
        ("transfer_out_ms_mean_ms", "Device-to-host", "#E45756"),
        ("postprocess_ms_mean_ms", "Post-process", "#B279A2"),
    )
    x = np.arange(len(summary))
    fig, axis = plt.subplots(figsize=(6.8, 3.5))
    bottom = np.zeros(len(summary), dtype=float)
    for key, label, color in stages:
        values = np.asarray([row[key] for row in summary], dtype=float)
        axis.bar(x, values, bottom=bottom, width=0.64, label=label, color=color)
        bottom += values
    p95 = np.asarray([row["complete_core_ms_p95_ms"] for row in summary])
    axis.scatter(x, p95, marker="_", s=180, linewidth=2, color="black", label="Complete p95")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Latency per field query (ms)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    axis.set_title("Paired field-core runtime, batch size 1", pad=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-globs",
        nargs="+",
        default=["evaluation/pinn_prospective_v2_cache/highwayenv_merge_v0_seed010*"],
    )
    parser.add_argument(
        "--checkpoint",
        default="rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt",
    )
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--frames-per-recording", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="evaluation/pinn_prospective_runtime_v3",
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    collection = CachedSnapshotCollection(_paths(args.cache_globs))
    examples = _selected_examples(collection, args.frames_per_recording)
    solver_specs = [recording.manifest["solver"] for recording in collection.recordings]
    if any(spec != solver_specs[0] for spec in solver_specs[1:]):
        raise RuntimeError("Selected caches use different prospective solvers")

    gpu_before = _gpu_load()
    rows = _benchmark_numerical(examples, solver_specs[0], warmup=args.warmup)
    startup = {}
    for requested in args.devices:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print(f"Skipping unavailable device {requested}", flush=True)
            continue
        values, startup_ms = _benchmark_pinn(
            examples,
            args.checkpoint,
            device_name=requested,
            warmup=args.warmup,
        )
        rows.extend(values)
        startup[requested] = float(startup_ms)
    gpu_after = _gpu_load()
    summary = _summarize(rows)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runtime_frame_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "runtime_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = {
        "scope": "paired field core after ego-local scene conditioning",
        "batch_size": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "caches": collection.recording_keys,
        "frames_per_recording": int(args.frames_per_recording),
        "torch_threads": int(torch.get_num_threads()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu_load_before": gpu_before,
        "gpu_load_after": gpu_after,
        "pinn_startup_ms": startup,
        "summary": summary,
        "interpretation": (
            "Kernel-only values must not be described as end-to-end decision latency. "
            "CUDA timings collected while another GPU job is active are diagnostic."
        ),
    }
    (output / "runtime_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_latex(output / "runtime_table.tex", summary)
    _plot(output / "runtime_decomposition", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
