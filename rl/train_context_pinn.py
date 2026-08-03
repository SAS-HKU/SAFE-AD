"""Train and validate a recording-disjoint, context-conditioned DRIFT PINN.

The command is intentionally staged.  Numerical teacher snapshots are cached
per recording, calibration recordings are used for optimization, and every
held-out recording is restored from disk for validation.  Time is reset within
each recording and normalized against an explicit deployment horizon.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config as TeacherConfig
from pde_solver import PDESolver
from pinn_risk_field import FlatSampleCache, Normalizer, PINNTrainer
from rl.risk.pinn_adapter import PINNRiskAdapter
from rl.risk.pinn_snapshot_cache import (
    CachedSnapshotCollection,
    cache_teacher_recording,
)


def _recording_ids(value: str) -> list[str]:
    ids = []
    for part in str(value).split(","):
        recording_id = part.strip()
        if not recording_id:
            continue
        if recording_id.isdigit():
            recording_id = recording_id.zfill(2)
        ids.append(recording_id)
    if not ids:
        raise argparse.ArgumentTypeError("At least one recording ID is required")
    return ids


def _paths(cache_root: Path, dataset: str, recording_ids: list[str]) -> list[Path]:
    return [cache_root / f"{dataset}_{recording_id}" for recording_id in recording_ids]


def _expand_cache_globs(patterns: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns or []:
        matches = sorted(Path(path) for path in glob.glob(str(pattern)))
        if not matches:
            raise FileNotFoundError(f"Cache glob matched no paths: {pattern}")
        paths.extend(matches)
    return list(dict.fromkeys(paths))


def _hardware() -> dict:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
        "torch_threads": int(torch.get_num_threads()),
    }


def cache_recordings(args, recording_ids: list[str]) -> list[Path]:
    cached = []
    for recording_id in recording_ids:
        path = cache_teacher_recording(
            data_root=args.data_root,
            dataset=args.dataset,
            recording_id=recording_id,
            cache_root=args.cache_dir,
            max_seconds=args.max_sec,
            warmup_seconds=args.warmup_sec,
            perception_range=args.perception_range,
            selection_mode=args.selection_mode,
            top_k=args.top_k,
            threshold_ratio=args.threshold_ratio,
            store_source_components=True,
            rebuild=args.rebuild_cache,
        )
        cached.append(path)
        print(f"[cache] ready: {path}")
    return cached


def _load_sample_cache(
    collection: CachedSnapshotCollection,
    *,
    cache_dir: Path,
    points_per_snapshot: int,
):
    cache_schema = "spatial-context-v1"
    signature = "_".join(collection.signatures)
    tag = hashlib.sha256(
        f"{signature}|{int(points_per_snapshot)}|{cache_schema}".encode("utf-8")
    ).hexdigest()[:16]
    data_path = cache_dir / f"sample_points_{tag}.npy"
    meta_path = cache_dir / f"sample_points_{tag}.json"
    if data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("signatures") == collection.signatures
            and meta.get("cache_schema") == cache_schema
        ):
            return FlatSampleCache.load(
                str(data_path), collection.x_grid, collection.y_grid, collection.times
            )

    cache = FlatSampleCache(
        collection,
        collection.x_grid,
        collection.y_grid,
        pts_per_snap=int(points_per_snapshot),
        seed=0,
    )
    cache.save(str(data_path))
    meta_path.write_text(
        json.dumps(
            {
                "signatures": collection.signatures,
                "recordings": collection.recording_keys,
                "points_per_snapshot": int(points_per_snapshot),
                "cache_schema": cache_schema,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cache


def _make_trainer(args, calibration: CachedSnapshotCollection, checkpoint=None):
    norm = Normalizer(calibration, calibration.x_grid, calibration.y_grid)
    observed_max = float(np.max(calibration.times))
    if observed_max > float(args.deployment_horizon) + 1e-5:
        raise ValueError(
            f"Calibration snapshots reach {observed_max:.3f}s, beyond the requested "
            f"deployment horizon {args.deployment_horizon:.3f}s"
        )
    norm.ranges["t"] = (0.0, float(args.deployment_horizon))

    if checkpoint is not None:
        norm.ranges = dict(checkpoint["norm_ranges"])
        hidden = int(checkpoint.get("hidden", args.hidden))
        depth = int(checkpoint.get("depth", args.depth))
        use_rff = bool(checkpoint.get("use_rff", args.use_rff))
        rff_features = int(checkpoint.get("rff_features", args.rff_features))
        rff_scale = float(checkpoint.get("rff_scale", args.rff_scale))
        use_context = bool(checkpoint.get("use_context", True))
        use_spatial_context = bool(checkpoint.get("use_spatial_context", False))
        use_distance_rbf = bool(checkpoint.get("use_distance_rbf", False))
        rff_include_raw = bool(checkpoint.get("rff_include_raw", False))
        output_bias_init = float(checkpoint.get("output_bias_init", 0.0))
        risk_sample_fraction = float(
            checkpoint.get("risk_sample_fraction", args.risk_sample_fraction)
        )
        risk_loss_boost = float(checkpoint.get("risk_loss_boost", args.risk_loss_boost))
        risk_halo_boost = float(checkpoint.get("risk_halo_boost", args.risk_halo_boost))
    else:
        hidden = args.hidden
        depth = args.depth
        use_rff = args.use_rff
        rff_features = args.rff_features
        rff_scale = args.rff_scale
        use_context = True
        use_spatial_context = args.spatial_context
        use_distance_rbf = args.distance_rbf_context
        rff_include_raw = args.rff_include_raw
        output_bias_init = args.output_bias_init
        risk_sample_fraction = args.risk_sample_fraction
        risk_loss_boost = args.risk_loss_boost
        risk_halo_boost = args.risk_halo_boost

    point_cache = _load_sample_cache(
        calibration,
        cache_dir=Path(args.cache_dir),
        points_per_snapshot=args.points_per_snapshot,
    )
    context_features = ["N_agents", "dist_nearest"]
    if use_distance_rbf:
        context_features.extend(["dist_rbf_2m", "dist_rbf_6m", "dist_rbf_15m"])
    if use_spatial_context:
        context_features.extend(["dist_dx", "dist_dy"])

    metadata = {
        "model_role": str(args.model_role),
        "dataset": args.dataset,
        "calibration_recordings": calibration.recording_keys,
        "heldout_recordings": list(args.heldout_recording_keys),
        "split_mode": "recording_disjoint",
        "coordinate_mode": str(args.coordinate_mode),
        "time_mode": "recording_relative",
        "deployment_horizon_s": float(args.deployment_horizon),
        "context_features": context_features,
        "rff_include_raw": bool(rff_include_raw),
        "output_bias_init": float(output_bias_init),
        "teacher_cache_signatures": calibration.signatures,
        "risk_balancing": {
            "focus_fraction": float(risk_sample_fraction),
            "risk_loss_boost": float(risk_loss_boost),
            "risk_halo_boost": float(risk_halo_boost),
            "risk_threshold": 0.05,
            "source_threshold": 0.05,
        },
        "loss_weights": {
            "data": args.w_data,
            "physics": args.w_phys,
            "initial": args.w_ic,
            "boundary": args.w_bc,
            "smooth": args.w_smooth,
            "gradient": args.w_grad,
            "temporal": args.w_temp,
            "behavior_longitudinal": args.w_beh_long,
            "behavior_lateral": args.w_beh_lat,
        },
        "hardware": _hardware(),
    }
    trainer = PINNTrainer(
        snapshots=calibration,
        norm=norm,
        interp=point_cache,
        hidden=hidden,
        depth=depth,
        use_rff=use_rff,
        rff_features=rff_features,
        rff_scale=rff_scale,
        use_context=use_context,
        use_spatial_context=use_spatial_context,
        use_distance_rbf=use_distance_rbf,
        rff_include_raw=rff_include_raw,
        output_bias_init=output_bias_init,
        device=args.device,
        w_data=args.w_data,
        w_phys=args.w_phys,
        w_ic=args.w_ic,
        w_bc=args.w_bc,
        w_smooth=args.w_smooth,
        w_grad=args.w_grad,
        w_temp=args.w_temp,
        w_beh_long=args.w_beh_long,
        w_beh_lat=args.w_beh_lat,
        n_colloc=args.n_colloc,
        n_data=args.n_data,
        risk_sample_fraction=risk_sample_fraction,
        risk_loss_boost=risk_loss_boost,
        risk_halo_boost=risk_halo_boost,
        selection_mode=args.selection_mode,
        top_k=args.top_k,
        threshold_ratio=args.threshold_ratio,
        checkpoint_metadata=metadata,
    )
    if checkpoint is not None:
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.history = dict(checkpoint.get("history", trainer.history))
        trainer.checkpoint_metadata = dict(checkpoint.get("metadata", metadata))
        trainer._optimizer_state = checkpoint.get("optimizer_state")
        trainer._scheduler_state = checkpoint.get("scheduler_state")
    return trainer


def _write_training_plot(history: dict[str, list[float]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for key, label in (
        ("loss", "Total"),
        ("L_data", "Data"),
        ("L_phys", "Physics"),
        ("L_grad", "Gradient"),
        ("L_temp", "Temporal"),
    ):
        values = np.asarray(history.get(key, []), dtype=float)
        if values.size:
            ax.plot(np.arange(1, values.size + 1), np.maximum(values, 1e-12), label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pinn_training_losses.png", dpi=300)
    fig.savefig(output_dir / "pinn_training_losses.pdf")
    plt.close(fig)


def train(args, calibration: CachedSnapshotCollection) -> PINNTrainer:
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and not args.resume:
        raise FileExistsError(
            f"Checkpoint exists: {model_path}. Use --resume or choose a new --model-out."
        )
    checkpoint = (
        torch.load(model_path, map_location=args.device, weights_only=False)
        if model_path.exists() and args.resume
        else None
    )
    trainer = _make_trainer(args, calibration, checkpoint=checkpoint)

    def save_progress(epoch: int) -> None:
        trainer.checkpoint_metadata.update(
            {
                "status": "in_progress",
                "last_completed_epoch_this_run": int(epoch),
                "history_epochs": int(len(trainer.history.get("loss", []))),
            }
        )
        temporary = model_path.with_suffix(model_path.suffix + ".tmp")
        trainer.save(str(temporary))
        os.replace(temporary, model_path)

    started = time.perf_counter()
    trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        print_every=args.print_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=save_progress,
        optimizer_state=(checkpoint or {}).get("optimizer_state"),
        scheduler_state=(checkpoint or {}).get("scheduler_state"),
    )
    trainer.checkpoint_metadata["training_wall_time_s"] = float(time.perf_counter() - started)
    trainer.checkpoint_metadata["epochs_this_run"] = int(args.epochs)
    trainer.checkpoint_metadata["history_epochs"] = int(len(trainer.history.get("loss", [])))
    trainer.checkpoint_metadata["status"] = "complete"
    trainer.save(str(model_path.with_suffix(model_path.suffix + ".tmp")))
    os.replace(model_path.with_suffix(model_path.suffix + ".tmp"), model_path)
    _write_training_plot(trainer.history, Path(args.output_dir))
    return trainer


def _field_metrics(pred: np.ndarray, true: np.ndarray, dx: float, dy: float) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    error = pred - true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    teacher_energy = float(np.sum(true**2))
    error_energy = float(np.sum(error**2))
    relative_l2 = (
        float(np.sqrt(error_energy / teacher_energy))
        if teacher_energy > 1e-10
        else float("nan")
    )
    if np.std(pred) > 1e-12 and np.std(true) > 1e-12:
        correlation = float(np.corrcoef(pred.reshape(-1), true.reshape(-1))[0, 1])
    else:
        correlation = float("nan")

    true_gy, true_gx = np.gradient(true, dy, dx)
    pred_gy, pred_gx = np.gradient(pred, dy, dx)
    gradient_error = np.sqrt((pred_gx - true_gx) ** 2 + (pred_gy - true_gy) ** 2)
    gradient_teacher_energy = float(np.sum(true_gx**2 + true_gy**2))
    gradient_error_energy = float(np.sum(gradient_error**2))
    gradient_relative_l2 = (
        float(np.sqrt(gradient_error_energy / gradient_teacher_energy))
        if gradient_teacher_energy > 1e-10
        else float("nan")
    )
    true_mag = np.sqrt(true_gx**2 + true_gy**2)
    active = (true_mag > 1e-8) & (true_mag >= np.percentile(true_mag, 75.0))
    dot = pred_gx * true_gx + pred_gy * true_gy
    denom = np.sqrt(pred_gx**2 + pred_gy**2) * true_mag
    cosine = np.clip(dot / np.maximum(denom, 1e-12), -1.0, 1.0)
    gradient_angle_deg = (
        float(np.degrees(np.arccos(cosine[active])).mean())
        if np.any(active)
        else float("nan")
    )

    if float(np.max(true)) > 1e-6:
        threshold = max(float(np.percentile(true, 90.0)), 0.05 * float(np.max(true)))
        true_hot = true >= threshold
        hot_count = int(true_hot.sum())
        pred_hot = np.zeros_like(true_hot)
        if hot_count > 0:
            top_indices = np.argpartition(pred.reshape(-1), -hot_count)[-hot_count:]
            pred_hot.reshape(-1)[top_indices] = True
        union = np.logical_or(true_hot, pred_hot).sum()
        hotspot_iou = float(np.logical_and(true_hot, pred_hot).sum() / max(union, 1))
    else:
        hotspot_iou = float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "relative_l2": relative_l2,
        "correlation": correlation,
        "gradient_relative_l2": gradient_relative_l2,
        "gradient_angle_deg": gradient_angle_deg,
        "hotspot_iou": hotspot_iou,
        "teacher_energy": teacher_energy,
        "field_error_energy": error_energy,
        "gradient_teacher_energy": gradient_teacher_energy,
        "gradient_error_energy": gradient_error_energy,
    }


def _plot_validation_panel(
    *,
    snapshot: dict,
    pred: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    title: str,
    output: Path,
) -> None:
    true = np.asarray(snapshot["R"])
    vmax = max(float(np.percentile(true, 99.0)), float(np.percentile(pred, 99.0)), 1e-6)
    error = np.abs(pred - true)
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), constrained_layout=True)
    images = [
        axes[0].imshow(true, origin="lower", extent=extent, cmap="turbo", vmin=0.0, vmax=vmax),
        axes[1].imshow(pred, origin="lower", extent=extent, cmap="turbo", vmin=0.0, vmax=vmax),
        axes[2].imshow(error, origin="lower", extent=extent, cmap="magma", vmin=0.0),
    ]
    for ax, panel in zip(axes, ("Numerical teacher", "Context PINN", "Absolute error")):
        ax.set_title(panel, fontsize=9)
        ax.set_xlabel("$x$ (m)")
    axes[0].set_ylabel("$y$ (m)")
    fig.colorbar(images[1], ax=axes[:2], shrink=0.78, label="Risk intensity")
    fig.colorbar(images[2], ax=axes[2], shrink=0.78, label="Absolute error")
    fig.suptitle(title, fontsize=10)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def validate(
    args,
    trainer: PINNTrainer,
    heldout: CachedSnapshotCollection,
) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    rows: list[dict[str, object]] = []
    plot_budget = int(args.validation_plots)
    old_snaps = trainer.snaps
    trainer.snaps = heldout
    dx = float(heldout.x_grid[1] - heldout.x_grid[0])
    dy = float(heldout.y_grid[1] - heldout.y_grid[0])
    sparse_adapter = PINNRiskAdapter(
        checkpoint_path=args.model_out,
        device=args.device,
        inference_x_range=(float(heldout.x_grid.min()), float(heldout.x_grid.max())),
        inference_y_range=(float(heldout.y_grid.min()), float(heldout.y_grid.max())),
        time_mode="error",
    )
    if not sparse_adapter.available:
        raise RuntimeError(f"Unable to load trained checkpoint: {args.model_out}")

    first_snapshot = heldout[0]
    validation_grid = SimpleNamespace(x=heldout.x_grid, y=heldout.y_grid)
    first_lane_centers = [
        float(first_snapshot["ego_y"]) - 3.5,
        float(first_snapshot["ego_y"]),
        float(first_snapshot["ego_y"]) + 3.5,
    ]
    sparse_adapter.query_risk_features(
        ego_x=float(first_snapshot["ego_x"]),
        ego_y=float(first_snapshot["ego_y"]),
        t=float(first_snapshot["t"]),
        Q_grid=first_snapshot["Q"],
        vx_grid=first_snapshot["vx"],
        vy_grid=first_snapshot["vy"],
        D_grid=first_snapshot["D"],
        sim_cfg=validation_grid,
        lane_centers=first_lane_centers,
        current_lane=1,
        N_agents=int(first_snapshot["N_agents"]),
        dist_nearest=first_snapshot["dist_nearest"],
    )

    offset = 0
    for recording_index, recording in enumerate(heldout.recordings):
        if int(args.validation_frames) <= 0:
            frame_indices = np.arange(len(recording), dtype=int)
        else:
            frame_indices = np.linspace(
                0,
                len(recording) - 1,
                min(int(args.validation_frames), len(recording)),
                dtype=int,
            )
        unique_frame_indices = np.unique(frame_indices)
        remaining_recordings = len(heldout.recordings) - recording_index
        n_recording_plots = (
            min(
                len(unique_frame_indices),
                max(1, plot_budget // max(1, remaining_recordings)),
            )
            if plot_budget > 0
            else 0
        )
        plot_indices = set(
            np.linspace(
                0,
                len(unique_frame_indices) - 1,
                n_recording_plots,
                dtype=int,
            ).tolist()
        )
        for frame_position, local_index in enumerate(unique_frame_indices):
            global_index = offset + int(local_index)
            snapshot = heldout[global_index]
            started = time.perf_counter()
            pred = trainer.predict_field(global_index)
            query_ms = 1000.0 * (time.perf_counter() - started)

            numerical_X, numerical_Y = np.meshgrid(
                np.asarray(heldout.x_grid), np.asarray(heldout.y_grid)
            )
            numerical_config = SimpleNamespace(
                X=numerical_X,
                Y=numerical_Y,
                x=np.asarray(heldout.x_grid),
                y=np.asarray(heldout.y_grid),
                dx=dx,
                dy=dy,
                tau=float(TeacherConfig.tau),
                lambda_decay=float(TeacherConfig.lambda_decay),
                L_decay=float(TeacherConfig.L_decay),
                sponge_length=float(getattr(TeacherConfig, "sponge_length", 0.0)),
                lambda_sponge=float(getattr(TeacherConfig, "lambda_sponge", 0.0)),
                post_smooth_sigma=float(
                    getattr(TeacherConfig, "post_smooth_sigma", 0.0)
                ),
            )
            numerical_solver = PDESolver(config=numerical_config)
            numerical_solver.R = np.asarray(snapshot["R"], dtype=np.float64).copy()
            started = time.perf_counter()
            numerical_substeps = max(1, int(args.numerical_substeps))
            for _ in range(numerical_substeps):
                numerical_solver.step(
                    snapshot["Q"],
                    snapshot["D"],
                    snapshot["vx"],
                    snapshot["vy"],
                    dt=float(snapshot["dt"]) / numerical_substeps,
                )
            numerical_ms = 1000.0 * (time.perf_counter() - started)

            lane_centers = [
                float(snapshot["ego_y"]) - 3.5,
                float(snapshot["ego_y"]),
                float(snapshot["ego_y"]) + 3.5,
            ]
            started = time.perf_counter()
            sparse_adapter.query_risk_features(
                ego_x=float(snapshot["ego_x"]),
                ego_y=float(snapshot["ego_y"]),
                t=float(snapshot["t"]),
                Q_grid=snapshot["Q"],
                vx_grid=snapshot["vx"],
                vy_grid=snapshot["vy"],
                D_grid=snapshot["D"],
                sim_cfg=validation_grid,
                lane_centers=lane_centers,
                current_lane=1,
                N_agents=int(snapshot["N_agents"]),
                dist_nearest=snapshot["dist_nearest"],
            )
            sparse_ms = 1000.0 * (time.perf_counter() - started)
            metrics = _field_metrics(pred, snapshot["R"], dx=dx, dy=dy)
            rows.append(
                {
                    "dataset": recording.dataset,
                    "recording_id": recording.recording_id,
                    "frame_id": int(snapshot["frame_id"]),
                    "time_s": float(snapshot["t"]),
                    "pinn_full_grid_ms": query_ms,
                    "pinn_sparse_descriptor_ms": sparse_ms,
                    "numerical_pde_step_ms": numerical_ms,
                    **metrics,
                }
            )
            if plot_budget > 0 and frame_position in plot_indices:
                stem = f"heldout_{recording.dataset}_{recording.recording_id}_frame_{int(snapshot['frame_id'])}"
                _plot_validation_panel(
                    snapshot=snapshot,
                    pred=pred,
                    x_grid=heldout.x_grid,
                    y_grid=heldout.y_grid,
                    title=f"Held-out {recording.dataset} recording {recording.recording_id}",
                    output=output_dir / f"{stem}.png",
                )
                plot_budget -= 1
        offset += len(recording)

    trainer.snaps = old_snaps
    csv_path = output_dir / "heldout_pinn_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        "rmse",
        "mae",
        "relative_l2",
        "correlation",
        "gradient_relative_l2",
        "gradient_angle_deg",
        "hotspot_iou",
        "pinn_full_grid_ms",
        "pinn_sparse_descriptor_ms",
        "numerical_pde_step_ms",
    ]

    def _summarize(selected):
        result = {
            metric: float(np.nanmean([float(row[metric]) for row in selected]))
            for metric in metric_keys
        }
        teacher_energy = float(sum(float(row["teacher_energy"]) for row in selected))
        field_error_energy = float(sum(float(row["field_error_energy"]) for row in selected))
        gradient_teacher_energy = float(
            sum(float(row["gradient_teacher_energy"]) for row in selected)
        )
        gradient_error_energy = float(
            sum(float(row["gradient_error_energy"]) for row in selected)
        )
        result["relative_l2"] = (
            float(np.sqrt(field_error_energy / teacher_energy))
            if teacher_energy > 1e-10
            else float("nan")
        )
        result["gradient_relative_l2"] = (
            float(np.sqrt(gradient_error_energy / gradient_teacher_energy))
            if gradient_teacher_energy > 1e-10
            else float("nan")
        )
        result["nontrivial_structural_frames"] = int(
            sum(np.isfinite(float(row["correlation"])) for row in selected)
        )
        return result

    by_recording = {}
    for key in heldout.recording_keys:
        dataset, recording_id = key.split(":", 1)
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset and row["recording_id"] == recording_id
        ]
        by_recording[key] = _summarize(selected)
    summary = {
        "checkpoint": str(Path(args.model_out).resolve()),
        "split_mode": "recording_disjoint",
        "calibration_recordings": trainer.checkpoint_metadata.get("calibration_recordings", []),
        "heldout_recordings": heldout.recording_keys,
        "n_validation_frames": len(rows),
        "aggregate": _summarize(rows),
        "per_recording": by_recording,
    }
    aggregate = summary["aggregate"]
    aggregate["sparse_speedup_over_numerical"] = float(
        aggregate["numerical_pde_step_ms"]
        / max(aggregate["pinn_sparse_descriptor_ms"], 1e-9)
    )
    aggregate["full_grid_speedup_over_numerical"] = float(
        aggregate["numerical_pde_step_ms"]
        / max(aggregate["pinn_full_grid_ms"], 1e-9)
    )
    (output_dir / "heldout_pinn_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cache", "train", "validate", "all"], default="all")
    parser.add_argument("--dataset", default="inD")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--calibration-recordings", type=_recording_ids, default=_recording_ids("01,02,03,04,05"))
    parser.add_argument("--heldout-recordings", type=_recording_ids, default=_recording_ids("06,07,08,09,10,11"))
    parser.add_argument("--cache-dir", default="evaluation/pinn_teacher_cache")
    parser.add_argument(
        "--calibration-cache-paths",
        nargs="+",
        help="Use explicit cached recordings instead of constructing dataset paths.",
    )
    parser.add_argument(
        "--heldout-cache-paths",
        nargs="+",
        help="Held-out cache paths paired with --calibration-cache-paths.",
    )
    parser.add_argument(
        "--calibration-cache-globs",
        nargs="+",
        help="Glob patterns for calibration caches, useful for multi-scenario seeds.",
    )
    parser.add_argument(
        "--heldout-cache-globs",
        nargs="+",
        help="Glob patterns for held-out caches paired with --calibration-cache-globs.",
    )
    parser.add_argument("--output-dir", default="evaluation/pinn_context_multirecord")
    parser.add_argument("--model-out", default="rl/checkpoints/pinn/pinn_context_ind_cal01-05.pt")
    parser.add_argument("--model-role", default="multi_recording_context_surrogate")
    parser.add_argument(
        "--coordinate-mode",
        choices=["native", "ego_local"],
        default="native",
        help="Coordinate frame represented by x and y in the cached snapshots.",
    )
    parser.add_argument("--max-sec", type=float, default=44.0)
    parser.add_argument("--warmup-sec", type=float, default=4.0)
    parser.add_argument("--deployment-horizon", type=float, default=40.0)
    parser.add_argument("--perception-range", type=float, default=80.0)
    parser.add_argument("--selection-mode", choices=["all", "soft_topk"], default="soft_topk")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold-ratio", type=float, default=0.15)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--use-rff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rff-features", type=int, default=96)
    parser.add_argument("--rff-scale", type=float, default=8.0)
    parser.add_argument(
        "--rff-include-raw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Concatenate normalized raw inputs with their Fourier embedding.",
    )
    parser.add_argument(
        "--output-bias-init",
        type=float,
        default=0.0,
        help="Initial bias of the nonnegative Softplus field head.",
    )
    parser.add_argument("--n-colloc", type=int, default=4096)
    parser.add_argument("--n-data", type=int, default=4096)
    parser.add_argument("--points-per-snapshot", type=int, default=400)
    parser.add_argument(
        "--spatial-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition the PINN on the nearest-agent direction field.",
    )
    parser.add_argument(
        "--distance-rbf-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add fixed 2, 6, and 15 m radial distance basis channels.",
    )
    parser.add_argument(
        "--risk-sample-fraction",
        type=float,
        default=0.65,
        help="Fraction of data/collocation points drawn from non-trivial risk/source regions.",
    )
    parser.add_argument(
        "--risk-loss-boost",
        type=float,
        default=15.0,
        help="Importance-weight multiplier for nonzero teacher-risk targets.",
    )
    parser.add_argument(
        "--risk-halo-boost",
        type=float,
        default=2.0,
        help="Extra loss weight for low-risk cells near selected agents.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="CPU intra-op threads; 0 preserves the PyTorch default.",
    )
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Atomically save model and optimizer state every N epochs; 0 disables.",
    )
    parser.add_argument(
        "--validation-frames",
        type=int,
        default=0,
        help="Frames per held-out recording; 0 validates every cached frame.",
    )
    parser.add_argument("--validation-plots", type=int, default=6)
    parser.add_argument(
        "--numerical-substeps",
        type=int,
        default=3,
        help="Teacher PDE substeps per environment update for timing parity.",
    )
    parser.add_argument("--w-data", type=float, default=1.0)
    parser.add_argument("--w-phys", type=float, default=0.8)
    parser.add_argument("--w-ic", type=float, default=0.2)
    parser.add_argument("--w-bc", type=float, default=0.2)
    parser.add_argument("--w-smooth", type=float, default=0.15)
    parser.add_argument("--w-grad", type=float, default=0.20)
    parser.add_argument("--w-temp", type=float, default=0.10)
    parser.add_argument("--w-beh-long", type=float, default=0.15)
    parser.add_argument("--w-beh-lat", type=float, default=0.10)
    args = parser.parse_args()
    args.calibration_recordings = list(args.calibration_recordings)
    args.heldout_recordings = list(args.heldout_recordings)
    overlap = set(args.calibration_recordings) & set(args.heldout_recordings)
    if overlap:
        parser.error(f"Calibration and held-out recordings overlap: {sorted(overlap)}")
    explicit_paths = bool(args.calibration_cache_paths or args.calibration_cache_globs)
    explicit_heldout = bool(args.heldout_cache_paths or args.heldout_cache_globs)
    if explicit_paths != explicit_heldout:
        parser.error(
            "Calibration and held-out cache paths/globs must be provided together"
        )
    if args.calibration_cache_paths and args.calibration_cache_globs:
        parser.error("Choose explicit calibration paths or globs, not both")
    if args.heldout_cache_paths and args.heldout_cache_globs:
        parser.error("Choose explicit held-out paths or globs, not both")
    if explicit_paths and args.stage in {"cache", "all"}:
        parser.error("Explicit cache paths support --stage train or --stage validate")
    if not 0.0 <= float(args.risk_sample_fraction) <= 1.0:
        parser.error("--risk-sample-fraction must lie in [0, 1]")
    if float(args.risk_loss_boost) < 0.0:
        parser.error("--risk-loss-boost must be non-negative")
    if float(args.risk_halo_boost) < 0.0:
        parser.error("--risk-halo-boost must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_ids = list(dict.fromkeys([*args.calibration_recordings, *args.heldout_recordings]))
    if args.stage in {"cache", "all"}:
        cache_recordings(args, all_ids)
    if args.stage == "cache":
        return 0

    if args.calibration_cache_paths or args.calibration_cache_globs:
        calibration_paths = (
            [Path(path) for path in args.calibration_cache_paths]
            if args.calibration_cache_paths
            else _expand_cache_globs(args.calibration_cache_globs)
        )
        heldout_paths = (
            [Path(path) for path in args.heldout_cache_paths]
            if args.heldout_cache_paths
            else _expand_cache_globs(args.heldout_cache_globs)
        )
    else:
        calibration_paths = _paths(Path(args.cache_dir), args.dataset, args.calibration_recordings)
        heldout_paths = _paths(Path(args.cache_dir), args.dataset, args.heldout_recordings)
    calibration = CachedSnapshotCollection(calibration_paths)
    heldout = CachedSnapshotCollection(heldout_paths)
    args.heldout_recording_keys = heldout.recording_keys

    if args.stage in {"train", "all"}:
        trainer = train(args, calibration)
    else:
        checkpoint = torch.load(args.model_out, map_location=args.device, weights_only=False)
        trainer = _make_trainer(args, calibration, checkpoint=checkpoint)

    if args.stage in {"validate", "all"}:
        summary = validate(args, trainer, heldout)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
