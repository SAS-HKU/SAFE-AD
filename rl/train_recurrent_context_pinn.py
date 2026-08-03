"""Train and validate a recording-disjoint recurrent DRIFT PINN operator."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from rl.risk.pinn_snapshot_cache import CachedSnapshotCollection
from rl.risk.recurrent_pinn_operator import (
    INPUT_CHANNELS,
    OperatorScales,
    RecurrentContextPINN,
    build_operator_input,
    finite_difference_physics_losses,
    infer_scales,
    load_recurrent_pinn_checkpoint,
    warp_ego_local_field,
)


def _paths(patterns: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(value) for value in glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Cache glob matched no recordings: {pattern}")
        resolved.extend(matches)
    return list(dict.fromkeys(resolved))


def _hardware() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": int(torch.get_num_threads()),
    }


def _require_temporal_fields(collection: CachedSnapshotCollection) -> None:
    required = {"R_prev", "R_t", "R_t_prev", "road_mask"}
    for recording in collection.recordings:
        missing = sorted(required - set(recording._fields))
        if missing:
            raise RuntimeError(
                f"{recording.path} lacks {missing}; rebuild the HighwayEnv teacher cache"
            )
        scalar_required = {"ego_world_x", "ego_world_y", "ego_world_heading"}
        scalar_missing = sorted(scalar_required - set(recording._scalars))
        if scalar_missing:
            raise RuntimeError(
                f"{recording.path} lacks {scalar_missing}; rebuild the teacher cache"
            )


def _snapshot_pose(snapshot: dict) -> tuple[float, float, float]:
    return (
        float(snapshot["ego_world_x"]),
        float(snapshot["ego_world_y"]),
        float(snapshot["ego_world_heading"]),
    )


def _scenario_examples(collection: CachedSnapshotCollection) -> dict[str, list[tuple]]:
    groups: dict[str, list[tuple]] = {}
    for recording in collection.recordings:
        rows = groups.setdefault(recording.dataset, [])
        rows.extend((recording, index) for index in range(len(recording)))
    empty = [name for name, values in groups.items() if not values]
    if empty:
        raise RuntimeError(f"No transitions available for {empty}")
    return groups


def _random_batch(
    *,
    groups: dict[str, list[tuple]],
    batch_size: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    scales: OperatorScales,
    crop_width: int,
    risk_patch_fraction: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scenario_names = sorted(groups)
    inputs = []
    targets: dict[str, list[np.ndarray | float]] = {
        key: []
        for key in ("R", "R_t", "R_prev", "R_t_prev", "Q", "vx", "vy", "D", "dt")
    }
    nx = len(x_grid)
    width = min(int(crop_width), nx)
    for _ in range(int(batch_size)):
        scenario = scenario_names[int(rng.integers(0, len(scenario_names)))]
        recording, index = groups[scenario][int(rng.integers(0, len(groups[scenario])))]
        snapshot = recording[index]
        initial = index == 0
        previous_risk = np.zeros_like(snapshot["R"]) if initial else None
        previous_rate = np.zeros_like(snapshot["R_t"]) if initial else None
        if (
            width < nx
            and float(rng.random()) < float(risk_patch_fraction)
            and float(np.max(snapshot["R"])) > 0.05
        ):
            _peak_y, peak_x = np.unravel_index(
                int(np.argmax(snapshot["R"])), np.asarray(snapshot["R"]).shape
            )
            start = int(np.clip(peak_x - width // 2, 0, nx - width))
        else:
            start = int(rng.integers(0, nx - width + 1)) if width < nx else 0
        stop = start + width
        inputs.append(
            build_operator_input(
                snapshot,
                x_grid=x_grid,
                y_grid=y_grid,
                scales=scales,
                previous_risk=previous_risk,
                previous_rate=previous_rate,
            )[:, :, start:stop]
        )
        for key in ("R", "R_t", "R_prev", "R_t_prev", "Q", "vx", "vy", "D"):
            value = snapshot[key]
            if initial and key in {"R_prev", "R_t_prev"}:
                value = np.zeros_like(snapshot["R"])
            targets[key].append(np.asarray(value, dtype=np.float32)[:, start:stop])
        targets["dt"].append(float(snapshot["dt"]))

    input_tensor = torch.from_numpy(np.stack(inputs))
    result = {
        key: torch.from_numpy(np.stack(values)).unsqueeze(1)
        for key, values in targets.items()
        if key != "dt"
    }
    result["dt"] = torch.tensor(targets["dt"], dtype=torch.float32).view(-1, 1, 1, 1)
    return input_tensor, result


def _gradient(value: torch.Tensor, *, dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
    gx = F.pad((value[..., 2:] - value[..., :-2]) / (2.0 * dx), (1, 1, 0, 0), mode="replicate")
    gy = F.pad((value[..., 2:, :] - value[..., :-2, :]) / (2.0 * dy), (0, 0, 1, 1), mode="replicate")
    return gx, gy


def _sobolev_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    dx: float,
    dy: float,
    hotspot_boost: float,
    active_quantile: float,
    direction_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match actionable gradients without averaging them into the background."""
    pred_gx, pred_gy = _gradient(prediction, dx=dx, dy=dy)
    true_gx, true_gy = _gradient(target, dx=dx, dy=dy)
    true_magnitude = torch.sqrt(true_gx.square() + true_gy.square() + 1e-12)
    pred_magnitude = torch.sqrt(pred_gx.square() + pred_gy.square() + 1e-12)

    flattened = true_magnitude.detach().flatten(start_dim=1)
    scale = torch.quantile(flattened, float(active_quantile), dim=1)
    scale = torch.clamp(scale, min=1e-4).view(-1, 1, 1, 1)
    weights = 1.0 + float(hotspot_boost) * torch.clamp(
        true_magnitude / scale,
        min=0.0,
        max=1.0,
    )

    normalized_x = (pred_gx - true_gx) / scale
    normalized_y = (pred_gy - true_gy) / scale
    component_error = F.smooth_l1_loss(
        normalized_x,
        torch.zeros_like(normalized_x),
        reduction="none",
    ) + F.smooth_l1_loss(
        normalized_y,
        torch.zeros_like(normalized_y),
        reduction="none",
    )
    component_loss = torch.sum(weights * component_error) / torch.sum(weights)

    active = (true_magnitude >= scale).to(prediction.dtype)
    cosine = (pred_gx * true_gx + pred_gy * true_gy) / torch.clamp(
        pred_magnitude * true_magnitude,
        min=1e-8,
    )
    direction_error = 1.0 - torch.clamp(cosine, min=-1.0, max=1.0)
    direction_weights = weights * active
    direction_loss = torch.sum(direction_weights * direction_error) / torch.clamp(
        torch.sum(direction_weights),
        min=1.0,
    )
    total = component_loss + float(direction_weight) * direction_loss
    return total, component_loss, direction_loss


def train(args, calibration: CachedSnapshotCollection) -> dict:
    model_path = Path(args.model_out)
    if model_path.exists() and not args.resume:
        raise FileExistsError(f"Checkpoint exists: {model_path}")
    groups = _scenario_examples(calibration)
    scales = infer_scales(calibration.recordings)
    device = torch.device(args.device)
    model = RecurrentContextPINN(
        width=args.width,
        output_mode=args.output_mode,
        max_normalized_risk=args.max_normalized_risk,
        max_normalized_rate=args.max_normalized_rate,
    ).to(device)
    history: dict[str, list[float]] = {
        key: [] for key in ("total", "field", "rate", "gradient", "kinematic", "dynamic")
    }
    start_step = 0
    if args.resume:
        loaded_model, checkpoint = load_recurrent_pinn_checkpoint(model_path, device=args.device)
        model = loaded_model
        scales = OperatorScales.from_dict(checkpoint["scales"])
        history = dict(checkpoint.get("history", history))
        start_step = int(checkpoint.get("steps", 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(args.steps))
    )
    rng = np.random.default_rng(args.seed)
    dx = float(calibration.x_grid[1] - calibration.x_grid[0])
    dy = float(calibration.y_grid[1] - calibration.y_grid[0])
    started = time.perf_counter()

    def save(step: int, status: str) -> None:
        payload = {
            "model_family": "recurrent_context_pinn",
            "model_state": model.state_dict(),
            "input_channels": len(INPUT_CHANNELS),
            "input_channel_names": list(INPUT_CHANNELS),
            "width": int(model.width),
            "dilations": list(model.dilations),
            "output_mode": str(model.output_mode),
            "max_normalized_risk": float(model.max_normalized_risk),
            "max_normalized_rate": float(model.max_normalized_rate),
            "scales": scales.to_dict(),
            "x_grid": np.asarray(calibration.x_grid, dtype=np.float32),
            "y_grid": np.asarray(calibration.y_grid, dtype=np.float32),
            "coordinate_mode": str(args.coordinate_mode),
            "calibration_recordings": calibration.recording_keys,
            "teacher_cache_signatures": calibration.signatures,
            "split_mode": "recording_disjoint",
            "steps": int(step),
            "history": history,
            "status": status,
            "loss_weights": {
                "field": args.w_field,
                "rate": args.w_rate,
                "gradient": args.w_gradient,
                "gradient_direction": args.gradient_direction_weight,
                "physics": args.w_physics,
            },
            "gradient_active_quantile": float(args.gradient_active_quantile),
            "physics": {
                "tau": float(Config.tau),
                "lambda_decay": float(Config.lambda_decay),
                "L_decay": float(Config.L_decay),
            },
            "hardware": _hardware(),
            "training_wall_time_s": float(time.perf_counter() - started),
        }
        temporary = model_path.with_suffix(model_path.suffix + ".tmp")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, temporary)
        os.replace(temporary, model_path)

    model.train()
    for local_step in range(1, int(args.steps) + 1):
        step = start_step + local_step
        batch, target = _random_batch(
            groups=groups,
            batch_size=args.batch_size,
            x_grid=calibration.x_grid,
            y_grid=calibration.y_grid,
            scales=scales,
            crop_width=args.crop_width,
            risk_patch_fraction=args.risk_patch_fraction,
            rng=rng,
        )
        batch = batch.to(device)
        target = {key: value.to(device) for key, value in target.items()}
        pred_r_norm, pred_rt_norm = model(batch)
        pred_r = pred_r_norm * scales.risk
        pred_rt = pred_rt_norm * scales.risk_rate
        true_r = target["R"]
        true_rt = target["R_t"]
        hotspot = 1.0 + args.hotspot_boost * torch.clamp(
            true_r / 0.05, 0.0, 1.0
        )
        field_loss = torch.sum(hotspot * F.smooth_l1_loss(pred_r, true_r, reduction="none")) / torch.sum(hotspot)
        rate_loss = F.smooth_l1_loss(pred_rt, true_rt)
        gradient_loss, gradient_component, gradient_direction = _sobolev_gradient_loss(
            pred_r,
            true_r,
            dx=dx,
            dy=dy,
            hotspot_boost=args.hotspot_boost,
            active_quantile=args.gradient_active_quantile,
            direction_weight=args.gradient_direction_weight,
        )
        kinematic, dynamic = finite_difference_physics_losses(
            risk=pred_r,
            risk_rate=pred_rt,
            previous_risk=target["R_prev"],
            previous_rate=target["R_t_prev"],
            source=target["Q"],
            vx=target["vx"],
            vy=target["vy"],
            diffusion=target["D"],
            dt=target["dt"],
            dx=dx,
            dy=dy,
            tau=float(Config.tau),
            lambda_decay=float(Config.lambda_decay),
            length_decay=float(Config.L_decay),
        )
        physics_loss = (
            kinematic / max(scales.risk_rate**2, 1e-6)
            + dynamic / max(scales.source**2, 1e-6)
        )
        total = (
            args.w_field * field_loss
            + args.w_rate * rate_loss
            + args.w_gradient * gradient_loss
            + args.w_physics * physics_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        for key, value in (
            ("total", total),
            ("field", field_loss),
            ("rate", rate_loss),
            ("gradient", gradient_loss),
            ("kinematic", kinematic),
            ("dynamic", dynamic),
        ):
            history[key].append(float(value.detach().cpu()))
        if local_step == 1 or local_step % args.print_every == 0:
            print(
                f"step {local_step:5d}/{args.steps} total={history['total'][-1]:.4e} "
                f"field={history['field'][-1]:.4e} grad={history['gradient'][-1]:.4e} "
                f"grad_comp={float(gradient_component.detach().cpu()):.3e} "
                f"grad_dir={float(gradient_direction.detach().cpu()):.3e} "
                f"phys={float(physics_loss.detach().cpu()):.4e}",
                flush=True,
            )
        if args.checkpoint_every and local_step % args.checkpoint_every == 0:
            save(step, "in_progress")
    save(start_step + int(args.steps), "complete")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for key in ("total", "field", "gradient", "kinematic", "dynamic"):
        values = np.maximum(np.asarray(history[key], dtype=float), 1e-12)
        ax.plot(values, label=key.capitalize())
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training_losses.png", dpi=300)
    fig.savefig(output / "training_losses.pdf")
    plt.close(fig)
    return {"checkpoint": str(model_path), "steps": start_step + int(args.steps)}


def _metrics(pred: np.ndarray, true: np.ndarray, *, dx: float, dy: float) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    error = pred - true
    energy = float(np.sum(true * true))
    gy_t, gx_t = np.gradient(true, dy, dx)
    gy_p, gx_p = np.gradient(pred, dy, dx)
    grad_energy = float(np.sum(gx_t * gx_t + gy_t * gy_t))
    grad_error = float(np.sum((gx_p - gx_t) ** 2 + (gy_p - gy_t) ** 2))
    mag = np.hypot(gx_t, gy_t)
    active = mag >= np.percentile(mag, 75.0)
    cosine = (gx_p * gx_t + gy_p * gy_t) / np.maximum(np.hypot(gx_p, gy_p) * mag, 1e-12)
    correlation = (
        float(np.corrcoef(pred.reshape(-1), true.reshape(-1))[0, 1])
        if np.std(pred) > 1e-10 and np.std(true) > 1e-10
        else float("nan")
    )
    threshold = max(float(np.percentile(true, 90.0)), 0.05 * float(np.max(true)))
    true_hot = true >= threshold
    count = int(true_hot.sum())
    pred_hot = np.zeros_like(true_hot)
    if count:
        selected = np.argpartition(pred.reshape(-1), -count)[-count:]
        pred_hot.reshape(-1)[selected] = True
    union = np.logical_or(true_hot, pred_hot).sum()
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "field_error_energy": float(np.sum(error * error)),
        "teacher_energy": energy,
        "correlation": correlation,
        "gradient_error_energy": grad_error,
        "gradient_teacher_energy": grad_energy,
        "gradient_angle_deg": float(np.degrees(np.arccos(np.clip(cosine[active], -1.0, 1.0))).mean()),
        "hotspot_iou": float(np.logical_and(true_hot, pred_hot).sum() / max(union, 1)),
    }


def _summary(rows: list[dict]) -> dict[str, float]:
    field_error = sum(row["field_error_energy"] for row in rows)
    field_energy = sum(row["teacher_energy"] for row in rows)
    grad_error = sum(row["gradient_error_energy"] for row in rows)
    grad_energy = sum(row["gradient_teacher_energy"] for row in rows)
    return {
        "rmse": float(np.sqrt(np.mean([row["rmse"] ** 2 for row in rows]))),
        "mae": float(np.mean([row["mae"] for row in rows])),
        "relative_l2": float(np.sqrt(field_error / max(field_energy, 1e-12))),
        "correlation": float(np.nanmean([row["correlation"] for row in rows])),
        "gradient_relative_l2": float(np.sqrt(grad_error / max(grad_energy, 1e-12))),
        "gradient_angle_deg": float(np.nanmean([row["gradient_angle_deg"] for row in rows])),
        "hotspot_iou": float(np.nanmean([row["hotspot_iou"] for row in rows])),
        "inference_ms": float(np.mean([row["inference_ms"] for row in rows])),
        "n_frames": int(len(rows)),
    }


def _plot_panel(path: Path, teacher: np.ndarray, prediction: np.ndarray, x, y, title: str) -> None:
    vmax = max(float(np.max(teacher)), float(np.max(prediction)), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
    for ax, field, label in zip(
        axes,
        (teacher, prediction, np.abs(prediction - teacher)),
        ("Numerical teacher", "Recurrent context PINN", "Absolute error"),
    ):
        image = ax.imshow(
            field,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="auto",
            cmap="turbo",
            vmin=0.0,
            vmax=vmax if label != "Absolute error" else None,
        )
        ax.set_title(label)
        ax.set_xlabel("Ego-longitudinal position (m)")
        ax.set_ylabel("Ego-lateral position (m)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def validate(args, heldout: CachedSnapshotCollection) -> dict:
    model, checkpoint = load_recurrent_pinn_checkpoint(args.model_out, device=args.device)
    scales = OperatorScales.from_dict(checkpoint["scales"])
    device = torch.device(args.device)
    dx = float(heldout.x_grid[1] - heldout.x_grid[0])
    dy = float(heldout.y_grid[1] - heldout.y_grid[0])
    rows: list[dict] = []
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plotted = 0
    model.eval()
    with torch.inference_mode():
        n_recordings = len(heldout.recordings)
        base_plot_budget = int(args.validation_plots) // max(n_recordings, 1)
        extra_plot_budget = int(args.validation_plots) % max(n_recordings, 1)
        for recording_index, recording in enumerate(heldout.recordings):
            previous_pred = np.asarray(recording[0]["R"], dtype=np.float32).copy()
            previous_rate = np.asarray(recording[0]["R_t"], dtype=np.float32).copy()
            previous_pose = _snapshot_pose(recording[0])
            selected_indices = np.linspace(
                1,
                len(recording) - 1,
                min(max(1, args.validation_frames), max(1, len(recording) - 1)),
                dtype=int,
            )
            selected = set(int(value) for value in selected_indices)
            local_plot_budget = base_plot_budget + int(recording_index < extra_plot_budget)
            if local_plot_budget == 1 and len(recording) > 1:
                plot_indices = {(1 + len(recording) - 1) // 2}
            elif local_plot_budget:
                plot_indices = set(
                    int(value)
                    for value in np.linspace(
                        1,
                        len(recording) - 1,
                        min(local_plot_budget, max(0, len(recording) - 1)),
                        dtype=int,
                    )
                )
            else:
                plot_indices = set()
            for index in range(1, len(recording)):
                snapshot = recording[index]
                current_pose = _snapshot_pose(snapshot)
                if checkpoint.get("coordinate_mode", "ego_local") == "ego_local":
                    aligned_r = warp_ego_local_field(
                        previous_pred,
                        x_grid=heldout.x_grid,
                        y_grid=heldout.y_grid,
                        previous_pose=previous_pose,
                        current_pose=current_pose,
                    )
                    aligned_rt = warp_ego_local_field(
                        previous_rate,
                        x_grid=heldout.x_grid,
                        y_grid=heldout.y_grid,
                        previous_pose=previous_pose,
                        current_pose=current_pose,
                    )
                else:
                    aligned_r = previous_pred
                    aligned_rt = previous_rate
                array = build_operator_input(
                    snapshot,
                    x_grid=heldout.x_grid,
                    y_grid=heldout.y_grid,
                    scales=scales,
                    previous_risk=aligned_r,
                    previous_rate=aligned_rt,
                )
                checkpoint_channels = tuple(
                    checkpoint.get("input_channel_names", INPUT_CHANNELS)
                )
                if checkpoint_channels != tuple(INPUT_CHANNELS):
                    channel_index = {
                        name: position for position, name in enumerate(INPUT_CHANNELS)
                    }
                    array = array[
                        [channel_index[name] for name in checkpoint_channels]
                    ]
                tensor = torch.from_numpy(array[None]).to(device)
                started = time.perf_counter()
                pred_norm, rate_norm = model(tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_ms = 1000.0 * (time.perf_counter() - started)
                prediction = pred_norm[0, 0].cpu().numpy() * scales.risk
                predicted_rate = rate_norm[0, 0].cpu().numpy() * scales.risk_rate
                previous_pred = prediction
                previous_rate = predicted_rate
                previous_pose = current_pose
                if index not in selected:
                    continue
                row = {
                    "dataset": recording.dataset,
                    "recording_id": recording.recording_id,
                    "frame_id": int(snapshot["frame_id"]),
                    "time_s": float(snapshot["t"]),
                    "inference_ms": inference_ms,
                    **_metrics(
                        prediction,
                        np.asarray(snapshot["R"]),
                        dx=dx,
                        dy=dy,
                    ),
                }
                rows.append(row)
                if index in plot_indices and plotted < args.validation_plots:
                    _plot_panel(
                        output / f"heldout_{recording.dataset}_{recording.recording_id}_frame_{int(snapshot['frame_id'])}.png",
                        np.asarray(snapshot["R"]),
                        prediction,
                        heldout.x_grid,
                        heldout.y_grid,
                        f"Held-out autoregressive rollout: {recording.dataset} {recording.recording_id}",
                    )
                    plotted += 1

    if not rows:
        raise RuntimeError("No held-out transitions were evaluated")
    csv_path = output / "heldout_recurrent_pinn_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_recording = {}
    for recording in heldout.recordings:
        selected = [
            row for row in rows
            if row["dataset"] == recording.dataset and row["recording_id"] == recording.recording_id
        ]
        by_recording[f"{recording.dataset}:{recording.recording_id}"] = _summary(selected)
    summary = {
        "checkpoint": str(Path(args.model_out).resolve()),
        "validation_mode": "autoregressive_recording_disjoint",
        "calibration_recordings": checkpoint["calibration_recordings"],
        "heldout_recordings": heldout.recording_keys,
        "aggregate": _summary(rows),
        "per_recording": by_recording,
    }
    (output / "heldout_recurrent_pinn_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "validate", "all"), default="all")
    parser.add_argument("--calibration-cache-globs", nargs="+", required=True)
    parser.add_argument("--heldout-cache-globs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coordinate-mode", choices=("ego_local", "native"), default="ego_local")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--crop-width", type=int, default=96)
    parser.add_argument("--risk-patch-fraction", type=float, default=0.70)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument(
        "--output-mode",
        choices=("residual", "absolute_bounded"),
        default="residual",
    )
    parser.add_argument("--max-normalized-risk", type=float, default=1.25)
    parser.add_argument("--max-normalized-rate", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hotspot-boost", type=float, default=4.0)
    parser.add_argument("--w-field", type=float, default=1.0)
    parser.add_argument("--w-rate", type=float, default=0.05)
    parser.add_argument("--w-gradient", type=float, default=0.20)
    parser.add_argument(
        "--gradient-active-quantile",
        type=float,
        default=0.85,
        help="Per-sample teacher-gradient quantile defining actionable pixels.",
    )
    parser.add_argument(
        "--gradient-direction-weight",
        type=float,
        default=0.25,
        help="Direction-cosine contribution within the Sobolev gradient loss.",
    )
    parser.add_argument("--w-physics", type=float, default=0.01)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-frames", type=int, default=30)
    parser.add_argument("--validation-plots", type=int, default=8)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def main() -> int:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    calibration = CachedSnapshotCollection(_paths(args.calibration_cache_globs))
    heldout = CachedSnapshotCollection(_paths(args.heldout_cache_globs))
    _require_temporal_fields(calibration)
    _require_temporal_fields(heldout)
    if set(calibration.recording_keys) & set(heldout.recording_keys):
        raise ValueError("Calibration and held-out recordings overlap")
    if args.stage in {"train", "all"}:
        train(args, calibration)
    if args.stage in {"validate", "all"}:
        validate(args, heldout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
