"""Train a recording-disjoint context PINN for the prospective-v2 teacher."""

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

from rl.risk.pinn_snapshot_cache import CachedSnapshotCollection
from rl.risk.prospective_solver import ProspectiveRiskSolver, ProspectiveSolverConfig
from rl.risk.recurrent_pinn_operator import (
    INPUT_CHANNELS,
    OperatorScales,
    RecurrentContextPINN,
    build_operator_input,
    checkpoint_domain_scales,
    load_recurrent_pinn_checkpoint,
    select_checkpoint_inputs,
)
from rl.train_recurrent_context_pinn import _sobolev_gradient_loss


PROSPECTIVE_INPUT_CHANNELS = (
    "Q",
    "Q_blur_2m",
    "Q_blur_6m",
    "Q_blur_15m",
    "vx",
    "vy",
    "D",
    "occ_mask",
    "road_mask",
    "dist_rbf_2m",
    "dist_rbf_6m",
    "dist_rbf_15m",
    "x",
    "y",
    "yaw_rate",
    "N_agents",
    "domain_highwayenv",
)


def _dilations(value: str) -> tuple[int, ...]:
    result = tuple(int(token) for token in str(value).split(",") if token.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("dilations must be positive comma-separated integers")
    return result


def _paths(patterns: list[str]) -> list[Path]:
    values: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in sorted(glob.glob(pattern))]
        if not matches:
            raise FileNotFoundError(f"Cache glob matched no recordings: {pattern}")
        values.extend(matches)
    return list(dict.fromkeys(values))


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


def _require_prospective(collection: CachedSnapshotCollection) -> dict:
    solver_specs = []
    for recording in collection.recordings:
        if recording.manifest.get("teacher_version") != "prospective_v2":
            raise RuntimeError(f"Not a prospective-v2 cache: {recording.path}")
        if recording.manifest.get("coordinate_mode") != "ego_local":
            raise RuntimeError(f"Cache is not ego-local: {recording.path}")
        missing = {"Q_terminal", "road_mask"} - set(recording._fields)
        if missing:
            raise RuntimeError(f"{recording.path} lacks {sorted(missing)}")
        solver_specs.append(recording.manifest["solver"])
    canonical = solver_specs[0]
    if any(value != canonical for value in solver_specs[1:]):
        raise RuntimeError("All caches must use the same prospective solver configuration")
    return canonical


def _domain(dataset: str) -> str:
    return "highwayenv" if str(dataset).startswith("highwayenv_") else "naturalistic"


def _sampled_abs(recordings, field: str, *, scalar: bool = False) -> np.ndarray:
    values = []
    for recording in recordings:
        source = recording._scalars if scalar else recording._fields
        if field not in source:
            continue
        array = np.abs(np.asarray(source[field], dtype=np.float32)).reshape(-1)
        stride = max(1, array.size // 100_000)
        values.append(array[::stride])
    return np.concatenate(values) if values else np.zeros(1, dtype=np.float32)


def _infer_domain_scales(collection: CachedSnapshotCollection) -> dict[str, OperatorScales]:
    """Infer scales from calibration recordings only, separately by domain."""
    grouped: dict[str, list] = {}
    for recording in collection.recordings:
        grouped.setdefault(_domain(recording.dataset), []).append(recording)
    result = {}
    for domain, recordings in grouped.items():
        risk = _sampled_abs(recordings, "R")
        rate = _sampled_abs(recordings, "R_t")
        source = _sampled_abs(recordings, "Q")
        velocity = np.concatenate(
            (_sampled_abs(recordings, "vx"), _sampled_abs(recordings, "vy"))
        )
        diffusion = _sampled_abs(recordings, "D")
        yaw_rate = _sampled_abs(recordings, "ego_yaw_rate", scalar=True)
        agents = _sampled_abs(recordings, "N_agents", scalar=True)
        dt = _sampled_abs(recordings, "dt", scalar=True)
        result[domain] = OperatorScales(
            risk=max(float(np.max(risk)), 1e-3),
            risk_rate=max(float(np.quantile(rate, 0.999)), 1e-3),
            source=max(float(np.max(source)), 1e-3),
            velocity=max(float(np.quantile(velocity, 0.995)), 1e-3),
            diffusion=max(float(np.quantile(diffusion, 0.999)), 1e-3),
            yaw_rate=max(float(np.quantile(yaw_rate, 0.995)), 1e-3),
            agents=max(float(np.max(agents)), 1.0),
            dt=max(float(np.max(dt)), 1e-3),
        )
    missing = {"naturalistic", "highwayenv"} - set(result)
    if missing:
        raise RuntimeError(f"Calibration split lacks domains: {sorted(missing)}")
    return result


def _examples(collection: CachedSnapshotCollection) -> dict[str, dict[str, list[tuple]]]:
    groups: dict[str, dict[str, list[tuple]]] = {}
    for recording in collection.recordings:
        scenario = str(recording.dataset)
        bucket = groups.setdefault(_domain(scenario), {}).setdefault(scenario, [])
        bucket.extend((recording, index) for index in range(len(recording)))
    return groups


def _select_inputs(array: np.ndarray, *, domain: str) -> np.ndarray:
    index = {name: position for position, name in enumerate(INPUT_CHANNELS)}
    channels = []
    for name in PROSPECTIVE_INPUT_CHANNELS:
        if name == "domain_highwayenv":
            channels.append(
                np.full(
                    array.shape[1:],
                    1.0 if domain == "highwayenv" else 0.0,
                    dtype=np.float32,
                )
            )
        else:
            channels.append(array[index[name]])
    return np.clip(np.stack(channels), -4.0, 4.0).astype(np.float32)


def _random_batch(
    groups,
    *,
    batch_size: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    domain_scales: dict[str, OperatorScales],
    crop_width: int,
    risk_patch_fraction: float,
    rng: np.random.Generator,
):
    inputs = []
    targets = {key: [] for key in ("R", "Q", "Q_terminal", "vx", "vy", "D", "road_mask")}
    risk_scales = []
    source_scales = []
    domains = sorted(groups)
    nx = len(x_grid)
    width = min(int(crop_width), nx)
    for _ in range(int(batch_size)):
        domain = domains[int(rng.integers(0, len(domains)))]
        scenarios = sorted(groups[domain])
        scenario = scenarios[int(rng.integers(0, len(scenarios)))]
        examples = groups[domain][scenario]
        recording, index = examples[int(rng.integers(0, len(examples)))]
        snapshot = recording[index]
        if (
            width < nx
            and rng.random() < float(risk_patch_fraction)
            and float(np.max(snapshot["R"])) > 0
        ):
            _peak_y, peak_x = np.unravel_index(
                int(np.argmax(snapshot["R"])), np.asarray(snapshot["R"]).shape
            )
            start = int(np.clip(peak_x - width // 2, 0, nx - width))
        else:
            start = int(rng.integers(0, nx - width + 1)) if width < nx else 0
        stop = start + width
        scales = domain_scales[domain]
        operator_input = build_operator_input(
            snapshot,
            x_grid=x_grid,
            y_grid=y_grid,
            scales=scales,
        )
        inputs.append(
            _select_inputs(operator_input, domain=domain)[:, :, start:stop]
        )
        for key in targets:
            targets[key].append(np.asarray(snapshot[key], dtype=np.float32)[:, start:stop])
        risk_scales.append(float(scales.risk))
        source_scales.append(float(scales.source))
    target_tensors = {
        key: torch.from_numpy(np.stack(values)).unsqueeze(1)
        for key, values in targets.items()
    }
    target_tensors["risk_scale"] = torch.tensor(
        risk_scales, dtype=torch.float32
    ).view(-1, 1, 1, 1)
    target_tensors["source_scale"] = torch.tensor(
        source_scales, dtype=torch.float32
    ).view(-1, 1, 1, 1)
    return (
        torch.from_numpy(np.stack(inputs)),
        target_tensors,
    )


def _derivative_x(value: torch.Tensor, dx: float) -> torch.Tensor:
    core = (value[..., 2:] - value[..., :-2]) / (2.0 * dx)
    return F.pad(core, (1, 1, 0, 0), mode="replicate")


def _derivative_y(value: torch.Tensor, dy: float) -> torch.Tensor:
    core = (value[..., 2:, :] - value[..., :-2, :]) / (2.0 * dy)
    return F.pad(core, (0, 0, 1, 1), mode="replicate")


def _finite_horizon_physics_loss(
    prediction: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    dx: float,
    dy: float,
    horizon_s: float,
    decay_rate: float,
    source_scale: torch.Tensor,
) -> torch.Tensor:
    grad_x = _derivative_x(prediction, dx)
    grad_y = _derivative_y(prediction, dy)
    div_advection = _derivative_x(target["vx"] * prediction, dx) + _derivative_y(
        target["vy"] * prediction, dy
    )
    diffusion_level = torch.quantile(
        target["D"].flatten(start_dim=1), 0.5, dim=1
    ).view(-1, 1, 1, 1)
    laplacian = _derivative_x(grad_x, dx) + _derivative_y(grad_y, dy)
    transport_operator = div_advection - diffusion_level * laplacian
    attenuation = float(np.exp(-float(decay_rate) * float(horizon_s)))
    normalizer = (
        float(horizon_s)
        if decay_rate <= 1e-12
        else (1.0 - attenuation) / float(decay_rate)
    )
    rhs = (target["Q"] - attenuation * target["Q_terminal"]) / max(normalizer, 1e-6)
    residual = transport_operator + float(decay_rate) * prediction - rhs
    scaled = residual / torch.clamp(source_scale, min=1e-6)
    mask = torch.clamp(target["road_mask"], 0.0, 1.0)
    return torch.sum(mask * F.smooth_l1_loss(scaled, torch.zeros_like(scaled), reduction="none")) / torch.clamp(mask.sum(), min=1.0)


def _metrics(prediction: np.ndarray, target: np.ndarray, *, dx: float, dy: float) -> dict:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    error = pred - true
    gy_t, gx_t = np.gradient(true, dy, dx)
    gy_p, gx_p = np.gradient(pred, dy, dx)
    magnitude = np.hypot(gx_t, gy_t)
    active = (magnitude >= np.percentile(magnitude, 75.0)) & (magnitude > 1e-8)
    cosine = (gx_p * gx_t + gy_p * gy_t) / np.maximum(
        np.hypot(gx_p, gy_p) * magnitude, 1e-12
    )
    maximum = float(np.max(true))
    threshold = max(float(np.percentile(true, 90.0)), 0.05 * maximum)
    true_hot = (true >= threshold) & (true > 1e-8)
    count = int(true_hot.sum())
    pred_hot = np.zeros_like(true_hot)
    if count:
        selected = np.argpartition(pred.reshape(-1), -count)[-count:]
        pred_hot.reshape(-1)[selected] = True
    union = np.logical_or(true_hot, pred_hot).sum()
    correlation = (
        float(np.corrcoef(pred.reshape(-1), true.reshape(-1))[0, 1])
        if np.std(pred) > 1e-10 and np.std(true) > 1e-10
        else float("nan")
    )
    true_norm = float(np.linalg.norm(true))
    gradient_norm = float(np.sum(gx_t**2 + gy_t**2))
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "relative_l2": (
            float(np.linalg.norm(error) / true_norm) if true_norm > 1e-8 else float("nan")
        ),
        "correlation": correlation,
        "gradient_relative_l2": float(
            np.sqrt(
                np.sum((gx_p - gx_t) ** 2 + (gy_p - gy_t) ** 2)
                / gradient_norm
            )
        ) if gradient_norm > 1e-12 else float("nan"),
        "gradient_angle_deg": float(
            np.degrees(np.arccos(np.clip(cosine[active], -1.0, 1.0))).mean()
        ) if np.any(active) else float("nan"),
        "hotspot_iou": float(np.logical_and(true_hot, pred_hot).sum() / max(union, 1)),
        "background_risk_mean": float(np.mean(np.clip(pred, 0.0, None)[~true_hot])),
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = (
        "rmse",
        "mae",
        "relative_l2",
        "correlation",
        "gradient_relative_l2",
        "gradient_angle_deg",
        "hotspot_iou",
        "background_risk_mean",
        "pinn_inference_ms",
        "numerical_solver_ms",
    )
    result = {
        key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys
    }
    result["solver_speedup"] = result["numerical_solver_ms"] / max(
        result["pinn_inference_ms"], 1e-9
    )
    result["n_frames"] = len(rows)
    return result


def _plot_panel(path: Path, target, prediction, x, y, title: str) -> None:
    vmax = max(float(np.max(target)), float(np.max(prediction)), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.75), constrained_layout=True)
    for axis, field, label in zip(
        axes,
        (target, prediction, np.abs(prediction - target)),
        ("Prospective numerical teacher", "Context PINN", "Absolute error"),
    ):
        image = axis.imshow(
            field,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="auto",
            cmap="turbo",
            vmin=0.0,
            vmax=vmax if label != "Absolute error" else None,
        )
        axis.set_title(label, fontsize=8)
        axis.set_xlabel("Ego-longitudinal position (m)")
        axis.set_ylabel("Ego-lateral position (m)")
        fig.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def train(args, calibration: CachedSnapshotCollection, solver_spec: dict) -> dict:
    model_path = Path(args.model_out)
    if model_path.exists() and not args.resume:
        raise FileExistsError(model_path)
    device = torch.device(args.device)
    domain_scales = _infer_domain_scales(calibration)
    scales = domain_scales["highwayenv"]
    model = RecurrentContextPINN(
        input_channels=len(PROSPECTIVE_INPUT_CHANNELS),
        channel_names=PROSPECTIVE_INPUT_CHANNELS,
        width=args.width,
        dilations=args.dilations,
        output_mode="absolute_bounded",
        max_normalized_risk=args.max_normalized_risk,
        max_normalized_rate=1.0,
    ).to(device)
    history = {key: [] for key in ("total", "field", "gradient", "physics")}
    start_step = 0
    if args.resume:
        model, checkpoint = load_recurrent_pinn_checkpoint(model_path, device=args.device)
        domain_scales = {
            domain: OperatorScales.from_dict(values)
            for domain, values in checkpoint["domain_scales"].items()
        }
        scales = domain_scales["highwayenv"]
        history = checkpoint.get("history", history)
        start_step = int(checkpoint.get("steps", 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps))
    rng = np.random.default_rng(args.seed)
    groups = _examples(calibration)
    dx = float(calibration.x_grid[1] - calibration.x_grid[0])
    dy = float(calibration.y_grid[1] - calibration.y_grid[0])
    started = time.perf_counter()

    def save(step: int, status: str) -> None:
        payload = {
            "model_family": "prospective_context_pinn",
            "model_state": model.state_dict(),
            "input_channels": len(PROSPECTIVE_INPUT_CHANNELS),
            "input_channel_names": list(PROSPECTIVE_INPUT_CHANNELS),
            "width": int(model.width),
            "dilations": list(model.dilations),
            "output_mode": model.output_mode,
            "max_normalized_risk": float(model.max_normalized_risk),
            "max_normalized_rate": float(model.max_normalized_rate),
            "stateful": False,
            "coordinate_mode": "ego_local",
            "teacher_version": "prospective_v2",
            "prospective_solver": solver_spec,
            "physics_identity": "finite_horizon_discounted_transport",
            "scales": scales.to_dict(),
            "domain_scales": {
                domain: values.to_dict() for domain, values in domain_scales.items()
            },
            "domain_conditioning": "binary_highwayenv_channel",
            "input_clip": 4.0,
            "x_grid": np.asarray(calibration.x_grid, dtype=np.float32),
            "y_grid": np.asarray(calibration.y_grid, dtype=np.float32),
            "calibration_recordings": calibration.recording_keys,
            "heldout_recordings": list(
                getattr(args, "heldout_recording_keys", [])
            ),
            "teacher_cache_signatures": calibration.signatures,
            "heldout_cache_signatures": list(
                getattr(args, "heldout_cache_signatures", [])
            ),
            "split_mode": "recording_disjoint",
            "training_seed": int(args.seed),
            "training_config": {
                "batch_size": int(args.batch_size),
                "crop_width": int(args.crop_width),
                "risk_patch_fraction": float(args.risk_patch_fraction),
                "learning_rate": float(args.lr),
                "torch_threads": int(args.torch_threads),
                "device": str(args.device),
            },
            "steps": int(step),
            "history": history,
            "status": status,
            "loss_weights": {
                "field": float(args.w_field),
                "gradient": float(args.w_gradient),
                "physics": float(args.w_physics),
            },
            "hardware": _hardware(),
            "training_wall_time_s": float(time.perf_counter() - started),
        }
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(model_path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, model_path)

    model.train()
    for local_step in range(1, int(args.steps) + 1):
        step = start_step + local_step
        batch, target = _random_batch(
            groups,
            batch_size=args.batch_size,
            x_grid=calibration.x_grid,
            y_grid=calibration.y_grid,
            domain_scales=domain_scales,
            crop_width=args.crop_width,
            risk_patch_fraction=args.risk_patch_fraction,
            rng=rng,
        )
        batch = batch.to(device)
        target = {key: value.to(device) for key, value in target.items()}
        prediction_norm, _unused_rate = model(batch)
        prediction = prediction_norm * target["risk_scale"]
        true = target["R"]
        true_norm = true / torch.clamp(target["risk_scale"], min=1e-6)
        hotspot_scale = 0.05
        weights = 1.0 + float(args.hotspot_boost) * torch.clamp(
            true_norm / hotspot_scale, 0.0, 1.0
        )
        field_loss = torch.sum(
            weights * F.smooth_l1_loss(prediction_norm, true_norm, reduction="none")
        ) / torch.sum(weights)
        gradient_loss, _component, _direction = _sobolev_gradient_loss(
            prediction_norm,
            true_norm,
            dx=dx,
            dy=dy,
            hotspot_boost=args.hotspot_boost,
            active_quantile=args.gradient_active_quantile,
            direction_weight=args.gradient_direction_weight,
        )
        physics_loss = _finite_horizon_physics_loss(
            prediction,
            target,
            dx=dx,
            dy=dy,
            horizon_s=float(solver_spec["horizon_s"]),
            decay_rate=float(solver_spec["decay_rate"]),
            source_scale=target["source_scale"],
        )
        total = (
            args.w_field * field_loss
            + args.w_gradient * gradient_loss
            + args.w_physics * physics_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        for key, value in (
            ("total", total),
            ("field", field_loss),
            ("gradient", gradient_loss),
            ("physics", physics_loss),
        ):
            history[key].append(float(value.detach().cpu()))
        if local_step == 1 or local_step % args.print_every == 0:
            print(
                f"step {local_step:5d}/{args.steps} total={history['total'][-1]:.4e} "
                f"field={history['field'][-1]:.4e} grad={history['gradient'][-1]:.4e} "
                f"physics={history['physics'][-1]:.4e}",
                flush=True,
            )
        if args.checkpoint_every and local_step % args.checkpoint_every == 0:
            save(step, "in_progress")
    save(start_step + int(args.steps), "complete")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 3.2))
    for key, values in history.items():
        axis.plot(np.maximum(values, 1e-12), label=key.capitalize())
    axis.set_yscale("log")
    axis.set_xlabel("Optimization step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(output / "training_losses.png", dpi=300)
    fig.savefig(output / "training_losses.pdf")
    plt.close(fig)
    return {"checkpoint": str(model_path), "steps": start_step + int(args.steps)}


def validate(args, heldout: CachedSnapshotCollection, solver_spec: dict) -> dict:
    model, checkpoint = load_recurrent_pinn_checkpoint(args.model_out, device=args.device)
    device = torch.device(args.device)
    x = np.asarray(heldout.x_grid)
    y = np.asarray(heldout.y_grid)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    solver = ProspectiveRiskSolver(
        x_grid=x,
        y_grid=y,
        config=ProspectiveSolverConfig(**solver_spec),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    plotted = 0
    warm = torch.zeros(
        (1, len(PROSPECTIVE_INPUT_CHANNELS), len(y), len(x)),
        dtype=torch.float32,
        device=device,
    )
    runtime_model = model
    if device.type == "cpu":
        runtime_model = torch.jit.freeze(
            torch.jit.trace(model, warm, check_trace=False).eval()
        )
    with torch.inference_mode():
        runtime_model(warm)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for recording in heldout.recordings:
            domain = _domain(recording.dataset)
            scales = checkpoint_domain_scales(checkpoint, domain)
            count = min(max(1, args.validation_frames), len(recording))
            selected = np.linspace(0, len(recording) - 1, count, dtype=int)
            plot_index = int(selected[len(selected) // 2])
            for index in selected:
                snapshot = recording[int(index)]
                inputs = select_checkpoint_inputs(
                    build_operator_input(snapshot, x_grid=x, y_grid=y, scales=scales),
                    checkpoint,
                    domain=domain,
                )
                tensor = torch.from_numpy(inputs[None]).to(device)
                started = time.perf_counter()
                prediction_norm, _ = runtime_model(tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                pinn_ms = 1000.0 * (time.perf_counter() - started)
                prediction = (
                    prediction_norm[0, 0].detach().cpu().numpy().astype(np.float32)
                    * scales.risk
                )
                numerical = solver.solve(
                    snapshot["Q"],
                    snapshot["vx"],
                    snapshot["vy"],
                    snapshot["D"],
                    road_mask=snapshot["road_mask"],
                )
                row = {
                    "domain": _domain(recording.dataset),
                    "dataset": recording.dataset,
                    "recording_id": recording.recording_id,
                    "frame_id": int(snapshot["frame_id"]),
                    "time_s": float(snapshot["t"]),
                    "pinn_inference_ms": float(pinn_ms),
                    "numerical_solver_ms": float(solver.last_timing_ms),
                    **_metrics(prediction, np.asarray(snapshot["R"]), dx=dx, dy=dy),
                }
                rows.append(row)
                if int(index) == plot_index and plotted < args.validation_plots:
                    _plot_panel(
                        output
                        / f"heldout_{recording.dataset}_{recording.recording_id}_frame_{int(snapshot['frame_id'])}.png",
                        np.asarray(snapshot["R"]),
                        prediction,
                        x,
                        y,
                        f"Held-out {recording.dataset}, recording {recording.recording_id}",
                    )
                    plotted += 1
    with (output / "heldout_prospective_pinn_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(Path(args.model_out).resolve()),
        "validation_mode": "stateless_recording_disjoint",
        "teacher_version": "prospective_v2",
        "calibration_recordings": checkpoint["calibration_recordings"],
        "heldout_recordings": heldout.recording_keys,
        "aggregate": _aggregate(rows),
        "per_domain": {
            domain: _aggregate([row for row in rows if row["domain"] == domain])
            for domain in sorted({str(row["domain"]) for row in rows})
        },
        "per_dataset": {
            dataset: _aggregate([row for row in rows if row["dataset"] == dataset])
            for dataset in sorted({str(row["dataset"]) for row in rows})
        },
    }
    (output / "heldout_prospective_pinn_validation_summary.json").write_text(
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
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-width", type=int, default=128)
    parser.add_argument("--risk-patch-fraction", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dilations", type=_dilations, default=_dilations("1,2,4,8,16,1"))
    parser.add_argument("--max-normalized-risk", type=float, default=1.10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hotspot-boost", type=float, default=8.0)
    parser.add_argument("--w-field", type=float, default=1.0)
    parser.add_argument("--w-gradient", type=float, default=0.20)
    parser.add_argument("--w-physics", type=float, default=0.002)
    parser.add_argument("--gradient-active-quantile", type=float, default=0.85)
    parser.add_argument("--gradient-direction-weight", type=float, default=0.25)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-frames", type=int, default=50)
    parser.add_argument("--validation-plots", type=int, default=10)
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
    solver_spec = _require_prospective(calibration)
    if _require_prospective(heldout) != solver_spec:
        raise RuntimeError("Calibration and held-out caches use different solvers")
    overlap = set(calibration.recording_keys) & set(heldout.recording_keys)
    if overlap:
        raise ValueError(f"Calibration/held-out recording overlap: {sorted(overlap)}")
    args.heldout_recording_keys = heldout.recording_keys
    args.heldout_cache_signatures = heldout.signatures
    if args.stage in {"train", "all"}:
        train(args, calibration, solver_spec)
    if args.stage in {"validate", "all"}:
        validate(args, heldout, solver_spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
