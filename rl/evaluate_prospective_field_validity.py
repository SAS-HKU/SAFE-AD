"""Independent recording-disjoint validation of prospective-v2 risk fields.

Future naturalistic trajectories are used only to construct occupancy labels.
The labels are expressed in the future ego frame and accumulated over the
deployment horizon, matching the causal ego-relative field semantics without
entering numerical-teacher or PINN training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from pinn_risk_field import ExiDLoader
from rl.risk.pinn_snapshot_cache import CachedRecording
from rl.risk.recurrent_pinn_operator import (
    build_operator_input,
    checkpoint_domain_scales,
    load_recurrent_pinn_checkpoint,
    select_checkpoint_inputs,
)


def _ids(value: str) -> list[str]:
    return [token.strip().zfill(2) for token in value.split(",") if token.strip()]


def _future_occupancy_envelope(
    *,
    frame_lookup: dict,
    frame_id: int,
    frame_rate: float,
    horizon_s: float,
    ego_track_id: int,
    X: np.ndarray,
    Y: np.ndarray,
    sample_step_s: float,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros_like(X, dtype=bool)
    roi = X * X + Y * Y <= float(radius_m) ** 2
    offsets = np.arange(
        float(sample_step_s),
        float(horizon_s) + 0.5 * float(sample_step_s),
        float(sample_step_s),
    )
    for elapsed in offsets:
        target_frame = int(round(frame_id + float(elapsed) * frame_rate))
        entries = frame_lookup.get(target_frame, [])
        ego = next(
            (entry for entry in entries if int(entry["trackId"]) == ego_track_id),
            None,
        )
        if ego is None:
            continue
        heading = float(ego.get("heading", 0.0))
        c, s = float(np.cos(heading)), float(np.sin(heading))
        for other in entries:
            if int(other["trackId"]) == ego_track_id:
                continue
            dx = float(other["x"] - ego["x"])
            dy = float(other["y"] - ego["y"])
            center_x = c * dx + s * dy
            center_y = -s * dx + c * dy
            if center_x * center_x + center_y * center_y > float(radius_m) ** 2:
                continue
            relative_heading = float(other.get("heading", 0.0)) - heading
            ch, sh = float(np.cos(relative_heading)), float(np.sin(relative_heading))
            local_dx = X - center_x
            local_dy = Y - center_y
            longitudinal = ch * local_dx + sh * local_dy
            lateral = -sh * local_dx + ch * local_dy
            half_length = 0.5 * float(other.get("length", 4.8)) + 1.5
            half_width = 0.5 * float(other.get("width", 1.9)) + 1.0
            positive |= (
                (np.abs(longitudinal) <= half_length)
                & (np.abs(lateral) <= half_width)
                & roi
            )
    return positive, roi


def _metrics(field: np.ndarray, positive: np.ndarray, roi: np.ndarray) -> dict | None:
    valid = np.asarray(roi, dtype=bool) & np.isfinite(field)
    label = np.asarray(positive[valid], dtype=np.uint8)
    if label.sum() == 0 or label.sum() == len(label):
        return None
    score = np.asarray(field[valid], dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(label, score)),
        "auprc": float(average_precision_score(label, score)),
        "positive_mean": float(np.mean(score[label == 1])),
        "background_mean": float(np.mean(score[label == 0])),
    }


def _bootstrap(values: np.ndarray, *, repetitions: int, seed: int):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (int(repetitions), len(values)), replace=True).mean(1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _predict(recording: CachedRecording, checkpoint_path: str, device: str):
    if not checkpoint_path:
        return None
    model, checkpoint = load_recurrent_pinn_checkpoint(checkpoint_path, device=device)
    if checkpoint.get("model_family") != "prospective_context_pinn":
        raise ValueError("Expected a prospective_context_pinn checkpoint")
    scales = checkpoint_domain_scales(checkpoint, "naturalistic")
    predictions = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(recording)):
            snapshot = recording[index]
            array = build_operator_input(
                snapshot,
                x_grid=np.asarray(recording.x_grid),
                y_grid=np.asarray(recording.y_grid),
                scales=scales,
            )
            array = select_checkpoint_inputs(
                array,
                checkpoint,
                domain="naturalistic",
            )
            risk_norm, _ = model(torch.from_numpy(array[None]).to(device))
            predictions.append(
                risk_norm[0, 0].cpu().numpy().astype(np.float32) * scales.risk
            )
    return predictions


def _plot(
    path: Path,
    fields: dict,
    positive,
    roi,
    x,
    y,
    title: str,
    *,
    column_layout: bool = False,
) -> None:
    vmax = max(float(np.percentile(field[roi], 99)) for field in fields.values())
    if column_layout:
        fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.65), constrained_layout=True)
        axes = np.asarray(axes).ravel()
    else:
        fig, axes = plt.subplots(
            1, len(fields), figsize=(3.15 * len(fields), 2.75), constrained_layout=True
        )
    axes = np.atleast_1d(axes)
    used_axes = []
    for panel_index, (axis, (label, field)) in enumerate(zip(axes, fields.items())):
        image = axis.imshow(
            np.where(roi, field, np.nan),
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="auto",
            cmap="turbo",
            vmin=0.0,
            vmax=max(vmax, 1e-6),
        )
        axis.contour(x, y, positive.astype(float), levels=[0.5], colors="white", linewidths=0.8)
        axis.set_title(f"({chr(97 + panel_index)}) {label}", fontsize=7.2)
        axis.tick_params(labelsize=6.3)
        if not column_layout or panel_index >= 1:
            axis.set_xlabel("Longitudinal position (m)", fontsize=6.5)
        if not column_layout or panel_index % 2 == 0:
            axis.set_ylabel("Lateral position (m)", fontsize=6.5)
        used_axes.append(axis)
    for axis in axes[len(fields) :]:
        axis.axis("off")
    if column_layout and len(axes) > len(fields):
        axes[-1].text(
            0.5,
            0.5,
            "White contour:\nobserved occupancy\nduring the next 3 s",
            ha="center",
            va="center",
            fontsize=7,
            transform=axes[-1].transAxes,
        )
    fig.colorbar(
        image,
        ax=used_axes,
        shrink=0.82,
        orientation="horizontal" if column_layout else "vertical",
        label="Risk intensity",
        pad=0.03,
    )
    if not column_layout:
        fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="inD")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path("evaluation/pinn_prospective_v2_cache")
    )
    parser.add_argument("--recordings", default="06,07,08,09,10,11")
    parser.add_argument("--horizon-s", type=float, default=3.0)
    parser.add_argument("--sample-step-s", type=float, default=0.2)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--radius-m", type=float, default=80.0)
    parser.add_argument("--pinn-checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/prospective_field_validity")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best_example = None
    best_example_score = -np.inf
    for recording_id in _ids(args.recordings):
        recording = CachedRecording(args.cache_root / f"{args.dataset}_{recording_id}")
        loader = ExiDLoader(
            data_dir=str(args.data_root / args.dataset),
            recording_id=recording_id,
            max_seconds=0.1,
            warmup_seconds=0.0,
        )
        tracks, tracks_meta, rec_meta = loader._read_csv()
        lookup = loader._build_frame_lookup(tracks, tracks_meta)
        frame_rate = float(rec_meta["frameRate"])
        X, Y = np.meshgrid(np.asarray(recording.x_grid), np.asarray(recording.y_grid))
        predictions = _predict(recording, args.pinn_checkpoint, args.device)
        for index in range(0, len(recording), max(1, int(args.frame_stride))):
            snapshot = recording[index]
            positive, roi = _future_occupancy_envelope(
                frame_lookup=lookup,
                frame_id=int(snapshot["frame_id"]),
                frame_rate=frame_rate,
                horizon_s=float(args.horizon_s),
                ego_track_id=int(round(snapshot["ego_trackId"])),
                X=X,
                Y=Y,
                sample_step_s=float(args.sample_step_s),
                radius_m=float(args.radius_m),
            )
            fields = {
                "Instantaneous source": np.asarray(snapshot["Q"]),
                "Prospective numerical field": np.asarray(snapshot["R"]),
            }
            if predictions is not None:
                fields["Context PINN"] = predictions[index]
            frame_metrics = {}
            for model_name, field in fields.items():
                result = _metrics(field, positive, roi)
                if result is not None:
                    frame_metrics[model_name] = result
                    rows.append(
                        {
                            "recording_id": recording_id,
                            "frame_id": int(snapshot["frame_id"]),
                            "horizon_s": float(args.horizon_s),
                            "model": model_name,
                            "positive_cells": int(positive.sum()),
                            **result,
                        }
                    )
            if {
                "Instantaneous source",
                "Prospective numerical field",
            } <= set(frame_metrics):
                central_positive = positive & (X >= -20.0) & (X <= 80.0) & (np.abs(Y) <= 17.0)
                score = (
                    frame_metrics["Prospective numerical field"]["auprc"]
                    - frame_metrics["Instantaneous source"]["auprc"]
                    + 1e-4 * float(central_positive.sum())
                )
                if central_positive.any() and score > best_example_score:
                    best_example_score = score
                    best_example = (
                        {key: np.asarray(value).copy() for key, value in fields.items()},
                        positive.copy(),
                        roi.copy(),
                        np.asarray(recording.x_grid).copy(),
                        np.asarray(recording.y_grid).copy(),
                        f"Held-out recording {recording_id}: observed occupancy over {args.horizon_s:g} s (white)",
                    )
    if not rows:
        raise RuntimeError("No valid future-occupancy labels were produced")
    if best_example is not None:
        _plot(
            args.output_dir / "heldout_future_occupancy_envelope.png",
            *best_example,
        )
        _plot(
            args.output_dir / "heldout_future_occupancy_envelope_column.png",
            *best_example,
            column_layout=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "prospective_field_validity_frames.csv", index=False)
    per_recording = (
        frame.groupby(["recording_id", "model"], as_index=False)[
            ["auroc", "auprc", "positive_mean", "background_mean"]
        ]
        .mean()
    )
    per_recording.to_csv(
        args.output_dir / "prospective_field_validity_recordings.csv", index=False
    )
    summary_rows = []
    for model_name, group in per_recording.groupby("model"):
        for metric in ("auroc", "auprc"):
            mean, low, high = _bootstrap(
                group[metric].to_numpy(),
                repetitions=args.bootstrap,
                seed=2026 + len(summary_rows),
            )
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_recordings": int(group["recording_id"].nunique()),
                }
            )
    contrasts = []
    pivot = per_recording.pivot(index="recording_id", columns="model", values=["auroc", "auprc"])
    for candidate in ("Prospective numerical field", "Context PINN"):
        if candidate not in pivot.columns.get_level_values(1):
            continue
        for metric in ("auroc", "auprc"):
            delta = (
                pivot[(metric, candidate)] - pivot[(metric, "Instantaneous source")]
            ).dropna().to_numpy()
            mean, low, high = _bootstrap(
                delta, repetitions=args.bootstrap, seed=4400 + len(contrasts)
            )
            contrasts.append(
                {
                    "candidate": candidate,
                    "baseline": "Instantaneous source",
                    "metric": metric,
                    "paired_delta": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_recordings": len(delta),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "prospective_field_validity_summary.csv", index=False
    )
    pd.DataFrame(contrasts).to_csv(
        args.output_dir / "prospective_field_validity_contrasts.csv", index=False
    )
    metadata = {
        "dataset": args.dataset,
        "recordings": _ids(args.recordings),
        "horizon_s": float(args.horizon_s),
        "future_trajectories_used_only_as_labels": True,
        "label_frame": "future_ego_relative",
        "bootstrap_unit": "recording",
        "pinn_checkpoint": str(Path(args.pinn_checkpoint).resolve()) if args.pinn_checkpoint else None,
    }
    (args.output_dir / "prospective_field_validity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(contrasts).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
