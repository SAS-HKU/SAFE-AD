"""Animate numerical and PINN risk fields on a cached naturalistic recording."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from pinn_risk_field import ExiDLoader
from rl.risk.pinn_adapter import PINNRiskAdapter
from rl.risk.pinn_snapshot_cache import CachedRecording
from rl.risk.recurrent_pinn_operator import (
    build_operator_input,
    checkpoint_domain_scales,
    load_recurrent_pinn_checkpoint,
    select_checkpoint_inputs,
)


def _vehicle_patch(vehicle: dict, *, ego: bool) -> Rectangle:
    length = float(vehicle.get("length", 4.5))
    width = float(vehicle.get("width", 1.9))
    heading_deg = float(np.degrees(vehicle.get("heading", 0.0)))
    return Rectangle(
        (float(vehicle["x"]) - length / 2.0, float(vehicle["y"]) - width / 2.0),
        length,
        width,
        angle=heading_deg,
        rotation_point="center",
        facecolor="#d62728" if ego else "white",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )


def _ego_local_vehicles(vehicles: list[dict], ego: dict):
    heading = float(ego.get("heading", 0.0))
    c, s = float(np.cos(heading)), float(np.sin(heading))

    def transform(vehicle: dict) -> dict:
        dx = float(vehicle["x"]) - float(ego["x"])
        dy = float(vehicle["y"]) - float(ego["y"])
        value = dict(vehicle)
        value.update(
            x=c * dx + s * dy,
            y=-s * dx + c * dy,
            heading=float(vehicle.get("heading", 0.0)) - heading,
        )
        return value

    return [transform(vehicle) for vehicle in vehicles], transform(ego)


def _render_frame(
    *,
    snapshot: dict,
    prediction: np.ndarray,
    vehicles: list[dict],
    ego: dict,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    dataset: str,
    recording_id: str,
    output: Path,
    view_x: float,
    view_y: float,
) -> None:
    teacher = np.asarray(snapshot["R"], dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    positive = np.concatenate([teacher[teacher > 0.0], prediction[prediction > 0.0]])
    vmax = max(float(np.percentile(positive, 99.0)) if positive.size else 1.0, 1e-3)
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        (teacher, prediction),
        (
            "Prospective numerical teacher"
            if snapshot.get("teacher_version") == "prospective_v2"
            else "Numerical teacher",
            "Context-conditioned PINN",
        ),
    ):
        image = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="turbo",
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            alpha=0.88,
        )
        for vehicle in vehicles:
            ax.add_patch(_vehicle_patch(vehicle, ego=False))
        ax.add_patch(_vehicle_patch(ego, ego=True))
        ax.set_xlim(float(ego["x"]) - view_x, float(ego["x"]) + view_x)
        ax.set_ylim(float(ego["y"]) - view_y, float(ego["y"]) + view_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Longitudinal position (m)")
    axes[0].set_ylabel("Lateral position (m)")
    colorbar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.06, pad=0.08)
    colorbar.set_label("Propagated risk intensity", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{dataset} recording {recording_id} | "
        f"frame {int(snapshot['frame_id'])} | $t={float(snapshot['t']):.2f}$ s",
        fontsize=10,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="inD")
    parser.add_argument("--recording", default="06")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-root", default="evaluation/pinn_teacher_cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--view-x", type=float, default=55.0)
    parser.add_argument("--view-y", type=float, default=28.0)
    parser.add_argument(
        "--output",
        default="evaluation/pinn_context_multirecord/heldout_ind06_teacher_vs_pinn.gif",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_root) / f"{args.dataset}_{args.recording}"
    recording = CachedRecording(cache_path)
    checkpoint_payload = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model_family = checkpoint_payload.get("model_family")
    context_operator = model_family in {
        "recurrent_context_pinn",
        "prospective_context_pinn",
    }
    stateful = model_family == "recurrent_context_pinn"
    if context_operator:
        recurrent_model, checkpoint_payload = load_recurrent_pinn_checkpoint(
            args.checkpoint, device=args.device
        )
        recurrent_scales = checkpoint_domain_scales(
            checkpoint_payload, "naturalistic"
        )
        previous_prediction = np.zeros(
            (len(recording.y_grid), len(recording.x_grid)), dtype=np.float32
        )
        previous_rate = np.zeros_like(previous_prediction)
        adapter = None
    else:
        adapter = PINNRiskAdapter(
            checkpoint_path=args.checkpoint,
            device=args.device,
            inference_x_range=(float(recording.x_grid.min()), float(recording.x_grid.max())),
            inference_y_range=(float(recording.y_grid.min()), float(recording.y_grid.max())),
            time_mode="error",
        )
        if not adapter.available:
            raise RuntimeError(f"Unable to load PINN checkpoint: {args.checkpoint}")
        adapter.warmup(compute_gradient=False)

    manifest = recording.manifest
    loader = ExiDLoader(
        data_dir=str(Path(args.data_root) / args.dataset),
        recording_id=args.recording,
        max_seconds=float(manifest["max_seconds"]),
        warmup_seconds=float(manifest["warmup_seconds"]),
        perception_range=float(manifest["perception_range"]),
        selection_mode=str(manifest["selection_mode"]),
        top_k=int(manifest["top_k"]),
        threshold_ratio=float(manifest["threshold_ratio"]),
    )
    tracks, tracks_meta, _recording_meta = loader._read_csv()
    frame_lookup = loader._build_frame_lookup(tracks, tracks_meta)
    all_frames = sorted(frame_lookup)
    loader._compute_coord_offset(frame_lookup[all_frames[len(all_frames) // 2]])

    X, Y = np.meshgrid(recording.x_grid, recording.y_grid)
    requested = np.arange(0, len(recording), max(1, int(args.stride)), dtype=int)
    requested = requested[: max(1, int(args.frames))]
    requested_set = set(int(value) for value in requested)
    rendered: list[Image.Image] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="safer_dataset_pinn_") as temporary:
        temporary_dir = Path(temporary)
        output_index = 0
        final_index = max(requested_set)
        for snapshot_index in range(final_index + 1):
            snapshot = dict(recording[int(snapshot_index)])
            snapshot["teacher_version"] = manifest.get("teacher_version", "legacy")
            frame_id = int(snapshot["frame_id"])
            vehicles, ego, _context = loader._frame_to_drift(frame_lookup[frame_id])
            if ego is None:
                continue
            if manifest.get("coordinate_mode") == "ego_local":
                vehicles, ego = _ego_local_vehicles(vehicles, ego)
            if context_operator:
                inputs = build_operator_input(
                    snapshot,
                    x_grid=recording.x_grid,
                    y_grid=recording.y_grid,
                    scales=recurrent_scales,
                    previous_risk=previous_prediction if stateful else None,
                    previous_rate=previous_rate if stateful else None,
                )
                inputs = select_checkpoint_inputs(
                    inputs,
                    checkpoint_payload,
                    domain="naturalistic",
                )
                with torch.inference_mode():
                    risk_norm, rate_norm = recurrent_model(
                        torch.from_numpy(inputs[None]).to(args.device)
                    )
                prediction = (
                    risk_norm[0, 0].cpu().numpy() * recurrent_scales.risk
                )
                if stateful:
                    previous_rate = (
                        rate_norm[0, 0].cpu().numpy() * recurrent_scales.risk_rate
                    )
                    previous_prediction = np.asarray(prediction, dtype=np.float32)
            else:
                prediction = adapter.query_grid(
                    X=X,
                    Y=Y,
                    t=float(snapshot["t"]),
                    Q=snapshot["Q"],
                    vx=snapshot["vx"],
                    vy=snapshot["vy"],
                    D=snapshot["D"],
                    vehicles=vehicles,
                    ego_vehicle=ego,
                    selection_mode=str(manifest["selection_mode"]),
                    top_k=int(manifest["top_k"]),
                    threshold_ratio=float(manifest["threshold_ratio"]),
                )
            if snapshot_index not in requested_set:
                continue
            png = temporary_dir / f"frame_{output_index:04d}.png"
            _render_frame(
                snapshot=snapshot,
                prediction=prediction,
                vehicles=vehicles,
                ego=ego,
                x_grid=recording.x_grid,
                y_grid=recording.y_grid,
                dataset=args.dataset,
                recording_id=args.recording,
                output=png,
                view_x=float(args.view_x),
                view_y=float(args.view_y),
            )
            with Image.open(png) as image:
                rendered.append(image.convert("RGB").copy())
            output_index += 1

    if not rendered:
        raise RuntimeError("No dataset frames were rendered")
    duration_ms = max(20, int(round(1000.0 / max(float(args.fps), 0.1))))
    rendered[0].save(
        output,
        save_all=True,
        append_images=rendered[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    rendered[len(rendered) // 2].save(output.with_name(f"{output.stem}_snapshot.png"))
    print(f"Saved {len(rendered)} synchronized frames to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
