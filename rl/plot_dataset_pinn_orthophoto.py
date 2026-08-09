"""Render numerical-teacher and PINN risk fields on official dataset maps.

The prospective caches store fields on an ego-local metric grid.  This script
uses the frame and ego track recorded in each cache to recover the matching
dataset pose, rotates the local grid into world coordinates, and then applies
the same world-to-pixel transform used by the official track visualizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

try:
    import scienceplots  # noqa: F401
except ImportError:
    scienceplots = None

from tracks_import import read_from_dataset
from rl.risk.pinn_snapshot_cache import CachedRecording
from rl.risk.recurrent_pinn_operator import (
    build_operator_input,
    checkpoint_domain_scales,
    load_recurrent_pinn_checkpoint,
    select_checkpoint_inputs,
)


@dataclass(frozen=True)
class SceneSpec:
    dataset: str
    recording: str
    cache: Path
    status: str


SCENES = (
    SceneSpec(
        "rounD",
        "00",
        Path("evaluation/pinn_prospective_multidataset_v4/rounD_00"),
        "external replay",
    ),
    SceneSpec(
        "inD",
        "06",
        Path("evaluation/pinn_prospective_v2_cache/inD_06"),
        "recording-disjoint test",
    ),
    SceneSpec(
        "exiD",
        "01",
        Path("evaluation/pinn_prospective_multidataset_v4/exiD_01"),
        "external replay",
    ),
)

EGO_COLOR = "#F0441E"
HEAVY_COLOR = "#FF9200"
SURROUNDING_COLOR = "#A9D2E8"
EDGE_COLOR = "#1F2933"


def _style() -> None:
    try:
        plt.style.use(["science", "no-latex"])
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.2,
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "axes.titlepad": 4.0,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_frame(
    recording: CachedRecording,
    allowed_ego_ids: set[int],
) -> tuple[int, dict]:
    """Select the highest integrated-risk frame with a road-vehicle ego."""
    candidates = [
        index
        for index in range(len(recording))
        if int(round(float(recording[index]["ego_trackId"]))) in allowed_ego_ids
    ]
    if not candidates:
        raise RuntimeError(f"No passenger/heavy ego frame in {recording.path}")
    index = max(
        candidates,
        key=lambda value: float(np.nansum(recording[value]["R"])),
    )
    return index, recording[index]


def _predict(model, checkpoint: dict, recording: CachedRecording, snapshot: dict, device: str):
    scales = checkpoint_domain_scales(checkpoint, "naturalistic")
    operator_input = build_operator_input(
        snapshot,
        x_grid=recording.x_grid,
        y_grid=recording.y_grid,
        scales=scales,
    )
    operator_input = select_checkpoint_inputs(
        operator_input,
        checkpoint,
        domain="naturalistic",
    )
    tensor = torch.from_numpy(operator_input[None]).to(device)
    with torch.inference_mode():
        prediction, _ = model(tensor)
    return prediction[0, 0].detach().cpu().numpy() * float(scales.risk)


def _background_path(data_dir: Path, recording: str) -> Path:
    for name in (
        f"{recording}_background.png",
        f"{recording}_background.jpg",
        f"{recording}_highway.png",
        f"{recording}_highway.jpg",
    ):
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No orthophoto found for {data_dir.name} recording {recording}")


def _display_scale(dataset: str, tracks: list[dict], image: np.ndarray) -> float:
    """Infer whether the supplied orthophoto is stored at a reduced pixel scale."""
    x = np.concatenate([np.asarray(track["xCenterVis"], dtype=float) for track in tracks])
    y = np.concatenate([np.asarray(track["yCenterVis"], dtype=float) for track in tracks])
    height, width = image.shape[:2]
    x_high = float(np.nanpercentile(x, 99.9))
    y_high = float(np.nanpercentile(y, 99.9))
    # exiD orthophotos in this workspace retain the native visual resolution.
    # If the recorded pixel coordinates already fit, no scale is needed.
    if x_high <= 1.02 * float(width) and y_high <= 1.02 * float(height):
        return 1.0
    # inD and rounD distribute pre-scaled orthophotos with the official
    # drone-dataset-tools display reductions.
    official_reductions = {"ind": 12.0, "round": 10.0, "roundd": 10.0}
    dataset_key = str(dataset).lower()
    if dataset_key in official_reductions:
        return official_reductions[dataset_key]
    ratio = max(
        x_high / max(float(width - 1), 1.0),
        y_high / max(float(height - 1), 1.0),
        1.0,
    )
    # Official drone-dataset orthophotos use integer display reductions.
    return float(max(1, int(np.ceil(ratio - 1e-6))))


def _track_lookup(tracks: list[dict], metadata: list[dict]):
    return (
        {int(track["trackId"]): track for track in tracks},
        {int(meta["trackId"]): meta for meta in metadata},
    )


def _frame_index(track: dict, meta: dict, frame_id: int) -> int:
    frames = np.asarray(track.get("frame", []), dtype=int)
    matches = np.flatnonzero(frames == int(frame_id))
    if matches.size:
        return int(matches[0])
    index = int(frame_id) - int(meta["initialFrame"])
    if not 0 <= index < len(track["xCenter"]):
        raise IndexError(
            f"Frame {frame_id} lies outside track {track['trackId']} "
            f"[{meta['initialFrame']}, {meta['finalFrame']}]"
        )
    return index


def _official_ego_pose(
    snapshot: dict,
    tracks_by_id: dict[int, dict],
    meta_by_id: dict[int, dict],
) -> tuple[float, float, float, int]:
    track_id = int(round(float(snapshot["ego_trackId"])))
    frame_id = int(round(float(snapshot["frame_id"])))
    track = tracks_by_id[track_id]
    index = _frame_index(track, meta_by_id[track_id], frame_id)
    return (
        float(track["xCenter"][index]),
        float(track["yCenter"][index]),
        float(np.deg2rad(track["heading"][index])),
        track_id,
    )


def _field_pixels(
    recording: CachedRecording,
    pose: tuple[float, float, float, int],
    ortho_px_to_m: float,
    display_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    local_x, local_y = np.meshgrid(recording.x_grid, recording.y_grid)
    ego_x, ego_y, heading, _ = pose
    cosine, sine = float(np.cos(heading)), float(np.sin(heading))
    world_x = ego_x + cosine * local_x - sine * local_y
    world_y = ego_y + sine * local_x + cosine * local_y
    pixel_x = world_x / (float(ortho_px_to_m) * display_scale)
    pixel_y = -world_y / (float(ortho_px_to_m) * display_scale)
    return pixel_x, pixel_y


def _risk_scale(teacher: np.ndarray, prediction: np.ndarray) -> float:
    values = np.concatenate((teacher.reshape(-1), prediction.reshape(-1)))
    values = values[np.isfinite(values) & (values > 0.0)]
    return max(float(np.percentile(values, 98.5)) if values.size else 1.0, 1e-6)


def _draw_vehicles(
    axis,
    tracks: list[dict],
    metadata: list[dict],
    frame_id: int,
    ego_track_id: int,
    display_scale: float,
) -> None:
    tracks_by_id, _ = _track_lookup(tracks, metadata)
    for meta in metadata:
        track_id = int(meta["trackId"])
        if not int(meta["initialFrame"]) <= frame_id <= int(meta["finalFrame"]):
            continue
        track = tracks_by_id[track_id]
        index = _frame_index(track, meta, frame_id)
        vehicle_class = str(meta.get("class", "car")).lower()
        is_ego = track_id == ego_track_id
        is_heavy = vehicle_class in {"truck", "bus", "van", "truck_bus"}
        face = EGO_COLOR if is_ego else HEAVY_COLOR if is_heavy else SURROUNDING_COLOR
        edge = "#FF1E00" if is_ego else EDGE_COLOR
        linewidth = 0.9 if is_ego else 0.45
        if track.get("bboxVis") is not None:
            vertices = np.asarray(track["bboxVis"][index], dtype=float) / display_scale
            axis.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=linewidth,
                    alpha=0.96,
                    zorder=8 if is_ego else 7,
                )
            )


def _draw_panel(
    axis,
    *,
    image: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    field: np.ndarray,
    scale: float,
    tracks: list[dict],
    metadata: list[dict],
    frame_id: int,
    ego_track_id: int,
    display_scale: float,
) -> None:
    axis.imshow(image, origin="upper", zorder=0)
    smoothed = gaussian_filter(np.maximum(np.asarray(field, dtype=float), 0.0), 1.15)
    normalized = np.clip(smoothed / scale, 0.0, 1.0)
    visible = np.ma.masked_less_equal(normalized, 0.025)
    if np.ma.count(visible):
        axis.contourf(
            pixel_x,
            pixel_y,
            visible,
            levels=np.linspace(0.025, 1.0, 64),
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
            alpha=0.54,
            antialiased=True,
            zorder=4,
        )
    _draw_vehicles(
        axis,
        tracks,
        metadata,
        frame_id,
        ego_track_id,
        display_scale,
    )
    finite_x = pixel_x[np.isfinite(pixel_x)]
    finite_y = pixel_y[np.isfinite(pixel_y)]
    x0, x1 = float(np.min(finite_x)), float(np.max(finite_x))
    y0, y1 = float(np.min(finite_y)), float(np.max(finite_y))
    pad_x = 0.035 * max(x1 - x0, 1.0)
    pad_y = 0.07 * max(y1 - y0, 1.0)
    height, width = image.shape[:2]
    axis.set_xlim(max(0.0, x0 - pad_x), min(float(width - 1), x1 + pad_x))
    axis.set_ylim(min(float(height - 1), y1 + pad_y), max(0.0, y0 - pad_y))
    axis.set_facecolor("black")
    # Keep every manuscript panel landscape.  ``datalim`` expands the visible
    # orthophoto context when the ego-local grid is diagonal instead of
    # distorting the map or shrinking the physical axes.
    axis.set_box_aspect(0.58)
    axis.set_aspect("equal", adjustable="datalim")
    axis.axis("off")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/pinn_revision_figures_v5"),
    )
    args = parser.parse_args()
    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_recurrent_pinn_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    model.eval()

    fig, axes = plt.subplots(
        len(SCENES),
        2,
        figsize=(7.15, 5.35),
        gridspec_kw={"hspace": 0.08, "wspace": 0.035},
    )
    provenance = []
    for row, scene in enumerate(SCENES):
        recording = CachedRecording(scene.cache)
        data_dir = args.data_root / scene.dataset
        tracks, metadata, recording_meta = read_from_dataset(
            str(data_dir),
            dataset_name=scene.dataset,
            recording=scene.recording,
            include_px_coordinates=True,
        )
        allowed_ego_ids = {
            int(meta["trackId"])
            for meta in metadata
            if str(meta.get("class", "car")).lower()
            in {"car", "van", "truck", "bus", "truck_bus"}
        }
        cache_index, snapshot = _active_frame(recording, allowed_ego_ids)
        prediction = _predict(model, checkpoint, recording, snapshot, args.device)
        teacher = np.asarray(snapshot["R"], dtype=float)
        scale = _risk_scale(teacher, prediction)
        background_path = _background_path(data_dir, scene.recording)
        raw = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError(f"Unable to read {background_path}")
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        display_scale = _display_scale(scene.dataset, tracks, image)
        tracks_by_id, meta_by_id = _track_lookup(tracks, metadata)
        pose = _official_ego_pose(snapshot, tracks_by_id, meta_by_id)
        pixel_x, pixel_y = _field_pixels(
            recording,
            pose,
            float(recording_meta["orthoPxToMeter"]),
            display_scale,
        )
        frame_id = int(round(float(snapshot["frame_id"])))
        for column, field in enumerate((teacher, prediction)):
            _draw_panel(
                axes[row, column],
                image=image,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                field=field,
                scale=scale,
                tracks=tracks,
                metadata=metadata,
                frame_id=frame_id,
                ego_track_id=pose[3],
                display_scale=display_scale,
            )
        axes[row, 0].text(
            -0.035,
            0.5,
            scene.dataset,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            fontstyle="italic",
        )
        provenance.append(
            {
                "dataset": scene.dataset,
                "recording": scene.recording,
                "frame_id": frame_id,
                "ego_track_id": pose[3],
                "cache_index": cache_index,
                "selection_rule": (
                    "maximum integrated numerical-teacher risk among frames "
                    "whose cached ego is a passenger or heavy road vehicle"
                ),
                "cache": scene.cache.as_posix(),
                "background": background_path.as_posix(),
                "display_scale": display_scale,
                "status": scene.status,
            }
        )

    axes[0, 0].set_title("Numerical prospective field", fontweight="bold")
    axes[0, 1].set_title(r"Context-conditioned PINN query $\widehat{R}_{\theta}$", fontweight="bold")
    color_axis = fig.add_axes([0.925, 0.19, 0.014, 0.62])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo"),
        cax=color_axis,
    )
    colorbar.set_label("Scene-normalized risk intensity")
    legend = (
        Patch(facecolor=EGO_COLOR, edgecolor="#FF1E00", label="ego vehicle"),
        Patch(facecolor=HEAVY_COLOR, edgecolor=EDGE_COLOR, label="heavy vehicle"),
        Patch(facecolor=SURROUNDING_COLOR, edgecolor=EDGE_COLOR, label="surrounding vehicle"),
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.49, 0.005),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.07, right=0.90, top=0.95, bottom=0.07)
    output = args.output_dir / "pinn_naturalistic_orthophoto_overlay"
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": args.device,
        "coordinate_pipeline": (
            "ego-local field -> official recorded ego pose -> dataset world coordinates "
            "-> x/orthoPxToMeter, -y/orthoPxToMeter -> orthophoto display scale"
        ),
        "normalization": "shared within each dataset row; 98.5th percentile of teacher and PINN",
        "panels": provenance,
        "interpretation_boundary": (
            "inD is recording-disjoint. rounD and exiD are external qualitative replays "
            "because this checkpoint was not calibrated on those datasets."
        ),
    }
    (args.output_dir / "pinn_naturalistic_orthophoto_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(output.with_suffix(".pdf").resolve())
    print(output.with_suffix(".png").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
