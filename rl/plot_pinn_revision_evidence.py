"""Create reviewer-facing PINN evidence figures from recorded evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import torch

try:
    import scienceplots  # noqa: F401
except ImportError:
    scienceplots = None

from rl.risk.pinn_snapshot_cache import CachedRecording
from rl.risk.recurrent_pinn_operator import (
    build_operator_input,
    checkpoint_domain_scales,
    load_recurrent_pinn_checkpoint,
    select_checkpoint_inputs,
)


SCENARIOS = (
    ("highwayenv_highway_v0", "Highway"),
    ("highwayenv_merge_v0", "Merge"),
    ("highwayenv_intersection_v0", "Intersection"),
    ("highwayenv_roundabout_v0", "Roundabout"),
)


def _style() -> None:
    try:
        plt.style.use(["science", "no-latex"])
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.6,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.6,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "axes.titlepad": 3.0,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_frame(recording: CachedRecording) -> tuple[int, dict]:
    """Select the maximum integrated-risk frame using a declared fixed rule."""
    masses = [float(np.nansum(recording[index]["R"])) for index in range(len(recording))]
    index = int(np.nanargmax(masses))
    return index, recording[index]


def _predict(model, checkpoint: dict, recording: CachedRecording, snapshot: dict, device: str):
    domain = (
        "highwayenv"
        if str(recording.dataset).startswith("highwayenv_")
        else "naturalistic"
    )
    scales = checkpoint_domain_scales(checkpoint, domain)
    inputs = build_operator_input(
        snapshot,
        x_grid=recording.x_grid,
        y_grid=recording.y_grid,
        scales=scales,
    )
    inputs = select_checkpoint_inputs(inputs, checkpoint, domain=domain)
    with torch.inference_mode():
        prediction, _ = model(torch.from_numpy(inputs[None]).to(device))
    return prediction[0, 0].detach().cpu().numpy() * float(scales.risk)


def _robust_scale(*fields: np.ndarray) -> float:
    values = np.concatenate(
        [np.asarray(field, dtype=float).reshape(-1) for field in fields]
    )
    values = values[np.isfinite(values) & (values > 0.0)]
    return max(float(np.percentile(values, 99.0)) if values.size else 1.0, 1e-6)


def _draw_field(
    axis,
    field: np.ndarray,
    snapshot: dict,
    recording: CachedRecording,
    *,
    scale: float,
    error: bool = False,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
) -> None:
    x = np.asarray(recording.x_grid)
    y = np.asarray(recording.y_grid)
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    road = np.asarray(snapshot.get("road_mask", np.ones_like(field)), dtype=float)
    base = np.where(road > 0.08, 0.28, 0.06)
    axis.imshow(
        base,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    normalized = np.clip(np.asarray(field, dtype=float) / max(scale, 1e-8), 0.0, 1.0)
    mask = (road > 0.05) & (normalized > 0.008)
    overlay = np.ma.masked_where(~mask, normalized)
    axis.imshow(
        overlay,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma" if error else "turbo",
        vmin=0.0,
        vmax=1.0,
        alpha=np.where(mask, 0.28 + 0.72 * np.sqrt(normalized), 0.0),
        interpolation="bilinear",
    )
    occupancy = np.asarray(snapshot.get("occ_mask", np.zeros_like(field)), dtype=float)
    if np.nanmax(occupancy) > 0.05:
        xx, yy = np.meshgrid(x, y)
        axis.contour(
            xx,
            yy,
            occupancy,
            levels=[max(0.1, 0.35 * float(np.nanmax(occupancy)))],
            colors="white",
            linewidths=0.45,
            alpha=0.9,
        )
    axis.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=22,
        color="white",
        edgecolors="black",
        linewidths=0.45,
        zorder=8,
    )
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.set_ylim(float(y[0]), float(y[-1]))
    axis.set_xlabel(r"$x_{\mathrm{ego}}$ (m)" if show_xlabel else "")
    axis.set_ylabel(r"$y_{\mathrm{ego}}$ (m)" if show_ylabel else "")
    if not show_xlabel:
        axis.set_xticklabels([])
    if not show_ylabel:
        axis.set_yticklabels([])
    axis.grid(False)


def _naturalistic_figure(
    output: Path,
    model,
    checkpoint: dict,
    device: str,
    cache_specs: list[tuple[str, str, Path, str]],
) -> list[dict]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.15, 3.15),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    provenance = []
    for column, (dataset, geometry, path, split) in enumerate(cache_specs):
        recording = CachedRecording(path)
        index, snapshot = _active_frame(recording)
        prediction = _predict(model, checkpoint, recording, snapshot, device)
        source = np.asarray(snapshot["Q"], dtype=float)
        source_scale = _robust_scale(source)
        prediction_scale = _robust_scale(prediction)
        _draw_field(
            axes[0, column],
            source,
            snapshot,
            recording,
            scale=source_scale,
            show_ylabel=column == 0,
        )
        _draw_field(
            axes[1, column],
            prediction,
            snapshot,
            recording,
            scale=prediction_scale,
            show_xlabel=True,
            show_ylabel=column == 0,
        )
        axes[0, column].set_title(
            f"({chr(97 + column)}) {dataset}: {geometry}\n{split}",
            fontweight="bold",
        )
        provenance.append(
            {
                "dataset": dataset,
                "geometry": geometry,
                "cache": path.as_posix(),
                "recording": recording.recording_id,
                "frame_index": index,
                "frame_id": int(snapshot["frame_id"]),
                "selection_rule": "maximum integrated prospective teacher risk",
                "split": split,
            }
        )
    axes[0, 0].text(
        -0.21,
        0.5,
        r"Scene source $Q$",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontweight="bold",
    )
    axes[1, 0].text(
        -0.21,
        0.5,
        r"PINN query $\widehat{R}_{\theta}$",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontweight="bold",
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo"),
        ax=axes,
        location="bottom",
        fraction=0.045,
        pad=0.04,
        aspect=45,
    )
    colorbar.set_label("Per-scene normalized source / risk intensity")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return provenance


def _validity_panel(axis, validity: pd.DataFrame) -> None:
    labels = ("Source", "Numerical", "PINN")
    models = (
        "Instantaneous source",
        "Prospective numerical field",
        "Context PINN",
    )
    colors = ("#7A7A7A", "#D55E00", "#0072B2")
    metrics = ("auroc", "auprc")
    x = np.arange(len(metrics), dtype=float)
    width = 0.24
    for index, (label, model, color) in enumerate(zip(labels, models, colors)):
        group = validity[validity["model"] == model].set_index("metric")
        mean = np.array([float(group.loc[metric, "mean"]) for metric in metrics])
        low = np.array([float(group.loc[metric, "ci_low"]) for metric in metrics])
        high = np.array([float(group.loc[metric, "ci_high"]) for metric in metrics])
        axis.bar(
            x + (index - 1) * width,
            mean,
            width,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            yerr=np.vstack((mean - low, high - mean)),
            capsize=1.8,
            error_kw={"elinewidth": 0.7},
            label=label,
        )
    axis.set_xticks(x, ("AUROC", "AUPRC"))
    axis.set_ylim(0.30, 1.0)
    axis.set_ylabel("Held-out score")
    axis.set_title("(e) Future-occupancy validity", fontweight="bold")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="y", alpha=0.2)


def _fidelity_panel(axis, summary: dict) -> None:
    colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
    markers = ("o", "s", "^", "D")
    for (dataset, label), color, marker in zip(SCENARIOS, colors, markers):
        values = summary["per_dataset"][dataset]
        axis.scatter(
            values["gradient_angle_deg"],
            values["hotspot_iou"],
            s=34,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=4,
        )
    axis.set_xlabel(r"Gradient angular error $\theta_{\nabla}$ (deg)")
    axis.set_ylabel("Hotspot IoU")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("(f) Spatial and directional fidelity", fontweight="bold")
    axis.legend(frameon=False, loc="best", ncol=2)
    axis.grid(alpha=0.2)


def _propagation_panel(axis, swap: dict) -> dict:
    traces = swap["frozen_state_action_trace"]
    descriptor = float(np.mean([row["descriptor_rmse"] for row in traces]))
    action_mean = float(np.mean([row["action_l2_mean"] for row in traces]))
    action_p95 = float(np.mean([row["action_l2_p95"] for row in traces]))
    exceedance = float(np.mean([row["action_l2_gt_0_05_rate"] for row in traces]))
    values = np.array((descriptor, action_mean, action_p95))
    labels = ("Descriptor\nRMSE", "Mean action\n$L_2$", "Action $L_2$\n95th pct.")
    axis.bar(
        np.arange(3),
        values,
        color=("#56B4E9", "#009E73", "#E69F00"),
        edgecolor="white",
        linewidth=0.5,
    )
    axis.set_yscale("log")
    axis.set_xticks(np.arange(3), labels)
    axis.set_ylabel("Absolute error (log scale)")
    axis.set_title("(g) Error propagation to policy", fontweight="bold")
    axis.grid(axis="y", alpha=0.2, which="both")
    axis.text(
        0.98,
        0.95,
        rf"$P(\Vert\Delta u\Vert_2>0.05)={100.0 * exceedance:.0f}\%$",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
    )
    return {
        "descriptor_rmse_mean": descriptor,
        "action_l2_mean": action_mean,
        "action_l2_p95_mean": action_p95,
        "action_l2_gt_0_05_rate": exceedance,
    }


def _highwayenv_figure(
    output: Path,
    model,
    checkpoint: dict,
    device: str,
    cache_root: Path,
    summary: dict,
    validity: pd.DataFrame,
    swap: dict,
) -> tuple[list[dict], dict]:
    fig = plt.figure(figsize=(7.15, 7.15), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(3.1, 1.25),
        left=0.075,
        right=0.90,
        bottom=0.075,
        top=0.965,
        hspace=0.34,
    )
    heat_grid = outer[0].subgridspec(3, 4, hspace=0.14, wspace=0.08)
    summary_grid = outer[1].subgridspec(1, 3, wspace=0.38)
    axes = np.empty((3, 4), dtype=object)
    provenance = []
    for column, (dataset, label) in enumerate(SCENARIOS):
        axes[:, column] = [
            fig.add_subplot(heat_grid[row, column]) for row in range(3)
        ]
        path = cache_root / f"{dataset}_seed0100"
        recording = CachedRecording(path)
        index, snapshot = _active_frame(recording)
        teacher = np.asarray(snapshot["R"], dtype=float)
        prediction = _predict(model, checkpoint, recording, snapshot, device)
        error = np.abs(prediction - teacher)
        scale = _robust_scale(teacher, prediction)
        for row, field in enumerate((teacher, prediction, error)):
            _draw_field(
                axes[row, column],
                field,
                snapshot,
                recording,
                scale=scale,
                error=row == 2,
                show_xlabel=row == 2,
                show_ylabel=column == 0,
            )
        metrics = summary["per_dataset"][dataset]
        axes[0, column].set_title(
            f"({chr(97 + column)}) {label}",
            fontweight="bold",
        )
        axes[0, column].text(
            0.02,
            0.96,
            rf"$\rho={metrics['correlation']:.2f}$, IoU$={metrics['hotspot_iou']:.2f}$"
            "\n"
            rf"$\theta_{{\nabla}}={metrics['gradient_angle_deg']:.1f}^\circ$",
            transform=axes[0, column].transAxes,
            ha="left",
            va="top",
            fontsize=6.1,
            color="white",
            bbox={
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.48,
                "pad": 1.2,
            },
        )
        provenance.append(
            {
                "dataset": dataset,
                "cache": path.as_posix(),
                "recording": recording.recording_id,
                "frame_index": index,
                "frame_id": int(snapshot["frame_id"]),
                "selection_rule": "maximum integrated prospective teacher risk",
            }
        )
    axes[0, 0].set_ylabel(r"Numerical $R^{\mathrm{num}}$" "\n" r"$y_{\mathrm{ego}}$ (m)")
    axes[1, 0].set_ylabel(r"PINN $\widehat{R}_{\theta}$" "\n" r"$y_{\mathrm{ego}}$ (m)")
    axes[2, 0].set_ylabel("Absolute error\n" r"$y_{\mathrm{ego}}$ (m)")
    field_bar_axis = fig.add_axes([0.915, 0.585, 0.012, 0.34])
    field_bar = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo"),
        cax=field_bar_axis,
    )
    field_bar.set_label("Normalized risk")
    error_bar_axis = fig.add_axes([0.915, 0.345, 0.012, 0.18])
    error_bar = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma"),
        cax=error_bar_axis,
    )
    error_bar.set_label("Normalized absolute error")

    bottom_axes = (
        fig.add_subplot(summary_grid[0, 0]),
        fig.add_subplot(summary_grid[0, 1]),
        fig.add_subplot(summary_grid[0, 2]),
    )
    _validity_panel(bottom_axes[0], validity)
    _fidelity_panel(bottom_axes[1], summary)
    propagation = _propagation_panel(bottom_axes[2], swap)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return provenance, propagation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--highway-cache-root",
        type=Path,
        default=Path("evaluation/pinn_prospective_v2_cache"),
    )
    parser.add_argument(
        "--fidelity-summary",
        type=Path,
        default=Path(
            "evaluation/pinn_prospective_context_v3_domain_conditioned/"
            "heldout_prospective_pinn_validation_summary.json"
        ),
    )
    parser.add_argument(
        "--validity-summary",
        type=Path,
        default=Path(
            "evaluation/prospective_field_validity_pinn_v3/"
            "prospective_field_validity_summary.csv"
        ),
    )
    parser.add_argument(
        "--backend-swap-summary",
        type=Path,
        default=Path(
            "evaluation/highwayenv_prospective_pinn_backend_swap_v3/"
            "backend_swap_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/pinn_revision_figures_v4"),
    )
    args = parser.parse_args()
    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_recurrent_pinn_checkpoint(
        args.checkpoint, device=args.device
    )
    model.eval()
    fidelity = json.loads(args.fidelity_summary.read_text(encoding="utf-8"))
    validity = pd.read_csv(args.validity_summary)
    swap = json.loads(args.backend_swap_summary.read_text(encoding="utf-8"))
    naturalistic_specs = [
        (
            "inD",
            "urban intersection",
            Path("evaluation/pinn_prospective_v2_cache/inD_06"),
            "recording-disjoint held-out",
        ),
        (
            "rounD",
            "roundabout",
            Path("evaluation/pinn_prospective_multidataset_v4/rounD_01"),
            "external zero-shot diagnostic",
        ),
        (
            "exiD",
            "freeway merge/exit",
            Path("evaluation/pinn_prospective_multidataset_v4/exiD_01"),
            "external zero-shot diagnostic",
        ),
    ]
    naturalistic_provenance = _naturalistic_figure(
        args.output_dir / "pinn_naturalistic_scene_conditioning",
        model,
        checkpoint,
        args.device,
        naturalistic_specs,
    )
    highway_provenance, propagation = _highwayenv_figure(
        args.output_dir / "pinn_highwayenv_fidelity_and_propagation",
        model,
        checkpoint,
        args.device,
        args.highway_cache_root,
        fidelity,
        validity,
        swap,
    )
    manifest = {
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": args.device,
        "style": "SciencePlots science/no-latex",
        "field_colormap": "turbo (blue-to-red)",
        "naturalistic_panels": naturalistic_provenance,
        "highwayenv_panels": highway_provenance,
        "quantitative_inputs": {
            "fidelity_summary": args.fidelity_summary.as_posix(),
            "validity_summary": args.validity_summary.as_posix(),
            "backend_swap_summary": args.backend_swap_summary.as_posix(),
        },
        "policy_error_propagation": propagation,
        "interpretation_boundary": (
            "inD and HighwayEnv are recording/seed-disjoint validation; rounD and exiD "
            "are external zero-shot diagnostics and are not pooled into the main fidelity claim."
        ),
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print((args.output_dir / "pinn_naturalistic_scene_conditioning.pdf").resolve())
    print((args.output_dir / "pinn_highwayenv_fidelity_and_propagation.pdf").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
