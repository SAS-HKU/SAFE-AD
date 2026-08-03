"""Generate publication schematics for the prospective field and context PINN."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import torch

from rl.risk.pinn_snapshot_cache import CachedRecording
from rl.risk.recurrent_pinn_operator import (
    build_operator_input,
    checkpoint_domain_scales,
    load_recurrent_pinn_checkpoint,
    select_checkpoint_inputs,
)


COLORS = {
    "navy": "#194B73",
    "blue": "#DCECF6",
    "orange": "#F3E1C3",
    "green": "#DCECDD",
    "gray": "#ECEDEF",
    "ink": "#17212B",
}


def _box(ax, xy, width, height, title, body, *, face, edge, title_size=8.0, body_size=7.0):
    x, y = xy
    patch = Rectangle((x, y), width, height, facecolor=face, edgecolor=edge, lw=1.1)
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height - 0.055,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    if body:
        ax.text(
            x + width / 2,
            y + height * 0.40,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["ink"],
            linespacing=1.15,
        )
    return patch


def _arrow(ax, start, end, *, color=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.15,
            color=color or COLORS["navy"],
        )
    )


def _field_example(cache: Path, checkpoint: Path):
    recording = CachedRecording(cache)
    index = min(8, len(recording) - 1)
    snapshot = recording[index]
    model, metadata = load_recurrent_pinn_checkpoint(checkpoint, device="cpu")
    scales = checkpoint_domain_scales(metadata, "highwayenv")
    context = build_operator_input(
        snapshot,
        x_grid=np.asarray(recording.x_grid),
        y_grid=np.asarray(recording.y_grid),
        scales=scales,
    )
    context = select_checkpoint_inputs(context, metadata, domain="highwayenv")
    model.eval()
    with torch.inference_mode():
        prediction, _ = model(torch.from_numpy(context[None]))
    prediction = prediction[0, 0].cpu().numpy() * float(scales.risk)
    return (
        np.asarray(snapshot["Q"], dtype=float),
        np.asarray(snapshot["R"], dtype=float),
        np.asarray(prediction, dtype=float),
    )


def _inset_field(ax, bounds, field, vmax):
    axis = ax.inset_axes(bounds, transform=ax.transData)
    axis.imshow(field, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=vmax)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#FFFFFF")
        spine.set_linewidth(0.7)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _overview(path: Path, fields):
    source, teacher, prediction = fields
    vmax = max(float(np.percentile(teacher, 99.5)), 1e-6)
    fig, ax = plt.subplots(figsize=(11.3, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.012, 0.855, "Questions", fontsize=9, fontweight="bold", color=COLORS["navy"], rotation=90, va="center")
    ax.text(0.012, 0.535, "Method", fontsize=9, fontweight="bold", color="#A55A00", rotation=90, va="center")
    ax.text(0.012, 0.185, "Evidence", fontsize=9, fontweight="bold", color="#30733C", rotation=90, va="center")

    questions = [
        "Does propagation add\nprospective structure beyond\nthe instantaneous source?",
        "Can one multi-recording\nsurrogate preserve fields\nand actionable gradients?",
        "Does a fixed policy retain\nits actions and outcomes\nafter the backend swap?",
    ]
    for i, body in enumerate(questions):
        _box(ax, (0.055 + i * 0.31, 0.735), 0.275, 0.22, f"Q{i + 1}", body, face=COLORS["gray"], edge="#707780")

    method = [
        ("Current BEV", "$Q_t,\\,\\mathbf{c}_t,\\,D_t,\\,M_t$", COLORS["blue"]),
        ("Prospective teacher", "$3$-s causal transport\nand diffusion", COLORS["orange"]),
        ("Context PINN", "$17$ channels, dilated\nresidual operator", COLORS["green"]),
        ("Field interface", "$8$-D descriptor and\nfield-aware objective", COLORS["blue"]),
        ("RL policy", "continuous or discrete\ntactical control", COLORS["orange"]),
    ]
    starts = [0.05, 0.245, 0.44, 0.635, 0.83]
    for i, (title, body, face) in enumerate(method):
        _box(ax, (starts[i], 0.405), 0.15, 0.25, title, body, face=face, edge=COLORS["navy"], body_size=6.7)
        if i < len(method) - 1:
            _arrow(ax, (starts[i] + 0.152, 0.53), (starts[i + 1] - 0.004, 0.53))
    _inset_field(ax, [0.07, 0.418, 0.11, 0.055], source, vmax)
    _inset_field(ax, [0.265, 0.418, 0.11, 0.055], teacher, vmax)
    _inset_field(ax, [0.46, 0.418, 0.11, 0.055], prediction, vmax)

    evidence = [
        ("Naturalistic replay", "inD 01--05 calibration;\n06--11 held out"),
        ("Field fidelity", "RMSE, correlation, gradient\nangle, hotspot overlap"),
        ("Construct validity", "future occupancy labels;\nrecording bootstrap"),
        ("Backend swap", "frozen-state action error;\npaired closed loop"),
    ]
    for i, (title, body) in enumerate(evidence):
        x = 0.055 + i * 0.235
        _box(ax, (x, 0.055), 0.205, 0.225, title, body, face="#F7FAF7", edge="#4C8A5A", body_size=6.7)
        if i < len(evidence) - 1:
            _arrow(ax, (x + 0.207, 0.168), (x + 0.232, 0.168), color="#4C8A5A")
    _save(fig, path)


def _pipeline(path: Path, fields):
    source, teacher, prediction = fields
    vmax = max(float(np.percentile(teacher, 99.5)), 1e-6)
    fig, ax = plt.subplots(figsize=(11.3, 4.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "Offline calibration", fontsize=9.5, fontweight="bold", color=COLORS["navy"])
    ax.plot([0.02, 0.98], [0.91, 0.91], color=COLORS["navy"], lw=1.0)

    top = [
        (0.03, 0.56, 0.20, "Teacher cache", "Current scene only\nmultiple recordings and maps"),
        (0.285, 0.56, 0.20, "Prospective solver", "$U_H=Z_H^{-1}\\int_0^H e^{-\\lambda s}\\,\\mathcal{T}_s(Q_t)\\,ds$"),
        (0.54, 0.56, 0.20, "Context tensor", "multiscale source, transport,\nmasks, radial bases, domain bit"),
        (0.795, 0.56, 0.175, "Neural operator", "stem + 6 dilated residual blocks\nfield + gradient + physics loss"),
    ]
    for i, (x, y, w, title, body) in enumerate(top):
        _box(ax, (x, y), w, 0.27, title, body, face=[COLORS["blue"], COLORS["orange"], COLORS["gray"], COLORS["green"]][i], edge=COLORS["navy"], body_size=6.65)
        if i < len(top) - 1:
            _arrow(ax, (x + w + 0.006, y + 0.135), (top[i + 1][0] - 0.006, y + 0.135))
    _inset_field(ax, [0.055, 0.578, 0.15, 0.06], source, vmax)
    _inset_field(ax, [0.31, 0.578, 0.15, 0.06], teacher, vmax)
    for x, dilation in zip(np.linspace(0.823, 0.93, 6), [1, 2, 4, 8, 16, 1]):
        ax.add_patch(
            Rectangle(
                (x, 0.575),
                0.012,
                0.035 + 0.003 * min(dilation, 8),
                facecolor="#6AAE7B",
                edgecolor="#2B5D37",
                lw=0.5,
            )
        )

    ax.text(0.02, 0.45, "Held-out validation and online deployment", fontsize=9.5, fontweight="bold", color="#30733C")
    ax.plot([0.02, 0.98], [0.42, 0.42], color="#4C8A5A", lw=1.0)
    bottom = [
        (0.03, 0.08, 0.20, "Restored held-out data", "inD 06--11 and unseen\nHighwayEnv traffic seeds"),
        (0.285, 0.08, 0.20, "Surrogate checks", "field magnitude, gradients,\nhotspots, future occupancy"),
        (0.54, 0.08, 0.20, "Actual backend swap", "same policy and traffic:\nnumerical field $\\leftrightarrow$ PINN"),
        (0.795, 0.08, 0.175, "Policy query", "$\\hat{U}_{\\theta,t}\\rightarrow\\hat{\\eta}_t\\rightarrow\\pi_\\phi$\naction and outcome deltas"),
    ]
    for i, (x, y, w, title, body) in enumerate(bottom):
        _box(ax, (x, y), w, 0.25, title, body, face="#F7FAF7", edge="#4C8A5A", body_size=6.65)
        if i < len(bottom) - 1:
            _arrow(ax, (x + w + 0.006, y + 0.125), (bottom[i + 1][0] - 0.006, y + 0.125), color="#4C8A5A")
    _inset_field(ax, [0.31, 0.095, 0.15, 0.055], prediction, vmax)
    _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("evaluation/pinn_prospective_v2_cache/highwayenv_merge_v0_seed0100"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/pinn_prospective_v3_paper"),
    )
    args = parser.parse_args()
    fields = _field_example(args.cache, args.checkpoint)
    _overview(args.output_dir / "fig1_safer_overview", fields)
    _pipeline(args.output_dir / "fig2_context_pinn_pipeline", fields)
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
