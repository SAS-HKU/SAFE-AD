"""Build compact paper plots for prospective-field and PINN validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
except ImportError:
    scienceplots = None


LABELS = {
    "Instantaneous source": "Source",
    "Prospective numerical field": "Numerical",
    "Context PINN": "PINN",
    "highwayenv_highway_v0": "Highway",
    "highwayenv_merge_v0": "Merge",
    "highwayenv_intersection_v0": "Intersection",
    "highwayenv_roundabout_v0": "Roundabout",
    "inD": "inD",
}


def _bootstrap(values: np.ndarray, *, repetitions: int, seed: int):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, (int(repetitions), len(values)), replace=True).mean(1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validity-dir", type=Path, default=Path("evaluation/prospective_field_validity_pinn_v3")
    )
    parser.add_argument(
        "--fidelity-dir", type=Path, default=Path("evaluation/pinn_prospective_context_v3_domain_conditioned")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/pinn_prospective_v3_paper/pinn_validity_runtime")
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    validity = pd.read_csv(args.validity_dir / "prospective_field_validity_summary.csv")
    fidelity = pd.read_csv(args.fidelity_dir / "heldout_prospective_pinn_validation.csv")
    summary = json.loads(
        (args.fidelity_dir / "heldout_prospective_pinn_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )

    try:
        plt.style.use(["science", "no-latex"])
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "figure.dpi": 180,
        }
    )

    colors = {"Source": "#6B7280", "Numerical": "#D55E00", "PINN": "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), constrained_layout=True)

    metric_names = ["auroc", "auprc"]
    models = ["Instantaneous source", "Prospective numerical field", "Context PINN"]
    x = np.arange(len(metric_names), dtype=float)
    width = 0.24
    for offset, model in enumerate(models):
        group = validity[validity["model"] == model].set_index("metric")
        means = np.array([float(group.loc[name, "mean"]) for name in metric_names])
        lower = np.array([float(group.loc[name, "ci_low"]) for name in metric_names])
        upper = np.array([float(group.loc[name, "ci_high"]) for name in metric_names])
        label = LABELS[model]
        axes[0].bar(
            x + (offset - 1) * width,
            means,
            width,
            color=colors[label],
            label=label,
            edgecolor="white",
            linewidth=0.5,
            yerr=np.vstack((means - lower, upper - means)),
            capsize=2,
            error_kw={"elinewidth": 0.7},
        )
    axes[0].set_xticks(x, ["AUROC", "AUPRC"])
    axes[0].set_ylim(0.3, 1.0)
    axes[0].set_ylabel("Held-out predictive score")
    axes[0].set_title("(a) Future-occupancy construct validity")
    axes[0].legend(frameon=False, ncol=3, loc="upper center")
    axes[0].grid(axis="y", alpha=0.22)

    datasets = [
        "inD",
        "highwayenv_highway_v0",
        "highwayenv_merge_v0",
        "highwayenv_intersection_v0",
        "highwayenv_roundabout_v0",
    ]
    positions = np.arange(len(datasets), dtype=float)
    runtime_rows = []
    for dataset in datasets:
        group = fidelity[fidelity["dataset"] == dataset]
        teacher = group["numerical_solver_ms"].to_numpy(float)
        pinn = group["pinn_inference_ms"].to_numpy(float)
        for backend, values in (("Numerical", teacher), ("PINN", pinn)):
            median = float(np.median(values))
            q1, q3 = np.quantile(values, [0.25, 0.75])
            runtime_rows.append(
                {
                    "dataset": dataset,
                    "backend": backend,
                    "median_ms": median,
                    "q1_ms": float(q1),
                    "q3_ms": float(q3),
                }
            )
    runtime = pd.DataFrame(runtime_rows)
    for offset, backend in enumerate(("Numerical", "PINN")):
        group = runtime[runtime["backend"] == backend].set_index("dataset")
        means = np.array([float(group.loc[name, "median_ms"]) for name in datasets])
        q1 = np.array([float(group.loc[name, "q1_ms"]) for name in datasets])
        q3 = np.array([float(group.loc[name, "q3_ms"]) for name in datasets])
        axes[1].bar(
            positions + (offset - 0.5) * 0.34,
            means,
            0.34,
            color=colors[backend],
            label=backend,
            edgecolor="white",
            linewidth=0.5,
            yerr=np.vstack((means - q1, q3 - means)),
            capsize=2,
            error_kw={"elinewidth": 0.7},
        )
    axes[1].set_xticks(
        positions,
        [LABELS[name] for name in datasets],
        rotation=18,
        ha="right",
    )
    axes[1].set_ylabel("Propagation / forward time (ms)")
    axes[1].set_title("(b) Numerical teacher versus PINN")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].grid(axis="y", alpha=0.22)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    runtime.to_csv(args.output.with_name(args.output.name + "_runtime.csv"), index=False)
    paper_rows = []
    for dataset, values in summary["per_dataset"].items():
        paper_rows.append(
            {
                "dataset": dataset,
                "rmse": values["rmse"],
                "correlation": values["correlation"],
                "gradient_angle_deg": values["gradient_angle_deg"],
                "hotspot_iou": values["hotspot_iou"],
                "pinn_inference_ms": values["pinn_inference_ms"],
                "numerical_solver_ms": values["numerical_solver_ms"],
                "solver_speedup": values["solver_speedup"],
            }
        )
    pd.DataFrame(paper_rows).to_csv(
        args.output.with_name(args.output.name + "_fidelity.csv"), index=False
    )
    print(args.output.with_suffix(".pdf").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
