"""Aggregate paired numerical-teacher/PINN HighwayEnv planning evaluations."""

from __future__ import annotations

import argparse
import glob
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


SCENARIOS = ("highway-v0", "merge-v0", "intersection-v0", "roundabout-v0")
SCENARIO_LABELS = {
    "highway-v0": "Highway",
    "merge-v0": "Merge",
    "intersection-v0": "Intersection",
    "roundabout-v0": "Roundabout",
}
METRICS = {
    "crashed": ("Safety", "Collision rate", "lower"),
    "ttc_min": ("Safety", "Minimum TTC (s)", "higher"),
    "corridor_risk_mean": ("Safety", "Mean corridor risk", "lower"),
    "progress": ("Efficiency", "Route progress (m)", "higher"),
    "mean_speed": ("Efficiency", r"Mean speed (m s$^{-1}$)", "higher"),
    "mean_jerk_abs": ("Comfort", r"Mean absolute jerk (m s$^{-3}$)", "lower"),
    "imposed_rear_decel_max": (
        "Sociality",
        r"Maximum imposed follower deceleration (m s$^{-2}$)",
        "lower",
    ),
    "risk_flux_mean": ("Sociality", "Mean backward risk flux", "lower"),
    "mean_field_backend_ms": ("Runtime", "Field-query latency (ms)", "lower"),
}
PLOT_METRICS = ("crashed", "progress", "mean_jerk_abs", "imposed_rear_decel_max")
BACKEND_LABELS = {
    "prospective": "Numerical field",
    "pinn": "PINN field",
}
COLORS = {"prospective": "#D55E00", "pinn": "#0072B2"}
MARKERS = {"prospective": "o", "pinn": "s"}


def _style() -> None:
    try:
        plt.style.use(["science", "no-latex"])
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.6,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "figure.dpi": 180,
            "savefig.dpi": 450,
        }
    )


def _paths(patterns: list[str]) -> list[Path]:
    values: list[Path] = []
    for pattern in patterns:
        matches = [Path(value) for value in sorted(glob.glob(pattern))]
        if not matches:
            raise FileNotFoundError(f"No episode CSV matches {pattern!r}")
        values.extend(matches)
    return list(dict.fromkeys(values))


def _bootstrap_mean(values: np.ndarray, repetitions: int, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    sampled = rng.choice(values, size=(int(repetitions), len(values)), replace=True).mean(1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def _paired_randomization_p(
    differences: np.ndarray,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size < 2 or np.allclose(differences, 0.0):
        return 1.0
    observed = abs(float(np.mean(differences)))
    signs = rng.choice((-1.0, 1.0), size=(int(repetitions), differences.size))
    null = np.abs(np.mean(signs * differences[None, :], axis=1))
    return float((1.0 + np.sum(null >= observed)) / (1.0 + len(null)))


def _holm(values: list[float]) -> list[float]:
    result = np.full(len(values), np.nan, dtype=float)
    finite = [index for index, value in enumerate(values) if np.isfinite(value)]
    order = sorted(finite, key=lambda index: values[index])
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(values[index]))
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def _paired_values(rows: pd.DataFrame, metric: str):
    pivot = rows.pivot_table(index="seed", columns="backend", values=metric, aggfunc="first")
    required = {"prospective", "pinn"}
    if not required.issubset(pivot.columns):
        return np.asarray([]), np.asarray([])
    pivot = pivot.dropna(subset=["prospective", "pinn"])
    return (
        pivot["prospective"].to_numpy(dtype=float),
        pivot["pinn"].to_numpy(dtype=float),
    )


def _summarize(data: pd.DataFrame, bootstrap: int, permutations: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for scenario in SCENARIOS:
        selected = data[data["env_id"] == scenario]
        if selected.empty:
            continue
        for metric, (pillar, label, direction) in METRICS.items():
            if metric not in selected.columns:
                continue
            teacher, pinn = _paired_values(selected, metric)
            if not len(teacher):
                continue
            delta = pinn - teacher
            delta_mean, delta_low, delta_high = _bootstrap_mean(delta, bootstrap, rng)
            teacher_mean, teacher_low, teacher_high = _bootstrap_mean(teacher, bootstrap, rng)
            pinn_mean, pinn_low, pinn_high = _bootstrap_mean(pinn, bootstrap, rng)
            denominator = max(abs(teacher_mean), 1e-9)
            relative = 100.0 * delta_mean / denominator
            favorable = delta_mean if direction == "higher" else -delta_mean
            rows.append(
                {
                    "scenario": scenario,
                    "pillar": pillar,
                    "metric": metric,
                    "metric_label": label,
                    "favorable_direction": direction,
                    "n_pairs": len(delta),
                    "numerical_mean": teacher_mean,
                    "numerical_ci_low": teacher_low,
                    "numerical_ci_high": teacher_high,
                    "pinn_mean": pinn_mean,
                    "pinn_ci_low": pinn_low,
                    "pinn_ci_high": pinn_high,
                    "delta_pinn_minus_numerical": delta_mean,
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    "relative_delta_percent": relative,
                    "favorable_signed_delta": favorable,
                    "paired_randomization_p": _paired_randomization_p(
                        delta,
                        repetitions=permutations,
                        rng=rng,
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["holm_adjusted_p"] = _holm(summary["paired_randomization_p"].tolist())
    return summary


def _plot(data: pd.DataFrame, output: Path, bootstrap: int, seed: int) -> None:
    available = [scenario for scenario in SCENARIOS if scenario in set(data["env_id"])]
    x = np.arange(len(available), dtype=float)
    offsets = {"prospective": -0.09, "pinn": 0.09}
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.25), constrained_layout=True)
    for panel, (axis, metric) in enumerate(zip(axes.flat, PLOT_METRICS)):
        _pillar, label, _direction = METRICS[metric]
        for backend in ("prospective", "pinn"):
            means, lows, highs = [], [], []
            for scenario in available:
                values = data.loc[
                    (data["env_id"] == scenario) & (data["backend"] == backend),
                    metric,
                ].to_numpy(dtype=float)
                mean, low, high = _bootstrap_mean(values, bootstrap, rng)
                means.append(mean)
                lows.append(low)
                highs.append(high)
            means_array = np.asarray(means)
            axis.errorbar(
                x + offsets[backend],
                means_array,
                yerr=np.vstack((means_array - lows, np.asarray(highs) - means_array)),
                color=COLORS[backend],
                marker=MARKERS[backend],
                markersize=4.0,
                linewidth=1.0,
                capsize=2.0,
                label=BACKEND_LABELS[backend],
            )
        axis.set_title(f"({chr(97 + panel)}) {label}", fontweight="bold")
        axis.set_xticks(x, [SCENARIO_LABELS[value] for value in available])
        axis.tick_params(axis="x", rotation=14)
        axis.grid(axis="y", alpha=0.22)
        if metric == "crashed":
            axis.set_ylim(-0.03, 1.03)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
    )
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-csv", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/highwayenv_pinn_planning_comparison"),
    )
    args = parser.parse_args()
    paths = _paths(args.episodes_csv)
    data = pd.concat((pd.read_csv(path).assign(source_file=str(path)) for path in paths), ignore_index=True)
    required_columns = {"env_id", "seed", "backend", *METRICS.keys()}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Episode files omit required columns: {missing_columns}")
    backend_values = set(data["backend"].astype(str))
    if not {"prospective", "pinn"}.issubset(backend_values):
        raise ValueError("Both prospective and pinn backend rows are required")
    available = set(data["env_id"].astype(str))
    missing_scenarios = [scenario for scenario in SCENARIOS if scenario not in available]
    if missing_scenarios and not args.allow_partial:
        raise ValueError(
            "Planning comparison is incomplete; missing "
            f"{missing_scenarios}. Pass --allow-partial only for diagnostics."
        )
    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summarize(data, args.bootstrap, args.permutations, args.seed)
    summary.to_csv(args.output_dir / "pinn_planning_paired_statistics.csv", index=False)
    _plot(
        data,
        args.output_dir / "pinn_planning_four_pillar_comparison",
        args.bootstrap,
        args.seed,
    )
    manifest = {
        "episode_files": [str(path.resolve()) for path in paths],
        "available_scenarios": [scenario for scenario in SCENARIOS if scenario in available],
        "missing_scenarios": missing_scenarios,
        "partial_diagnostic": bool(missing_scenarios),
        "paired_bootstrap_repetitions": int(args.bootstrap),
        "paired_randomization_repetitions": int(args.permutations),
        "multiple_comparison_correction": "Holm across reported scenario-metric tests",
        "claim_rule": (
            "Efficiency superiority requires a positive paired progress effect; safety and "
            "social outcomes must be reported concurrently and cannot be inferred from field magnitude."
        ),
    }
    (args.output_dir / "pinn_planning_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print((args.output_dir / "pinn_planning_paired_statistics.csv").resolve())
    print((args.output_dir / "pinn_planning_four_pillar_comparison.pdf").resolve())
    if missing_scenarios:
        print(f"PARTIAL DIAGNOSTIC ONLY: missing {missing_scenarios}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
