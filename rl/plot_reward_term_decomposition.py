"""
Reward-term decomposition plots for the MetaDrive social_full RL.

Answers two questions for the paper:
  1. *Which reward term dominates* the shaped reward, during training and at
     convergence?
  2. *Which term does the policy improve the most* (i.e. shrinks the penalty)
     over training -- the term that is "trained best"?

Data sources
------------
  - Training: SB3 ``progress.csv`` files written by `train_metadrive_sb3.py`,
    which now carry per-episode penalty sums:
      ep_risk_penalty, ep_steer_penalty, ep_jerk_penalty, ep_throttle_penalty,
      ep_hard_brake_penalty, ep_courtesy_penalty, ep_rear_ttc_penalty,
      ep_backward_flux_penalty
  - Evaluation: ``eval_summary.json`` written by `eval_metadrive.py`
    (per ``planner@protocol`` block, each term as ``<key>_mean``).

Outputs (scienceplots IEEE style when available, graceful fallback otherwise):
  <out>_training.png/pdf     stacked-area of |penalty| vs timesteps, one panel/run
  <out>_improvement.png/pdf  first- vs last-decile penalty per term, per run
  <out>_eval.png/pdf         per-planner stacked bar of mean |penalty| (if summary)
  <out>_data.csv             tidy long-form table of the binned training data

Usage
-----
python rl/plot_reward_term_decomposition.py ^
  --train-runs "PPO=rl/logs/metadrive/social_int_ppo_cont_1m/progress.csv" ^
               "SAC=rl/logs/metadrive/social_int_sac_1m/progress.csv" ^
  --eval-summary rl/logs/metadrive/eval_intersection_respawn_all_algos/eval_summary.json ^
  --out rl/logs/figures/reward_term_decomposition
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401

    _HAS_SCIENCEPLOTS = True
except ImportError:  # pragma: no cover
    _HAS_SCIENCEPLOTS = False


# (csv column, eval-summary key, display label, colour). Ordered by the
# shaped-reward composition in metadrive_drift_wrapper._shape_reward.
TERMS = [
    ("ep_risk_penalty",          "ep_risk_penalty_mean",          "Risk",          "#0072B2"),
    ("ep_steer_penalty",         "ep_steer_penalty_mean",         "Steering",      "#D55E00"),
    ("ep_jerk_penalty",          "ep_jerk_penalty_mean",          "Jerk",          "#009E73"),
    ("ep_throttle_penalty",      "ep_throttle_penalty_mean",      "Throttle",      "#CC79A7"),
    ("ep_hard_brake_penalty",    "ep_hard_brake_penalty_mean",    "Hard brake",    "#56B4E9"),
    ("ep_courtesy_penalty",      "ep_courtesy_penalty_mean",      "Courtesy",      "#E69F00"),
    ("ep_rear_ttc_penalty",      "ep_rear_ttc_penalty_mean",      "Rear TTC",      "#F0E442"),
    ("ep_backward_flux_penalty", "ep_backward_flux_penalty_mean", "Backward flux", "#999999"),
]


def _apply_style() -> None:
    try:
        if _HAS_SCIENCEPLOTS:
            plt.style.use(["science", "ieee", "grid", "no-latex"])
        else:
            plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:  # pragma: no cover - never let styling break a figure
        pass


def _to_float(value, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec.strip())
    label = path.parent.name if path.name == "progress.csv" else path.stem
    return label, path


def _load_training(path: Path) -> dict:
    """Return {'t': np.ndarray, '<col>': np.ndarray(|penalty|)} for one run."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    t = np.asarray([_to_float(r.get("timesteps"), idx) for idx, r in enumerate(rows)], dtype=float)
    order = np.argsort(t)
    out = {"t": t[order]}
    for col, _ekey, _label, _color in TERMS:
        # Penalties are <= 0; plot magnitudes so stacking reads as "cost share".
        vals = np.asarray([abs(_to_float(r.get(col), 0.0)) for r in rows], dtype=float)
        out[col] = vals[order]
    return out


def _bin_means(t: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if t.size == 0:
        return t, y
    edges = np.linspace(float(t.min()), float(t.max()) + 1e-6, n_bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        chunk = y[idx == b]
        if chunk.size:
            means[b] = float(np.mean(chunk))
    return centers, means


def _plot_training(runs: dict[str, dict], out_base: Path, n_bins: int) -> list[dict]:
    _apply_style()
    n = len(runs)
    ncol = min(2, n)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.8 * nrow),
                             squeeze=False, constrained_layout=True)
    tidy: list[dict] = []
    for ax_idx, (label, data) in enumerate(runs.items()):
        ax = axes[ax_idx // ncol][ax_idx % ncol]
        t = data["t"]
        stacks, colors, labels = [], [], []
        centers = None
        for col, _ekey, term_label, color in TERMS:
            centers, m = _bin_means(t, data[col], n_bins)
            m = np.nan_to_num(m, nan=0.0)
            stacks.append(m)
            colors.append(color)
            labels.append(term_label)
            for c, val in zip(centers, m):
                tidy.append({"run": label, "term": term_label,
                             "timesteps": float(c), "penalty_abs": float(val)})
        if centers is not None and stacks:
            ax.stackplot(centers, np.vstack(stacks), colors=colors, labels=labels, alpha=0.9)
        ax.set_title(label)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("|penalty| (episode sum)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 6))
    # Hide any unused axes.
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)),
                   frameon=False, bbox_to_anchor=(0.5, 1.10))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_name(out_base.name + "_training.png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_name(out_base.name + "_training.pdf"), bbox_inches="tight")
    plt.close(fig)
    return tidy


def _plot_improvement(runs: dict[str, dict], out_base: Path, decile: float = 0.1) -> None:
    """First-decile vs last-decile mean |penalty| per term, grouped by run.

    A term whose bar drops markedly from first to last decile is one the policy
    "trained best" (it learned to avoid that penalty)."""
    _apply_style()
    term_labels = [t[2] for t in TERMS]
    x = np.arange(len(term_labels))
    n = len(runs)
    width = 0.8 / max(1, n)
    fig, ax = plt.subplots(figsize=(7.6, 3.2), constrained_layout=True)
    for r_idx, (label, data) in enumerate(runs.items()):
        t = data["t"]
        if t.size == 0:
            continue
        k = max(1, int(decile * t.size))
        first = {}
        last = {}
        for col, _e, term_label, _c in TERMS:
            y = data[col]
            first[term_label] = float(np.mean(y[:k]))
            last[term_label] = float(np.mean(y[-k:]))
        deltas = [first[tl] - last[tl] for tl in term_labels]  # >0 = penalty reduced
        ax.bar(x + r_idx * width - 0.4 + width / 2, deltas, width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(term_labels, rotation=30, ha="right")
    ax.set_ylabel("|penalty| reduced\n(first - last decile)")
    ax.set_title("Reward-term improvement over training")
    ax.legend(frameon=False, ncol=min(3, max(1, n)))
    fig.savefig(out_base.with_name(out_base.name + "_improvement.png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_name(out_base.name + "_improvement.pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_eval(summary_path: Path, out_base: Path) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        print(f"[warn] empty/invalid eval summary: {summary_path}")
        return
    _apply_style()
    planners = list(data.keys())
    x = np.arange(len(planners))
    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * len(planners)), 3.4), constrained_layout=True)
    bottom = np.zeros(len(planners))
    any_data = False
    for col, ekey, term_label, color in TERMS:
        vals = np.asarray([abs(_to_float(data[p].get(ekey), 0.0)) for p in planners], dtype=float)
        if not np.any(vals > 0):
            continue
        any_data = True
        ax.bar(x, vals, bottom=bottom, color=color, label=term_label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([p.split("@")[0] for p in planners], rotation=30, ha="right")
    ax.set_ylabel("Mean |penalty| (episode sum)")
    ax.set_title("Reward-term composition at evaluation")
    if any_data:
        ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.savefig(out_base.with_name(out_base.name + "_eval.png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_name(out_base.name + "_eval.pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_tidy(tidy: list[dict], path: Path) -> None:
    if not tidy:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "term", "timesteps", "penalty_abs"])
        writer.writeheader()
        writer.writerows(tidy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-runs", nargs="*", default=[],
                        help="Run specs 'Label=path/progress.csv' for the training views.")
    parser.add_argument("--eval-summary", default=None,
                        help="Path to eval_summary.json for the evaluation breakdown.")
    parser.add_argument("--out", default="rl/logs/figures/reward_term_decomposition")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    out_base = Path(args.out)
    runs: dict[str, dict] = {}
    for spec in args.train_runs:
        label, path = _parse_run_spec(spec)
        try:
            runs[label] = _load_training(path)
            print(f"[load] {label}: {runs[label]['t'].size} episodes from {path}")
        except FileNotFoundError:
            print(f"[warn] missing training log: {path}")

    if runs:
        tidy = _plot_training(runs, out_base, n_bins=max(2, int(args.bins)))
        _plot_improvement(runs, out_base)
        _write_tidy(tidy, out_base.with_name(out_base.name + "_data.csv"))
        print(f"[plot] wrote {out_base.name}_training.(png|pdf)")
        print(f"[plot] wrote {out_base.name}_improvement.(png|pdf)")

    if args.eval_summary:
        eval_path = Path(args.eval_summary)
        if eval_path.exists():
            _plot_eval(eval_path, out_base)
            print(f"[plot] wrote {out_base.name}_eval.(png|pdf)")
        else:
            print(f"[warn] missing eval summary: {eval_path}")

    if not runs and not args.eval_summary:
        raise SystemExit("Nothing to plot: pass --train-runs and/or --eval-summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
