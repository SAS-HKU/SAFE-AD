"""Training safety/efficiency curves vs timesteps (scienceplots).

Plots success-rate, crash-rate, and route-completion as a function of training
timesteps (binned over episodes) for one or more runs, overlaid. Unlike shaped
reward, these rates are comparable across reward arms, so this doubles as the
stock/risk/social ablation curve the RL-report reviewers expect.

Usage:
  python rl/plot_rl_training_safety_curves.py \
    --runs "Stock SAC=rl/logs/metadrive/matched_stock_intersection_respawn_sac_1m/progress.csv" \
           "Risk SAC=rl/logs/metadrive/matched_social_risk_intersection_respawn_sac_1m/progress.csv" \
           "Social SAC=rl/logs/metadrive/socialbench_intersection_social_sac_decoupled_1m/progress.csv" \
    --out rl/logs/figures/intersection_sac_safety_curves
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    _HAS = True
except ImportError:
    _HAS = False

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]


def _style():
    try:
        plt.style.use(["science", "ieee", "grid", "no-latex"] if _HAS else "seaborn-v0_8-whitegrid")
    except Exception:
        pass


def _f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([_f(r.get("timesteps"), i) for i, r in enumerate(rows)])
    o = np.argsort(t)
    return {
        "t": t[o],
        "success": np.array([_f(r.get("success"), 0) for r in rows])[o],
        "crash": np.array([_f(r.get("crash_vehicle"), 0) for r in rows])[o],
        "route": np.array([_f(r.get("route_completion"), 0) for r in rows])[o],
    }


def _binned(t, y, nb):
    edges = np.linspace(t.min(), t.max() + 1e-6, nb + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    m = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(nb)])
    return ctr, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="'Label=path/progress.csv'")
    ap.add_argument("--out", default="rl/logs/figures/safety_curves")
    ap.add_argument("--bins", type=int, default=25)
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        label, path = spec.split("=", 1)
        runs.append((label.strip(), _load(path.strip())))

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True)
    panels = [("success", "Success rate"), ("crash", "Crash rate"), ("route", "Route completion")]
    for i, (lbl, data) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        for ax, (key, title) in zip(axes, panels):
            ctr, m = _binned(data["t"], data[key], args.bins)
            ax.plot(ctr, m, color=c, linewidth=1.6, label=lbl)
            ax.set_title(title); ax.set_xlabel("Timesteps")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 6))
    axes[0].set_ylabel("rate / fraction")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(4, len(runs)), frameon=False, bbox_to_anchor=(0.5, 1.12))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}.png/.pdf")


if __name__ == "__main__":
    main()
