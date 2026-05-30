"""Traffic-density violin plots (scienceplots): distribution of a per-episode
metric (default route_completion) across Easy/Moderate/Hard density, grouped by
planner. Reads an eval_episodes.csv that spans multiple densities.

Usage:
  python rl/plot_rl_density_violin.py \
    --episodes rl/logs/metadrive/eval_intersection_sac_density/eval_episodes.csv \
    --metric route_completion --out rl/logs/figures/intersection_density_violin
"""
from __future__ import annotations
import argparse, csv, collections
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

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
DENS_LABEL = {"0.1": "Easy (0.1)", "0.3": "Moderate (0.3)", "0.5": "Hard (0.5)"}


def _style():
    try:
        plt.style.use(["science", "ieee", "grid", "no-latex"] if _HAS else "seaborn-v0_8-whitegrid")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--metric", default="route_completion")
    ap.add_argument("--out", default="rl/logs/figures/density_violin")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.episodes, encoding="utf-8")))
    data = collections.defaultdict(lambda: collections.defaultdict(list))  # planner -> dens -> [vals]
    for r in rows:
        pl = r["planner"].split("@")[0]
        d = str(r.get("traffic_density", "")).rstrip("0").rstrip(".") if "." in str(r.get("traffic_density","")) else str(r.get("traffic_density",""))
        # normalise density key to one decimal
        try:
            d = f"{float(r.get('traffic_density')):g}"
        except (TypeError, ValueError):
            pass
        data[pl][d].append(float(r.get(args.metric, "nan")))
    planners = list(data.keys())
    densities = sorted({d for pl in data.values() for d in pl}, key=lambda x: float(x))

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    n = len(planners); w = 0.8 / max(1, n)
    for pi, pl in enumerate(planners):
        positions = [di + (pi - (n - 1) / 2) * w for di in range(len(densities))]
        vals = [data[pl].get(d, [np.nan]) for d in densities]
        vp = ax.violinplot(vals, positions=positions, widths=w * 0.95, showmeans=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(COLORS[pi % len(COLORS)]); body.set_alpha(0.55); body.set_edgecolor("none")
        if "cmeans" in vp:
            vp["cmeans"].set_color("black"); vp["cmeans"].set_linewidth(1.0)
        ax.plot([], [], color=COLORS[pi % len(COLORS)], linewidth=6, alpha=0.55, label=pl)
    ax.set_xticks(range(len(densities)))
    ax.set_xticklabels([DENS_LABEL.get(d, d) for d in densities])
    ax.set_ylabel(args.metric.replace("_", " ")); ax.set_xlabel("Traffic density")
    ax.set_title(f"{args.metric.replace('_',' ').title()} by traffic density")
    ax.legend(frameon=False, ncol=n, loc="lower left")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}.png/.pdf  planners={planners} densities={densities}")


if __name__ == "__main__":
    main()
