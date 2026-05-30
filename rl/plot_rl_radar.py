"""Four-pillar radar chart (Safety / Efficiency / Comfort / Courtesy) from an
eval_summary.json (scienceplots). Lets reviewers see safety-vs-efficiency
trade-offs at a glance (over-conservative vs aggressive).

Usage:
  python rl/plot_rl_radar.py --summary rl/logs/metadrive/eval_intersection_3arm_full/eval_summary.json \
    --planners stock_sac,risk_sac,social_sac,idm --out rl/logs/figures/intersection_radar
"""
from __future__ import annotations
import argparse, json
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

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#000000"]
AXES = ["Safety", "Efficiency", "Comfort", "Courtesy"]


def _style():
    try:
        plt.style.use(["science", "ieee", "no-latex"] if _HAS else "seaborn-v0_8-whitegrid")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--planners", required=True, help="comma-separated labels (match eval labels)")
    ap.add_argument("--out", default="rl/logs/figures/radar")
    args = ap.parse_args()
    s = json.load(open(args.summary, encoding="utf-8"))
    by = {k.split("@")[0]: v for k, v in s.items()}
    labels = [x.strip() for x in args.planners.split(",")]

    # raw values
    safety = [by[l].get("safety_score_mean", float("nan")) for l in labels]
    eff = [by[l].get("progress_score_mean", float("nan")) for l in labels]
    courtesy = [by[l].get("courtesy_score_mean", float("nan")) for l in labels]
    jerk = np.array([by[l].get("mean_jerk_abs_mean", float("nan")) for l in labels])
    # comfort = inverted, min-max normalised jerk across the plotted set
    jmin, jmax = np.nanmin(jerk), np.nanmax(jerk)
    comfort = 1.0 - (jerk - jmin) / (jmax - jmin) if jmax > jmin else np.ones_like(jerk)

    data = {l: [safety[i], eff[i], comfort[i], courtesy[i]] for i, l in enumerate(labels)}

    _style()
    ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    for i, l in enumerate(labels):
        vals = data[l] + data[l][:1]
        c = COLORS[i % len(COLORS)]
        ax.plot(ang, vals, color=c, linewidth=1.6, label=l)
        ax.fill(ang, vals, color=c, alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(AXES, fontsize=9)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), frameon=False, fontsize=8)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}.png/.pdf  (Comfort = min-max inverted jerk across plotted set)")


if __name__ == "__main__":
    main()
