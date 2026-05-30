"""Compact 1x3 return figure: SAC episode return across intersection / merge /
roundabout, with stock / risk / social arms overlaid. Style mimics standard RL
curves — raw per-episode trace in light alpha + a bold smoothed mean on top
(scienceplots).
"""
import csv, math
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

L = "rl/logs/metadrive"
OUT = "docs/figRR_sac_metadrive_compact"
SCEN = ["intersection", "merge", "roundabout"]
TITLES = {"intersection": "Intersection", "merge": "Merge", "roundabout": "Roundabout"}
ARMS = [("Stock", "#0072B2"), ("Risk", "#D55E00"), ("Social", "#009E73")]
SMOOTH = 30  # rolling-mean window (episodes)


def _style():
    try:
        plt.style.use(["science", "ieee", "grid", "no-latex"] if _HAS else "seaborn-v0_8-whitegrid")
    except Exception:
        pass


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _run(arm, scen):
    if arm == "Stock":
        return f"{L}/matched_stock_{scen}_respawn_sac_1m/progress.csv"
    if arm == "Risk":
        return f"{L}/matched_social_risk_{scen}_respawn_sac_1m/progress.csv"
    return f"{L}/socialbench_{scen}_social_sac_decoupled_1m/progress.csv"


def _load(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    t = np.array([_f(r.get("timesteps")) for r in rows])
    y = np.array([_f(r.get("ep_reward")) for r in rows])
    o = np.argsort(t)
    return t[o], y[o]


def _smooth(y, w):
    if y.size == 0 or w <= 1:
        return y
    k = np.ones(w) / w
    # use 'same' convolution with edge handling via reflection
    y_pad = np.pad(y, w // 2, mode="edge")
    return np.convolve(y_pad, k, mode="valid")[: y.size]


def main():
    _style()
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 2.9), constrained_layout=True, sharey=False)
    for ci, scen in enumerate(SCEN):
        for arm, color in ARMS:
            p = _run(arm, scen)
            if not Path(p).exists():
                continue
            t, y = _load(p)
            # raw per-episode trace (faint)
            ax[ci].plot(t, y, color=color, alpha=0.20, linewidth=0.55)
            # smoothed mean (bold)
            ys = _smooth(y, SMOOTH)
            ax[ci].plot(t, ys, color=color, linewidth=1.8, label=arm)
        ax[ci].set_title(TITLES[scen])
        ax[ci].set_xlabel("Timesteps")
        ax[ci].ticklabel_format(axis="x", style="sci", scilimits=(3, 6))
    ax[0].set_ylabel("Episode return (shaped)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
