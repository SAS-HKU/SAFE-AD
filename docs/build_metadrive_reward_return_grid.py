"""Per-algorithm reward/return training grids for MetaDrive (scienceplots).

For a given algorithm, a 2x3 grid: rows = [avg reward per step, episode return],
cols = [intersection, merge, roundabout]; each cell overlays the three reward arms
(stock / risk-only / social-tuned). One figure per algorithm.

NOTE: reward/return here are the *shaped* training objective, which differs per arm
(social subtracts comfort/courtesy penalties), so curves are comparable in *shape/
convergence* within an arm but not in absolute level across arms — for cross-arm
performance use the success/route curves and the eval tables.
"""
import csv, math, argparse
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
SCEN = ["intersection", "merge", "roundabout"]
TITLES = {"intersection": "Intersection", "merge": "Merge", "roundabout": "Roundabout"}
ARMS = [("Stock", "#0072B2"), ("Risk", "#D55E00"), ("Social", "#009E73")]


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


def _run(arm, scen, algo):
    if arm == "Stock":
        return f"{L}/matched_stock_{scen}_respawn_{algo}_1m/progress.csv"
    if arm == "Risk":
        return f"{L}/matched_social_risk_{scen}_respawn_{algo}_1m/progress.csv"
    return f"{L}/socialbench_{scen}_social_{algo}_decoupled_1m/progress.csv"


def _load(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    t = np.array([_f(r.get("timesteps")) for r in rows]); o = np.argsort(t)
    rew = np.array([_f(r.get("ep_reward")) for r in rows])[o]
    ln = np.array([max(1.0, _f(r.get("ep_len"))) for r in rows])[o]
    return t[o], rew, rew / ln  # return, episode-return, avg-reward-per-step


def _bin(t, y, edges):
    idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
    return np.array([np.nanmean(y[idx == b]) if np.any(idx == b) else np.nan for b in range(len(edges) - 1)])


def build(algo):
    _style()
    edges = np.linspace(0, 1_000_000, 26); ctr = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(2, 3, figsize=(11.0, 5.2), constrained_layout=True, sharex=True)
    for ci, scen in enumerate(SCEN):
        for arm, color in ARMS:
            p = _run(arm, scen, algo)
            if not Path(p).exists():
                continue
            t, ret, avg = _load(p)
            ax[0][ci].plot(ctr, _bin(t, avg, edges), color=color, lw=1.5, label=arm)
            ax[1][ci].plot(ctr, _bin(t, ret, edges), color=color, lw=1.5, label=arm)
        ax[0][ci].set_title(TITLES[scen])
        ax[1][ci].set_xlabel("Timesteps")
        for r in (0, 1):
            ax[r][ci].ticklabel_format(axis="x", style="sci", scilimits=(3, 6))
    ax[0][0].set_ylabel("Avg reward / step")
    ax[1][0].set_ylabel("Episode return")
    h, l = ax[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"MetaDrive {algo.upper()} — shaped reward & return across scenarios (stock/risk/social)",
                 y=1.10, fontsize=11)
    out = f"docs/figRR_{algo}_metadrive"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.png/.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["sac", "td3", "ddpg"])
    for a in ap.parse_args().algos:
        build(a)
