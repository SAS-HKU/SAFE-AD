"""Figure A (MetaDrive): one 3-panel training figure — breadth + stability.

(a) shaded-band success-rate vs steps for stock/risk/social SAC (3 seeds, mean±95%CI)
(b) per-algorithm success-rate vs steps (social arm: SAC/TD3/DDPG/PPO/DQN) — family breadth
(c) social-SAC reward-term decomposition (stacked |penalty| vs steps) — credit assignment
All from progress.csv (scienceplots styling).
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
OUT = "docs/figA_metadrive_training"
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
TERMS = [("ep_risk_penalty", "Risk", "#0072B2"), ("ep_steer_penalty", "Steering", "#D55E00"),
         ("ep_jerk_penalty", "Jerk", "#009E73"), ("ep_throttle_penalty", "Throttle", "#CC79A7"),
         ("ep_hard_brake_penalty", "Hard brake", "#56B4E9"), ("ep_courtesy_penalty", "Courtesy", "#E69F00"),
         ("ep_rear_ttc_penalty", "Rear-TTC", "#F0E442"), ("ep_backward_flux_penalty", "Back-flux", "#999999")]


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


def _load(path, cols):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    t = np.array([_f(r.get("timesteps")) for r in rows]); o = np.argsort(t)
    d = {"t": t[o]}
    for c in cols:
        d[c] = np.array([_f(r.get(c)) for r in rows])[o]
    return d


def _bin(t, y, edges):
    idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
    return np.array([np.nanmean(y[idx == b]) if np.any(idx == b) else np.nan for b in range(len(edges) - 1)])


def main():
    _style()
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.1), constrained_layout=True)

    # ---- (a) shaded success, 3 arms x 3 seeds ----
    arms = {
        "Stock": ["matched_stock_intersection_respawn_sac_1m", "msd_stock_sac_int_s1", "msd_stock_sac_int_s2"],
        "Risk": ["matched_social_risk_intersection_respawn_sac_1m", "msd_risk_sac_int_s1", "msd_risk_sac_int_s2"],
        "Social": ["socialbench_intersection_social_sac_decoupled_1m", "msd_social_sac_int_s1", "msd_social_sac_int_s2"],
    }
    edges = np.linspace(0, 1_000_000, 26)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for i, (label, runs) in enumerate(arms.items()):
        series = []
        for rn in runs:
            p = f"{L}/{rn}/progress.csv"
            if Path(p).exists():
                d = _load(p, ["success"]); series.append(_bin(d["t"], d["success"], edges))
        if series:
            S = np.vstack(series); m = np.nanmean(S, 0)
            sd = np.nanstd(S, 0, ddof=1) if len(series) > 1 else np.zeros_like(m)
            band = 1.96 * sd / max(1, math.sqrt(len(series)))
            ax[0].plot(ctr, m, color=C[i], lw=1.6, label=f"{label} (n={len(series)})")
            ax[0].fill_between(ctr, m - band, m + band, color=C[i], alpha=0.18, lw=0)
    ax[0].set_title("(a) Stability: success vs steps (SAC, 3 seeds)")
    ax[0].set_xlabel("Timesteps"); ax[0].set_ylabel("Success rate")
    ax[0].ticklabel_format(axis="x", style="sci", scilimits=(3, 6)); ax[0].legend(frameon=False, fontsize=7)

    # ---- (b) per-algorithm success (social arm) ----
    for i, A in enumerate(["sac", "td3", "ddpg", "ppo", "dqn"]):
        p = f"{L}/socialbench_intersection_social_{A}_decoupled_1m/progress.csv"
        if Path(p).exists():
            d = _load(p, ["success"]); ax[1].plot(ctr, _bin(d["t"], d["success"], edges), color=C[i], lw=1.4, label=A.upper())
    ax[1].set_title("(b) Breadth: success vs steps by algorithm (social)")
    ax[1].set_xlabel("Timesteps"); ax[1].set_ylabel("Success rate")
    ax[1].ticklabel_format(axis="x", style="sci", scilimits=(3, 6)); ax[1].legend(frameon=False, fontsize=7, ncol=2)

    # ---- (c) reward-term decomposition (social SAC) ----
    p = f"{L}/socialbench_intersection_social_sac_decoupled_1m/progress.csv"
    d = _load(p, [t[0] for t in TERMS])
    stacks, colors, labels = [], [], []
    for col, lab, col_hex in TERMS:
        stacks.append(np.nan_to_num(_bin(d["t"], np.abs(d[col]), edges))); colors.append(col_hex); labels.append(lab)
    ax[2].stackplot(ctr, np.vstack(stacks), colors=colors, labels=labels, alpha=0.9)
    ax[2].set_title("(c) Credit assignment: reward terms (social SAC)")
    ax[2].set_xlabel("Timesteps"); ax[2].set_ylabel("|penalty| (episode)")
    ax[2].ticklabel_format(axis="x", style="sci", scilimits=(3, 6)); ax[2].legend(frameon=False, fontsize=6, ncol=2, loc="upper right")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
