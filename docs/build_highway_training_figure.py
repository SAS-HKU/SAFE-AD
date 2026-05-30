"""Figure B (Highway-env): one 3-panel training figure — depth of the social objective.

(a) return vs steps (social PPO & DQN) — convergence
(b) reward-component evolution vs steps (c_social, c_field, c_comfort, c_ttc) — how the
    social/risk penalties shrink as the policy stabilizes (credit assignment)
(c) safety-metric evolution on highway_v0 AND merge_v0 (collision rate + imposed rear-decel)
    — externality reduction + generalization while learning
Reads summary.json 'records' from social_ppo_a5 / social_dqn_a5 (scienceplots styling).
"""
import json
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

L = "rl/logs"
OUT = "docs/figB_highway_training"
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]


def _style():
    try:
        plt.style.use(["science", "ieee", "grid", "no-latex"] if _HAS else "seaborn-v0_8-whitegrid")
    except Exception:
        pass


def _records(run):
    p = f"{L}/{run}/summary.json"
    if not Path(p).exists():
        return None
    return json.load(open(p, encoding="utf-8")).get("records", [])


def _series(recs, key):
    t = np.array([r.get("timesteps", np.nan) for r in recs], float)
    y = np.array([r.get(key, np.nan) for r in recs], float)
    return t, y


def main():
    _style()
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.1), constrained_layout=True)

    # ---- (a) return vs steps (social PPO, DQN) ----
    for i, (run, lab) in enumerate([("social_ppo_a5", "Social PPO"), ("social_dqn_a5", "Social DQN")]):
        recs = _records(run)
        if recs:
            t, y = _series(recs, "highway_v0_return_mean")
            ax[0].plot(t, y, color=C[i], lw=1.6, marker="o", ms=3, label=lab)
    ax[0].set_title("(a) Convergence: eval return vs steps")
    ax[0].set_xlabel("Timesteps"); ax[0].set_ylabel("Eval return (highway_v0)")
    ax[0].ticklabel_format(axis="x", style="sci", scilimits=(3, 6)); ax[0].legend(frameon=False, fontsize=7)

    # ---- (b) reward-component evolution (social PPO, highway_v0) ----
    recs = _records("social_ppo_a5")
    comps = [("highway_v0_c_field", "Risk-field cost"), ("highway_v0_c_comfort", "Comfort cost"),
             ("highway_v0_c_ttc", "TTC cost"), ("highway_v0_c_thw", "THW cost")]
    if recs:
        for i, (k, lab) in enumerate(comps):
            t, y = _series(recs, k)
            if np.any(np.isfinite(y)):
                ax[1].plot(t, y, color=C[i], lw=1.4, marker="o", ms=2.5, label=lab)
    ax[1].set_title("(b) Credit assignment: cost terms decay")
    ax[1].set_xlabel("Timesteps"); ax[1].set_ylabel("Cost component")
    ax[1].ticklabel_format(axis="x", style="sci", scilimits=(3, 6)); ax[1].legend(frameon=False, fontsize=7)

    # ---- (c) safety-metric evolution across scenarios (social PPO) ----
    if recs:
        for i, (k, lab) in enumerate([("highway_v0_collision_rate", "Highway collision"),
                                      ("merge_v0_collision_rate", "Merge collision")]):
            t, y = _series(recs, k)
            if np.any(np.isfinite(y)):
                ax[2].plot(t, y, color=C[i], lw=1.6, marker="o", ms=3, label=lab)
        ax2b = ax[2].twinx()
        t, y = _series(recs, "highway_v0_corridor_risk_mean")
        if np.any(np.isfinite(y)):
            ax2b.plot(t, y, color=C[2], lw=1.2, ls="--", marker="s", ms=2.5, label="Corridor risk (highway)")
        ax2b.set_ylabel("corridor risk exposure", fontsize=8)
        ax2b.legend(frameon=False, fontsize=6, loc="upper right")
    ax[2].set_title("(c) Externality + generalization while learning")
    ax[2].set_xlabel("Timesteps"); ax[2].set_ylabel("Collision rate")
    ax[2].ticklabel_format(axis="x", style="sci", scilimits=(3, 6)); ax[2].legend(frameon=False, fontsize=7, loc="upper left")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
