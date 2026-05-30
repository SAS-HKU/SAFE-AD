"""Zero-shot transfer bars (scienceplots): a source-trained policy evaluated on
unseen target scenarios vs. the scenario-native policy. Reads the zero-shot
eval_summary.json and the native per-scenario eval summaries.

Usage:
  python rl/plot_rl_zeroshot.py \
    --zeroshot rl/logs/metadrive/eval_zeroshot_social_sac/eval_summary.json \
    --native-template rl/logs/metadrive/eval_{S}_3arm_full/eval_summary.json \
    --native-label social_sac --out rl/logs/figures/zeroshot_social_sac
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


def _style():
    try:
        plt.style.use(["science", "ieee", "grid", "no-latex"] if _HAS else "seaborn-v0_8-whitegrid")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeroshot", required=True)
    ap.add_argument("--native-template", required=True, help="path with {S} placeholder")
    ap.add_argument("--native-label", default="social_sac")
    ap.add_argument("--out", default="rl/logs/figures/zeroshot")
    args = ap.parse_args()

    zs = json.load(open(args.zeroshot, encoding="utf-8"))
    zsl = {k.split("@")[0]: v for k, v in zs.items()}
    # labels look like zeroshot_<scenario>_sac (e.g. zeroshot_round_sac)
    ALIAS = [("merge", "merge"), ("round", "roundabout"), ("inter", "intersection")]
    targets = []
    for lab in sorted(zsl):
        for sub, S in ALIAS:
            if sub in lab.lower():
                targets.append((S, lab)); break

    metrics = [("success", "Success"), ("route_completion", "Route"), ("social_friendliness_score", "Social")]
    scen = [S for S, _ in targets]
    _style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.6, 3.0), constrained_layout=True)
    x = np.arange(len(scen)); w = 0.38
    for ax, (mk, mt) in zip(axes, metrics):
        zvals, nvals = [], []
        for S, lab in targets:
            zvals.append(zsl[lab].get(mk + "_mean", np.nan))
            nat = json.load(open(args.native_template.format(S=S), encoding="utf-8"))
            natb = {k.split("@")[0]: v for k, v in nat.items()}.get(args.native_label, {})
            nvals.append(natb.get(mk + "_mean", np.nan))
        ax.bar(x - w / 2, zvals, w, label="Zero-shot (intersection→)", color="#D55E00")
        ax.bar(x + w / 2, nvals, w, label="Native (scenario-trained)", color="#0072B2")
        ax.set_xticks(x); ax.set_xticklabels([s.title() for s in scen])
        ax.set_title(mt); ax.set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}.png/.pdf  targets={scen}")


if __name__ == "__main__":
    main()
