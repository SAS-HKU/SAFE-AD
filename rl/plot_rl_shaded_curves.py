"""Multi-seed shaded-band learning curves (scienceplots) + cross-seed final-stats.

For each group (reward arm) given as several seed runs, bins a per-episode metric
over timesteps, then plots the across-seed mean with a shaded band (std or 95% CI).
Also prints/saves the cross-seed final-decile mean ± std table and a Welch t-test
of the social vs stock final performance.

Usage:
  python rl/plot_rl_shaded_curves.py \
    --group "Stock SAC=rl/logs/metadrive/matched_stock_intersection_respawn_sac_1m/progress.csv,rl/logs/metadrive/msd_stock_sac_int_s1/progress.csv,rl/logs/metadrive/msd_stock_sac_int_s2/progress.csv" \
    --group "Risk SAC=...s0,...s1,...s2" \
    --group "Social SAC=...s0,...s1,...s2" \
    --metrics success route_completion --band ci \
    --out rl/logs/figures/intersection_sac_multiseed
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

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
TITLES = {"success": "Success rate", "route_completion": "Route completion",
          "crash_vehicle": "Crash rate", "out_of_road": "Off-road rate"}


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
    out = {"t": t[o]}
    for k in ("success", "route_completion", "crash_vehicle", "out_of_road"):
        out[k] = np.array([_f(r.get(k), 0.0) for r in rows])[o]
    return out


def _bin(t, y, edges):
    idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
    return np.array([y[idx == b].mean() if np.any(idx == b) else np.nan
                     for b in range(len(edges) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help="'Label=csv1,csv2,...'")
    ap.add_argument("--metrics", nargs="+", default=["success", "route_completion"])
    ap.add_argument("--band", choices=["std", "ci"], default="ci")
    ap.add_argument("--bins", type=int, default=25)
    ap.add_argument("--out", default="rl/logs/figures/multiseed")
    args = ap.parse_args()

    groups = []  # (label, [seed dicts])
    for spec in args.group:
        label, paths = spec.split("=", 1)
        seeds = [_load(p.strip()) for p in paths.split(",") if Path(p.strip()).exists()]
        groups.append((label.strip(), seeds))

    # common timestep range across all seeds
    tmax = max((s["t"].max() for _, ss in groups for s in ss), default=1.0)
    edges = np.linspace(0, tmax + 1e-6, args.bins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    _style()
    fig, axes = plt.subplots(1, len(args.metrics), figsize=(4.8 * len(args.metrics), 3.2),
                             squeeze=False, constrained_layout=True)
    axes = axes[0]
    final = {}  # label -> metric -> [per-seed final-decile mean]
    for gi, (label, seeds) in enumerate(groups):
        c = COLORS[gi % len(COLORS)]
        nseed = len(seeds)
        final[label] = {}
        for ax, mk in zip(axes, args.metrics):
            series = np.vstack([_bin(s["t"], s[mk], edges) for s in seeds]) if seeds else np.full((1, len(ctr)), np.nan)
            mean = np.nanmean(series, axis=0)
            sd = np.nanstd(series, axis=0, ddof=1) if nseed > 1 else np.zeros_like(mean)
            band = sd if args.band == "std" else (1.96 * sd / max(1, math.sqrt(nseed)))
            ax.plot(ctr, mean, color=c, linewidth=1.6, label=f"{label} (n={nseed})")
            ax.fill_between(ctr, mean - band, mean + band, color=c, alpha=0.18, linewidth=0)
            ax.set_title(TITLES.get(mk, mk)); ax.set_xlabel("Timesteps")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 6))
            # final-decile per-seed means (last 10% of episodes)
            vals = []
            for s in seeds:
                k = max(1, len(s["t"]) // 10)
                vals.append(float(np.nanmean(s[mk][-k:])))
            final[label][mk] = vals
    axes[0].set_ylabel("rate / fraction")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(groups), frameon=False, bbox_to_anchor=(0.5, 1.13))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}.png/.pdf")

    # cross-seed final-decile table + Welch test (social vs stock) on first metric
    print("\n=== cross-seed final-decile mean ± std ===")
    for label, md in final.items():
        cells = "  ".join(f"{m}={np.mean(v):.3f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.3f}(n={len(v)})"
                          for m, v in md.items())
        print(f"  {label:14s} {cells}")
    try:
        from scipy import stats
        st = next((l for l in final if l.lower().startswith("stock")), None)
        so = next((l for l in final if l.lower().startswith("social")), None)
        mk = args.metrics[0]
        if st and so and len(final[st][mk]) > 1 and len(final[so][mk]) > 1:
            t, p = stats.ttest_ind(final[so][mk], final[st][mk], equal_var=False)
            print(f"\nWelch t-test (social vs stock, {mk}): t={t:.3f} p={p:.3f} "
                  f"(n={len(final[so][mk])} vs {len(final[st][mk])})")
    except ImportError:
        print("\n[note] scipy not available — skipping Welch test (report mean±std).")


if __name__ == "__main__":
    main()
