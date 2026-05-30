"""
Publication-quality training-curve plots for the DREAM decision-RL policy.

Reads one or more `decision_ppo_*_log.json` files produced by
`rl/train_decision_ppo.py` and plots, with SciencePlots styling:

    (a) average episode return       (ep_return_mean) vs. env steps
    (b) average training reward      (pi_loss-proxy / running mean) vs. env steps
    (c) policy entropy               vs. env steps
    (d) lane-change exploration      (lc_frac, lc_rejected/lc_actions) vs. env steps

The (a) and (b) panels are the ones to drop into the paper's experiment
section.  (c) and (d) are diagnostic; --diagnostic toggles their inclusion.

Usage
-----
# Single run
python -m rl.plot_training_curves --logs rl/logs/decision_ppo_v2_log.json \
    --labels "DREAM-RL (ours)" --out figures/ppo_training.pdf

# Compare v1 vs v2
python -m rl.plot_training_curves \
    --logs rl/logs/decision_ppo_log.json rl/logs/decision_ppo_v2_log.json \
    --labels "baseline" "rebalanced reward (ours)" \
    --out figures/ppo_training_compare.pdf --diagnostic

The script will emit both PDF (vector, for LaTeX) and PNG (300 dpi, for
slides / Markdown) with the same stem.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401  (registers styles)
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def ema(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Exponential moving average for smoother curves."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def rolling_mean(x: np.ndarray, k: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or k <= 1:
        return x
    kern = np.ones(k) / k
    # 'same' keeps length; pad-reflect at edges so first points are valid
    pad = k // 2
    xp = np.pad(x, pad, mode='edge')
    y = np.convolve(xp, kern, mode='valid')
    return y[:x.size]


# ---------------------------------------------------------------------------
# Log loader
# ---------------------------------------------------------------------------

def load_log(path: str) -> dict:
    with open(path, "r") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"Empty log: {path}")
    # Union of all keys across rows (older logs may be missing some fields)
    keys = set()
    for r in rows:
        keys.update(r.keys())
    out = {}
    for k in keys:
        vals = [r.get(k, np.nan) for r in rows]
        try:
            out[k] = np.array(vals, dtype=np.float64)
        except (TypeError, ValueError):
            # Ignore metadata-style string fields such as env_id.
            continue

    # Backfill `reward_mean` for legacy logs that didn't record it.
    # Approximation: avg per-step reward ≈ ep_return_mean / assumed_ep_len.
    # MAX_STEPS defaults to 400 but many episodes end early — use 200 as a
    # conservative middle estimate. Mark as approximate so the plot legend
    # can distinguish it.
    if "reward_mean" not in out and "ep_return_mean" in out:
        assumed_len = 200.0
        out["reward_mean"] = out["ep_return_mean"] / assumed_len
        out["_reward_mean_approx"] = np.ones_like(out["reward_mean"])
    else:
        out["_reward_mean_approx"] = np.zeros_like(out.get("reward_mean",
                                                           np.array([0.0])))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _use_style():
    if _HAS_SCIENCEPLOTS:
        # IEEE + no-latex (portable on Windows without a LaTeX install)
        plt.style.use(['science', 'ieee', 'no-latex'])
    else:
        plt.style.use('seaborn-v0_8-whitegrid')


_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']


def _plot_panel(ax, logs, labels, key, smooth_k, ylabel, title=None,
                shade=True):
    """Plot one metric across all runs onto axis `ax`."""
    for i, (log, lab) in enumerate(zip(logs, labels)):
        if key not in log:
            continue
        x = log["steps"]
        y = log[key]
        y_s = rolling_mean(y, k=smooth_k)
        c = _COLORS[i % len(_COLORS)]
        # Mark approximated reward_mean curves with dashed line
        is_approx = (key == "reward_mean"
                     and bool(np.any(log.get("_reward_mean_approx", 0) > 0.5)))
        ls = '--' if is_approx else '-'
        suffix = " (est.)" if is_approx else ""
        # Raw trace (light)
        ax.plot(x, y, color=c, alpha=0.18, linewidth=0.7, linestyle=ls)
        # Smoothed trace
        ax.plot(x, y_s, color=c, linewidth=1.5, linestyle=ls,
                label=lab + suffix)
        # ±1 std shade (over a short sliding window)
        if shade and y.size > smooth_k:
            std = np.array([np.std(y[max(0, j - smooth_k):j + 1])
                            for j in range(y.size)])
            ax.fill_between(x, y_s - std, y_s + std, color=c, alpha=0.12,
                            linewidth=0)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 6))


def plot_training_curves(log_paths, labels, out_path, diagnostic=False,
                         smooth_k=5, figsize=None):
    _use_style()

    logs = [load_log(p) for p in log_paths]

    # Primary paper panels: per-step reward (rises from low values, converges)
    # and per-episode return (monotone-ish improvement over training).
    panels = [
        ("reward_mean",    "Avg. training reward (per step)",
                                                    "Average training reward"),
        ("ep_return_mean", "Episode return",        "Average episode return"),
    ]
    if diagnostic:
        panels += [
            ("entropy",    "Policy entropy",        "Policy entropy (9-action)"),
            ("lc_frac",    "LC-action fraction",    "Lane-change exploration"),
            ("pi_loss",    "Policy-gradient loss",  "PPO policy loss"),
        ]

    n = len(panels)
    ncols = 2
    nrows = (n + 1) // 2
    if figsize is None:
        figsize = (3.4 * ncols, 2.4 * nrows)   # IEEE column-friendly
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax, (key, ylab, title) in zip(axes.flat, panels):
        _plot_panel(ax, logs, labels, key, smooth_k, ylab, title)

    # Hide unused axes
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    # Single legend at figure top
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc='upper center',
                   ncol=min(len(labels), 4), frameon=False,
                   bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PDF for LaTeX
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    # PNG for slides / previews
    fig.savefig(out_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    print(f"[plot] wrote {out_path.with_suffix('.pdf')}")
    print(f"[plot] wrote {out_path.with_suffix('.png')}")

    # Also emit a numeric summary for the paper
    _print_summary(logs, labels)
    return fig


def _print_summary(logs, labels):
    print("\n=== Training summary ===")
    hdr = f"{'run':<30s} {'final_return':>14s} {'final_ent':>12s} " \
          f"{'mean_lc_frac':>14s} {'reject_rate':>14s}"
    print(hdr)
    print("-" * len(hdr))
    for log, lab in zip(logs, labels):
        final_ret = np.mean(log["ep_return_mean"][-10:]) \
            if log["ep_return_mean"].size else np.nan
        final_ent = np.mean(log.get("entropy", np.array([np.nan]))[-10:])
        if "lc_frac" in log and log["lc_frac"].size:
            mean_lc = float(np.mean(log["lc_frac"]))
        else:
            mean_lc = float('nan')
        if "lc_actions" in log and "lc_rejected" in log:
            la = log["lc_actions"].sum()
            lr = log["lc_rejected"].sum()
            rej = lr / la if la > 0 else float('nan')
        else:
            rej = float('nan')
        print(f"{lab:<30s} {final_ret:>14.2f} {final_ent:>12.3f} "
              f"{mean_lc:>14.3f} {rej:>14.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", required=True,
                   help="One or more PPO training-log JSON files")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Display labels (default: file stems)")
    p.add_argument("--out", default="figures/ppo_training.pdf",
                   help="Output path (pdf/png are both written)")
    p.add_argument("--diagnostic", action="store_true",
                   help="Also plot entropy and LC exploration panels")
    p.add_argument("--smooth", type=int, default=5,
                   help="Rolling-mean window (rollouts)")
    return p.parse_args()


def main():
    args = _parse_args()
    labels = args.labels or [Path(p).stem for p in args.logs]
    if len(labels) != len(args.logs):
        raise SystemExit("--labels must match --logs in count")
    plot_training_curves(args.logs, labels, args.out,
                         diagnostic=args.diagnostic,
                         smooth_k=args.smooth)


if __name__ == "__main__":
    main()
