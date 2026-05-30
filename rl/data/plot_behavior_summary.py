"""
Behaviour-summary figure for human-driving BC datasets.
========================================================

Reads one or more ``.npz`` files produced by
:mod:`rl.data.historical_extractor` (schema v3 preferred; v2 partially
supported) and writes a paper-style figure that visualises *tactical*
human driving behaviour rather than raw trajectories.

The figure has six panels in a 3 x 2 grid and uses the SciencePlots
``['science', 'ieee', 'no-latex']`` style already adopted elsewhere in
``rl/`` (see :mod:`rl.plot_training_curves`).

Panels
------
(a) 9-way action histogram
(b) Lane-delta and speed-mode marginals
(c) Outcome behaviour bars
    (lane-change success, lane-change advantage, blocked-by-leader,
     escape | blocked, near-miss, collision)
(d) ``future_risk_change`` distribution grouped by keep / left / right
(e) Advantage calibration:  P(human LC) vs binned ``best_adv``
(f) Dataset comparison heatmap:  rows = datasets, cols = key fractions

Each panel is rendered by a small function and *fails soft* — missing
keys (e.g. running on a schema-v2 dataset) skip the offending panel and
print a one-line warning instead of aborting the whole figure.

Usage
-----
Single dataset from a pre-extracted ``.npz``::

    python -m rl.data.plot_behavior_summary \
        --inputs rl/checkpoints/bc_highd_full.npz \
        --labels highD \
        --out figures/behavior_summary_highd

Direct from a dataset directory (no ``.npz`` round-trip; runs the
extractor in memory)::

    python -m rl.data.plot_behavior_summary \
        --from-dataset highD data/highD 01,02 \
        --out figures/behavior_summary_highd

Multi-dataset comparison; mix .npz and live extraction freely.  The
heatmap activates whenever there are >=2 sources::

    python -m rl.data.plot_behavior_summary \
        --inputs rl/checkpoints/bc_combined.npz \
        --from-dataset highD data/highD all \
        --from-dataset exiD  auto         00,01,02 \
        --labels combined highD exiD \
        --out figures/behavior_summary_compare \
        --cache-npz-dir rl/checkpoints/

Use ``auto`` as the data dir to fall back to ``data/<format>``.  Use
``all`` for the recordings to auto-discover every ``*_tracks.csv``.

Notes on schema
---------------
The extractor's schema-v3 keys this script consumes::

    action_9way / actions
    lane_delta_label, speed_mode_label   (auto-derived from action_9way if absent)
    future_risk_change
    lane_change_success / lc_success
    near_miss_future, collision_future
    blocked_by_leader_flag, escape_success_flag
    lane_change_advantage_flag
    adv_left, adv_right, best_adv

If a v2 dataset is supplied, panels (c)-(f) are skipped silently.  Re-run
the extractor (`python -m rl.data.historical_extractor ... --out-path
bc_*.npz`) to upgrade to v3.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401  (registers styles)
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False


# ---------------------------------------------------------------------------
# Style + palette (kept in sync with rl/plot_training_curves.py)
# ---------------------------------------------------------------------------

_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
_LANE_COLORS = {
    'keep':  '#0072B2',
    'left':  '#009E73',
    'right': '#D55E00',
}
_SPEED_COLORS = {
    'maintain': '#0072B2',
    'slower':   '#CC79A7',
    'faster':   '#E69F00',
}

# 9-way action labels (must match rl.policy.decision_policy.encode_action):
#   lane_block 0 = keep, 1 = right (-1), 2 = left (+1); speed 0/1/2 = maintain/slower/faster
ACTION_LABELS_9 = [
    'K-mt', 'K-sl', 'K-fa',     # keep
    'R-mt', 'R-sl', 'R-fa',     # right
    'L-mt', 'L-sl', 'L-fa',     # left
]
LANE_DELTA_NAMES  = ['keep', 'right', 'left']      # index = lane_delta_label
SPEED_MODE_NAMES  = ['maintain', 'slower', 'faster']  # index = speed_mode_label


def _use_style() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(['science', 'ieee', 'no-latex'])
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
    # Force-disable tight savefig bbox so the saved canvas matches
    # figsize exactly even when annotations sit just outside the axes
    # (per-agent figures with sparse data exposed this).
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['savefig.bbox'] = 'standard'


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# v2 -> v3 alias map.  Read order: prefer v3 key, fall back to v2 alias.
_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    'action_9way':                ('action_9way', 'actions'),
    'lane_change_success':        ('lane_change_success', 'lc_success'),
    'short_horizon_return_proxy': ('short_horizon_return_proxy', 'future_return'),
}


def _resolve(d: dict, name: str) -> np.ndarray | None:
    """Return ``d[name]`` honouring the schema-v3 alias table; None if absent."""
    for k in _KEY_ALIASES.get(name, (name,)):
        if k in d:
            return np.asarray(d[k])
    return None


def load_npz(path: str) -> dict:
    """Load a BC ``.npz`` produced by the extractor; values are numpy arrays."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


_DATASET_LOWER_TO_DISPLAY = {
    'exid': 'exiD', 'highd': 'highD', 'ind': 'inD',
    'round': 'rounD', 'roundd': 'rounD', 'unid': 'uniD',
    'sqm-n-4': 'SQM-N-4', 'ytdj-3': 'YTDJ-3',
    'xam-n-5': 'XAM-N-5', 'xam-n-6': 'XAM-N-6',
}


def clean_label(raw: str) -> str:
    """
    Strip filename artefacts (``bc_``, ``_smoke``, ``_v5``, ``.npz``)
    from a dataset label so plot titles read like ``highD`` rather than
    ``bc_v5_smoke``.  Falls back to the input if no canonical name can
    be detected.
    """
    if raw is None:
        return 'dataset'
    s = str(raw).strip()
    s = s.replace('.npz', '')
    base = os.path.basename(s).lower()
    for prefix in ('bc_', 'bc-', 'historical_'):
        if base.startswith(prefix):
            base = base[len(prefix):]
    for suffix in ('_smoke', '_smoke2', '_full', '_padtest', '_v3', '_v4',
                   '_v5', '_v6', '_combined'):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    # If the trimmed name matches a known dataset key, return its display form.
    if base in _DATASET_LOWER_TO_DISPLAY:
        return _DATASET_LOWER_TO_DISPLAY[base]
    # ``highd_01`` → ``highD-01``
    for key, disp in _DATASET_LOWER_TO_DISPLAY.items():
        if base.startswith(key + '_') or base.startswith(key + '-'):
            tail = base[len(key) + 1:]
            return f"{disp}-{tail}" if tail else disp
    return base or 'dataset'


def filter_by_ego_id(d: dict, ego_id: int) -> dict:
    """Return a new dict with all per-sample arrays filtered to one ego."""
    if 'ego_id' not in d and 'track_ids' not in d:
        raise ValueError("Dataset has no ego_id / track_ids; cannot filter")
    ids = np.asarray(d.get('ego_id', d.get('track_ids')), dtype=np.int64)
    mask = ids == int(ego_id)
    if not mask.any():
        raise ValueError(
            f"ego_id={ego_id} not found in dataset "
            f"(unique ids: {np.unique(ids).size})"
        )
    n = int(mask.sum())
    out: dict = {}
    for k, v in d.items():
        a = np.asarray(v)
        if a.ndim >= 1 and a.shape[0] == ids.size:
            out[k] = a[mask]
        else:
            out[k] = a               # scalars / sidecars unchanged
    print(f"[plot] filtered to ego_id={ego_id}: {n} samples")
    return out


def list_top_egos(d: dict, top_k: int = 20) -> list:
    """Return the ``top_k`` ego ids with the most samples in the dataset."""
    ids = np.asarray(d.get('ego_id', d.get('track_ids')), dtype=np.int64)
    if ids.size == 0:
        return []
    uniq, counts = np.unique(ids, return_counts=True)
    order = np.argsort(-counts)
    return [(int(uniq[k]), int(counts[k])) for k in order[:top_k]]


def extract_live(
    dataset_format: str,
    data_dir: str | None = None,
    recordings: str = 'all',
    horizon_sec: float = 1.5,
    outcome_horizon_sec: float = 3.0,
    limit_tracks: int | None = None,
    save_npz: str | None = None,
) -> dict:
    """
    Extract behaviour features from a dataset *directory* in memory.

    This bypasses the ``.npz`` round-trip by calling
    :func:`rl.data.historical_extractor.extract_many` directly.  Schema is
    always v3 (the extractor's current schema), so every panel of the
    behaviour-summary figure is fully populated.

    Args:
        dataset_format     : one of ``exiD``, ``highD``, ``inD``, ``rounD``,
                             ``uniD``, ``SQM-N-4``, ``YTDJ-3``, ``XAM-N-5``,
                             ``XAM-N-6``  (case-insensitive — same parsing
                             rules as ``rl.data.historical_extractor``).
        data_dir           : path to the dataset directory.  When ``None``
                             the extractor's default of
                             ``<repo>/data/<canonical-name>`` is used.
        recordings         : comma-separated recording IDs (e.g.
                             ``"00,01,02"``) or the literal ``"all"`` to
                             discover every ``*_tracks.csv`` in the
                             directory.  Special datasets (SQM-N-4 / YTDJ-3
                             / XAM-*) ignore this and use ``"00"``.
        horizon_sec        : action-label horizon (forwarded).
        outcome_horizon_sec: outcome-label horizon (forwarded).
        limit_tracks       : per-recording cap on ego candidates — useful
                             for fast smoke runs.
        save_npz           : when set, also writes the extracted dict to
                             this path so subsequent figures can reuse it.

    Returns:
        Dict identical in shape to ``np.load(out_path)`` for an extractor
        ``--out-path`` run; passes straight to
        :func:`render_figure`.
    """
    # Lazy import — the extractor pulls in pandas and (transitively) the
    # dataset reader, which we don't want as a hard dependency for users
    # who only ever consume pre-extracted ``.npz`` files.
    from rl.data import historical_extractor as he

    canonical = he._canonical_dataset_name(dataset_format)
    if data_dir is None:
        data_dir = he._default_data_dir(canonical)
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"--data-dir not found: {data_dir}")

    if str(recordings).strip().lower() == 'all':
        rids = he._discover_recordings(data_dir, canonical)
        if not rids:
            raise SystemExit(f"No recordings discovered under {data_dir}")
        print(f"[plot] live-extract: {canonical} discovered {len(rids)} recordings "
              f"({rids[0]}..{rids[-1]})")
    else:
        rids = [r.strip() for r in str(recordings).split(',') if r.strip()]
        print(f"[plot] live-extract: {canonical} recordings={rids}")

    out = he.extract_many(
        data_dir, rids,
        horizon_sec=horizon_sec,
        outcome_horizon_sec=outcome_horizon_sec,
        limit_tracks=limit_tracks,
        dataset_format=canonical,
    )
    print(f"[plot] live-extract: {canonical} produced {out['obs'].shape[0]} samples")

    if save_npz:
        os.makedirs(os.path.dirname(save_npz) or '.', exist_ok=True)
        np.savez_compressed(save_npz, **out)
        print(f"[plot] live-extract: cached to {save_npz}")

    return out


# ---------------------------------------------------------------------------
# Per-dataset summary statistics (also used by the heatmap)
# ---------------------------------------------------------------------------

# Columns in the dataset-comparison heatmap.  Each entry is
# (display_name, callable(data) -> float in [0,1] or NaN).  Order matters.
def _frac(arr: np.ndarray | None, n: int) -> float:
    return float(arr.mean()) if (arr is not None and arr.size == n) else float('nan')


def _frac_in_mask(arr: np.ndarray | None, mask: np.ndarray) -> float:
    if arr is None or arr.size != mask.size or not mask.any():
        return float('nan')
    return float(arr[mask].mean())


def summary_row(d: dict) -> Dict[str, float]:
    """Compute the heatmap row for one dataset."""
    actions = _resolve(d, 'action_9way')
    if actions is None:
        return {}
    n = int(actions.size)
    lane_bin = (actions.astype(np.int64) // 3)
    lc_mask  = lane_bin != 0

    lc_succ = _resolve(d, 'lane_change_success')
    lc_adv  = _resolve(d, 'lane_change_advantage_flag')
    blocked = _resolve(d, 'blocked_by_leader_flag')
    escape  = _resolve(d, 'escape_success_flag')
    near_m  = _resolve(d, 'near_miss_future')
    coll    = _resolve(d, 'collision_future')
    frc     = _resolve(d, 'future_risk_change')

    blocked_mask = (blocked == 1) if (blocked is not None and blocked.size == n) else np.zeros(n, dtype=bool)

    return {
        'lane_change_frac':      float(lc_mask.mean()),
        'lc_success_frac_lc':    _frac_in_mask(lc_succ, lc_mask),
        'lc_advantage_frac_lc':  _frac_in_mask(lc_adv,  lc_mask),
        'blocked_frac':          _frac(blocked, n),
        'escape_given_blocked':  _frac_in_mask(escape, blocked_mask),
        'near_miss_frac':        _frac(near_m, n),
        'collision_frac':        _frac(coll,   n),
        'mean_future_risk_change': float(frc.mean()) if (frc is not None and frc.size == n) else float('nan'),
    }


HEATMAP_COLUMNS: Sequence[Tuple[str, str]] = (
    ('lane_change_frac',        'P(LC)'),
    ('lc_success_frac_lc',      'LC success | LC'),
    ('lc_advantage_frac_lc',    'LC adv | LC'),
    ('blocked_frac',            'P(blocked)'),
    ('escape_given_blocked',    'escape | blocked'),
    ('near_miss_frac',          'P(near-miss)'),
    ('collision_frac',          'P(collision)'),
)


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def _ensure_len(name: str, arr: np.ndarray | None, n: int, ax: plt.Axes) -> bool:
    """Common guard; if ``arr`` is missing/wrong length, print a hint and hide ax."""
    if arr is None or arr.size != n:
        ax.set_visible(False)
        print(f"[plot] skip panel: missing or shape-mismatched key '{name}'")
        return False
    return True


def panel_action_histogram(ax: plt.Axes, d: dict, label: str) -> None:
    """(a) 9-way action histogram (counts as % of all samples)."""
    actions = _resolve(d, 'action_9way')
    if actions is None or actions.size == 0:
        ax.set_visible(False)
        return
    hist = np.bincount(actions.astype(np.int64), minlength=9)[:9]
    frac = hist / max(1, hist.sum())
    bar_colors = ([_LANE_COLORS['keep']] * 3
                  + [_LANE_COLORS['right']] * 3
                  + [_LANE_COLORS['left']]  * 3)
    xs = np.arange(9)
    ax.bar(xs, frac * 100.0, color=bar_colors, edgecolor='black', linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(ACTION_LABELS_9, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('share of samples (%)')
    ax.set_title('(a) 9-way Action Distribution')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    for x0, x1, name in [(-0.4, 2.4, 'keep'), (2.6, 5.4, 'right'), (5.6, 8.4, 'left')]:
        ax.axvspan(x0, x1, ymin=0.0, ymax=0.02, color=_LANE_COLORS[name], alpha=0.6, zorder=0)


def panel_lane_speed_marginals(ax: plt.Axes, d: dict, label: str) -> None:
    """(b) Lane-delta and speed-mode marginals."""
    actions = _resolve(d, 'action_9way')
    if actions is None or actions.size == 0:
        ax.set_visible(False)
        return
    lane = _resolve(d, 'lane_delta_label')
    speed = _resolve(d, 'speed_mode_label')
    # Derive from action_9way if labels not present (matches encode_action)
    if lane is None or lane.size != actions.size:
        lane = (actions.astype(np.int64) // 3).astype(np.int8)  # 0=keep,1=right,2=left
    if speed is None or speed.size != actions.size:
        speed = (actions.astype(np.int64) %  3).astype(np.int8)  # 0=maint,1=slow,2=fast

    lane_frac = np.bincount(lane.astype(np.int64), minlength=3)[:3] / max(1, lane.size)
    speed_frac = np.bincount(speed.astype(np.int64), minlength=3)[:3] / max(1, speed.size)

    # Two side-by-side bar groups
    width = 0.35
    xs_lane  = np.array([0.0, 1.0, 2.0]) - width / 2
    xs_speed = np.array([0.0, 1.0, 2.0]) + width / 2

    ax.bar(xs_lane,  lane_frac * 100.0,  width=width,
           color=[_LANE_COLORS[n]  for n in LANE_DELTA_NAMES],
           edgecolor='black', linewidth=0.4, label='lane delta')
    ax.bar(xs_speed, speed_frac * 100.0, width=width,
           color=[_SPEED_COLORS[n] for n in SPEED_MODE_NAMES],
           edgecolor='black', linewidth=0.4, hatch='//', label='speed mode')

    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_xticklabels([f"{l}\n/{s}" for l, s in zip(LANE_DELTA_NAMES, SPEED_MODE_NAMES)],
                       fontsize=7)
    ax.set_ylabel('share (%)')
    ax.set_title('(b) Lane-Delta and Speed-Mode Marginals')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc='upper right')


def panel_outcome_bars(ax: plt.Axes, d: dict, label: str) -> None:
    """(c) Outcome behaviour bars on the same axis."""
    actions = _resolve(d, 'action_9way')
    if actions is None or actions.size == 0:
        ax.set_visible(False)
        print("[plot] skip panel (c): no actions array")
        return
    n = int(actions.size)
    lane_bin = actions.astype(np.int64) // 3
    lc_mask = lane_bin != 0
    blocked = _resolve(d, 'blocked_by_leader_flag')
    blocked_mask = (blocked == 1) if (blocked is not None and blocked.size == n) else None

    lc_succ = _resolve(d, 'lane_change_success')
    lc_adv  = _resolve(d, 'lane_change_advantage_flag')
    escape  = _resolve(d, 'escape_success_flag')
    near_m  = _resolve(d, 'near_miss_future')
    coll    = _resolve(d, 'collision_future')

    # Each entry: (display name, fraction, denominator label)
    entries = []
    if lc_succ is not None and lc_succ.size == n:
        entries.append(('LC success | LC', _frac_in_mask(lc_succ, lc_mask), '|LC'))
    if lc_adv is not None and lc_adv.size == n:
        entries.append(('LC adv | LC',     _frac_in_mask(lc_adv,  lc_mask), '|LC'))
    if blocked is not None and blocked.size == n:
        entries.append(('blocked',         _frac(blocked, n), 'all'))
    if escape is not None and escape.size == n and blocked_mask is not None:
        entries.append(('escape | blocked', _frac_in_mask(escape, blocked_mask), '|blocked'))
    if near_m is not None and near_m.size == n:
        entries.append(('near-miss',       _frac(near_m, n), 'all'))
    if coll is not None and coll.size == n:
        entries.append(('collision',       _frac(coll,   n), 'all'))

    if not entries:
        ax.set_visible(False)
        print("[plot] skip panel (c): no outcome flags found (schema v2?)")
        return

    names = [e[0] for e in entries]
    fracs = np.asarray([e[1] for e in entries], dtype=np.float64)
    denoms = [e[2] for e in entries]
    color_map = {'all': '#0072B2', '|LC': '#009E73', '|blocked': '#D55E00'}
    colors = [color_map[d] for d in denoms]

    xs = np.arange(len(entries))
    bars = ax.bar(xs, fracs * 100.0, color=colors, edgecolor='black', linewidth=0.4)
    for bar, val in zip(bars, fracs):
        if not np.isfinite(val):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=6)
    ax.set_ylabel('rate (%)')
    ax.set_title('(c) Outcome Behaviour Rates')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    # Mini legend for the denominator coding
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec='black', lw=0.4)
               for c in color_map.values()]
    ax.legend(handles, list(color_map.keys()), title='denominator',
              frameon=False, fontsize=6, title_fontsize=6, loc='upper right')


def panel_risk_change_violin(ax: plt.Axes, d: dict, label: str) -> None:
    """(d) future_risk_change distribution grouped by lane action."""
    actions = _resolve(d, 'action_9way')
    frc     = _resolve(d, 'future_risk_change')
    if (actions is None or actions.size == 0
            or frc is None or frc.size != actions.size):
        ax.set_visible(False)
        print("[plot] skip panel (d): future_risk_change not available")
        return

    lane_bin = (actions.astype(np.int64) // 3)
    groups = []
    group_names = []
    group_colors = []
    for code, name in enumerate(LANE_DELTA_NAMES):  # 0=keep, 1=right, 2=left
        mask = (lane_bin == code) & np.isfinite(frc)
        if mask.sum() < 5:
            continue
        groups.append(frc[mask].astype(np.float64))
        group_names.append(name)
        group_colors.append(_LANE_COLORS[name])

    if not groups:
        ax.set_visible(False)
        print("[plot] skip panel (d): no group has enough samples")
        return

    parts = ax.violinplot(groups, positions=np.arange(len(groups)),
                          showmeans=False, showmedians=True, widths=0.75)
    for body, color in zip(parts['bodies'], group_colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.6)
        body.set_linewidth(0.4)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(0.8)
    for key in ('cbars', 'cmins', 'cmaxes'):
        if key in parts:
            parts[key].set_color('black')
            parts[key].set_linewidth(0.5)

    ax.axhline(0.0, color='grey', linewidth=0.6, linestyle='--', alpha=0.7)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(group_names)
    ax.set_ylabel(r'$\Delta$ corridor risk ($\tau{=}2$s)')
    ax.set_title('(d) Future-Corridor Risk Change by Action')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    # Robust y-limits: clip outside ±[1st, 99th] percentile of the union
    all_v = np.concatenate(groups)
    q_lo, q_hi = np.percentile(all_v, [1.0, 99.0])
    span = max(abs(q_lo), abs(q_hi), 1e-3)
    ax.set_ylim(-1.1 * span, 1.1 * span)


def panel_advantage_calibration(
    ax: plt.Axes, d: dict, label: str, n_bins: int = 10,
) -> None:
    """(e) P(human takes LC) vs binned best_adv."""
    actions = _resolve(d, 'action_9way')
    best_adv = _resolve(d, 'best_adv')
    if (actions is None or actions.size == 0
            or best_adv is None or best_adv.size != actions.size):
        ax.set_visible(False)
        print("[plot] skip panel (e): best_adv not available")
        return

    lane_bin = (actions.astype(np.int64) // 3)
    lc_taken = (lane_bin != 0).astype(np.int64)
    finite = np.isfinite(best_adv)
    if finite.sum() < 50:
        ax.set_visible(False)
        return
    ba = best_adv[finite]
    lc = lc_taken[finite]

    # Use empirical 1st-99th percentile to avoid sensitivity to long tails
    lo, hi = np.percentile(ba, [1.0, 99.0])
    if hi - lo < 1e-3:
        ax.set_visible(False)
        return
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    p_lc = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        m = (ba >= edges[k]) & (ba < edges[k + 1] if k < n_bins - 1 else ba <= edges[k + 1])
        counts[k] = int(m.sum())
        if counts[k] >= 30:
            p_lc[k] = float(lc[m].mean())

    valid = np.isfinite(p_lc)
    if valid.sum() < 2:
        ax.set_visible(False)
        return

    ax.plot(centres[valid], p_lc[valid] * 100.0,
            marker='o', color=_COLORS[0], linewidth=1.2, markersize=4,
            label='P(LC) (binned)')
    # Overall LC rate as a horizontal reference
    overall = float(lc.mean())
    ax.axhline(overall * 100.0, color='grey', linestyle='--', linewidth=0.8,
               alpha=0.7, label=f'overall {overall*100:.1f}%')
    # Sample count strip on a twin-y for transparency
    ax2 = ax.twinx()
    ax2.bar(centres, counts, width=(edges[1] - edges[0]) * 0.85,
            color=_COLORS[2], alpha=0.18, edgecolor='none', zorder=0,
            label='samples per bin')
    ax2.set_ylabel('samples', color=_COLORS[2], fontsize=7)
    ax2.tick_params(axis='y', labelcolor=_COLORS[2], labelsize=6)
    for spine in ('top',):
        ax2.spines[spine].set_visible(False)

    ax.set_xlabel('best_adv  (utility advantage of best adjacent lane)')
    ax.set_ylabel('P(human takes LC)  [%]')
    ax.set_title('(e) Lane-Change Advantage Calibration')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc='upper left')


def panel_dataset_heatmap(
    ax: plt.Axes,
    summaries: Dict[str, Dict[str, float]],
) -> None:
    """(f) Dataset comparison heatmap.  Cells are fractions in [0, 1]."""
    if len(summaries) < 1:
        ax.set_visible(False)
        return

    labels = list(summaries.keys())
    col_keys = [k for k, _ in HEATMAP_COLUMNS]
    col_disp = [disp for _, disp in HEATMAP_COLUMNS]

    matrix = np.full((len(labels), len(col_keys)), np.nan, dtype=np.float64)
    for r, lab in enumerate(labels):
        row = summaries[lab]
        for c, k in enumerate(col_keys):
            v = row.get(k, np.nan)
            if v is None or not np.isfinite(v):
                continue
            matrix[r, c] = v

    if not np.any(np.isfinite(matrix)):
        ax.set_visible(False)
        print("[plot] skip panel (f): no v3 outcome flags in any input")
        return

    cmap = plt.get_cmap('viridis')
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0,
                   origin='upper')
    ax.set_xticks(np.arange(len(col_keys)))
    ax.set_xticklabels(col_disp, rotation=30, ha='right', fontsize=6)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    # Annotate cells
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            if not np.isfinite(v):
                ax.text(c, r, '—', ha='center', va='center', fontsize=6,
                        color='lightgrey')
                continue
            txt_color = 'white' if v < 0.55 else 'black'
            ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                    fontsize=6, color=txt_color)
    ax.set_title('(f) Dataset Comparison')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('rate', fontsize=7)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def render_figure(
    sources: List[Tuple[str, "str | dict"]],
    out_path: str,
    n_calibration_bins: int = 10,
    figsize: Tuple[float, float] | None = None,
) -> None:
    """
    Render the 6-panel behaviour-summary figure.

    Args:
        sources : list of ``(label, source)`` pairs.  Each ``source`` is
                  either an ``.npz`` path (loaded with :func:`load_npz`)
                  or an already-prepared dict (e.g. returned by
                  :func:`extract_live`).
        out_path : output stem; ``.pdf`` and ``.png`` are written.
    """
    if not sources:
        raise ValueError("Need at least one source (--inputs or --from-dataset)")

    _use_style()

    # Pre-compute summary rows for the heatmap (and the title strip)
    datasets: Dict[str, dict] = {}
    summaries: Dict[str, Dict[str, float]] = {}
    for lab, src in sources:
        d = load_npz(src) if isinstance(src, str) else src
        datasets[lab] = d
        summaries[lab] = summary_row(d)
        n = int(d.get('obs').shape[0]) if 'obs' in d else -1
        sv = int(d['schema_version']) if 'schema_version' in d else -1
        print(f"[plot] loaded {lab}: n={n} samples, schema_v={sv}")
    labels = list(datasets.keys())

    # The four single-dataset panels (a,b,c,d,e) use the FIRST input.
    # The heatmap panel (f) uses every input.
    primary_label = labels[0]
    primary = datasets[primary_label]

    if figsize is None:
        # IEEE column-friendly: 2 cols * 3.4in, 3 rows * 2.6in
        figsize = (7.0, 8.4)
    fig, axes = plt.subplots(3, 2, figsize=figsize, squeeze=False)

    panel_action_histogram      (axes[0, 0], primary, primary_label)
    panel_lane_speed_marginals  (axes[0, 1], primary, primary_label)
    panel_outcome_bars          (axes[1, 0], primary, primary_label)
    panel_risk_change_violin    (axes[1, 1], primary, primary_label)
    panel_advantage_calibration (axes[2, 0], primary, primary_label,
                                 n_bins=n_calibration_bins)
    panel_dataset_heatmap       (axes[2, 1], summaries)

    fig.suptitle(
        f'Tactical Behaviour Summary — {primary_label}',
        fontsize=10, y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    print(f"[plot] wrote {out_path.with_suffix('.pdf')}")
    print(f"[plot] wrote {out_path.with_suffix('.png')}")

    # Also dump a tiny console summary so the user gets numbers without
    # opening the figure.
    _print_summary_table(summaries)


def _print_summary_table(summaries: Dict[str, Dict[str, float]]) -> None:
    if not summaries:
        return
    cols = [k for k, _ in HEATMAP_COLUMNS]
    header = f"{'dataset':<20s}" + ''.join(f"{c[:18]:>20s}" for c in cols)
    print('\n=== Behaviour summary ===')
    print(header)
    print('-' * len(header))
    for lab, row in summaries.items():
        line = f"{lab:<20s}"
        for k in cols:
            v = row.get(k, float('nan'))
            line += f"{('--' if not np.isfinite(v) else f'{v:.3f}'):>20s}"
        print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split('\n', 1)[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )

    src = p.add_argument_group(
        'data sources',
        'Provide pre-extracted .npz files via --inputs OR raw dataset\n'
        'directories via --from-dataset (or both — they are merged).',
    )
    src.add_argument(
        '--inputs', nargs='+', default=None,
        help='One or more BC .npz files produced by the extractor.',
    )
    src.add_argument(
        '--from-dataset', nargs='+', action='append', default=None,
        metavar='FORMAT [DATA_DIR [RECORDINGS]]',
        help=(
            'Live-extract from a dataset directory.  Repeatable.  Tokens:\n'
            '  FORMAT      (required) exiD/highD/inD/rounD/uniD/SQM-N-4/\n'
            '              YTDJ-3/XAM-N-5/XAM-N-6 (case-insensitive).\n'
            '  DATA_DIR    (optional, default "auto") path to the dir\n'
            '              holding *_tracks.csv.  "auto" -> data/<format>.\n'
            '  RECORDINGS  (optional, default "all") comma-separated ids\n'
            '              (e.g. 00,01,02) or "all".  Special datasets\n'
            '              (SQM-N-4/YTDJ-3/XAM-*) always use one\n'
            '              recording — this token is ignored for them.\n'
            'Examples:\n'
            '  --from-dataset SQM-N-4                # everything defaults\n'
            '  --from-dataset highD data/highD 01,02\n'
            '  --from-dataset exiD  auto         all'
        ),
    )

    p.add_argument(
        '--labels', nargs='+', default=None,
        help='Display labels.  Order = --inputs first, then --from-dataset.\n'
             '(default: .npz file stems / "<format>:<recordings>")',
    )
    p.add_argument(
        '--out', default='figures/behavior_summary',
        help='Output stem (.pdf and .png written next to it)',
    )
    p.add_argument(
        '--calibration-bins', type=int, default=10,
        help='Number of bins for the advantage calibration panel',
    )
    p.add_argument(
        '--ego-id', type=int, default=None,
        help='Restrict every panel to one ego_id (case-study mode).  When '
             'set with multiple sources, the same ego_id is filtered out '
             'of each.  Use --list-egos to see candidates.',
    )
    p.add_argument(
        '--list-egos', action='store_true',
        help='Print the 20 ego_ids with the most samples and exit.',
    )
    p.add_argument(
        '--dataset-label', default=None,
        help='Override the figure label.  Default: derived from the input '
             'filename via clean_label() — strips bc_/_smoke/_v* artefacts '
             'and maps to canonical names (highD, exiD, etc).',
    )

    extr = p.add_argument_group(
        'extractor knobs (only used by --from-dataset)',
    )
    extr.add_argument('--horizon-sec', type=float, default=1.5,
                      help='Action-label horizon (s)')
    extr.add_argument('--outcome-horizon-sec', type=float, default=3.0,
                      help='Outcome-label horizon (s)')
    extr.add_argument('--limit-tracks', type=int, default=None,
                      help='Cap ego candidates per recording (smoke runs)')
    extr.add_argument('--cache-npz-dir', default=None,
                      help='When set, every --from-dataset extraction is\n'
                           'also cached as <dir>/bc_<format>_<rids>.npz so\n'
                           'subsequent runs can use --inputs instead.')
    return p.parse_args()


def _build_sources(args: argparse.Namespace) -> List[Tuple[str, "str | dict"]]:
    """Merge --inputs and --from-dataset into a (label, source) list."""
    inputs: List[Tuple[str, "str | dict"]] = []

    if args.inputs:
        for path in args.inputs:
            inputs.append((clean_label(Path(path).stem), path))

    if args.from_dataset:
        for spec in args.from_dataset:
            if len(spec) == 0 or len(spec) > 3:
                raise SystemExit(
                    f"--from-dataset takes 1-3 tokens "
                    f"(FORMAT [DATA_DIR [RECORDINGS]]); got: {spec}"
                )
            fmt = spec[0]
            ddir = spec[1] if len(spec) >= 2 else 'auto'
            recs = spec[2] if len(spec) >= 3 else 'all'
            ddir_arg = None if ddir.lower() == 'auto' else ddir
            cache_path = None
            if args.cache_npz_dir:
                rid_tag = recs.replace(',', '-').replace(' ', '')
                rid_tag = (rid_tag[:24] + '+') if len(rid_tag) > 24 else rid_tag
                cache_path = os.path.join(
                    args.cache_npz_dir, f"bc_{fmt.lower()}_{rid_tag}.npz",
                )
            data = extract_live(
                dataset_format=fmt,
                data_dir=ddir_arg,
                recordings=recs,
                horizon_sec=args.horizon_sec,
                outcome_horizon_sec=args.outcome_horizon_sec,
                limit_tracks=args.limit_tracks,
                save_npz=cache_path,
            )
            disp = clean_label(fmt)
            inputs.append((disp, data))

    if not inputs:
        raise SystemExit(
            "No data sources provided.  Use --inputs <bc.npz>... and/or\n"
            "--from-dataset <FORMAT> <DATA_DIR> <RECORDINGS>."
        )

    # Apply user-supplied labels (in order).
    if args.labels:
        if len(args.labels) != len(inputs):
            raise SystemExit(
                f"--labels has {len(args.labels)} entries but {len(inputs)} "
                "data sources were resolved (--inputs first, --from-dataset "
                "second)."
            )
        inputs = [(lab, src) for lab, (_, src) in zip(args.labels, inputs)]

    return inputs


def main() -> None:
    args = _parse_args()
    sources = _build_sources(args)

    # Materialise sources to dicts (so --ego-id and --list-egos work
    # uniformly across path and live-extracted inputs).
    materialised: List[Tuple[str, dict]] = []
    for lab, src in sources:
        d = load_npz(src) if isinstance(src, str) else src
        materialised.append((lab, d))

    if args.list_egos:
        for lab, d in materialised:
            print(f"\n[ego list] {lab}")
            for tid, n in list_top_egos(d):
                print(f"   ego_id={tid:>10d}   n_samples={n}")
        return

    if args.ego_id is not None:
        filtered: List[Tuple[str, dict]] = []
        for lab, d in materialised:
            try:
                filtered.append((f"{lab} · ego {args.ego_id}",
                                 filter_by_ego_id(d, args.ego_id)))
            except ValueError as exc:
                print(f"[plot] skip {lab}: {exc}")
        materialised = filtered
        if not materialised:
            raise SystemExit(f"--ego-id {args.ego_id} matched no source")

    if args.dataset_label:
        # Override the first (primary) source's label.  Keep extra
        # sources as-is so the heatmap row labels stay distinct.
        materialised = [(args.dataset_label, materialised[0][1])] + materialised[1:]

    render_figure(
        sources=materialised,
        out_path=args.out,
        n_calibration_bins=args.calibration_bins,
    )


if __name__ == '__main__':
    main()
