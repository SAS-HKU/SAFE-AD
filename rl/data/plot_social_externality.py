"""
Social-externality figure (Figure 2) for schema-v4 BC datasets.
================================================================

Companion to :mod:`rl.data.plot_behavior_summary`.  Where Figure 1
visualises the *ego's tactical choice* (action distribution, lane
utility, outcome rates), this figure visualises **what the ego's
maneuver did to the surrounding traffic** — the courtesy / cooperation
/ field-externality features added in schema v4 by
:mod:`rl.data.social_features`.

Layout (3 x 2 grid)
-------------------
(a) Imposed-braking distribution by lane action
    Violin/box of ``rear_decel_peak_3s`` grouped by keep / right / left.

(b) Cut-in burden bars
    Rates of ``hard_brake_imposed_flag``, ``bad_cut_in_flag``,
    ``rear_ttc_delta < 0`` (TTC loss),
    ``rear_thw_delta < 0`` (THW loss).

(c) Decision-quality bars
    ``missed_opportunity_flag``, ``bad_lane_change_flag``,
    ``escape_success_flag`` (v3, kept for context),
    ``lane_change_advantage_flag`` (v3).

(d) Risk-field externality scatter
    ``risk_mass_others`` vs ``risk_gradient_peak``, coloured by
    ``social_class``.  Marker shows whether the maneuver caused
    ``hard_brake_imposed_flag``.

(e) Social-class breakdown
    Stacked bar of the five-class label
    (good / defensive / aggressive / passive / harmful).

(f) Social-benefit Pareto
    x = ``progress_score``, y = ``courtesy_score``,
    coloured by ``safety_score``.  Best human samples sit upper-right.

Usage
-----
Single dataset, average across all egos::

    python -m rl.data.plot_social_externality \
        --inputs rl/checkpoints/bc_v4_smoke.npz \
        --out figures/social_externality

Per-agent case study (one ego_id only)::

    python -m rl.data.plot_social_externality \
        --inputs rl/checkpoints/bc_v4_smoke.npz \
        --ego-id 12 \
        --out figures/social_externality_ego12

Live extraction (auto-includes social features)::

    python -m rl.data.plot_social_externality \
        --from-dataset highD data/highD 01,02 \
        --out figures/social_externality_highd
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False

# Reuse the source-resolution and ego-filter machinery from Figure 1.
from rl.data.plot_behavior_summary import (
    load_npz, filter_by_ego_id, list_top_egos, extract_live, _build_sources,
    _resolve, LANE_DELTA_NAMES, _LANE_COLORS, clean_label,
)
from rl.data.social_features import SOCIAL_CLASS_NAMES


_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
_CLASS_COLORS = {
    0: '#1f9e6e',   # good — green
    1: '#0072B2',   # defensive — blue
    2: '#D55E00',   # aggressive — orange-red
    3: '#9b9b9b',   # passive — grey
    4: '#a30013',   # harmful — dark red
}


def _use_style() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(['science', 'ieee', 'no-latex'])
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
    # SciencePlots' "ieee" style sets:
    #   * figure.autolayout = True (sometimes) — fights subplots_adjust;
    #   * savefig.bbox = 'tight' — expands the saved canvas if any
    #     artist sits just outside the axes (annotations on small bars,
    #     colorbar labels, etc.), which produces the duplicated-title
    #     artefact we saw on per-agent renders.
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['savefig.bbox'] = 'standard'


def _check_v4(d: dict) -> bool:
    """Return True iff this dict carries the v4 social keys we need."""
    sv = int(d.get('schema_version', 0))
    return sv >= 4 or all(k in d for k in (
        'social_friendliness_score', 'rear_decel_peak_3s', 'social_class',
    ))


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_imposed_braking(ax: plt.Axes, d: dict, label: str) -> None:
    actions = _resolve(d, 'action_9way')
    decel   = d.get('rear_decel_peak_3s')
    if actions is None or decel is None or actions.size == 0:
        ax.set_visible(False); return
    decel = np.asarray(decel)
    lane_bin = actions.astype(np.int64) // 3
    groups, names, colors = [], [], []
    for code, n in enumerate(LANE_DELTA_NAMES):
        m = (lane_bin == code) & np.isfinite(decel)
        if m.sum() < 5:
            continue
        groups.append(decel[m].astype(np.float64))
        names.append(n)
        colors.append(_LANE_COLORS[n])
    if not groups:
        ax.set_visible(False); return
    parts = ax.violinplot(groups, positions=np.arange(len(groups)),
                          showmedians=True, widths=0.75)
    for body, c in zip(parts['bodies'], colors):
        body.set_facecolor(c); body.set_edgecolor('black'); body.set_alpha(0.6)
        body.set_linewidth(0.4)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(0.8)
    for k in ('cbars', 'cmins', 'cmaxes'):
        if k in parts:
            parts[k].set_color('black'); parts[k].set_linewidth(0.5)
    ax.axhline(0.0, color='grey', ls='--', lw=0.6, alpha=0.7)
    ax.axhline(-3.0, color='red', ls=':', lw=0.6, alpha=0.5,
               label='hard-brake threshold')
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(names)
    ax.set_ylabel(r'rear $a_{lon,\min}$ over 3 s [m/s$^2$]')
    ax.set_title('(a) Follower Peak Deceleration (DRAC, 3 s) by Lane Action')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc='lower right')
    all_v = np.concatenate(groups)
    if np.isfinite(all_v).any():
        q_lo, q_hi = np.percentile(all_v[np.isfinite(all_v)], [1.0, 99.0])
        span = max(abs(q_lo), abs(q_hi), 1.0)
        ax.set_ylim(-1.1 * span, 1.1 * span)


def panel_cutin_burden(ax: plt.Axes, d: dict, label: str) -> None:
    n = int(_resolve(d, 'action_9way').size if _resolve(d, 'action_9way') is not None else 0)
    if n == 0:
        ax.set_visible(False); return

    def frac(key: str, denom_mask: np.ndarray | None = None) -> float:
        a = d.get(key)
        if a is None: return float('nan')
        a = np.asarray(a)
        if denom_mask is None:
            return float(a.mean())
        if not denom_mask.any(): return float('nan')
        return float(a[denom_mask].mean())

    rear_ttc_delta = np.asarray(d.get('rear_ttc_delta', np.full(n, np.nan)))
    rear_thw_delta = np.asarray(d.get('rear_thw_delta', np.full(n, np.nan)))
    ttc_loss_flag = np.where(np.isfinite(rear_ttc_delta), rear_ttc_delta < 0, 0).astype(np.int8)
    thw_loss_flag = np.where(np.isfinite(rear_thw_delta), rear_thw_delta < 0, 0).astype(np.int8)

    bars = [
        ('hard brake imposed', frac('hard_brake_imposed_flag')),
        ('bad cut-in',         frac('bad_cut_in_flag')),
        ('rear TTC↓',          float(ttc_loss_flag.mean())),
        ('rear THW↓',          float(thw_loss_flag.mean())),
    ]
    names = [b[0] for b in bars]
    vals = np.asarray([b[1] for b in bars], dtype=np.float64)

    xs = np.arange(len(bars))
    bs = ax.bar(xs, vals * 100.0, color=_COLORS[1],
                edgecolor='black', linewidth=0.4)
    for b, v in zip(bs, vals):
        if not np.isfinite(v): continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f'{v*100:.1f}%', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7)
    ax.set_ylabel('rate (%)')
    ax.set_title('(b) Cut-In Burden Rates')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)


def panel_decision_quality(ax: plt.Axes, d: dict, label: str) -> None:
    actions = _resolve(d, 'action_9way')
    if actions is None: ax.set_visible(False); return
    n = int(actions.size)
    lc_mask = (actions.astype(np.int64) // 3) != 0

    def frac(key: str, denom_mask: np.ndarray | None = None) -> float:
        a = d.get(key)
        if a is None: return float('nan')
        a = np.asarray(a)
        if a.size != n: return float('nan')
        if denom_mask is None: return float(a.mean())
        if not denom_mask.any(): return float('nan')
        return float(a[denom_mask].mean())

    bars = [
        ('missed opportunity',  frac('missed_opportunity_flag'),  '#9b9b9b'),
        ('bad LC',              frac('bad_lane_change_flag'),     '#a30013'),
        ('LC adv | LC',         frac('lane_change_advantage_flag', lc_mask), '#1f9e6e'),
        ('escape | blocked',    frac('escape_success_flag',
                                     (np.asarray(d.get('blocked_by_leader_flag', np.zeros(n))) == 1)),
                                                                  '#0072B2'),
    ]
    names = [b[0] for b in bars]
    vals = np.asarray([b[1] for b in bars], dtype=np.float64)
    xs = np.arange(len(bars))
    bs = ax.bar(xs, vals * 100.0, color=[b[2] for b in bars],
                edgecolor='black', linewidth=0.4)
    for b, v in zip(bs, vals):
        if not np.isfinite(v): continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f'{v*100:.1f}%', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7)
    ax.set_ylabel('rate (%)')
    ax.set_title('(c) Decision-Quality Flags')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)


def panel_field_scatter(ax: plt.Axes, d: dict, label: str) -> None:
    rmo = d.get('risk_mass_others')
    rgp = d.get('risk_gradient_peak')
    sc  = d.get('social_class')
    hb  = d.get('hard_brake_imposed_flag')
    if rmo is None or rgp is None or sc is None:
        ax.set_visible(False); return
    rmo = np.asarray(rmo); rgp = np.asarray(rgp); sc = np.asarray(sc)
    n = rmo.size
    if n == 0: ax.set_visible(False); return
    finite = np.isfinite(rmo) & np.isfinite(rgp)
    rmo = rmo[finite]; rgp = rgp[finite]; sc = sc[finite]
    hb = (np.asarray(hb)[finite] if hb is not None else np.zeros(rmo.size, dtype=np.int8))
    # Subsample if huge
    rng = np.random.default_rng(0)
    if rmo.size > 5000:
        idx = rng.choice(rmo.size, size=5000, replace=False)
        rmo, rgp, sc, hb = rmo[idx], rgp[idx], sc[idx], hb[idx]

    for cls in (0, 1, 2, 3, 4):
        m = (sc == cls)
        if not m.any(): continue
        ax.scatter(rmo[m], rgp[m], s=4, alpha=0.4,
                   color=_CLASS_COLORS[cls], label=SOCIAL_CLASS_NAMES[cls],
                   edgecolors='none')
    if hb.any():
        ax.scatter(rmo[hb == 1], rgp[hb == 1], s=12, marker='x',
                   color='black', alpha=0.6, label='hard-brake imposed',
                   linewidths=0.5)

    ax.set_xlabel('risk_mass_others')
    ax.set_ylabel(r'risk_gradient_peak  $\max\|\nabla R\|$')
    ax.set_title('(d) Imposed Risk Potential vs Risk Gradient Peak')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=5, loc='upper right',
              markerscale=1.5, ncol=2)


def panel_social_class_bar(ax: plt.Axes, d: dict, label: str) -> None:
    sc = d.get('social_class')
    if sc is None: ax.set_visible(False); return
    sc = np.asarray(sc, dtype=np.int64)
    n = sc.size
    if n == 0: ax.set_visible(False); return

    counts = np.bincount(sc, minlength=5)[:5]
    frac = counts / max(1, counts.sum())

    xs = np.arange(5)
    colors = [_CLASS_COLORS[k] for k in range(5)]
    bs = ax.bar(xs, frac * 100.0, color=colors,
                edgecolor='black', linewidth=0.4)
    for b, f in zip(bs, frac):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f'{f*100:.1f}%', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace('social_', '') for n in SOCIAL_CLASS_NAMES],
                       rotation=20, ha='right', fontsize=7)
    ax.set_ylabel('share (%)')
    ax.set_title('(e) 5-Class Breakdown (experimental)')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)


def panel_pareto(ax: plt.Axes, d: dict, label: str) -> None:
    p = d.get('progress_score'); c = d.get('courtesy_score')
    s = d.get('safety_score')
    if p is None or c is None or s is None:
        ax.set_visible(False); return
    p = np.asarray(p); c = np.asarray(c); s = np.asarray(s)
    finite = np.isfinite(p) & np.isfinite(c) & np.isfinite(s)
    p = p[finite]; c = c[finite]; s = s[finite]
    if p.size == 0: ax.set_visible(False); return
    rng = np.random.default_rng(0)
    if p.size > 5000:
        idx = rng.choice(p.size, size=5000, replace=False)
        p, c, s = p[idx], c[idx], s[idx]

    sc = ax.scatter(p, c, c=s, cmap='viridis', vmin=0.0, vmax=1.0,
                    s=4, alpha=0.5, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('safety_score', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    # Mark the upper-right "good" region
    ax.axvline(0.5, color='grey', ls='--', lw=0.6, alpha=0.4)
    ax.axhline(0.5, color='grey', ls='--', lw=0.6, alpha=0.4)
    ax.set_xlabel('progress_score'); ax.set_ylabel('courtesy_score')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title('(f) Progress × Courtesy Pareto (colour = safety)')
    ax.grid(True, alpha=0.3, linewidth=0.5)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_figure(
    sources: List[Tuple[str, "str | dict"]],
    out_path: str,
) -> None:
    if not sources:
        raise ValueError("Need at least one source (--inputs or --from-dataset)")

    _use_style()
    primary_label, primary_src = sources[0]
    primary = load_npz(primary_src) if isinstance(primary_src, str) else primary_src
    n = int(primary.get('obs').shape[0]) if 'obs' in primary else 0
    sv = int(primary.get('schema_version', 0))
    print(f"[plot] primary {primary_label}: n={n} samples, schema_v={sv}")

    if not _check_v4(primary):
        raise SystemExit(
            f"[plot] {primary_label} is not schema v4 — re-extract with "
            "`python -m rl.data.historical_extractor ... --include-social`"
        )

    plt.close('all')                # avoid stale state from previous calls
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 9.6), squeeze=False)

    panel_imposed_braking   (axes[0, 0], primary, primary_label)
    panel_cutin_burden      (axes[0, 1], primary, primary_label)
    panel_decision_quality  (axes[1, 0], primary, primary_label)
    panel_field_scatter     (axes[1, 1], primary, primary_label)
    panel_social_class_bar  (axes[2, 0], primary, primary_label)
    panel_pareto            (axes[2, 1], primary, primary_label)

    fig.suptitle(f'Social Externality Summary — {primary_label}',
                 fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.945, bottom=0.07,
                        hspace=0.55, wspace=0.32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # SciencePlots sets savefig.bbox='tight' globally which expands the
    # canvas when scattered points/legends sit just outside the axes
    # box.  Force standard bbox so the PNG dims match figsize exactly.
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches=None)
    fig.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches=None)
    print(f"[plot] wrote {out_path.with_suffix('.pdf')}")
    print(f"[plot] wrote {out_path.with_suffix('.png')}")

    # Console summary
    if 'social_friendliness_score' in primary:
        s = np.asarray(primary['social_friendliness_score'])
        s = s[np.isfinite(s)]
        if s.size:
            print(f"[plot] social_friendliness_score  mean={float(s.mean()):+.3f}  "
                  f"median={float(np.median(s)):+.3f}  std={float(s.std()):.3f}")
    sc = np.asarray(primary.get('social_class', []), dtype=np.int64)
    if sc.size:
        cnt = np.bincount(sc, minlength=5)[:5]
        for k, name in enumerate(SOCIAL_CLASS_NAMES):
            print(f"[plot]   {name:<20s} {cnt[k]:>7d}  ({100*cnt[k]/sc.size:5.2f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split('\n', 1)[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    src = p.add_argument_group('data sources')
    src.add_argument('--inputs', nargs='+', default=None,
                     help='One or more BC .npz files (schema v4 required).')
    src.add_argument('--from-dataset', nargs='+', action='append', default=None,
                     metavar='FORMAT [DATA_DIR [RECORDINGS]]',
                     help='Live-extract via historical_extractor --include-social.')
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--out', default='figures/social_externality')
    p.add_argument('--ego-id', type=int, default=None,
                   help='Restrict every panel to one ego_id (case study).')
    p.add_argument('--list-egos', action='store_true',
                   help='Print the 20 ego_ids with the most samples and exit.')
    p.add_argument('--dataset-label', default=None,
                   help='Override the figure label.  Default: derived from '
                        'the input filename via clean_label().')
    extr = p.add_argument_group('extractor knobs (only with --from-dataset)')
    extr.add_argument('--horizon-sec', type=float, default=1.5)
    extr.add_argument('--outcome-horizon-sec', type=float, default=3.0)
    extr.add_argument('--limit-tracks', type=int, default=None)
    extr.add_argument('--cache-npz-dir', default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.include_social = True   # implied by being in this script

    # Build sources via the same helper Figure 1 uses, but force
    # `--include-social` on every live extraction.
    inputs: List[Tuple[str, "str | dict"]] = []
    if args.inputs:
        for path in args.inputs:
            inputs.append((clean_label(Path(path).stem), path))
    if args.from_dataset:
        from rl.data.historical_extractor import (
            _canonical_dataset_name, _default_data_dir, _discover_recordings,
            extract_many,
        )
        for spec in args.from_dataset:
            if not spec or len(spec) > 3:
                raise SystemExit(f"--from-dataset takes 1-3 tokens; got {spec}")
            fmt = spec[0]
            ddir = spec[1] if len(spec) >= 2 else 'auto'
            recs = spec[2] if len(spec) >= 3 else 'all'
            canonical = _canonical_dataset_name(fmt)
            if ddir.lower() == 'auto':
                ddir = _default_data_dir(canonical)
            if recs.lower() == 'all':
                rids = _discover_recordings(ddir, canonical)
            else:
                rids = [r.strip() for r in recs.split(',') if r.strip()]
            data = extract_many(
                ddir, rids,
                horizon_sec=args.horizon_sec,
                outcome_horizon_sec=args.outcome_horizon_sec,
                limit_tracks=args.limit_tracks,
                dataset_format=canonical,
                include_social=True,
            )
            if args.cache_npz_dir:
                rid_tag = (recs.replace(',', '-')[:24] + '+'
                           if len(recs) > 24 else recs.replace(',', '-'))
                cache_path = os.path.join(
                    args.cache_npz_dir, f"bc_{fmt.lower()}_{rid_tag}.npz",
                )
                np.savez_compressed(cache_path, **data)
                print(f"[plot] cached extraction → {cache_path}")
            inputs.append((clean_label(fmt), data))

    if not inputs:
        raise SystemExit(
            "No data sources — use --inputs <bc.npz> or --from-dataset FORMAT ...")

    # Apply user labels in order (inputs first, then from_dataset).
    if args.labels:
        if len(args.labels) != len(inputs):
            raise SystemExit("--labels count must match data sources")
        inputs = [(lab, src) for lab, (_, src) in zip(args.labels, inputs)]

    # Materialise to dicts.
    materialised: List[Tuple[str, dict]] = [
        (lab, load_npz(src) if isinstance(src, str) else src)
        for lab, src in inputs
    ]

    if args.list_egos:
        for lab, d in materialised:
            print(f"\n[ego list] {lab}")
            for tid, cnt in list_top_egos(d):
                print(f"   ego_id={tid:>10d}   n_samples={cnt}")
        return

    if args.ego_id is not None:
        filtered = []
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
        materialised = [(args.dataset_label, materialised[0][1])] + materialised[1:]

    render_figure(materialised, args.out)


if __name__ == '__main__':
    main()
