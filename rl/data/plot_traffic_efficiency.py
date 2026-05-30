"""
Traffic-efficiency figure (Figure 3) — system-level externality view.
======================================================================

Companion to Figure 1 (`plot_behavior_summary`) and Figure 2
(`plot_social_externality`).  This figure visualises the **PDE-
propagated, frame-level metrics** added by schema v5:

* agent-normalised risk (`risk_mass_per_agent`,
  `risk_per_close_pair`, etc.);
* interaction-graph statistics (`close_pair_count`,
  `interaction_density`);
* temporal propagation (`risk_mass_growth_rate_frame`,
  `shockwave_onset_flag`);
* the composite `social_traffic_efficiency_index` (STEI).

Whereas Figure 2 measures *one ego's externality on its rear*, Figure 3
measures *the whole frame*: how risk is being created, transported and
dissipated as the scene evolves.

Layout (3 x 2)
--------------
(a) Risk-vs-density scatter
    ``risk_mass_per_agent`` vs ``interaction_density``, coloured by
    STEI.  Bottom-left = orderly; top-right = unstable.

(b) Backward-flux ratio histogram by lane action
    Distribution of ``backward_risk_flux_ratio`` grouped by ego's
    chosen lane action — quantifies how much risk the ego dumps
    behind itself when it changes lane vs. stays.

(c) Risk-adjusted progress vs frame agent count
    ``risk_adjusted_progress`` against ``num_agents_frame`` — under
    "good" highway driving the curve should rise sub-linearly with
    agent count, not collapse.

(d) STEI distribution
    Histogram of ``social_traffic_efficiency_index`` for the dataset.
    A long left tail signals frames where progress is bought with
    risk.

(e) Shockwave time-series (per recording)
    ``risk_mass_growth_rate_frame`` over time, with shockwave-onset
    frames marked.  When more than one recording is supplied, only
    the first is shown — use ``--ego-id`` for case-study slicing.

(f) Recording-level summary table
    Compact table of:
      mean ``num_agents_frame``,
      mean ``risk_mass_per_agent``,
      mean ``backward_risk_flux_ratio``,
      mean STEI,
      shockwave onset rate,
      total progress  /  total risk mass.

Usage
-----

    python -m rl.data.plot_traffic_efficiency \
        --inputs rl/checkpoints/bc_v5_smoke.npz \
        --out figures/traffic_efficiency_smoke

    # case study on one ego id
    python -m rl.data.plot_traffic_efficiency \
        --inputs rl/checkpoints/bc_v5_smoke.npz \
        --ego-id 27 --out figures/traffic_efficiency_ego27
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

from rl.data.plot_behavior_summary import (
    load_npz, filter_by_ego_id, list_top_egos, _resolve,
    LANE_DELTA_NAMES, _LANE_COLORS, clean_label,
)


_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']


def _use_style() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(['science', 'ieee', 'no-latex'])
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['savefig.bbox'] = 'standard'


def _has_v5(d: dict) -> bool:
    sv = int(d.get('schema_version', 0))
    return sv >= 5 or all(k in d for k in (
        'risk_mass_per_agent', 'social_traffic_efficiency_index',
    ))


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_risk_density(ax: plt.Axes, d: dict, label: str) -> None:
    """(a) Mean Risk per Vehicle vs Interaction Density, colour = SEI."""
    rmp = d.get('risk_mass_per_agent')
    idn = d.get('interaction_density')
    sei = d.get('safety_efficiency_index_sei')
    # Fall back to STEI if SEI absent (older v5 npz)
    if sei is None:
        sei = d.get('social_traffic_efficiency_index')
    if rmp is None or idn is None or sei is None:
        ax.set_visible(False); return
    rmp = np.asarray(rmp); idn = np.asarray(idn); sei = np.asarray(sei)
    finite = np.isfinite(rmp) & np.isfinite(idn) & np.isfinite(sei)
    rmp = rmp[finite]; idn = idn[finite]; sei = sei[finite]
    if rmp.size == 0:
        ax.set_visible(False); return

    rng = np.random.default_rng(0)
    if rmp.size > 5000:
        idx = rng.choice(rmp.size, size=5000, replace=False)
        rmp, idn, sei = rmp[idx], idn[idx], sei[idx]

    sc = ax.scatter(idn, rmp, c=sei, cmap='viridis',
                    s=4, alpha=0.5, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('SEI', fontsize=7); cbar.ax.tick_params(labelsize=6)
    ax.set_xlabel('Interaction Density (close pairs / $N_t$)')
    ax.set_ylabel('Mean Risk per Vehicle')
    ax.set_title('(a) Mean Risk per Vehicle vs Interaction Density')
    ax.grid(True, alpha=0.3, linewidth=0.5)


def panel_ttc_by_action(ax: plt.Axes, d: dict, label: str) -> None:
    """(b) Time-To-Collision distribution by ego lane action.

    Uses ``ttc_min_s`` (per-frame min TTC over closing pairs).  Critical
    threshold drawn at 1.5 s per Mahmud et al. 2017.
    """
    actions = _resolve(d, 'action_9way')
    ttc = d.get('ttc_min_s')
    if actions is None or ttc is None:
        ax.set_visible(False); return
    ttc = np.asarray(ttc)
    lane_bin = actions.astype(np.int64) // 3
    groups, names, colors = [], [], []
    for code, n in enumerate(LANE_DELTA_NAMES):
        m = (lane_bin == code) & np.isfinite(ttc)
        if m.sum() < 5: continue
        clipped = np.clip(ttc[m].astype(np.float64), 0.0, 30.0)  # tail clip for plotting
        groups.append(clipped)
        names.append(n); colors.append(_LANE_COLORS[n])
    if not groups: ax.set_visible(False); return

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
    ax.axhline(1.5, color='red', linestyle=':', linewidth=0.7,
               alpha=0.7, label='critical TTC = 1.5 s')
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(names)
    ax.set_ylabel('Min TTC over closing pairs (s)')
    ax.set_title('(b) Time-To-Collision (TTC) by Lane Action')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc='upper right')


def panel_progress_vs_density(ax: plt.Axes, d: dict, label: str) -> None:
    rap = d.get('risk_adjusted_progress')
    n_a = d.get('num_agents_frame')
    if rap is None or n_a is None:
        ax.set_visible(False); return
    rap = np.asarray(rap); n_a = np.asarray(n_a, dtype=np.int64)
    finite = np.isfinite(rap)
    rap = rap[finite]; n_a = n_a[finite]
    if rap.size == 0: ax.set_visible(False); return

    # Bin by agent count and show median + IQR
    uniq = np.unique(n_a)
    if uniq.size == 0:
        ax.set_visible(False); return
    medians, q1, q3 = [], [], []
    counts = []
    for v in uniq:
        m = (n_a == v)
        if m.sum() < 3:
            medians.append(np.nan); q1.append(np.nan); q3.append(np.nan)
        else:
            arr = rap[m]
            medians.append(np.median(arr))
            q1.append(np.percentile(arr, 25))
            q3.append(np.percentile(arr, 75))
        counts.append(int(m.sum()))
    medians = np.asarray(medians); q1 = np.asarray(q1); q3 = np.asarray(q3)
    valid = np.isfinite(medians)
    ax.fill_between(uniq[valid], q1[valid], q3[valid],
                    color=_COLORS[0], alpha=0.18, linewidth=0,
                    label='IQR')
    ax.plot(uniq[valid], medians[valid], color=_COLORS[0],
            marker='o', linewidth=1.2, markersize=3,
            label='median')
    ax.set_xlabel('Agents in Frame  $N_t$')
    ax.set_ylabel('Risk-adjusted Progress')
    ax.set_title('(c) Risk-Adjusted Progress vs Density (Andreotti EI)')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc='best')


def panel_drac_hist(ax: plt.Axes, d: dict, label: str) -> None:
    """(d) DRAC (Deceleration Rate to Avoid Collision) distribution.

    Per Cooper & Ferguson 1976 / Jiang et al. 2023, DRAC > 4 m/s² is
    the standard ``critical event'' threshold.
    """
    drac = d.get('max_drac_mps2')
    if drac is None: ax.set_visible(False); return
    drac = np.asarray(drac, dtype=np.float64)
    drac = drac[np.isfinite(drac) & (drac >= 0)]
    if drac.size == 0: ax.set_visible(False); return
    lo, hi = np.percentile(drac, [0.5, 99.5])
    hi = max(hi, 5.0)                          # always show the threshold
    ax.hist(drac, bins=40, range=(lo, hi),
            color=_COLORS[1], edgecolor='black', linewidth=0.3, alpha=0.85)
    median = float(np.median(drac))
    ax.axvline(median, color='black', ls='--', lw=0.8,
               label=f'median {median:.2f} m/s²')
    ax.axvline(4.0, color='red', ls=':', lw=0.8,
               label='critical = 4 m/s²')
    ax.set_xlabel('Max DRAC over closing pairs (m/s²)')
    ax.set_ylabel('frame count')
    ax.set_title('(d) Required Follower Deceleration (DRAC)')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc='upper right')


def panel_shockwave_timeline(ax: plt.Axes, d: dict, label: str) -> None:
    growth = d.get('risk_mass_growth_rate_frame')
    sk = d.get('shockwave_onset_flag')
    t = d.get('t_sec')
    if t is None:
        t = d.get('t_rel')
    rec = d.get('recording_ids')
    if growth is None or t is None:
        ax.set_visible(False); return
    growth = np.asarray(growth); t = np.asarray(t)
    rec = np.asarray(rec, dtype=np.int64) if rec is not None else None
    sk = np.asarray(sk, dtype=np.int8) if sk is not None else None

    # Pick the first recording for time-domain clarity
    if rec is not None:
        first = int(np.unique(rec)[0])
        m = rec == first
        growth = growth[m]; t = t[m]
        if sk is not None: sk = sk[m]
        label = f"{label} rec={first}"

    if growth.size == 0:
        ax.set_visible(False); return
    order = np.argsort(t)
    t = t[order]; growth = growth[order]
    if sk is not None: sk = sk[order]

    ax.plot(t, growth, color=_COLORS[1], linewidth=0.6, alpha=0.7,
            label='dR/dt')
    if sk is not None and sk.any():
        ax.scatter(t[sk == 1], growth[sk == 1], s=10, marker='x',
                   color='black', linewidths=0.6, label='shockwave onset')
    ax.axhline(0.0, color='grey', ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'$\partial R_{\mathrm{total}}/\partial t$')
    ax.set_title('(e) Total Risk Potential Growth (shockwave detection)')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc='upper right')


def panel_summary_table(ax: plt.Axes, d: dict, label: str) -> None:
    """Compact text-table summary instead of a chart."""
    ax.set_axis_off()

    def _mean(key: str) -> float:
        a = d.get(key)
        if a is None: return float('nan')
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float('nan')

    def _rate(key: str) -> float:
        a = d.get(key)
        if a is None: return float('nan')
        a = np.asarray(a)
        return float(a.mean())

    def _sum(key: str) -> float:
        a = d.get(key)
        if a is None: return float('nan')
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        return float(a.sum()) if a.size else float('nan')

    rows = [
        ('Avg agents per frame',       f"{_mean('num_agents_frame'):.2f}"),
        ('Avg speed (m/s)',            f"{_mean('mean_speed_frame'):.2f}"),
        ('Min TTC (s, mean)',          f"{_mean('ttc_min_s'):.2f}"),
        ('Frac TTC < 1.5 s',           f"{100*_mean('frac_critical_ttc'):.2f} %"),
        ('Mean THW (s)',               f"{_mean('mean_thw_s'):.2f}"),
        ('Frac THW < 1.0 s',           f"{100*_mean('frac_tailgate_thw'):.2f} %"),
        ('Max DRAC (m/s², mean)',      f"{_mean('max_drac_mps2'):.2f}"),
        ('Frac DRAC > 4 m/s²',         f"{100*_mean('frac_critical_drac'):.2f} %"),
        ('Efficiency Index (EI)',      f"{_mean('efficiency_index_ei'):.3f}"),
        ('Safety–Efficiency (SEI)',    f"{_mean('safety_efficiency_index_sei'):.3f}"),
        ('Mean Risk per Vehicle',      f"{_mean('risk_mass_per_agent'):.3f}"),
        ('Backward Flux Ratio',        f"{_mean('backward_risk_flux_ratio'):.3f}"),
        ('Shockwave onset rate',       f"{100*_rate('shockwave_onset_flag'):.2f} %"),
    ]
    cell_text = [[name, value] for name, value in rows]
    table = ax.table(cellText=cell_text, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='left', colLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.15)
    ax.set_title('(f) Recording-level Metrics', fontsize=8)


# ---------------------------------------------------------------------------
# Driver + CLI
# ---------------------------------------------------------------------------

def render_figure(
    sources: List[Tuple[str, "str | dict"]],
    out_path: str,
) -> None:
    if not sources:
        raise ValueError("Need at least one source")
    _use_style()

    primary_label, primary_src = sources[0]
    primary = load_npz(primary_src) if isinstance(primary_src, str) else primary_src
    n = int(primary.get('obs').shape[0]) if 'obs' in primary else 0
    sv = int(primary.get('schema_version', 0))
    print(f"[plot] primary {primary_label}: n={n} samples, schema_v={sv}")
    if not _has_v5(primary):
        raise SystemExit(
            f"[plot] {primary_label} is not schema v5 — re-extract with "
            "`python -m rl.data.historical_extractor ... --include-social`"
        )

    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 9.6), squeeze=False)
    panel_risk_density       (axes[0, 0], primary, primary_label)
    panel_ttc_by_action      (axes[0, 1], primary, primary_label)
    panel_progress_vs_density(axes[1, 0], primary, primary_label)
    panel_drac_hist          (axes[1, 1], primary, primary_label)
    panel_shockwave_timeline (axes[2, 0], primary, primary_label)
    panel_summary_table      (axes[2, 1], primary, primary_label)

    fig.suptitle(f'Traffic-Efficiency Summary — {primary_label}',
                 fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.945, bottom=0.07,
                        hspace=0.55, wspace=0.32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix('.pdf'))
    fig.savefig(out_path.with_suffix('.png'), dpi=300)
    print(f"[plot] wrote {out_path.with_suffix('.pdf')}")
    print(f"[plot] wrote {out_path.with_suffix('.png')}")

    stei = np.asarray(primary.get('social_traffic_efficiency_index', []),
                      dtype=np.float64)
    stei = stei[np.isfinite(stei)]
    if stei.size:
        print(f"[plot] STEI:  mean={stei.mean():+.3f}  median={np.median(stei):+.3f}  "
              f"std={stei.std():.3f}  q05={np.percentile(stei, 5):+.3f}  "
              f"q95={np.percentile(stei, 95):+.3f}")
    sk = np.asarray(primary.get('shockwave_onset_flag', []), dtype=np.int64)
    if sk.size:
        print(f"[plot] shockwave_onset rate: {100 * sk.mean():.2f}%  "
              f"({int(sk.sum())} / {sk.size} frames)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split('\n', 1)[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    src = p.add_argument_group('data sources')
    src.add_argument('--inputs', nargs='+', default=None)
    src.add_argument('--from-dataset', nargs='+', action='append', default=None,
                     metavar='FORMAT [DATA_DIR [RECORDINGS]]',
                     help='Live-extract via historical_extractor --include-social')
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--out', default='figures/traffic_efficiency')
    p.add_argument('--ego-id', type=int, default=None)
    p.add_argument('--list-egos', action='store_true')
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
        raise SystemExit("No data sources — use --inputs or --from-dataset")

    if args.labels:
        if len(args.labels) != len(inputs):
            raise SystemExit("--labels count must match data sources")
        inputs = [(lab, src) for lab, (_, src) in zip(args.labels, inputs)]

    materialised = [
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
