"""
pinn_compare_fields.py
======================
Side-by-side visualisation of the hand-crafted numerical PDE risk field
versus the PINN-learned surrogate, driven by real exiD trajectory data.

Produces:
  pinn_compare_<dataset>_<rec>/
    overview_t<T>.png/.pdf   – full-field comparison at selected timesteps
    profile_longitudinal.png – 1-D risk profile along road centre-line
    scatter.png              – scatter plot: PINN vs numerical R (all test pts)
    source_fields_t<T>.png   – Q / vx / vy / D panels (what drives the PDE)

Usage
-----
  python pinn_compare_fields.py --dataset exiD --recording 00
  python pinn_compare_fields.py --dataset inD  --recording 03 --model pinn_inD_03.pt
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scienceplots          # registers 'science', 'bright', etc.

DREAM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DREAM_ROOT)

# Import shared PINN infrastructure
import torch
from pinn_risk_field import (
    ExiDLoader, Normalizer, FieldInterpolator, FlatSampleCache,
    RiskFieldNet, PINNTrainer,
    load_all_recordings, parse_recording_ids,
    KNOWN_DATASETS, _PERCEPTION_RANGE_DEFAULT,
)
from config import Config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extent(loader):
    """Return imshow extent [x0, x1, y0, y1] and axis unit label."""
    xg, yg = loader.x_grid, loader.y_grid
    return [xg[0], xg[-1], yg[0], yg[-1]], 'm'


def _vmax(*arrays, pct=99):
    """Robust colour-scale ceiling: p-th percentile of all non-zero values."""
    all_vals = np.concatenate([a.ravel() for a in arrays])
    pos = all_vals[all_vals > 0]
    return float(np.percentile(pos, pct)) if len(pos) > 0 else 1.0


def _sci(styles=('science', 'bright')):
    return plt.style.context(list(styles))


def _cbar(fig, ax, im, label, size='4%', pad=0.04):
    div = make_axes_locatable(ax)
    cb  = plt.colorbar(im, cax=div.append_axes('right', size=size, pad=pad))
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return cb


# ---------------------------------------------------------------------------
# Plot 1 — Full-field overview at one timestep
# ---------------------------------------------------------------------------

def plot_overview(snap, R_pinn, loader, snap_idx, save_dir, label=''):
    """
    4-panel figure:
      [Numerical PDE]  [PINN Surrogate]  [Difference]  [Source Q]
    """
    R_num  = snap['R']
    Q      = snap['Q']
    diff   = R_pinn - R_num
    ext, unit = _extent(loader)
    vm = _vmax(R_num, R_pinn)
    qm = _vmax(Q)

    with _sci():
        fig = plt.figure(figsize=(14, 3.4))
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)

        # ── Panel A: Numerical PDE ─────────────────────────────────────
        ax0  = fig.add_subplot(gs[0])
        im0  = ax0.imshow(R_num, origin='lower', aspect='auto',
                          extent=ext, vmin=0, vmax=vm, cmap='inferno')
        ax0.set_title(r'(a) Numerical PDE  $\mathcal{R}_\mathrm{num}$', fontsize=8)
        ax0.set_xlabel(f'$x$ [{unit}]', fontsize=7)
        ax0.set_ylabel(f'$y$ [{unit}]', fontsize=7)
        ax0.tick_params(labelsize=6)
        _cbar(fig, ax0, im0, r'$\mathcal{R}$')

        # ── Panel B: PINN ──────────────────────────────────────────────
        ax1  = fig.add_subplot(gs[1])
        im1  = ax1.imshow(R_pinn, origin='lower', aspect='auto',
                          extent=ext, vmin=0, vmax=vm, cmap='inferno')
        ax1.set_title(r'(b) PINN Surrogate  $\hat{\mathcal{R}}_\theta$', fontsize=8)
        ax1.set_xlabel(f'$x$ [{unit}]', fontsize=7)
        ax1.set_ylabel(f'$y$ [{unit}]', fontsize=7)
        ax1.tick_params(labelsize=6)
        _cbar(fig, ax1, im1, r'$\hat{\mathcal{R}}$')

        # ── Panel C: Signed difference ─────────────────────────────────
        ax2   = fig.add_subplot(gs[2])
        dlim  = float(np.abs(diff).max()) or 0.1
        im2   = ax2.imshow(diff, origin='lower', aspect='auto',
                           extent=ext, vmin=-dlim, vmax=dlim, cmap='RdBu_r')
        ax2.set_title(r'(c) Difference  $\hat{\mathcal{R}}-\mathcal{R}$', fontsize=8)
        ax2.set_xlabel(f'$x$ [{unit}]', fontsize=7)
        ax2.set_ylabel(f'$y$ [{unit}]', fontsize=7)
        ax2.tick_params(labelsize=6)
        cb2 = _cbar(fig, ax2, im2, r'$\Delta\mathcal{R}$')
        # Add zero contour for clarity
        ny, nx = R_num.shape
        x_lin = np.linspace(ext[0], ext[1], nx)
        y_lin = np.linspace(ext[2], ext[3], ny)
        ax2.contour(x_lin, y_lin, diff, levels=[0], colors='k',
                    linewidths=0.4, linestyles='--')

        # ── Panel D: Source term Q ──────────────────────────────────────
        ax3  = fig.add_subplot(gs[3])
        im3  = ax3.imshow(Q, origin='lower', aspect='auto',
                          extent=ext, vmin=0, vmax=qm, cmap='YlOrRd')
        ax3.set_title(r'(d) Source $Q(x,t)$ (vehicle risk)', fontsize=8)
        ax3.set_xlabel(f'$x$ [{unit}]', fontsize=7)
        ax3.set_ylabel(f'$y$ [{unit}]', fontsize=7)
        ax3.tick_params(labelsize=6)
        _cbar(fig, ax3, im3, r'$Q$')

        t_str = f't = {snap["t"]:.1f}\\,\\mathrm{{s}}'
        corr  = float(np.corrcoef(R_pinn.ravel(), R_num.ravel())[0, 1])
        l2r   = float(np.linalg.norm(diff) / (np.linalg.norm(R_num) + 1e-8))
        fig.suptitle(
            rf'Numerical PDE vs.\ PINN Surrogate — {label} — ${t_str}$'
            rf'   $\rho={corr:.3f}$,  $L_2^\mathrm{{rel}}={l2r:.3f}$',
            fontsize=9, y=1.01)

        stem = f"overview_t{snap_idx:03d}"
        for ext_fmt in ('pdf', 'png'):
            plt.savefig(os.path.join(save_dir, f"{stem}.{ext_fmt}"),
                        dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"  [overview] snap {snap_idx}  t={snap['t']:.1f}s  "
          f"ρ={corr:.3f}  L2_rel={l2r:.3f}")


# ---------------------------------------------------------------------------
# Plot 2 — 1-D longitudinal risk profile along road centre-line
# ---------------------------------------------------------------------------

def plot_longitudinal_profiles(test_snaps, R_pinn_list, loader, save_dir):
    """
    Average risk along the y-axis at each x position, comparing numerical
    vs PINN at multiple timesteps on one plot.
    """
    xg = loader.x_grid
    # Colour cycle: one colour per timestep
    n  = min(len(test_snaps), 6)
    times = [test_snaps[i]['t'] for i in range(n)]

    with _sci():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)

        for i in range(n):
            snap  = test_snaps[i]
            R_num = snap['R']
            R_pin = R_pinn_list[i]
            # Mean along y-axis → 1-D profile along x
            profile_num  = R_num.mean(axis=0)
            profile_pinn = R_pin.mean(axis=0)
            c = f'C{i}'

            axes[0].plot(xg, profile_num,  color=c, lw=1.3,
                         label=f'$t={times[i]:.1f}$s')
            axes[1].plot(xg, profile_pinn, color=c, lw=1.3,
                         label=f'$t={times[i]:.1f}$s')

        for ax, title in zip(axes,
                             [r'Numerical PDE  $\langle\mathcal{R}\rangle_y$',
                              r'PINN Surrogate  $\langle\hat{\mathcal{R}}\rangle_y$']):
            ax.set_xlabel(r'$x$ [m]', fontsize=8)
            ax.set_ylabel(r'Mean risk', fontsize=8)
            ax.set_title(title, fontsize=8)
            ax.legend(fontsize=6, ncol=2)

        fig.suptitle(r'Longitudinal risk profile $\langle R\rangle_y(x)$ '
                     r'— Numerical vs.\ PINN', fontsize=9)
        plt.tight_layout()
        for ext_fmt in ('pdf', 'png'):
            plt.savefig(os.path.join(save_dir, f"profile_longitudinal.{ext_fmt}"),
                        dpi=150, bbox_inches='tight')
        plt.close(fig)
    print("  [profile] longitudinal risk profiles saved")


# ---------------------------------------------------------------------------
# Plot 3 — Scatter: PINN vs numerical (all test-set grid points)
# ---------------------------------------------------------------------------

def plot_scatter(test_snaps, R_pinn_list, save_dir):
    """
    Scatter plot of PINN prediction vs numerical R for all test points.
    Points coloured by local source Q (shows where Q drives risk).
    Includes identity line and regression line.
    """
    R_num_all  = np.concatenate([s['R'].ravel()   for s in test_snaps])
    R_pinn_all = np.concatenate([r.ravel()         for r in R_pinn_list])
    Q_all      = np.concatenate([s['Q'].ravel()    for s in test_snaps])

    # Subsample for legibility
    rng   = np.random.default_rng(0)
    n_max = 8000
    if len(R_num_all) > n_max:
        idx = rng.choice(len(R_num_all), n_max, replace=False)
        R_num_all  = R_num_all[idx]
        R_pinn_all = R_pinn_all[idx]
        Q_all      = Q_all[idx]

    vmax = float(np.percentile(np.maximum(R_num_all, R_pinn_all), 99))
    corr = float(np.corrcoef(R_num_all, R_pinn_all)[0, 1])
    # Simple least-squares fit
    A = np.vstack([R_num_all, np.ones_like(R_num_all)]).T
    slope, intercept = np.linalg.lstsq(A, R_pinn_all, rcond=None)[0]

    with _sci():
        fig, ax = plt.subplots(figsize=(5, 4.5))

        sc = ax.scatter(R_num_all, R_pinn_all, c=Q_all,
                        s=3, alpha=0.4, cmap='YlOrRd',
                        vmin=0, vmax=float(np.percentile(Q_all, 98)))
        cb = plt.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r'Source $Q$', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Identity line
        lim = [0, vmax * 1.05]
        ax.plot(lim, lim, 'k--', lw=0.9, label='Identity ($y=x$)')

        # Regression line
        x_fit = np.array(lim)
        ax.plot(x_fit, slope * x_fit + intercept, 'b-', lw=1.2,
                label=rf'Fit: $y={slope:.2f}x + {intercept:.2f}$')

        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(r'Numerical PDE  $\mathcal{R}$', fontsize=8)
        ax.set_ylabel(r'PINN Surrogate  $\hat{\mathcal{R}}$', fontsize=8)
        ax.set_title(rf'PINN vs.\ Numerical Risk  ($\rho = {corr:.3f}$)', fontsize=9)
        ax.legend(fontsize=7)

        plt.tight_layout()
        for ext_fmt in ('pdf', 'png'):
            plt.savefig(os.path.join(save_dir, f"scatter.{ext_fmt}"),
                        dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"  [scatter]  ρ={corr:.3f}  slope={slope:.3f}  intercept={intercept:.3f}")


# ---------------------------------------------------------------------------
# Plot 4 — Source / transport fields driving the PDE
# ---------------------------------------------------------------------------

def plot_source_fields(snap, loader, snap_idx, save_dir, label=''):
    """
    4-panel: Q (source), vx (longitudinal flow), vy (lateral flow), D (diffusion).
    These are the hand-crafted physics inputs that drive the PDE.
    """
    Q  = snap['Q']
    vx = snap['vx']
    vy = snap['vy']
    D  = snap['D']
    ext, unit = _extent(loader)

    panels = [
        (Q,  r'Source $Q(x,t)$',           'YlOrRd',  r'$Q$'),
        (vx, r'Flow $v_x$ (longitudinal)',  'RdBu_r',  r'$v_x$ [m/s]'),
        (vy, r'Flow $v_y$ (lateral)',       'RdBu_r',  r'$v_y$ [m/s]'),
        (D,  r'Diffusion $D(x,t)$',         'Blues',   r'$D$ [m²/s]'),
    ]

    with _sci():
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
        fig.subplots_adjust(wspace=0.38)

        for ax, (data, title, cmap, clabel) in zip(axes, panels):
            # Symmetric colour scale for signed fields
            if data.min() < 0:
                vlim = float(np.abs(data).max()) or 0.1
                kwargs = dict(vmin=-vlim, vmax=vlim)
            else:
                kwargs = dict(vmin=0, vmax=_vmax(data))

            im = ax.imshow(data, origin='lower', aspect='auto',
                           extent=ext, cmap=cmap, **kwargs)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(f'$x$ [{unit}]', fontsize=7)
            ax.set_ylabel(f'$y$ [{unit}]', fontsize=7)
            ax.tick_params(labelsize=6)
            _cbar(fig, ax, im, clabel)

        t_str = f't = {snap["t"]:.1f}\\,\\mathrm{{s}}'
        fig.suptitle(
            rf'Hand-crafted PDE source/transport fields — {label} — ${t_str}$',
            fontsize=9, y=1.01)

        stem = f"source_fields_t{snap_idx:03d}"
        for ext_fmt in ('pdf', 'png'):
            plt.savefig(os.path.join(save_dir, f"{stem}.{ext_fmt}"),
                        dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"  [source]   snap {snap_idx}  t={snap['t']:.1f}s")


# ---------------------------------------------------------------------------
# Plot 5 — Error map over training vs test epochs (heatmap)
# ---------------------------------------------------------------------------

def plot_error_heatmap(train_snaps, test_snaps, R_pinn_train, R_pinn_test, save_dir):
    """
    Show how point-wise absolute error is spatially distributed,
    averaged over all train and test snapshots separately.
    Reveals which parts of the grid the PINN struggles with.
    """
    err_train = np.mean(
        np.stack([np.abs(R_pinn_train[i] - s['R'])
                  for i, s in enumerate(train_snaps)], axis=0), axis=0)
    err_test  = np.mean(
        np.stack([np.abs(R_pinn_test[i]  - s['R'])
                  for i, s in enumerate(test_snaps)],  axis=0), axis=0)

    vm = _vmax(err_train, err_test)

    with _sci():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
        fig.subplots_adjust(wspace=0.3)

        for ax, err, title in zip(
                axes,
                [err_train, err_test],
                [r'Mean $|\hat{\mathcal{R}}-\mathcal{R}|$ — train set',
                 r'Mean $|\hat{\mathcal{R}}-\mathcal{R}|$ — test set']):
            im = ax.imshow(err, origin='lower', aspect='auto',
                           vmin=0, vmax=vm, cmap='hot')
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(r'$x$ grid index', fontsize=7)
            ax.set_ylabel(r'$y$ grid index', fontsize=7)
            ax.tick_params(labelsize=6)
            _cbar(fig, ax, im, r'Mean abs.\ error')

        fig.suptitle(r'Spatial distribution of PINN prediction error',
                     fontsize=9)
        plt.tight_layout()
        for ext_fmt in ('pdf', 'png'):
            plt.savefig(os.path.join(save_dir, f"error_heatmap.{ext_fmt}"),
                        dpi=150, bbox_inches='tight')
        plt.close(fig)
    print("  [heatmap]  spatial error maps saved")


# ---------------------------------------------------------------------------
# Plot 6 — Agent selection validation
# ---------------------------------------------------------------------------

def plot_agent_selection(snapshots, train_split, save_dir, perc_range=0.0):
    """
    Show how many agents were included in PINN training at each snapshot.
    Uses the N_agents field stored per snapshot during data loading.

    Also prints a table of snapshot index / time / N_agents / Q_max so the
    researcher can verify that the perception filter is selecting sensibly.
    """
    times    = np.array([s['t']         for s in snapshots], dtype=float)
    n_agents = np.array([s.get('N_agents', 0) for s in snapshots], dtype=float)
    q_max    = np.array([float(s['Q'].max()) for s in snapshots], dtype=float)

    n_train = train_split
    n_test  = len(snapshots) - n_train

    with _sci():
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [2, 1]})

        # Top: N_agents over time with train/test split marker
        ax0 = axes[0]
        ax0.plot(times[:n_train], n_agents[:n_train],
                 color="C0", lw=1.3, label=f"Train ({n_train} snaps)")
        ax0.plot(times[n_train:], n_agents[n_train:],
                 color="C1", lw=1.3, label=f"Test ({n_test} snaps)")
        ax0.axvline(times[n_train - 1], color="grey", lw=0.8, ls="--",
                    label="train/test split")
        ax0.set_ylabel("N agents in PINN")
        perc_str = f"  (≤{perc_range:.0f} m filter)" if perc_range > 0 else "  (no filter)"
        ax0.set_title(r"Agent selection per snapshot" + perc_str, fontsize=9)
        ax0.legend(fontsize=7, ncol=3)
        ax0.grid(True, lw=0.4, alpha=0.4)
        ax0.set_xlim(times[0], times[-1])

        # Add mean/min annotations
        mu = n_agents.mean()
        mn = n_agents.min()
        ax0.axhline(mu, color="C0", lw=0.6, ls=":", alpha=0.7)
        ax0.text(times[-1], mu + 0.1, f"  μ={mu:.1f}", fontsize=6,
                 color="C0", va="bottom", ha="right")
        ax0.text(times[int(np.argmin(n_agents))], mn - 0.3,
                 f"min={int(mn)}", fontsize=6, color="C3", ha="center")

        # Bottom: Q_max over time — large spikes indicate close vehicle encounters
        ax1 = axes[1]
        ax1.plot(times[:n_train], q_max[:n_train], color="C0", lw=1.1)
        ax1.plot(times[n_train:], q_max[n_train:], color="C1", lw=1.1)
        ax1.axvline(times[n_train - 1], color="grey", lw=0.8, ls="--")
        ax1.set_xlabel("t [s]")
        ax1.set_ylabel(r"$Q_\mathrm{max}$")
        ax1.set_title(r"Peak source term $Q_\mathrm{max}$ (spikes = near agent)", fontsize=8)
        ax1.grid(True, lw=0.4, alpha=0.4)
        ax1.set_xlim(times[0], times[-1])

        for ext_fmt in ('pdf', 'png'):
            plt.savefig(os.path.join(save_dir, f"agent_selection.{ext_fmt}"),
                        dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Console summary table (first 10 + last 5 snapshots)
    print("\n  --- Agent selection summary ---")
    print(f"  {'idx':>4}  {'t':>6}  {'N_agents':>8}  {'Q_max':>7}")
    idx_show = list(range(min(10, len(snapshots)))) + list(range(max(10, len(snapshots)-5), len(snapshots)))
    for k in sorted(set(idx_show)):
        s = snapshots[k]
        split_tag = " [test]" if k >= n_train else ""
        print(f"  {k:>4}  {s['t']:>6.1f}  {int(s.get('N_agents',0)):>8}  "
              f"{float(s['Q'].max()):>7.2f}{split_tag}")
    print(f"\n  Mean N_agents={n_agents.mean():.1f}  "
          f"min={int(n_agents.min())}  max={int(n_agents.max())}")
    print(f"  Snapshots with 0 agents: {int((n_agents == 0).sum())}")
    print(f"  [agent_selection] saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise PINN vs numerical risk field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          Examples
          --------
            python pinn_compare_fields.py --dataset exiD --recording 00
            python pinn_compare_fields.py --dataset inD  --recording 03
            python pinn_compare_fields.py --dataset rounD --recording all --max_sec 10
        """))
    parser.add_argument('--dataset',    default='exiD', choices=KNOWN_DATASETS)
    parser.add_argument('--data_root',  default=os.path.join(DREAM_ROOT, 'data'))
    parser.add_argument('--recording',  default='00',
                        help='Recording ID, comma list, or "all"')
    parser.add_argument('--model',      default=None,
                        help='Path to trained .pt file (auto-inferred if omitted)')
    parser.add_argument('--max_sec',    type=float, default=40.0)
    parser.add_argument('--warmup_sec', type=float, default=4.0)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--hidden',     type=int,   default=128)
    parser.add_argument('--depth',      type=int,   default=6)
    parser.add_argument('--n_overview', type=int,   default=4,
                        help='Number of overview snapshots to plot')
    args = parser.parse_args()

    # ── paths ──────────────────────────────────────────────────────────
    data_dir = os.path.join(args.data_root, args.dataset)
    rec_ids  = parse_recording_ids(args.recording, data_dir)
    rec_tag  = args.recording.replace(',', '+') if args.recording.lower() != 'all' else 'all'
    save_dir = os.path.join(DREAM_ROOT, f"pinn_compare_{args.dataset}_{rec_tag}")
    os.makedirs(save_dir, exist_ok=True)

    # Auto-find model if not specified
    model_path = args.model
    if model_path is None:
        candidates = [
            f"pinn_{args.dataset}_{rec_tag}.pt",
            "pinn_risk_field.pt",
        ]
        for c in candidates:
            fp = os.path.join(DREAM_ROOT, c)
            if os.path.isfile(fp):
                model_path = fp
                break
    if model_path is None or not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"No model file found. Train first with pinn_risk_field.py, "
            f"then pass --model <path>.")

    label = f'{args.dataset} / {rec_tag}'
    print(f"Dataset  : {label}  ({data_dir})")
    print(f"Model    : {model_path}")
    print(f"Saving   : {save_dir}/")

    # ── Load checkpoint first (need arch + normalizer + perception_range) ─
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt   = torch.load(model_path, map_location=device, weights_only=False)

    # Restore original normalizer — MUST come from checkpoint, not recomputed.
    # Recomputing from current data gives different Q_max (because filter may
    # differ between training and evaluation runs) → wrong PINN input → zero R.
    norm = Normalizer.__new__(Normalizer)
    norm.ranges       = ckpt['norm_ranges']
    norm.lambda_decay = cfg.lambda_decay
    norm.tau          = cfg.tau

    # Architecture params from checkpoint (with safe defaults for old ckpts)
    _hidden      = int(ckpt.get('hidden',      args.hidden))
    _depth       = int(ckpt.get('depth',       args.depth))
    _use_rff     = bool(ckpt.get('use_rff',    False))
    _rff_feat    = int(ckpt.get('rff_features', 64))
    _rff_scale   = float(ckpt.get('rff_scale',  10.0))
    _use_ctx     = bool(ckpt.get('use_context', False))
    # Perception range used during training (0.0 → no filter = old model)
    _perc_range  = float(ckpt.get('perception_range', 0.0))
    perc_load    = _perc_range if _perc_range > 0 else float('inf')

    print(f"  arch: {_hidden}×{_depth}  rff={_use_rff}  ctx={_use_ctx}  "
          f"perc={_perc_range:.0f}m ({'+filter' if _perc_range > 0 else 'no filter'})")

    # ── Phase 1: load data with matching perception range ──────────────
    print("\n[Load] Generating numerical PDE snapshots...")
    snapshots, last_loader = load_all_recordings(
        recording_ids=rec_ids,
        data_dir=data_dir,
        max_sec=args.max_sec,
        warmup_sec=args.warmup_sec,
        perception_range=perc_load,   # match training filter
    )
    split       = int(len(snapshots) * args.train_frac)
    train_snaps = snapshots[:split]
    test_snaps  = snapshots[split:]
    print(f"  {len(train_snaps)} train / {len(test_snaps)} test snapshots")

    # ── Phase 2: rebuild PINN with checkpoint arch and normalizer ──────
    # Use a dummy cache — predict_field() reads from snap dicts directly,
    # so the cache is only needed for the PINNTrainer constructor.
    _n_cols = len(FlatSampleCache.KEYS)
    dummy_cache = FlatSampleCache.__new__(FlatSampleCache)
    dummy_cache.x_grid = last_loader.x_grid
    dummy_cache.y_grid = last_loader.y_grid
    dummy_cache.times  = np.array([0.0, 1.0])
    dummy_cache._buf   = np.zeros((2, _n_cols), dtype=np.float32)
    dummy_cache._N     = 2

    trainer = PINNTrainer(
        snapshots=train_snaps, norm=norm, interp=dummy_cache,
        hidden=_hidden, depth=_depth,
        use_rff=_use_rff, rff_features=_rff_feat, rff_scale=_rff_scale,
        use_context=_use_ctx, device=device,
    )
    trainer.model.load_state_dict(ckpt['model_state'])
    trainer.model.eval()
    trainer.snaps = snapshots   # allow predict_field over full range
    # predict_field() needs interp.y_grid / x_grid for meshgrid
    trainer.interp = dummy_cache

    # ── Phase 3: predict PINN fields for all test snapshots ────────────
    print("\n[Predict] Running PINN on test set...")
    R_pinn_test  = []
    for s in test_snaps:
        idx = snapshots.index(s)
        R_pinn_test.append(trainer.predict_field(idx))

    print("[Predict] Running PINN on train set (for error heatmap)...")
    R_pinn_train = []
    for s in train_snaps[:min(len(train_snaps), 80)]:   # cap at 80 for speed
        idx = snapshots.index(s)
        R_pinn_train.append(trainer.predict_field(idx))
    train_snaps_sub = train_snaps[:len(R_pinn_train)]

    # ── Visualisations ──────────────────────────────────────────────────
    print(f"\n[Plot] Saving figures to {save_dir}/ ...")

    # Select evenly spaced overview timesteps from the test set
    n_ov  = min(args.n_overview, len(test_snaps))
    ov_idx = np.linspace(0, len(test_snaps) - 1, n_ov, dtype=int)

    # Plot 1: overview comparison at each selected timestep
    for i in ov_idx:
        s_idx = snapshots.index(test_snaps[i])
        plot_overview(test_snaps[i], R_pinn_test[i], last_loader,
                      s_idx, save_dir, label=label)

    # Plot 2: longitudinal profile
    plot_longitudinal_profiles(
        [test_snaps[i] for i in ov_idx],
        [R_pinn_test[i] for i in ov_idx],
        last_loader, save_dir)

    # Plot 3: scatter
    plot_scatter(test_snaps, R_pinn_test, save_dir)

    # Plot 4: source fields at first overview snapshot
    s_idx0 = snapshots.index(test_snaps[ov_idx[0]])
    plot_source_fields(test_snaps[ov_idx[0]], last_loader,
                       s_idx0, save_dir, label=label)

    # Plot 5: spatial error heatmap
    plot_error_heatmap(train_snaps_sub, test_snaps,
                       R_pinn_train, R_pinn_test, save_dir)

    # Plot 6: agent selection validation
    plot_agent_selection(snapshots, split, save_dir, _perc_range)

    print(f"\nDone — all figures in {save_dir}/")


if __name__ == "__main__":
    main()
