"""
PRIDEAM Visualization Utilities
===============================
Visualization tools for DRIFT risk field overlay on IDEAM simulations.

Provides:
- Risk field heatmap overlay
- Risk contour plotting
- Horizon risk visualization
- Combined ego + risk visualization

Author: PRIDEAM Integration
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import Config as cfg


# =============================================================================
# COLOR MAPS
# =============================================================================

def get_risk_cmap(style='hot'):
    """
    Get colormap for risk visualization.

    Args:
        style: 'hot', 'cold', 'traffic', or 'custom'

    Returns:
        matplotlib colormap
    """
    if style == 'hot':
        # Red-yellow-white (high risk = bright)
        colors = [
            (0.0, '#000000'),   # Black (no risk)
            (0.2, '#1a1a2e'),   # Dark blue
            (0.4, '#e94560'),   # Red
            (0.7, '#f9a825'),   # Orange
            (1.0, '#ffffff'),   # White (max risk)
        ]
    elif style == 'cold':
        # Blue-cyan-white
        colors = [
            (0.0, '#000000'),
            (0.3, '#0d47a1'),
            (0.6, '#00bcd4'),
            (1.0, '#ffffff'),
        ]
    elif style == 'traffic':
        # Green-yellow-red (traffic light)
        colors = [
            (0.0, '#1b5e20'),   # Dark green (safe)
            (0.3, '#4caf50'),   # Green
            (0.5, '#ffeb3b'),   # Yellow
            (0.7, '#ff9800'),   # Orange
            (1.0, '#f44336'),   # Red (danger)
        ]
    else:  # custom/default
        colors = [
            (0.0, '#0D1117'),
            (0.15, '#1a1a2e'),
            (0.35, '#4a148c'),
            (0.55, '#e91e63'),
            (0.75, '#ff5722'),
            (1.0, '#ffeb3b'),
        ]

    positions = [c[0] for c in colors]
    color_values = [c[1] for c in colors]

    return LinearSegmentedColormap.from_list('risk', list(zip(positions, color_values)))


# =============================================================================
# RISK FIELD VISUALIZATION
# =============================================================================

def plot_risk_field(ax, risk_field, X=None, Y=None, vmin=0, vmax=2.0,
                    cmap='hot', alpha=0.7, add_colorbar=True,
                    colorbar_label='Risk'):
    """
    Plot risk field as heatmap.

    Args:
        ax: Matplotlib axes
        risk_field: 2D risk array
        X, Y: Meshgrid coordinates (uses cfg.X, cfg.Y if None)
        vmin, vmax: Color scale limits
        cmap: Colormap name or object
        alpha: Transparency
        add_colorbar: Whether to add colorbar
        colorbar_label: Label for colorbar

    Returns:
        pcm: Pcolormesh object
    """
    if X is None or Y is None:
        X, Y = cfg.X, cfg.Y

    if isinstance(cmap, str):
        cmap = get_risk_cmap(cmap)

    # Smooth for visualization
    R_smooth = gaussian_filter(risk_field, sigma=0.5)

    pcm = ax.pcolormesh(X, Y, R_smooth, cmap=cmap, shading='gouraud',
                        vmin=vmin, vmax=vmax, alpha=alpha, zorder=1)

    if add_colorbar:
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(colorbar_label, fontsize=10)

    return pcm


def plot_risk_contours(ax, risk_field, X=None, Y=None, levels=None,
                       colors='white', linewidths=1.0, alpha=0.6):
    """
    Plot risk field contours.

    Args:
        ax: Matplotlib axes
        risk_field: 2D risk array
        X, Y: Meshgrid coordinates
        levels: Contour levels (auto if None)
        colors: Contour colors
        linewidths: Line widths
        alpha: Transparency

    Returns:
        cs: Contour set
    """
    if X is None or Y is None:
        X, Y = cfg.X, cfg.Y

    R_smooth = gaussian_filter(risk_field, sigma=0.5)

    if levels is None:
        max_val = R_smooth.max()
        if max_val > 0.2:
            levels = np.linspace(0.2, min(max_val, 2.0) * 0.9, 5)
        else:
            levels = [0.1, 0.15, 0.2]

    cs = ax.contour(X, Y, R_smooth, levels=levels, colors=colors,
                    linewidths=linewidths, alpha=alpha, zorder=2)

    return cs


def plot_risk_overlay(ax, prideam_controller, ego_state=None,
                      show_heatmap=True, show_contours=True,
                      vmax=2.0, alpha=0.6):
    """
    Plot risk field overlay from PRIDEAM controller.

    Args:
        ax: Matplotlib axes
        prideam_controller: PRIDEAMController instance
        ego_state: [x, y, psi] for centering view (optional)
        show_heatmap: Whether to show heatmap
        show_contours: Whether to show contours
        vmax: Max value for color scale
        alpha: Transparency

    Returns:
        dict: Plot objects {pcm, cs}
    """
    result = {}

    risk_field = prideam_controller.risk_field
    X, Y = prideam_controller.drift.grid

    if show_heatmap:
        result['pcm'] = plot_risk_field(
            ax, risk_field, X, Y,
            vmax=vmax, alpha=alpha, add_colorbar=True
        )

    if show_contours:
        result['cs'] = plot_risk_contours(
            ax, risk_field, X, Y, alpha=0.7
        )

    return result


# =============================================================================
# EGO + RISK VISUALIZATION
# =============================================================================

def plot_ego_with_risk(ax, prideam_controller, ego_state, ego_global,
                       show_risk_value=True, show_modulation=True):
    """
    Plot ego vehicle with local risk information.

    Args:
        ax: Matplotlib axes
        prideam_controller: PRIDEAMController instance
        ego_state: [vx, vy, w, s, ey, epsi]
        ego_global: [x, y, psi]
        show_risk_value: Show numerical risk value
        show_modulation: Show CBF modulation info

    Returns:
        dict: Plot objects
    """
    result = {}

    x, y, psi = ego_global[0], ego_global[1], ego_global[2]

    # Get local risk
    risk = prideam_controller.drift.get_risk_cartesian(x, y)

    # Determine risk color
    if risk > 1.0:
        color = '#f44336'  # Red
        risk_level = 'HIGH'
    elif risk > 0.5:
        color = '#ff9800'  # Orange
        risk_level = 'MEDIUM'
    elif risk > 0.2:
        color = '#ffeb3b'  # Yellow
        risk_level = 'LOW'
    else:
        color = '#4caf50'  # Green
        risk_level = 'SAFE'

    # Plot ego vehicle
    L = prideam_controller.vehicle_length
    W = prideam_controller.vehicle_width

    rect = mpatches.FancyBboxPatch(
        (x - L/2, y - W/2), L, W,
        boxstyle="round,pad=0.1",
        facecolor='#2ECC71',
        edgecolor=color,
        linewidth=3.0,
        zorder=10
    )
    ax.add_patch(rect)
    result['ego_patch'] = rect

    # Risk value text
    if show_risk_value:
        text = ax.text(
            x, y - W/2 - 1.5,
            f'Risk: {risk:.2f} ({risk_level})',
            ha='center', va='top',
            fontsize=8, fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            zorder=11
        )
        result['risk_text'] = text

    # Modulation info
    if show_modulation and prideam_controller.last_modulated_params:
        params = prideam_controller.last_modulated_params
        mod_text = f"CBF: {params['a_l']:.2f}x{params['b_l']:.2f}"
        text2 = ax.text(
            x, y + W/2 + 0.5,
            mod_text,
            ha='center', va='bottom',
            fontsize=6, color='cyan',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
            zorder=11
        )
        result['mod_text'] = text2

    return result


def plot_horizon_risk(ax, prideam_controller, oS, oey, path_index,
                      show_profile=True):
    """
    Plot risk along MPC prediction horizon.

    Args:
        ax: Matplotlib axes
        prideam_controller: PRIDEAMController instance
        oS: Predicted s trajectory
        oey: Predicted ey trajectory
        path_index: Path index
        show_profile: Whether to plot as line profile

    Returns:
        dict: Plot objects
    """
    result = {}

    if oS is None or oey is None:
        return result

    # Get risk along horizon
    risks = prideam_controller.drift.get_risk_along_horizon(oS, oey, path_index)

    if show_profile:
        # Plot as line
        timesteps = np.arange(len(risks)) * prideam_controller.dt
        line, = ax.plot(timesteps, risks, 'r-', linewidth=2, label='Horizon Risk')
        ax.fill_between(timesteps, 0, risks, alpha=0.3, color='red')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Risk')
        ax.legend()
        result['line'] = line
    else:
        # Store for other use
        result['risks'] = risks

    return result


# =============================================================================
# COMBINED VISUALIZATION
# =============================================================================

def create_prideam_figure(prideam_controller, ego_state, ego_global,
                          vehicles_dict=None, title=None,
                          xlim=None, ylim=None, figsize=(14, 8)):
    """
    Create a complete PRIDEAM visualization figure.

    Args:
        prideam_controller: PRIDEAMController instance
        ego_state: Ego Frenet state
        ego_global: Ego Cartesian state [x, y, psi]
        vehicles_dict: Dict of surrounding vehicles
        title: Figure title
        xlim, ylim: Axis limits (auto if None)
        figsize: Figure size

    Returns:
        fig, axes: Figure and axes dict
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('#0D1117')

    # Create grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.25, wspace=0.2)

    # Main risk field view
    ax_main = fig.add_subplot(gs[0, :2])
    ax_main.set_facecolor('#161B22')

    # Info panel
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.set_facecolor('#161B22')
    ax_info.axis('off')

    # Horizon risk profile
    ax_horizon = fig.add_subplot(gs[1, :])
    ax_horizon.set_facecolor('#161B22')

    # Plot risk field
    plot_risk_overlay(ax_main, prideam_controller, alpha=0.7)

    # Plot ego
    plot_ego_with_risk(ax_main, prideam_controller, ego_state, ego_global)

    # Set limits
    if xlim is None:
        xlim = (ego_global[0] - 40, ego_global[0] + 40)
    if ylim is None:
        ylim = (ego_global[1] - 15, ego_global[1] + 15)

    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('x [m]', color='white')
    ax_main.set_ylabel('y [m]', color='white')
    ax_main.tick_params(colors='white')

    if title:
        ax_main.set_title(title, fontsize=12, fontweight='bold', color='white')

    # Info text
    risk = prideam_controller.drift.get_risk_cartesian(ego_global[0], ego_global[1])
    stats = prideam_controller.get_stats()

    info_text = f"""PRIDEAM Status
━━━━━━━━━━━━━━━━━━
Ego Position:
  x: {ego_global[0]:.1f} m
  y: {ego_global[1]:.1f} m

Local Risk: {risk:.3f}

Statistics:
  Updates: {stats['risk_field_updates']}
  Solves: {stats['mpc_solves']}
  Blocked: {stats['lane_changes_blocked']}

Weights:
  CBF mod: {prideam_controller.weights.cbf_modulation}
  Headway: {prideam_controller.weights.headway_modulation}
  Threshold: {prideam_controller.weights.decision_threshold}
"""

    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=9, color='white', verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9))

    # Horizon risk (if available)
    if prideam_controller.last_risk_along_horizon is not None:
        risks = prideam_controller.last_risk_along_horizon
        timesteps = np.arange(len(risks)) * prideam_controller.dt
        ax_horizon.fill_between(timesteps, 0, risks, alpha=0.4, color='#e91e63')
        ax_horizon.plot(timesteps, risks, 'w-', linewidth=1.5)
        ax_horizon.axhline(y=prideam_controller.weights.decision_threshold,
                          color='red', linestyle='--', linewidth=1, alpha=0.7,
                          label='Threshold')
        ax_horizon.set_xlabel('Horizon Time [s]', color='white')
        ax_horizon.set_ylabel('Risk', color='white')
        ax_horizon.set_title('Risk Along MPC Horizon', color='white', fontsize=10)
        ax_horizon.tick_params(colors='white')
        ax_horizon.set_xlim(0, timesteps[-1])
        ax_horizon.set_ylim(0, max(risks.max() * 1.2, prideam_controller.weights.decision_threshold * 1.2))
        ax_horizon.legend(loc='upper right', fontsize=8)
    else:
        ax_horizon.text(0.5, 0.5, 'No horizon data', transform=ax_horizon.transAxes,
                       ha='center', va='center', color='gray', fontsize=12)

    axes = {
        'main': ax_main,
        'info': ax_info,
        'horizon': ax_horizon,
    }

    return fig, axes


# =============================================================================
# ANIMATION HELPERS
# =============================================================================

def update_prideam_plot(ax, prideam_controller, ego_global, clear=True):
    """
    Update existing plot with new risk field data.

    For use in animation loops.

    Args:
        ax: Matplotlib axes
        prideam_controller: PRIDEAMController
        ego_global: [x, y, psi]
        clear: Whether to clear axes first

    Returns:
        list: Artist objects
    """
    artists = []

    if clear:
        ax.clear()

    # Plot risk field
    result = plot_risk_overlay(ax, prideam_controller, alpha=0.6)
    if 'pcm' in result:
        artists.append(result['pcm'])

    return artists


def save_risk_frame(prideam_controller, ego_state, ego_global, frame_num,
                    output_dir, prefix='prideam'):
    """
    Save a single frame of PRIDEAM visualization.

    Args:
        prideam_controller: PRIDEAMController
        ego_state: Ego Frenet state
        ego_global: Ego Cartesian state
        frame_num: Frame number
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        str: Saved filename
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, _ = create_prideam_figure(
        prideam_controller, ego_state, ego_global,
        title=f'PRIDEAM - Frame {frame_num}'
    )

    filename = os.path.join(output_dir, f'{prefix}_{frame_num:04d}.png')
    fig.savefig(filename, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)

    return filename
