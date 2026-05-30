"""
PRIDEAM Integration Layer
=========================
PDE-Risk Integrated Dynamic Emergency Assessment Model

This package provides the integration between IDEAM's MPC+CBF control
and DRIFT's propagating risk field.

Components:
- DRIFTInterface: Coordinate transforms, risk queries, caching
- PRIDEAMController: Combined MPC controller with risk awareness
- Visualization utilities for risk overlay

Usage:
    from Integration import PRIDEAMController, DRIFTInterface

    # Initialize controller
    controller = PRIDEAMController()

    # Update risk field each timestep
    controller.update_risk_field(vehicles, ego_state, dt)

    # Solve MPC with risk awareness
    result = controller.solve_with_risk(...)
"""

from .drift_interface import DRIFTInterface
from .prideam_controller import PRIDEAMController, create_prideam_controller, RiskWeights
from .visualization import (
    plot_risk_field,
    plot_risk_contours,
    plot_risk_overlay,
    plot_ego_with_risk,
    plot_horizon_risk,
    create_prideam_figure,
    get_risk_cmap,
)

__all__ = [
    'DRIFTInterface',
    'PRIDEAMController',
    'create_prideam_controller',
    'RiskWeights',
    'plot_risk_field',
    'plot_risk_contours',
    'plot_risk_overlay',
    'plot_ego_with_risk',
    'plot_horizon_risk',
    'create_prideam_figure',
    'get_risk_cmap',
]

__version__ = '1.0.0'
