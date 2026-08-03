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

__all__ = ['DRIFTInterface']

# The public RL/PINN release does not require the synthetic MPC stack. Keep
# those exports available when its optional dependencies are installed without
# making a field-only import fail in a clean environment.
try:
    from .prideam_controller import (
        PRIDEAMController,
        RiskWeights,
        create_prideam_controller,
    )
except ModuleNotFoundError:
    pass
else:
    __all__.extend(
        ['PRIDEAMController', 'create_prideam_controller', 'RiskWeights']
    )

try:
    from .visualization import (
        create_prideam_figure,
        get_risk_cmap,
        plot_ego_with_risk,
        plot_horizon_risk,
        plot_risk_contours,
        plot_risk_field,
        plot_risk_overlay,
    )
except ModuleNotFoundError:
    pass
else:
    __all__.extend(
        [
            'plot_risk_field',
            'plot_risk_contours',
            'plot_risk_overlay',
            'plot_ego_with_risk',
            'plot_horizon_risk',
            'create_prideam_figure',
            'get_risk_cmap',
        ]
    )

__version__ = '1.0.0'
