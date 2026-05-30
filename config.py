"""
PDE Risk Field Configuration
Advection-Diffusion-Telegrapher Model for Traffic Risk Propagation
"""
import numpy as np


class Config:
    """Configuration for the PDE risk field solver."""
    
    # ============ Spatial Domain ============
    # NOTE: Grid is in WORLD-FIXED coordinates aligned with IDEAM coordinate system
    # IDEAM paths are defined in a specific coordinate frame - grid must match
    x_min, x_max = -150.0, 255.2   # Longitudinal range [m] (matches IDEAM path geometry)
    y_min, y_max = -225.2, -45.3   # Lateral range [m] (matches IDEAM path geometry)
    nx, ny = 250, 80               # Grid resolution

    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # ============ PDE Parameters ============
    # Tuned for IDEAM highway scenario - clearer vehicle-specific risk patterns

    # Diffusion (reduced to keep risk localized around vehicles)
    D0 = 0.3              # Base diffusion coefficient [m²/s] (reduced from 1.0)
    D_occ = 6.0           # Additional diffusion in occluded regions [m²/s] (spreads ~7m with λ=0.15)

    # Decay (increased to prevent risk accumulation)
    lambda_decay = 0.15   # Base decay rate [1/s] (increased from 0.05)
    L_decay = 25.0        # Distance half-life [m] (reduced from 40.0 for faster decay)

    # Sponge layer (absorbing boundary to prevent reflections)
    sponge_length = 15.0  # [m] - width of absorbing boundary layer
    lambda_sponge = 1.5   # [1/s] - stronger absorption (increased from 1.0)

    # Telegrapher (wave-like propagation)
    # Set tau=0 to disable (recommended for testing)
    # Set tau=0.2-0.5 to enable smooth wave-like propagation with inertia
    tau = 0.2             # Inertia/reaction time [s] (0 = disabled)

    # Post-processing
    post_smooth_sigma = 0.0   # Smoothing sigma (0.0 = disabled; use 0.1-0.2 for viz only)

    # ============ Source Parameters ============
    # GVF-style Gaussian kernel - more anisotropic for clearer forward-facing risk
    sigma_x = 8.0         # Longitudinal spread [m] (reduced from 12.0 for sharper peaks)
    sigma_y = 2.5         # Lateral spread [m] (reduced from 3.0 for narrower risk zones)
    
    # ============ Merge Zone ============
    merge_x_start = 30.0  # Merge zone start [m]
    merge_x_end = 70.0    # Merge zone end (gore point) [m]
    merge_y_ramp = 6.0    # Ramp lane y-position [m]
    
    # ============ Lane Geometry ============
    lane_width = 4.0
    lane_centers = [-4.0, 0.0, 4.0, 8.0]  # Lane center y-positions
    
    # ============ Vehicle Dimensions ============
    car_length = 5.0
    car_width = 2.0
    truck_length = 12.0
    truck_width = 2.5
