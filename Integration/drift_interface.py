"""
DRIFT Interface for IDEAM Integration
=====================================
Provides risk field queries in both Cartesian and Frenet coordinates.
Manages PDE solver lifecycle and coordinate transformations.

This is the key integration layer between:
- DRIFT: PDE-based risk field propagation (Cartesian grid)
- IDEAM: MPC+CBF controller (Frenet coordinates along paths)

Author: PRIDEAM Integration
"""

import sys
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import DRIFT components
from config import Config as cfg
from pde_solver import (
    PDESolver,
    compute_total_Q,
    compute_velocity_field,
    compute_diffusion_field,
    create_vehicle as drift_create_vehicle,
)


class DRIFTInterface:
    """
    Interface between IDEAM and DRIFT risk field.

    Provides:
    - Risk queries at arbitrary (x, y) or (s, ey) coordinates
    - Gradient computation for CBF modulation
    - Batch queries for MPC horizon
    - Automatic coordinate transformation Frenet <-> Cartesian

    Attributes:
        solver: PDESolver instance
        path_funcs: Dict of path objects for Frenet conversion
        X, Y: Meshgrid coordinates
        last_Q: Last computed source term (for debugging)
    """

    def __init__(self, path_funcs=None):
        """
        Initialize the DRIFT interface.

        Args:
            path_funcs: Dict of path objects for Frenet conversion
                        {0: path_left, 1: path_center, 2: path_right}
        """
        self.solver = PDESolver()
        self.path_funcs = path_funcs or {}

        # Grid info from config
        self.X, self.Y = cfg.X, cfg.Y
        self.x_grid = cfg.x
        self.y_grid = cfg.y
        self.dx, self.dy = cfg.dx, cfg.dy

        # Interpolators (updated after each step)
        self._interpolator = None
        self._grad_x_interp = None
        self._grad_y_interp = None

        # Store last computed values for debugging/visualization
        self.last_Q = None
        self.last_D = None
        self.last_vx = None
        self.last_vy = None

        # Caching for repeated queries at same positions
        self._query_cache = {}
        self._cache_valid = False

    def set_road_mask(self, mask):
        """
        Set road boundary mask — Dirichlet BC confining risk to road.

        R(x,t) = 0 for x outside Ω_road, enforced each PDE step via:
            R^{n+1}(x) ← R^{n+1}(x) · M(x)

        Args:
            mask: 2D array (ny, nx) with values in [0,1].
                  1.0 = on road, 0.0 = off road, smooth taper at edges.
        """
        self.solver.set_road_mask(mask)

    def reset(self):
        """Reset the risk field to zero."""
        self.solver.reset()
        self._interpolator = None
        self._grad_x_interp = None
        self._grad_y_interp = None
        self._cache_valid = False
        self.last_Q = None
        self.last_D = None
        self.last_vx = None
        self.last_vy = None

    def warmup(self, vehicles, ego, dt=0.1, duration=5.0, substeps=3,
               source_fn=None):
        """
        Warm up risk field by pre-evolving PDE to quasi-equilibrium.

        Solves the cold start problem where risk field starts at zero.
        Pre-evolves the field for several seconds so it reaches realistic
        values before main simulation begins.

        Args:
            vehicles: Initial surrounding vehicles (DRIFT format)
            ego: Initial ego vehicle dict
            dt: Timestep [s]
            duration: Warm-up duration [s]
            substeps: PDE substeps per timestep
            source_fn: Optional callable replacing compute_total_Q.
                       Signature: source_fn(vehicles, ego, X, Y)
                       Returns (Q_total, Q_veh, Q_occ, occ_mask).

        Returns:
            final_risk: Risk field after warm-up
        """
        n_steps = int(duration / dt)
        print(f"Warming up DRIFT risk field for {duration:.1f}s ({n_steps} steps)...", end='', flush=True)

        for i in range(n_steps):
            # Evolve PDE with current vehicle configuration
            # (vehicles are static during warm-up)
            _ = self.step(vehicles, ego, dt=dt, substeps=substeps,
                          source_fn=source_fn)

            # Progress indicator
            if (i + 1) % max(1, n_steps // 5) == 0:
                print('.', end='', flush=True)

        # Query final risk at ego position
        final_risk_ego = self.get_risk_cartesian(ego['x'], ego['y'])
        print(f" Done!")
        print(f"  Initial risk at ego: {final_risk_ego:.3f}")
        print(f"  Field max risk: {np.max(self.solver.R):.3f}")
        print(f"  Field mean risk: {np.mean(self.solver.R):.3f}")

        return self.solver.R.copy()

    def step(self, vehicles, ego, dt=0.1, substeps=3, source_fn=None):
        """
        Advance the PDE one timestep.

        Args:
            vehicles: List of vehicle dicts in DRIFT format
                      Each dict: {id, x, y, vx, vy, heading, class, length, width}
            ego: Ego vehicle dict
            dt: Time step [s]
            substeps: Number of sub-steps for numerical stability
            source_fn: Optional callable replacing compute_total_Q.
                       Signature: source_fn(vehicles, ego, X, Y)
                       Returns (Q_total, Q_veh, Q_occ, occ_mask).
                       When None (default) the standard GVF formulation is used.

        Returns:
            R: Updated risk field (2D array on cfg.X, cfg.Y grid)
        """
        # Compute source terms Q(x,t)
        if source_fn is None:
            Q_total, Q_veh, Q_occ, occ_mask = compute_total_Q(
                vehicles, ego, self.X, self.Y
            )
        else:
            Q_total, Q_veh, Q_occ, occ_mask = source_fn(
                vehicles, ego, self.X, self.Y
            )
        self.last_Q = Q_total

        # Compute velocity field for advection
        vx, vy, vx_flow, vy_flow, vx_topo, vy_topo = compute_velocity_field(
            vehicles, ego, self.X, self.Y
        )
        self.last_vx, self.last_vy = vx, vy

        # Compute spatially-varying diffusion coefficient (with braking-enhanced diffusion)
        D = compute_diffusion_field(occ_mask, self.X, self.Y, vehicles, ego)
        self.last_D = D

        # Advance PDE with sub-stepping for stability
        sub_dt = dt / substeps
        for _ in range(substeps):
            R = self.solver.step(Q_total, D, vx, vy, dt=sub_dt)

        # Update interpolators for fast queries
        self._update_interpolators(R)
        self._cache_valid = False

        return R

    def step_with_ideam_vehicles(self, vehicle_left, vehicle_centre, vehicle_right,
                                  ego_x, ego_y, ego_vx, ego_psi, dt=0.1):
        """
        Convenience method: step using IDEAM's vehicle array format.

        Args:
            vehicle_left, vehicle_centre, vehicle_right: IDEAM vehicle arrays
                Each row: [s, ey, epsi, x, y, psi, vx, a]
            ego_x, ego_y: Ego position in Cartesian
            ego_vx: Ego longitudinal velocity
            ego_psi: Ego heading
            dt: Time step

        Returns:
            R: Updated risk field
        """
        # Convert IDEAM format to DRIFT format
        vehicles = self._convert_ideam_vehicles(
            vehicle_left, vehicle_centre, vehicle_right
        )

        # Create ego dict
        ego = drift_create_vehicle(
            vid=0, x=ego_x, y=ego_y, vx=ego_vx, vy=0, vclass='car'
        )
        ego['heading'] = ego_psi

        return self.step(vehicles, ego, dt)

    def _convert_ideam_vehicles(self, vehicle_left, vehicle_centre, vehicle_right):
        """
        Convert IDEAM vehicle arrays to DRIFT vehicle list.

        IDEAM format: [s, ey, epsi, x, y, psi, vx, a] per row
        DRIFT format: {id, x, y, vx, vy, heading, class, length, width}
        """
        vehicles = []
        vid = 1

        for vehicle_array in [vehicle_left, vehicle_centre, vehicle_right]:
            if vehicle_array is None:
                continue

            for row in vehicle_array:
                if len(row) < 7:
                    continue

                # Extract Cartesian state from IDEAM array
                x_cart = row[3]
                y_cart = row[4]
                psi = row[5]
                vx = row[6]

                # Create DRIFT vehicle
                v = drift_create_vehicle(
                    vid=vid,
                    x=x_cart,
                    y=y_cart,
                    vx=vx,
                    vy=0,  # IDEAM assumes vy ≈ 0 for surrounding vehicles
                    vclass='car'
                )
                v['heading'] = psi

                vehicles.append(v)
                vid += 1

        return vehicles

    def _update_interpolators(self, R):
        """
        Update interpolators for fast queries.

        Creates:
        - Risk value interpolator
        - Risk gradient interpolators (∂R/∂x, ∂R/∂y)
        """
        # Smooth slightly for numerical stability
        R_smooth = gaussian_filter(R, sigma=0.3)

        # Risk value interpolator
        self._interpolator = RegularGridInterpolator(
            (self.y_grid, self.x_grid), R_smooth,
            method='linear', bounds_error=False, fill_value=0.0
        )

        # Compute gradients for CBF modulation
        grad_y, grad_x = np.gradient(R_smooth, self.dy, self.dx)

        self._grad_x_interp = RegularGridInterpolator(
            (self.y_grid, self.x_grid), grad_x,
            method='linear', bounds_error=False, fill_value=0.0
        )
        self._grad_y_interp = RegularGridInterpolator(
            (self.y_grid, self.x_grid), grad_y,
            method='linear', bounds_error=False, fill_value=0.0
        )

    # =====================================================================
    # QUERY METHODS - Cartesian Coordinates
    # =====================================================================

    def get_risk_cartesian(self, x, y):
        """
        Query risk at Cartesian coordinates.

        Args:
            x, y: Scalars or arrays of positions [m]

        Returns:
            R: Risk values at queried points (same shape as input)
        """
        if self._interpolator is None:
            if hasattr(x, '__len__'):
                return np.zeros_like(np.atleast_1d(x), dtype=float)
            return 0.0

        # Handle scalar vs array input
        x_arr = np.atleast_1d(x)
        y_arr = np.atleast_1d(y)

        # RegularGridInterpolator expects (y, x) order
        points = np.column_stack([y_arr, x_arr])
        result = self._interpolator(points)

        # Return scalar if input was scalar
        if not hasattr(x, '__len__'):
            return float(result[0])
        return result

    def get_risk_gradient_cartesian(self, x, y):
        """
        Get risk gradient at Cartesian coordinates.

        Args:
            x, y: Scalars or arrays of positions

        Returns:
            dR_dx, dR_dy: Gradient components (same shape as input)
        """
        if self._grad_x_interp is None:
            if hasattr(x, '__len__'):
                zeros = np.zeros_like(np.atleast_1d(x), dtype=float)
                return zeros, zeros.copy()
            return 0.0, 0.0

        x_arr = np.atleast_1d(x)
        y_arr = np.atleast_1d(y)
        points = np.column_stack([y_arr, x_arr])

        dR_dx = self._grad_x_interp(points)
        dR_dy = self._grad_y_interp(points)

        if not hasattr(x, '__len__'):
            return float(dR_dx[0]), float(dR_dy[0])
        return dR_dx, dR_dy

    # =====================================================================
    # QUERY METHODS - Frenet Coordinates
    # =====================================================================

    def get_risk_frenet(self, s, ey, path_index=1):
        """
        Query risk at Frenet coordinates (s, ey) along a path.

        Args:
            s: Arc length coordinate(s) [m]
            ey: Lateral error coordinate(s) [m]
            path_index: Which path (0=left, 1=center, 2=right)

        Returns:
            R: Risk values at queried points
        """
        if path_index not in self.path_funcs:
            raise ValueError(f"Path {path_index} not registered. "
                           f"Available: {list(self.path_funcs.keys())}")

        path = self.path_funcs[path_index]

        # Convert Frenet to Cartesian
        s_arr = np.atleast_1d(s)
        ey_arr = np.atleast_1d(ey)

        x_arr = np.zeros_like(s_arr, dtype=float)
        y_arr = np.zeros_like(s_arr, dtype=float)

        for i, (si, eyi) in enumerate(zip(s_arr, ey_arr)):
            try:
                x_cart, y_cart = path.get_cartesian_coords(float(si), float(eyi))
                x_arr[i] = x_cart
                y_arr[i] = y_cart
            except Exception:
                # Fallback: use path center point if conversion fails
                try:
                    center = path(float(si))
                    x_arr[i] = center[0]
                    y_arr[i] = center[1]
                except Exception:
                    x_arr[i] = 0.0
                    y_arr[i] = 0.0

        result = self.get_risk_cartesian(x_arr, y_arr)

        # Return scalar if input was scalar
        if not hasattr(s, '__len__'):
            return float(result) if hasattr(result, '__len__') else result
        return result

    def get_risk_gradient_frenet(self, s, ey, path_index=1):
        """
        Get risk gradient in Frenet coordinates.

        Note: This returns the gradient projected onto the Frenet frame.

        Args:
            s, ey: Frenet coordinates
            path_index: Path index

        Returns:
            dR_ds, dR_dey: Gradient in Frenet frame
        """
        if path_index not in self.path_funcs:
            raise ValueError(f"Path {path_index} not registered")

        path = self.path_funcs[path_index]
        s_arr = np.atleast_1d(s)
        ey_arr = np.atleast_1d(ey)

        dR_ds = np.zeros_like(s_arr, dtype=float)
        dR_dey = np.zeros_like(ey_arr, dtype=float)

        for i, (si, eyi) in enumerate(zip(s_arr, ey_arr)):
            try:
                # Get Cartesian position and gradient
                x_cart, y_cart = path.get_cartesian_coords(float(si), float(eyi))
                dR_dx, dR_dy = self.get_risk_gradient_cartesian(x_cart, y_cart)

                # Get path tangent direction
                theta_r = path.get_theta_r(float(si))
                cos_t, sin_t = np.cos(theta_r), np.sin(theta_r)

                # Project gradient onto Frenet frame
                # Tangent direction: (cos_t, sin_t)
                # Normal direction: (-sin_t, cos_t)
                dR_ds[i] = dR_dx * cos_t + dR_dy * sin_t
                dR_dey[i] = -dR_dx * sin_t + dR_dy * cos_t

            except Exception:
                dR_ds[i] = 0.0
                dR_dey[i] = 0.0

        if not hasattr(s, '__len__'):
            return float(dR_ds[0]), float(dR_dey[0])
        return dR_ds, dR_dey

    # =====================================================================
    # MPC HORIZON QUERIES
    # =====================================================================

    def get_risk_along_horizon(self, trajectory_s, trajectory_ey, path_index=1):
        """
        Query risk along MPC prediction horizon.

        Args:
            trajectory_s: Array of s values over horizon [T+1]
            trajectory_ey: Array of ey values over horizon [T+1]
            path_index: Path index for Frenet conversion

        Returns:
            risk_profile: Array of risk values [T+1]
        """
        return self.get_risk_frenet(trajectory_s, trajectory_ey, path_index)

    def get_risk_cost_vector(self, oS, oey, path_index, weight=1.0):
        """
        Compute risk cost vector for MPC optimization.

        For use in cost function: sum_t weight * R(x_t)

        Args:
            oS: Predicted s trajectory [T+1]
            oey: Predicted ey trajectory [T+1]
            path_index: Current path index
            weight: Risk weight in cost function

        Returns:
            risk_costs: Array of risk costs for each timestep [T+1]
        """
        risks = self.get_risk_frenet(oS, oey, path_index)
        return weight * np.asarray(risks)

    # =====================================================================
    # IDEAM INTEGRATION HELPERS
    # =====================================================================

    def get_cbf_margin_modulation(self, x, y, base_a, base_b, alpha=0.3, max_scale=2.0,
                                   risk_norm=1.5):
        """
        Modulate CBF ellipse margins based on local risk.

        Higher risk -> larger safety ellipses -> more conservative behavior.

        Args:
            x, y: Position to evaluate (Cartesian)
            base_a, base_b: Base ellipse semi-axes
            alpha: Modulation factor (0-1), higher = more responsive
            max_scale: Maximum scaling factor
            risk_norm: Risk saturation reference. Scale = 1+alpha when risk = risk_norm.
                       Increase for dense traffic to prevent permanent saturation.

        Returns:
            a_mod, b_mod: Modulated ellipse parameters
        """
        risk = self.get_risk_cartesian(x, y)

        # Normalize risk to [0, 1] range; saturates at risk_norm
        risk_normalized = np.clip(risk / risk_norm, 0, 1)

        # Compute scale factor
        scale = 1 + alpha * risk_normalized
        scale = np.clip(scale, 1.0, max_scale)

        a_mod = base_a * scale
        b_mod = base_b * scale

        return float(a_mod), float(b_mod)

    def get_cbf_margin_modulation_frenet(self, s, ey, path_index, base_a, base_b,
                                          alpha=0.3, max_scale=2.0, risk_norm=2.0):
        """
        Modulate CBF ellipse margins based on local risk (Frenet input).

        Args:
            s, ey: Frenet coordinates
            path_index: Path index
            base_a, base_b: Base ellipse semi-axes
            alpha: Modulation factor
            max_scale: Maximum scaling factor
            risk_norm: Risk saturation reference (same semantics as Cartesian variant).

        Returns:
            a_mod, b_mod: Modulated ellipse parameters
        """
        risk = self.get_risk_frenet(s, ey, path_index)
        risk_normalized = np.clip(risk / risk_norm, 0, 1)

        scale = 1 + alpha * risk_normalized
        scale = np.clip(scale, 1.0, max_scale)

        a_mod = base_a * scale
        b_mod = base_b * scale

        return float(a_mod), float(b_mod)

    def get_headway_modulation(self, x, y, base_Th, base_d0, beta=0.5, max_scale=2.0,
                               risk_norm=1.5):
        """
        Modulate time headway and min distance based on local risk.

        Higher risk -> larger following distance -> more conservative.

        Args:
            x, y: Position to evaluate (Cartesian)
            base_Th: Base time headway [s]
            base_d0: Base minimum distance [m]
            beta: Modulation factor
            max_scale: Maximum scaling factor
            risk_norm: Risk saturation reference.

        Returns:
            Th_mod, d0_mod: Modulated parameters
        """
        risk = self.get_risk_cartesian(x, y)
        risk_normalized = np.clip(risk / risk_norm, 0, 1)

        scale = 1 + beta * risk_normalized
        scale = np.clip(scale, 1.0, max_scale)

        Th_mod = base_Th * scale
        d0_mod = base_d0 * scale

        return float(Th_mod), float(d0_mod)

    def evaluate_lane_change_risk(self, ego_s, ego_ey, target_ey, path_index,
                                   n_samples=10, lookahead=30.0):
        """
        Evaluate risk along a lane change trajectory.

        Samples points along an interpolated lane change path and
        returns aggregate risk metrics.

        Args:
            ego_s: Current s position [m]
            ego_ey: Current lateral position [m]
            target_ey: Target lateral position [m]
            path_index: Path index
            n_samples: Number of samples along trajectory
            lookahead: Lookahead distance [m]

        Returns:
            max_risk: Maximum risk along trajectory
            mean_risk: Mean risk along trajectory
            risk_profile: Full risk profile array
        """
        # Sample points along lane change (linear interpolation)
        ey_samples = np.linspace(ego_ey, target_ey, n_samples)
        s_samples = np.linspace(ego_s, ego_s + lookahead, n_samples)

        risks = self.get_risk_frenet(s_samples, ey_samples, path_index)
        risks = np.asarray(risks)

        return float(np.max(risks)), float(np.mean(risks)), risks

    def get_decision_risk_score(self, ego_state, current_path, target_path,
                                 lane_width=3.5, lookahead=30.0):
        """
        Compute a risk score for decision making (lane change feasibility).

        Args:
            ego_state: [vx, vy, w, s, ey, epsi] or similar
            current_path: Current path index
            target_path: Target path index
            lane_width: Lane width for ey calculation
            lookahead: Lookahead distance

        Returns:
            risk_score: Combined risk score (higher = more dangerous)
            details: Dict with breakdown
        """
        # Extract ego state
        if len(ego_state) >= 5:
            ego_s = ego_state[3]
            ego_ey = ego_state[4]
        else:
            ego_s = 0.0
            ego_ey = 0.0

        # Compute target ey (relative to current path)
        delta_path = target_path - current_path
        target_ey = ego_ey + delta_path * lane_width

        # Evaluate lane change risk
        max_risk, mean_risk, profile = self.evaluate_lane_change_risk(
            ego_s, ego_ey, target_ey, current_path, lookahead=lookahead
        )

        # Combined score (weighted)
        risk_score = 0.6 * max_risk + 0.4 * mean_risk

        details = {
            'max_risk': max_risk,
            'mean_risk': mean_risk,
            'risk_profile': profile,
            'ego_s': ego_s,
            'ego_ey': ego_ey,
            'target_ey': target_ey,
        }

        return float(risk_score), details

    # =====================================================================
    # PATH REGISTRATION
    # =====================================================================

    def register_path(self, path_index, path_obj):
        """
        Register a path object for Frenet coordinate conversion.

        The path object must have:
        - get_cartesian_coords(s, ey) -> (x, y)
        - get_theta_r(s) -> heading angle
        - __call__(s) -> (x, y) for center point

        Args:
            path_index: Integer index (0=left, 1=center, 2=right)
            path_obj: Path object with required methods
        """
        self.path_funcs[path_index] = path_obj

    def register_paths(self, paths_dict):
        """
        Register multiple paths at once.

        Args:
            paths_dict: {index: path_obj, ...}
        """
        for idx, path in paths_dict.items():
            self.register_path(idx, path)

    # =====================================================================
    # RL QUERY HELPERS (Stage 2 additions)
    # =====================================================================

    def get_risk_corridor(self, x_start: float, y_lane: float, heading: float,
                          length: float = 25.0, n_samples: int = 6) -> float:
        """
        Return the maximum risk along a forward corridor centred on a lane.

        Samples `n_samples` points spaced evenly from x_start to
        x_start + length, all at lateral position y_lane.  The heading
        argument is reserved for future curved-road support; for the
        straight-highway scenario pass 0.0.

        Args:
            x_start  : Longitudinal start of the corridor [m]
            y_lane   : Lateral position (lane centre y) [m]
            heading  : Road heading at this point [rad] (unused for straight road)
            length   : Corridor look-ahead length [m]
            n_samples: Number of sample points

        Returns:
            max_risk : Maximum R along the corridor (scalar float)
        """
        if self._interpolator is None:
            return 0.0

        # Uniform samples along x, constant y
        xs = np.linspace(x_start, x_start + length, n_samples)
        ys = np.full(n_samples, y_lane)
        points = np.column_stack([ys, xs])  # (y, x) convention
        vals = self._interpolator(points)
        return float(np.max(vals))

    def get_risk_gradient(self, x: float, y: float) -> tuple:
        """
        Return (dR/dx, dR/dy) at a Cartesian point.  Thin alias over
        get_risk_gradient_cartesian for convenience in RL observation code.

        Args:
            x, y: Cartesian position [m]

        Returns:
            (grad_x, grad_y): Risk gradient components (floats)
        """
        return self.get_risk_gradient_cartesian(x, y)

    # =====================================================================
    # PROPERTIES
    # =====================================================================

    @property
    def risk_field(self):
        """Get current risk field (2D array)."""
        return self.solver.R.copy()

    @property
    def grid(self):
        """Get grid coordinates (X, Y meshgrids)."""
        return self.X.copy(), self.Y.copy()

    @property
    def grid_bounds(self):
        """Get grid bounds as dict."""
        return {
            'x_min': cfg.x_min, 'x_max': cfg.x_max,
            'y_min': cfg.y_min, 'y_max': cfg.y_max,
            'nx': cfg.nx, 'ny': cfg.ny,
        }
