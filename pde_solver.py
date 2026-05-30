"""
PDE Risk Field Solver
Advection-Diffusion-Telegrapher Equation for Traffic Risk Propagation

PDE: τ∂²R/∂t² + ∂R/∂t + ∇·(vR) = ∇·(D∇R) + Q(x,t) - λR

Based on Zhang et al. (2021) GVF formulation extended with:
- Occlusion-induced hazard sources
- Merge topology drift
- Telegrapher (damped wave) dynamics
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from config import Config as cfg


# ============================================================
# VEHICLE UTILITIES
# ============================================================

def create_vehicle(vid, x, y, vx, vy, vclass='car'):
    """
    Create a vehicle dictionary.
    
    Args:
        vid: Vehicle ID
        x, y: Position [m]
        vx, vy: Velocity [m/s]
        vclass: 'car' or 'truck'
    
    Returns:
        dict: Vehicle info
    """
    length = cfg.truck_length if vclass == 'truck' else cfg.car_length
    width = cfg.truck_width if vclass == 'truck' else cfg.car_width
    heading = np.arctan2(vy, vx) if (vx != 0 or vy != 0) else 0
    
    return {
        'id': vid,
        'x': x, 'y': y,
        'vx': vx, 'vy': vy,
        'heading': heading,
        'class': vclass,
        'length': length,
        'width': width
    }


def move_vehicle(v, dt, ax=0, ay=0):
    """
    Move vehicle forward in time with optional acceleration.
    
    Args:
        v: Vehicle dict
        dt: Time step [s]
        ax, ay: Acceleration [m/s²]
    
    Returns:
        dict: Updated vehicle
    """
    if v is None:
        return None
    
    new_x = v['x'] + v['vx'] * dt + 0.5 * ax * dt**2
    new_y = v['y'] + v['vy'] * dt + 0.5 * ay * dt**2
    new_vx = v['vx'] + ax * dt
    new_vy = v['vy'] + ay * dt
    
    return create_vehicle(v['id'], new_x, new_y, new_vx, new_vy, v['class'])


# ============================================================
# SOURCE TERM Q(x,t)
# ============================================================

def compute_Q_vehicle(vehicles, ego, X, Y):
    """
    Compute vehicle-induced risk source using GVF-style Gaussian kernels.

    Q_veh(x,t) = Σ_i w_i(t) * N(x; x_i(t), Σ_i(t))

    ENHANCED: Added braking effect - vehicles decelerating ahead create elevated risk.

    Args:
        vehicles: List of vehicle dicts
        ego: Ego vehicle dict
        X, Y: Meshgrid coordinates

    Returns:
        ndarray: Vehicle source field Q_veh
    """
    Q = np.zeros_like(X)

    ego_vx, ego_vy = ego['vx'], ego['vy']

    for v in vehicles:
        if v is None or v['id'] == ego['id']:
            continue

        # Relative velocity (determines risk weight)
        rel_vx = v['vx'] - ego_vx
        rel_vy = v['vy'] - ego_vy
        rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)

        # Distance to ego
        dx = v['x'] - ego['x']
        dy = v['y'] - ego['y']
        dist = np.sqrt(dx**2 + dy**2)

        # Directional component: is vehicle ahead of ego?
        ego_heading = np.arctan2(ego_vy, ego_vx) if (ego_vx != 0 or ego_vy != 0) else 0
        to_vehicle_angle = np.arctan2(dy, dx)
        angle_diff = np.abs(np.arctan2(np.sin(to_vehicle_angle - ego_heading),
                                        np.cos(to_vehicle_angle - ego_heading)))
        is_ahead = angle_diff < np.pi / 2  # Within ±90° of ego heading

        # Risk weight: higher for closer vehicles with higher relative speed
        # w_i = ω_class * ω_dist * ω_rel * ω_brake
        omega_class = 2.5 if v['class'] == 'truck' else 2.0  # INCREASED: truck 3.5, car 2.5
        omega_dist = np.exp(-dist / 70.0)  # Even longer range (60→70m)
        omega_rel = 1.5 + rel_speed / 3.5  # Stronger relative speed effect

        # BRAKING EFFECT: Create STRONG forward-propagating risk wave
        omega_brake = 3.0
        sigma_brake_boost = 1.0  # Additional kernel expansion for braking
        if 'a' in v and is_ahead:  # If acceleration data available and vehicle is ahead
            accel = v['a']
            if accel < -0.3:  # Even light braking (0.3 m/s²) triggers
                # STRONG amplification: up to 6x for emergency braking
                brake_intensity = abs(accel)
                if brake_intensity > 2.0:  # Emergency braking
                    omega_brake = 1.0 + 5.0  # 6x total
                    sigma_brake_boost = 2.5  # Huge forward spread
                elif brake_intensity > 1.0:  # Moderate braking
                    omega_brake = 1.0 + 3.0  # 4x total
                    sigma_brake_boost = 1.8
                else:  # Light braking
                    omega_brake = 1.0 + 1.5  # 2.5x total
                    sigma_brake_boost = 1.3

        weight = omega_class * omega_dist * omega_rel * omega_brake

        # Anisotropic Gaussian kernel (aligned with vehicle heading)
        heading = v['heading']
        cos_h, sin_h = np.cos(heading), np.sin(heading)

        # Rotate coordinates to vehicle-aligned frame
        dX = X - v['x']
        dY = Y - v['y']
        dX_rot = cos_h * dX + sin_h * dY
        dY_rot = -sin_h * dX + cos_h * dY

        # Kernel parameters (inflate with relative speed AND braking)
        # For braking vehicles, create elongated forward wake
        sigma_par = cfg.sigma_x * (1 + 0.05 * np.abs(rel_vx)) * sigma_brake_boost
        sigma_perp = cfg.sigma_y

        # Gaussian kernel
        gaussian = np.exp(-0.5 * (dX_rot**2 / sigma_par**2 + dY_rot**2 / sigma_perp**2))

        Q += weight * gaussian

        # CRITICAL FIX: Add "closing speed" risk in approach zone
        # If ego is BEHIND vehicle and CATCHING UP, create elevated risk in the gap
        if is_ahead and rel_vx < -1.0:  # Ego approaching (closing speed > 1 m/s)
            closing_speed = abs(rel_vx)
            # Time to collision (rough estimate)
            ttc = max(0.1, dist / closing_speed)

            # Create "approach corridor" risk between ego and vehicle
            # This fills the SPACE BETWEEN with elevated risk
            approach_weight = omega_class * 3.0 * np.exp(-ttc / 3.0)  # High risk if TTC < 3s

            # Create elongated Gaussian along line of approach
            # Extends BACKWARD from vehicle toward ego
            dx_approach = X - v['x']
            dy_approach = Y - v['y']

            # Project onto vehicle heading (backward direction)
            proj_along = -(cos_h * dx_approach + sin_h * dy_approach)  # Negative = behind vehicle
            proj_perp = np.abs(-sin_h * dx_approach + cos_h * dy_approach)

            # Only apply in approach corridor (behind vehicle, aligned with heading)
            in_corridor = (proj_along > 0) & (proj_along < dist * 1.5) & (proj_perp < 4.0)
            approach_gaussian = np.exp(-0.5 * (proj_along**2 / (dist/2)**2 + proj_perp**2 / 4))

            Q += approach_weight * approach_gaussian * in_corridor.astype(float)

    return Q


def compute_Q_occlusion(vehicles, ego, X, Y):
    """
    Compute occlusion-induced latent hazard source.

    Q_occ(x,t) = 1_{x ∈ Ω_occ} * ω_sev * p_emerge(x,t)

    Shadow cone is aligned with the truck HEADING (road tangent direction)
    so it follows the road on curves, not the ego-to-truck chord.

    Args:
        vehicles: List of vehicle dicts (trucks create occlusion)
        ego: Ego vehicle dict
        X, Y: Meshgrid coordinates

    Returns:
        Q_occ: Occlusion source field
        occ_mask: Boolean mask of occluded regions
    """
    Q_occ = np.zeros_like(X)
    occ_mask = np.zeros_like(X, dtype=bool)

    for v in vehicles:
        if v is None or v['class'] != 'truck':
            continue

        dist_to_truck = np.sqrt((v['x'] - ego['x'])**2 + (v['y'] - ego['y'])**2)
        if dist_to_truck < 1:
            continue

        dx = X - v['x']
        dy = Y - v['y']

        # Use truck HEADING (road-tangent) for both longitudinal and lateral axes.
        # This keeps the shadow cone aligned with the road on curves.
        heading = v['heading']
        cos_h, sin_h = np.cos(heading), np.sin(heading)

        # Longitudinal distance along truck heading (positive = ahead on road)
        along_road = cos_h * dx + sin_h * dy

        # Lateral distance from truck's heading axis
        lateral_dist = np.abs(-sin_h * dx + cos_h * dy)

        dist_from_truck = np.sqrt(dx**2 + dy**2)

        # Shadow cone: extends AHEAD of truck along road direction
        # Width starts at truck half-width and expands at 0.3 per meter
        shadow_width = v['width'] * 0.5 + dist_from_truck * 0.3
        in_shadow = ((along_road > 0) &
                     (lateral_dist < shadow_width) &
                     (dist_from_truck < 60))

        occ_mask |= in_shadow

        # Occlusion risk: decays with distance from occluding vehicle
        p_emerge = np.exp(-dist_from_truck / 30)
        Q_occ += in_shadow.astype(float) * 5.0 * p_emerge

    return Q_occ, occ_mask


def compute_Q_merge(vehicles, ego, X, Y):
    """
    Compute merge-zone conflict source, gated by vehicle density.

    Q_merge = ρ_vehicles(x) * [k_merge * ρ_merge(x)]

    Only injects risk where vehicles are actually present.

    Args:
        vehicles: List of vehicle dicts
        ego: Ego vehicle dict
        X, Y: Meshgrid coordinates

    Returns:
        ndarray: Merge zone source field (gated by vehicle density)
    """
    # Merge zone geometry (base risk pattern)
    s = np.clip((X - cfg.merge_x_start) / (cfg.merge_x_end - cfg.merge_x_start), 0, 1)
    ramp = 3 * s**2 - 2 * s**3  # Smooth onset g(s)

    # Lateral: concentrated between ramp and mainline
    y_center = cfg.merge_y_ramp / 2
    lateral = np.exp(-0.5 * ((Y - y_center)**2 / 16))

    # Gore point intensity
    gore = np.exp(-((X - cfg.merge_x_end)**2 + (Y - 4)**2) / 100)

    Q_merge_base = 0.6 * ramp * lateral + 1.0 * gore  # INCREASED from 0.3/0.5 to 0.6/1.0

    # NEW: Compute vehicle density field (only inject risk where vehicles present)
    rho_vehicles = np.zeros_like(X)
    for v in vehicles:
        if v is None or v['id'] == ego['id']:
            continue
        # Gaussian density around each vehicle (30m x 4m influence region)
        dist_sq = ((X - v['x'])**2 / 900 + (Y - v['y'])**2 / 16)
        rho_vehicles += np.exp(-0.5 * dist_sq)

    # Normalize and gate
    rho_norm = np.clip(rho_vehicles, 0, 1.0)

    # Only inject merge risk where vehicles are present
    Q_merge = Q_merge_base * rho_norm

    return Q_merge


def compute_total_Q(vehicles, ego, X, Y):
    """
    Compute total source term Q = Q_veh + Q_occ + Q_merge.

    Args:
        vehicles: List of vehicle dicts
        ego: Ego vehicle dict
        X, Y: Meshgrid coordinates

    Returns:
        Q_total: Total source field
        Q_veh: Vehicle source
        Q_occ: Occlusion source
        occ_mask: Occlusion mask
    """
    Q_veh = compute_Q_vehicle(vehicles, ego, X, Y)
    Q_occ, occ_mask = compute_Q_occlusion(vehicles, ego, X, Y)
    Q_merge = compute_Q_merge(vehicles, ego, X, Y)  # Now uses vehicle density

    # Road mask boundary condition now prevents off-road diffusion,
    # so no need to scale down Q_veh. Full source strength maintained.
    Q_total = Q_veh + Q_occ + Q_merge

    return Q_total, Q_veh, Q_occ, occ_mask


# ============================================================
# VELOCITY FIELD v_eff = v_flow + v_topo
# ============================================================

def compute_velocity_field(vehicles, ego, X, Y):
    """
    Compute effective velocity field for advection.

    v_eff = v_flow (GVF) + v_topo (merge drift)

    For WORLD-FIXED frame: Uses absolute velocities so risk propagates with traffic.

    Args:
        vehicles: List of vehicle dicts
        ego: Ego vehicle dict (included for compatibility, not used for world frame)
        X, Y: Meshgrid coordinates

    Returns:
        vx, vy: Effective velocity components
        vx_flow, vy_flow: GVF flow field
        vx_topo, vy_topo: Topology drift field
    """
    # ---- GVF-style flow field ----
    vx_flow = np.zeros_like(X)
    vy_flow = np.zeros_like(Y)
    total_weight = np.zeros_like(X) + 1e-6

    for v in vehicles:
        if v is None:
            continue

        # ABSOLUTE velocity (for world-fixed frame)
        abs_vx = v['vx']
        abs_vy = v['vy']

        # Gaussian influence kernel (aligned with vehicle heading)
        dx = X - v['x']
        dy = Y - v['y']
        heading = v.get('heading', np.arctan2(abs_vy, abs_vx + 1e-8))
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dx_rot = cos_h * dx + sin_h * dy    # longitudinal
        dy_rot = -sin_h * dx + cos_h * dy   # lateral
        dist_sq = dx_rot**2 / 400 + dy_rot**2 / 9  # 20m along heading, 3m across
        weight = np.exp(-0.5 * dist_sq)

        vx_flow += weight * abs_vx
        vy_flow += weight * abs_vy
        total_weight += weight

    vx_flow = vx_flow / total_weight
    vy_flow = vy_flow / total_weight
    
    # ---- Topology drift (merge zone) ----
    # Smooth onset function
    s = np.clip((X - cfg.merge_x_start) / (cfg.merge_x_end - cfg.merge_x_start), 0, 1)
    urgency = s**2 * (3 - 2*s)
    
    # Only apply in merge region
    in_merge = (Y > 2) & (X > cfg.merge_x_start) & (X < cfg.merge_x_end)
    
    vx_topo = np.zeros_like(X)
    vy_topo = -2.0 * urgency * in_merge.astype(float)  # Drift toward mainline
    
    # Combine
    vx = vx_flow + vx_topo
    vy = vy_flow + vy_topo
    
    return vx, vy, vx_flow, vy_flow, vx_topo, vy_topo


def compute_diffusion_field(occ_mask, X, Y, vehicles=None, ego=None):
    """
    Compute spatially-varying diffusion coefficient.

    D(x,t) = D_0 + D_occ * 1_{x ∈ Ω_occ} + D_brake * 1_{x near braking vehicle}

    Higher diffusion in:
    - Occluded regions (uncertainty from hidden hazards)
    - Around braking vehicles (forward-propagating warning wave)

    Args:
        occ_mask: Boolean mask of occluded regions
        X, Y: Meshgrid coordinates
        vehicles: List of vehicle dicts (optional, for braking detection)
        ego: Ego vehicle dict (optional)

    Returns:
        D: Diffusion coefficient field
    """
    D = cfg.D0 * np.ones_like(X)
    D[occ_mask] += cfg.D_occ

    # ENHANCED: Add diffusion boost around braking vehicles
    if vehicles is not None and ego is not None:
        for v in vehicles:
            if v is None or v['id'] == ego['id']:
                continue

            # Check if vehicle is braking
            if 'a' in v and v['a'] < -0.3:  # Braking detected
                brake_intensity = abs(v['a'])
                # Create diffusion zone ahead of braking vehicle (toward followers)
                dist_sq = (X - v['x'])**2 + (Y - v['y'])**2
                # Larger zone for harder braking (up to 40m radius for emergency)
                D_brake_boost = 15.0 * np.exp(-dist_sq / (400 + 200 * brake_intensity))
                D += D_brake_boost

    D = gaussian_filter(D, sigma=1.0)  # Smooth transitions
    return D


# ============================================================
# PDE SOLVER
# ============================================================

class PDESolver:
    """
    Solver for the advection-diffusion-telegrapher PDE.

    τ∂²R/∂t² + ∂R/∂t + ∇·(vR) = ∇·(D∇R) + Q - λR

    Road boundary condition (Dirichlet):
        R(x,t) = 0,  ∀x ∉ Ω_road

    Enforced via a smooth road mask M(x):
        R^{n+1}(x) ← R^{n+1}(x) · M(x)

    where M(x) = 1 inside road, tapering to 0 at road edges.
    This confines risk propagation to the drivable corridor and
    prevents spurious diffusion into off-road regions.

    Attributes:
        R: Current risk field
        R_t: Time derivative (for telegrapher term)
        road_mask: Smooth mask [0,1] — 1 on road, 0 off road (None = no masking)
    """

    def __init__(self):
        self.X, self.Y = cfg.X, cfg.Y
        self.dx, self.dy = cfg.dx, cfg.dy
        self.R = np.zeros_like(self.X)
        self.R_t = np.zeros_like(self.X)
        self.road_mask = None  # Set via set_road_mask() to enable boundary

    def set_road_mask(self, mask):
        """
        Set road boundary mask for Dirichlet BC: R=0 off-road.

        The mask should be a 2D array (same shape as X, Y) with values
        in [0, 1]. Typically 1.0 on-road, smoothly tapering to 0 at edges.

        Mathematically, this enforces:
            R(x, t) · M(x) = 0  for x ∉ Ω_road

        where M is the indicator/taper function of the road domain.

        Args:
            mask: 2D array matching grid shape, values in [0,1]
        """
        self.road_mask = mask

    def reset(self):
        """Reset risk field to zero."""
        self.R = np.zeros_like(self.X)
        self.R_t = np.zeros_like(self.X)

    def step(self, Q, D, vx, vy, dt=0.05, tau=None):
        """
        Advance PDE one time step.

        Uses operator splitting with fixes:
        1. Source + Flow-Dependent Decay (λ = λ₀ + |v|/L_decay)
        2. Proper Variable Diffusion (∇·(D∇R) with gradient term)
        3. Conservative Advection (flux form ∇·(vR))
        4. Telegrapher damping (consistent update)

        Args:
            Q: Source field
            D: Diffusion coefficient field
            vx, vy: Velocity field components
            dt: Time step [s]
            tau: Telegrapher inertia (None uses config value)

        Returns:
            R: Updated risk field
        """
        if tau is None:
            tau = cfg.tau

        R = self.R.copy()
        R_t = self.R_t.copy()

        # ===================================================================
        # STEP 1: Source + Flow-Dependent Decay
        # ===================================================================
        # Compute spatially-varying decay: λ(x,t) = λ₀ + |v(x,t)|/L_decay
        speed = np.sqrt(vx**2 + vy**2)
        lambda_field = cfg.lambda_decay + speed / cfg.L_decay

        # Add sponge layer at downstream boundary (right edge)
        if hasattr(cfg, 'sponge_length') and cfg.sponge_length > 0:
            x_max = self.X[:, -1].mean()  # Right boundary
            x_sponge_start = x_max - cfg.sponge_length
            w_sponge = np.clip((self.X - x_sponge_start) / cfg.sponge_length, 0, 1)
            lambda_field = lambda_field + cfg.lambda_sponge * w_sponge**2

        # Apply source and decay
        R = R + dt * (Q - lambda_field * R)

        # ===================================================================
        # STEP 2: Proper Variable Diffusion ∇·(D∇R) = D∇²R + ∇D·∇R
        # ===================================================================
        # Compute gradients of R (central differences)
        dR_dx = np.zeros_like(R)
        dR_dy = np.zeros_like(R)
        dR_dx[:, 1:-1] = (R[:, 2:] - R[:, :-2]) / (2 * self.dx)
        dR_dy[1:-1, :] = (R[2:, :] - R[:-2, :]) / (2 * self.dy)

        # Compute gradients of D
        dD_dx = np.zeros_like(D)
        dD_dy = np.zeros_like(D)
        dD_dx[:, 1:-1] = (D[:, 2:] - D[:, :-2]) / (2 * self.dx)
        dD_dy[1:-1, :] = (D[2:, :] - D[:-2, :]) / (2 * self.dy)

        # Laplacian of R
        laplacian = np.zeros_like(R)
        laplacian[1:-1, 1:-1] = (
            (R[1:-1, 2:] - 2*R[1:-1, 1:-1] + R[1:-1, :-2]) / self.dx**2 +
            (R[2:, 1:-1] - 2*R[1:-1, 1:-1] + R[:-2, 1:-1]) / self.dy**2
        )

        # Full variable diffusion: ∇·(D∇R) = D∇²R + ∇D·∇R
        diffusion_term = D * laplacian + dD_dx * dR_dx + dD_dy * dR_dy

        # Apply
        R = R + dt * diffusion_term

        # ===================================================================
        # STEP 3: Conservative Advection -∇·(vR) using flux form
        # ===================================================================
        # Compute fluxes: F_x = vx * R, F_y = vy * R
        F_x = vx * R
        F_y = vy * R

        # Divergence of flux using upwind scheme
        div_flux = np.zeros_like(R)

        # ∂F_x/∂x (upwind)
        div_flux[:, 1:-1] += np.where(
            vx[:, 1:-1] > 0,
            (F_x[:, 1:-1] - F_x[:, :-2]) / self.dx,
            (F_x[:, 2:] - F_x[:, 1:-1]) / self.dx
        )

        # ∂F_y/∂y (upwind)
        div_flux[1:-1, :] += np.where(
            vy[1:-1, :] > 0,
            (F_y[1:-1, :] - F_y[:-2, :]) / self.dy,
            (F_y[2:, :] - F_y[1:-1, :]) / self.dy
        )

        # Update R
        R = R - dt * div_flux

        # ===================================================================
        # STEP 4: Telegrapher term (wave-like propagation with inertia)
        # ===================================================================
        if tau > 0:
            # Telegrapher equation as first-order system:
            # ∂R/∂t = R_t  (field evolves with velocity R_t)
            # ∂R_t/∂t = (RHS - R_t)/τ  (velocity relaxes to RHS with time constant τ)

            # Compute full RHS using updated R from steps 1-3
            full_rhs = Q - lambda_field * R + diffusion_term - div_flux

            # Steps 1-3 gave us: R_direct = R_old + dt * full_rhs
            # But with telegrapher, we need: R = R_old + dt * R_t_new

            # Store the "instantaneous" update for comparison
            R_instantaneous = R.copy()

            # Update telegrapher velocity toward full_rhs
            R_t = R_t + (dt / tau) * (full_rhs - R_t)

            # Apply telegrapher: replace instantaneous update with damped evolution
            # R should evolve according to R_t, not instantaneously
            R = self.R + dt * R_t

        # ===================================================================
        # Post-processing
        # ===================================================================
        # Enforce non-negativity
        R = np.clip(R, 0, 10)

        # Grid boundary conditions (zero flux at grid edges)
        R[0, :] = R[1, :]
        R[-1, :] = R[-2, :]
        R[:, 0] = R[:, 1]
        R[:, -1] = R[:, -2]

        # Road boundary condition (Dirichlet): R = 0 outside road
        # R^{n+1}(x) = R^{n+1}(x) · M(x)  where M=0 off-road
        if self.road_mask is not None:
            R *= self.road_mask
            if tau > 0:
                R_t *= self.road_mask

        # Optional smoothing (disabled by default)
        if hasattr(cfg, 'post_smooth_sigma') and cfg.post_smooth_sigma > 0:
            R = gaussian_filter(R, sigma=cfg.post_smooth_sigma)

        self.R = R
        self.R_t = R_t if tau > 0 else np.zeros_like(R)

        return self.R.copy()
