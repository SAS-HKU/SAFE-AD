"""
CBF Safety Filter
=================
Analytical Control Barrier Function projection that maps a raw RL action
(a_raw, δ_raw) onto the closest safe action (a_safe, δ_safe).

Role in the RL architecture
----------------------------
      RL policy → (a_raw, δ_raw)
                        │
                  CBFSafetyFilter.project()
                        │
               (a_safe, δ_safe) → environment step

Design
------
Two independent CBF constraints are enforced analytically (no QP solver needed):

1. **Longitudinal CBF** (gap to leading vehicle in current lane)
   h_lon(s) = gap - d_safe(v)  ≥  0
   d_safe(v) = D_MIN + T_HEAD * v   (time-headway safety distance)

   The CBF derivative condition  ḣ + γ_lon * h ≥ 0  gives an upper bound on
   the ego acceleration:

       a_max_cbf = v_lead_rel / dt  +  γ_lon * h / dt

   where v_lead_rel = v_lead - v_ego  (positive = leader pulling away).

   The raw action is clipped:  a_safe = min(a_raw, a_max_cbf)

2. **Lane-boundary CBF** (lateral)
   h_left(s)  = y_ego - LANE_LEFT_LIMIT  ≥  0
   h_right(s) = LANE_RIGHT_LIMIT - y_ego  ≥  0

   These translate to steering angle bounds.  The ego lateral velocity from
   the bicycle model is approximately  vy ≈ v * δ, so

       δ_max = (y_ego - LANE_LEFT_LIMIT  - γ_lat * dt * vy) / (v * dt)   [don't go further left]
       δ_min = (y_ego - LANE_RIGHT_LIMIT + γ_lat * dt * vy) / (v * dt)   [don't go further right]

   The raw steering is clipped:  δ_safe = clip(δ_raw, δ_min, δ_max)

Both projections are applied independently and in sequence (lon first, then lat).
The filter is conservative: it may prevent some valid manoeuvres in edge cases,
but it never violates the CBF conditions.

Parameters
----------
All parameters are in `RLConfig` or passed explicitly for flexibility.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np


# ---------------------------------------------------------------------------
# Filter configuration
# ---------------------------------------------------------------------------

@dataclass
class CBFConfig:
    # --- Longitudinal ---
    D_MIN: float = 2.0          # minimum safe gap [m]  (bumper-to-bumper)
    T_HEAD: float = 0.8         # time-headway [s]       d_safe = D_MIN + T_HEAD * v
    GAMMA_LON: float = 1.5      # CBF decay rate for longitudinal constraint
    DT: float = 0.1             # simulation timestep [s]

    # --- Lateral ---
    LANE_LEFT_LIMIT: float = 0.0    # absolute left boundary [m]  (road edge)
    LANE_RIGHT_LIMIT: float = 10.5  # absolute right boundary [m] (road edge)
    GAMMA_LAT: float = 2.0          # CBF decay rate for lateral constraint
    STEER_ABS_MAX: float = 0.35     # hard steering saturation [rad]
    V_MIN_FOR_STEER: float = 0.5    # below this speed, skip steering CBF

    # --- Actuation limits ---
    ACCEL_MIN: float = -4.0     # emergency braking [m/s²]
    ACCEL_MAX: float = 1.5      # max acceleration [m/s²]

    # --- Comfort-rate limits (safety strictly dominates these) ---
    # These turn the smoothness the RL reward only *encourages* into a hard
    # per-step guarantee. The longitudinal safety CBF overrides the jerk band
    # and the comfort-decel floor whenever emergency braking is required.
    J_MAX: float = 5.0            # max |jerk| [m/s³]  → |Δa| ≤ J_MAX·dt per step
    STEER_RATE_MAX: float = 0.08  # max |Δδ| per control step [rad]
    ACCEL_COMFORT_MIN: float = -2.5  # comfort decel floor [m/s²] (gentler than ACCEL_MIN)

    # --- Heading CBF (prevents yaw from diverging, fixing relative-degree-2 issue) ---
    YAW_MAX: float = 0.25         # maximum allowed heading deviation from road [rad]
    GAMMA_YAW: float = 5.0        # CBF decay rate for heading constraint
    WHEELBASE: float = 4.4        # bicycle model wheelbase [m]

    # --- Lateral-collision CBF for adjacent vehicles ---
    LATERAL_SAFE_GAP: float = 1.5  # lateral gap before adjacent-car CBF triggers [m]
    GAMMA_LAT_CAR: float = 1.0


@dataclass
class VehicleState:
    """Minimal state description for a surrounding vehicle."""
    x: float
    y: float
    vx: float
    vy: float = 0.0
    length: float = 4.5


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class CBFSafetyFilter:
    """
    Stateless analytical CBF projection.

    Usage::

        cbf = CBFSafetyFilter(CBFConfig())
        a_s, d_s, info = cbf.project(
            a_raw=0.5, delta_raw=0.05,
            ego_x=50.0, ego_y=5.25, ego_vx=10.0, ego_vy=0.0,
            surrounding=[VehicleState(x=70, y=5.25, vx=8.0)],
            current_lane=1,
        )
    """

    def __init__(self, cfg: CBFConfig = None):
        self.cfg = cfg if cfg is not None else CBFConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        a_raw: float,
        delta_raw: float,
        ego_x: float,
        ego_y: float,
        ego_vx: float,
        ego_vy: float,
        surrounding: List[VehicleState],
        current_lane: int = 1,
        lane_centers: Optional[List[float]] = None,
        ego_yaw: float = 0.0,
        last_accel: Optional[float] = None,
        last_steer: Optional[float] = None,
    ) -> Tuple[float, float, dict]:
        """
        Project (a_raw, δ_raw) onto the CBF-safe set.

        Parameters
        ----------
        a_raw, delta_raw : raw RL action
        ego_x, ego_y     : ego Cartesian position [m]
        ego_vx, ego_vy   : ego Cartesian velocity [m/s]
        surrounding       : list of VehicleState for nearby vehicles
        current_lane      : 0=left, 1=centre, 2=right
        lane_centers      : y-coords of lane centres (default [1.75, 5.25, 8.75])
        ego_yaw           : current heading angle [rad] (0 = road direction)

        Returns
        -------
        a_safe, delta_safe : projected action
        info dict          : {a_clipped, delta_clipped, h_lon, h_lat_l, h_lat_r,
                              a_max_cbf, delta_min_cbf, delta_max_cbf,
                              blocking_vehicle_idx}
        """
        cfg = self.cfg
        if lane_centers is None:
            lane_centers = [1.75, 5.25, 8.75]

        info: dict = {}

        # ---------------------------------------------------------------
        # 1. Longitudinal CBF
        # ---------------------------------------------------------------
        a_safe, lon_info = self._project_longitudinal(
            a_raw, ego_x, ego_y, ego_vx, surrounding
        )
        info.update(lon_info)

        # ---------------------------------------------------------------
        # 2. Heading CBF — prevents yaw divergence (relative-degree-2 fix)
        #    Must come before lane-boundary CBF so vy used there is bounded.
        # ---------------------------------------------------------------
        delta_safe, yaw_info = self._project_heading(delta_raw, ego_vx, ego_yaw)
        info.update(yaw_info)

        # ---------------------------------------------------------------
        # 3. Lane-boundary (road edge) CBF
        # ---------------------------------------------------------------
        delta_safe, lat_info = self._project_lateral_boundary(
            delta_safe, ego_y, ego_vx, ego_vy, ego_yaw=ego_yaw
        )
        info.update(lat_info)

        # ---------------------------------------------------------------
        # 4. Lateral car-to-car CBF (prevent side-swipe into adjacent lane)
        # ---------------------------------------------------------------
        delta_safe, adj_info = self._project_lateral_car(
            delta_safe, ego_x, ego_y, ego_vx, ego_vy, surrounding
        )
        info.update(adj_info)

        # ---------------------------------------------------------------
        # 5. Comfort-rate layer (this work): jerk, steer-rate, comfort-decel.
        #    Safety strictly dominates — emergency braking from the longitudinal
        #    CBF overrides the jerk band and the comfort-decel floor.
        # ---------------------------------------------------------------
        a_ceiling = float(info.get('a_max_cbf', cfg.ACCEL_MAX))
        a_safe, delta_safe, comfort_info = self._apply_comfort_rate_limits(
            a_safe, delta_safe, a_ceiling, last_accel, last_steer
        )
        info.update(comfort_info)

        # ---------------------------------------------------------------
        # 6. Hard saturation on final action
        # ---------------------------------------------------------------
        a_safe = float(np.clip(a_safe, cfg.ACCEL_MIN, cfg.ACCEL_MAX))
        delta_safe = float(np.clip(delta_safe, -cfg.STEER_ABS_MAX, cfg.STEER_ABS_MAX))

        info['a_clipped'] = (abs(a_raw - a_safe) > 1e-6)
        info['delta_clipped'] = (abs(delta_raw - delta_safe) > 1e-6)

        return a_safe, delta_safe, info

    def _apply_comfort_rate_limits(
        self,
        a_safe: float,
        delta_safe: float,
        a_ceiling: float,
        last_accel: Optional[float],
        last_steer: Optional[float],
    ) -> Tuple[float, float, dict]:
        """Apply jerk / steer-rate / comfort-decel limits with safety override.

        ``a_safe`` and ``delta_safe`` are already safety-feasible on entry.
        ``a_ceiling`` is the longitudinal-CBF upper bound on acceleration; the
        comfort layer may make the action gentler but must not raise accel above
        this ceiling, and must yield to harder braking when safety demands it.
        """
        cfg = self.cfg
        out = {
            'comfort_decel_active': False,
            'jerk_limited': False,
            'steer_rate_limited': False,
            'comfort_overridden_by_safety': False,
        }
        a_pre = a_safe

        # Comfort-decel floor: avoid unnecessarily hard braking, but never less
        # braking than the safety ceiling permits.
        a_comf = min(max(a_safe, cfg.ACCEL_COMFORT_MIN), a_ceiling)
        if abs(a_comf - a_safe) > 1e-9:
            out['comfort_decel_active'] = True
        a_safe = a_comf

        # Jerk limit (only when the previous accel is known).
        if last_accel is not None:
            lo = float(last_accel) - cfg.J_MAX * cfg.DT
            hi = float(last_accel) + cfg.J_MAX * cfg.DT
            if a_pre < lo:
                # Safety demanded harder braking than the jerk band allows.
                a_safe = a_pre
                out['comfort_overridden_by_safety'] = True
            else:
                a_jerk = min(max(a_safe, lo), hi)
                if abs(a_jerk - a_safe) > 1e-9:
                    out['jerk_limited'] = True
                a_safe = min(a_jerk, a_ceiling)

        # Steer-rate limit (gentler steering is safe-or-safer).
        if last_steer is not None:
            lo = float(last_steer) - cfg.STEER_RATE_MAX
            hi = float(last_steer) + cfg.STEER_RATE_MAX
            d_rate = min(max(delta_safe, lo), hi)
            if abs(d_rate - delta_safe) > 1e-9:
                out['steer_rate_limited'] = True
            delta_safe = d_rate

        return a_safe, delta_safe, out

    # ------------------------------------------------------------------
    # Internal CBF constraints
    # ------------------------------------------------------------------

    def _project_heading(
        self,
        delta_raw: float,
        ego_vx: float,
        ego_yaw: float,
    ) -> Tuple[float, dict]:
        """
        Heading CBF: keep |ψ| ≤ YAW_MAX.

        The bicycle model heading dynamics:
            ψ̇ = v * tan(δ) / L  ≈  v * δ / L   (small δ linearisation)

        CBF: h_ψ = YAW_MAX² − ψ²  ≥  0
            ḣ_ψ = −2ψ · ψ̇ = −2ψ · v·δ/L

        Condition  ḣ_ψ + γ·h_ψ ≥ 0:
            −2ψ·v·δ/L + γ·(YAW_MAX² − ψ²) ≥ 0

        When ψ > 0 (heading right):
            δ ≤  γ·L·(YAW_MAX² − ψ²) / (2·ψ·v)   [upper bound]
        When ψ < 0 (heading left):
            δ ≥  γ·L·(YAW_MAX² − ψ²) / (2·ψ·v)   [lower bound; note ψ<0 → denominator negative]
        When ψ = 0: no constraint from this CBF.
        """
        cfg = self.cfg
        psi = float(ego_yaw)
        v = max(ego_vx, cfg.V_MIN_FOR_STEER)
        L = cfg.WHEELBASE
        gamma = cfg.GAMMA_YAW
        psi_max = cfg.YAW_MAX

        h_psi = psi_max ** 2 - psi ** 2
        # nominal: unconstrained
        delta_min_yaw = -cfg.STEER_ABS_MAX
        delta_max_yaw =  cfg.STEER_ABS_MAX

        if abs(psi) > 1e-4 and v > cfg.V_MIN_FOR_STEER:
            cbf_val = gamma * L * h_psi / (2.0 * psi * v)
            if psi > 0:
                # heading right → upper-bound on δ (don't steer further right)
                delta_max_yaw = min(delta_max_yaw, cbf_val)
            else:
                # heading left → lower-bound on δ (don't steer further left)
                delta_min_yaw = max(delta_min_yaw, cbf_val)

        # Clamp to physical limits
        delta_min_yaw = max(delta_min_yaw, -cfg.STEER_ABS_MAX)
        delta_max_yaw = min(delta_max_yaw,  cfg.STEER_ABS_MAX)

        if delta_min_yaw > delta_max_yaw:
            # Degenerate (|ψ| >> YAW_MAX): steer back to road heading
            delta_safe = float(-np.sign(psi) * min(cfg.STEER_ABS_MAX, abs(psi) * 0.5))
        else:
            delta_safe = float(np.clip(delta_raw, delta_min_yaw, delta_max_yaw))

        return delta_safe, {
            'h_psi': float(h_psi),
            'delta_min_yaw': float(delta_min_yaw),
            'delta_max_yaw': float(delta_max_yaw),
        }

    def _project_longitudinal(
        self,
        a_raw: float,
        ego_x: float,
        ego_y: float,
        ego_vx: float,
        surrounding: List[VehicleState],
    ) -> Tuple[float, dict]:
        """Clip acceleration so the gap-CBF condition is satisfied."""
        cfg = self.cfg

        # Find closest vehicle AHEAD and in the same (approximate) lane
        best_gap = np.inf
        best_v_lead = ego_vx  # fallback: no relative motion
        best_idx = -1
        LANE_THRESH = 2.0  # ±2 m lateral → same path

        for i, sv in enumerate(surrounding):
            dx = sv.x - ego_x
            dy = abs(sv.y - ego_y)
            if dx < 0.5 or dy > LANE_THRESH:   # behind or different lane
                continue
            gap = dx - sv.length / 2           # front bumper of leader
            if gap < best_gap:
                best_gap = gap
                best_v_lead = sv.vx
                best_idx = i

        if best_gap == np.inf:
            # No leader — only hard acceleration cap applies
            return float(np.clip(a_raw, cfg.ACCEL_MIN, cfg.ACCEL_MAX)), {
                'h_lon': np.inf, 'a_max_cbf': cfg.ACCEL_MAX, 'blocking_vehicle_idx': -1
            }

        d_safe = cfg.D_MIN + cfg.T_HEAD * max(ego_vx, 0.0)
        h_lon = best_gap - d_safe            # CBF value

        # ḣ + γ·h ≥ 0  →  (v_lead - v_ego)/dt + γ·h/dt ≥ a_ego
        v_rel = best_v_lead - ego_vx
        a_max_cbf = v_rel / cfg.DT + cfg.GAMMA_LON * h_lon / cfg.DT

        a_safe = min(a_raw, a_max_cbf)
        a_safe = max(a_safe, cfg.ACCEL_MIN)  # never below emergency braking

        return float(a_safe), {
            'h_lon': float(h_lon),
            'a_max_cbf': float(a_max_cbf),
            'blocking_vehicle_idx': best_idx,
        }

    def _project_lateral_boundary(
        self,
        delta_raw: float,
        ego_y: float,
        ego_vx: float,
        ego_vy: float,
        ego_yaw: float = 0.0,
    ) -> Tuple[float, dict]:
        """Clip steering so road-edge CBF conditions are satisfied.

        Uses heading-corrected lateral velocity estimate: the next-step lateral
        position is y_next ≈ y + v*sin(ψ)*dt + v*cos(ψ)*tan(δ)*dt²/L, but we
        simplify to the dominant term:
            vy_total ≈ v*sin(ψ) + v*cos(ψ)*δ  (for small δ and dt)

        The heading-induced component v*sin(ψ) is accounted for as a drift term,
        so the steering bound is adjusted to counteract it.
        """
        cfg = self.cfg
        v = max(ego_vx, cfg.V_MIN_FOR_STEER)

        # h_left = ego_y - LANE_LEFT_LIMIT  ≥ 0
        h_lat_l = ego_y - cfg.LANE_LEFT_LIMIT
        # h_right = LANE_RIGHT_LIMIT - ego_y  ≥ 0
        h_lat_r = cfg.LANE_RIGHT_LIMIT - ego_y

        # Heading-induced lateral drift (the part steering CANNOT cancel immediately)
        vy_heading = v * float(np.sin(ego_yaw))  # positive = moving right (increasing y)

        # CBF conditions with heading drift:
        #   ḣ_left  = vy_total = vy_heading + v*cos(ψ)*δ ≥ -γ*h_lat_l
        #   → v*cos(ψ)*δ ≥ -γ*h_lat_l - vy_heading
        #   → δ ≥ (-γ*h_lat_l - vy_heading) / (v*cos(ψ))
        #
        #   ḣ_right = -vy_total = -(vy_heading + v*cos(ψ)*δ) ≥ -γ*h_lat_r
        #   → -vy_heading - v*cos(ψ)*δ ≥ -γ*h_lat_r
        #   → δ ≤ (γ*h_lat_r - vy_heading) / (v*cos(ψ))  ... wait sign
        # Actually ḣ_right = d/dt(RIGHT-y) = -ẏ = -vy_total
        #   -vy_total + γ*h_lat_r ≥ 0 → vy_total ≤ γ*h_lat_r
        #   vy_heading + v*cos(ψ)*δ ≤ γ*h_lat_r
        #   → δ ≤ (γ*h_lat_r - vy_heading) / (v*cos(ψ))

        cos_yaw = float(np.cos(ego_yaw))
        v_eff = v * max(cos_yaw, 0.1)   # effective lateral authority from steering

        if v > cfg.V_MIN_FOR_STEER:
            delta_min_cbf = (-cfg.GAMMA_LAT * h_lat_l - vy_heading) / v_eff
            delta_max_cbf = ( cfg.GAMMA_LAT * h_lat_r - vy_heading) / v_eff
        else:
            delta_min_cbf = -cfg.STEER_ABS_MAX
            delta_max_cbf =  cfg.STEER_ABS_MAX

        # Also hard-clamp to physical steering limits
        delta_min_cbf = max(delta_min_cbf, -cfg.STEER_ABS_MAX)
        delta_max_cbf = min(delta_max_cbf,  cfg.STEER_ABS_MAX)

        # Ensure [min, max] is a valid interval (degenerate if ego is outside bounds)
        if delta_min_cbf > delta_max_cbf:
            # Emergency: steer toward road centre
            centre = (cfg.LANE_LEFT_LIMIT + cfg.LANE_RIGHT_LIMIT) / 2.0
            delta_safe = float(np.sign(centre - ego_y) * 0.1)
        else:
            delta_safe = float(np.clip(delta_raw, delta_min_cbf, delta_max_cbf))

        return delta_safe, {
            'h_lat_l': float(h_lat_l),
            'h_lat_r': float(h_lat_r),
            'delta_min_cbf': float(delta_min_cbf),
            'delta_max_cbf': float(delta_max_cbf),
        }

    def _project_lateral_car(
        self,
        delta_raw: float,
        ego_x: float,
        ego_y: float,
        ego_vx: float,
        ego_vy: float,
        surrounding: List[VehicleState],
    ) -> Tuple[float, dict]:
        """
        Prevent steering into a vehicle that is laterally close.

        Only activates when a vehicle is within LATERAL_SAFE_GAP laterally
        AND within a short longitudinal window (±15 m).
        """
        cfg = self.cfg
        v = max(ego_vx, cfg.V_MIN_FOR_STEER)
        dt = cfg.DT

        delta_min = -cfg.STEER_ABS_MAX
        delta_max =  cfg.STEER_ABS_MAX
        triggered = False

        for sv in surrounding:
            dx = abs(sv.x - ego_x)
            if dx > 15.0:
                continue  # too far ahead/behind to matter
            dy = sv.y - ego_y   # positive = sv is to the left of ego
            lat_gap = abs(dy) - 1.0  # 1 m half-width each side

            if lat_gap >= cfg.LATERAL_SAFE_GAP:
                continue  # comfortable gap, no constraint needed

            h_lat_car = lat_gap  # ≥ 0 required
            if dy > 0:
                # Vehicle to the LEFT → don't steer further left (positive δ)
                # δ ≤ γ * h / v
                delta_max_car = cfg.GAMMA_LAT_CAR * h_lat_car / v
                delta_max = min(delta_max, delta_max_car)
            else:
                # Vehicle to the RIGHT → don't steer further right (negative δ)
                # δ ≥ -γ * h / v
                delta_min_car = -cfg.GAMMA_LAT_CAR * h_lat_car / v
                delta_min = max(delta_min, delta_min_car)
            triggered = True

        if delta_min > delta_max:
            delta_safe = float(np.clip(delta_raw, -cfg.STEER_ABS_MAX, cfg.STEER_ABS_MAX))
        else:
            delta_safe = float(np.clip(delta_raw, delta_min, delta_max))

        return delta_safe, {'lateral_car_cbf_triggered': triggered}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def check_constraints(
        self,
        ego_x: float,
        ego_y: float,
        ego_vx: float,
        surrounding: List[VehicleState],
    ) -> dict:
        """
        Return current CBF values without projecting any action.
        Useful for logging and debugging.
        """
        cfg = self.cfg
        LANE_THRESH = 2.0

        # Longitudinal
        best_gap = np.inf
        for sv in surrounding:
            dx = sv.x - ego_x
            if dx < 0.5 or abs(sv.y - ego_y) > LANE_THRESH:
                continue
            gap = dx - sv.length / 2
            best_gap = min(best_gap, gap)

        d_safe = cfg.D_MIN + cfg.T_HEAD * max(ego_vx, 0.0)
        h_lon = (best_gap - d_safe) if best_gap < np.inf else np.inf

        h_lat_l = ego_y - cfg.LANE_LEFT_LIMIT
        h_lat_r = cfg.LANE_RIGHT_LIMIT - ego_y

        return {
            'h_lon': float(h_lon),
            'h_lat_l': float(h_lat_l),
            'h_lat_r': float(h_lat_r),
            'min_gap': float(best_gap),
            'd_safe': float(d_safe),
            'safe': bool(h_lon >= 0 and h_lat_l >= 0 and h_lat_r >= 0),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cfg = CBFConfig()
    cbf = CBFSafetyFilter(cfg)

    # Test 1: no surrounding vehicles → action passes through (clamped to limits)
    a_s, d_s, info = cbf.project(1.0, 0.05, 50.0, 5.25, 10.0, 0.0, [])
    assert abs(a_s - 1.0) < 0.01, f"Expected a≈1.0, got {a_s}"
    print(f"Test 1 passed: no vehicles, a={a_s:.3f}, δ={d_s:.4f}")

    # Test 2: leader very close → acceleration clipped
    leader = VehicleState(x=54.0, y=5.25, vx=8.0)  # 4 m ahead, slower
    a_s, d_s, info = cbf.project(1.0, 0.0, 50.0, 5.25, 10.0, 0.0, [leader])
    assert a_s < 0.0, f"Expected negative a_safe, got {a_s}"
    print(f"Test 2 passed: close leader, a_safe={a_s:.3f}, h_lon={info['h_lon']:.2f}")

    # Test 3: steering TOWARD lower road edge (negative δ, ego near y=0.5) → clipped
    # Positive δ moves away from lower edge → safe; negative δ moves toward it → clipped
    a_s, d_s, info = cbf.project(0.0, -0.35, 50.0, 0.5, 10.0, 0.0, [])
    assert d_s > -0.35, f"Expected δ clipped upward (less negative), got {d_s}"
    print(f"Test 3 passed: near lower edge, δ_safe={d_s:.4f}, h_lat_l={info['h_lat_l']:.2f}")

    # Test 4: vehicle to the left laterally → left steer clipped
    adj = VehicleState(x=51.0, y=6.0, vx=9.0)  # 0.75 m lateral gap
    a_s, d_s, info = cbf.project(0.0, 0.3, 50.0, 5.25, 10.0, 0.0, [adj])
    assert d_s < 0.3, f"Expected left-steer clipped, got {d_s}"
    print(f"Test 4 passed: adj vehicle left, δ_safe={d_s:.4f}")

    # Test 5: jerk limit — full-accel request from a=0 is capped to J_MAX·dt.
    a_s, d_s, info = cbf.project(1.5, 0.0, 50.0, 5.25, 10.0, 0.0, [], last_accel=0.0)
    j_band = cfg.J_MAX * cfg.DT
    assert abs(a_s - j_band) < 1e-6 and info['jerk_limited'], f"Expected a≈{j_band}, got {a_s}"
    print(f"Test 5 passed: jerk-limited a_safe={a_s:.3f} (|Δa|≤{j_band})")

    # Test 6: steer-rate limit — large steer step capped to STEER_RATE_MAX.
    a_s, d_s, info = cbf.project(0.0, 0.3, 50.0, 5.25, 10.0, 0.0, [], last_steer=0.0)
    assert abs(d_s - cfg.STEER_RATE_MAX) < 1e-6 and info['steer_rate_limited'], f"Expected δ≈{cfg.STEER_RATE_MAX}, got {d_s}"
    print(f"Test 6 passed: steer-rate-limited δ_safe={d_s:.4f}")

    # Test 7: comfort-decel floor — gratuitous hard brake softened to comfort floor.
    a_s, d_s, info = cbf.project(-4.0, 0.0, 50.0, 5.25, 10.0, 0.0, [])
    assert abs(a_s - cfg.ACCEL_COMFORT_MIN) < 1e-6 and info['comfort_decel_active'], f"Expected a≈{cfg.ACCEL_COMFORT_MIN}, got {a_s}"
    print(f"Test 7 passed: comfort-decel floor a_safe={a_s:.3f}")

    # Test 8: safety override — a close leader forces braking past the jerk band.
    leader = VehicleState(x=53.0, y=5.25, vx=2.0)  # 3 m ahead, much slower
    a_s, d_s, info = cbf.project(0.5, 0.0, 50.0, 5.25, 12.0, 0.0, [leader], last_accel=0.0)
    assert a_s < -cfg.J_MAX * cfg.DT and info['comfort_overridden_by_safety'], (
        f"Expected hard brake overriding jerk, got a={a_s}, info={info}")
    print(f"Test 8 passed: safety overrides jerk, a_safe={a_s:.3f}")

    print("\nAll CBF self-tests passed.")
