"""
PRIDEAM Controller
==================
PDE-Risk Integrated Dynamic Emergency Assessment Model

Wraps IDEAM's LMPC+CBF controller with DRIFT risk field integration.
Provides risk-aware MPC solving, decision gating, and parameter modulation.

Author: PRIDEAM Integration
"""

import sys
import os
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Control.MPC import LMPC
from Control.constraint_params import constraint_params
from Integration.drift_interface import DRIFTInterface


class RiskWeights:
    """Default risk integration weights."""

    def __init__(self):
        # MPC cost function weight for risk penalty
        self.mpc_cost = 0.5

        # CBF ellipse expansion factor (0-1)
        # Higher = more aggressive expansion in risky areas
        self.cbf_modulation = 0.6

        # Headway/distance increase factor (0-1)
        # Higher = larger following distances in risky areas
        self.headway_modulation = 0.4

        # Risk threshold for blocking lane changes
        # Lane changes with risk > threshold are rejected
        self.decision_threshold = 1.5

        # Maximum scale factors to prevent extreme values
        self.max_cbf_scale = 2.0
        self.max_headway_scale = 2.0

        # Risk saturation reference for all modulation functions.
        # At risk = cbf_risk_normalization the CBF/headway scale reaches its maximum.
        # Increase this for dense-traffic scenarios where aggregate risk is high.
        self.cbf_risk_normalization = 1.5

    def to_dict(self):
        return {
            'mpc_cost': self.mpc_cost,
            'cbf_modulation': self.cbf_modulation,
            'headway_modulation': self.headway_modulation,
            'decision_threshold': self.decision_threshold,
            'max_cbf_scale': self.max_cbf_scale,
            'max_headway_scale': self.max_headway_scale,
            'cbf_risk_normalization': self.cbf_risk_normalization,
        }

    @classmethod
    def from_dict(cls, d):
        w = cls()
        for k, v in d.items():
            if hasattr(w, k):
                setattr(w, k, v)
        return w


class PRIDEAMController:
    """
    PRIDEAM: PDE-Risk Integrated Dynamic Emergency Assessment Model

    Combines IDEAM's MPC+CBF control with DRIFT's propagating risk field.

    The controller provides three levels of risk integration:
    1. Decision Level: Gate lane changes based on trajectory risk
    2. Constraint Level: Modulate CBF ellipse sizes based on local risk
    3. Cost Level: Add risk penalty to MPC cost function (via horizon queries)

    Attributes:
        mpc: LMPC controller instance
        drift: DRIFTInterface instance
        weights: RiskWeights configuration
        last_risk_field: Last computed risk field (for visualization)
        last_risk_along_horizon: Risk values along last MPC solution
    """

    def __init__(self, mpc_params=None, risk_weights=None, paths=None):
        """
        Initialize PRIDEAM controller.

        Args:
            mpc_params: Dict to override constraint_params() values
            risk_weights: RiskWeights instance or dict of weights
            paths: Dict of path objects {index: path_obj} for Frenet conversion
        """
        # Initialize base IDEAM MPC controller
        params = constraint_params()
        if mpc_params:
            params.update(mpc_params)
        self.mpc = LMPC(**params)

        # Initialize DRIFT interface
        self.drift = DRIFTInterface()
        if paths:
            self.drift.register_paths(paths)

        # Risk integration weights
        if isinstance(risk_weights, dict):
            self.weights = RiskWeights.from_dict(risk_weights)
        elif isinstance(risk_weights, RiskWeights):
            self.weights = risk_weights
        else:
            self.weights = RiskWeights()

        # Store original CBF parameters for restoration
        self._orig_a_l = self.mpc.a_l
        self._orig_b_l = self.mpc.b_l
        self._orig_a_f = self.mpc.a_f
        self._orig_b_f = self.mpc.b_f

        # Store for visualization/debugging
        self.last_risk_field = None
        self.last_risk_along_horizon = None
        self.last_modulated_params = None
        self.last_decision_risk = None

        # Statistics
        self.stats = {
            'risk_field_updates': 0,
            'mpc_solves': 0,
            'lane_changes_blocked': 0,
            'total_lane_change_attempts': 0,
        }

    # =====================================================================
    # PROPERTIES (expose MPC attributes)
    # =====================================================================

    @property
    def T(self):
        """MPC prediction horizon."""
        return self.mpc.T

    @property
    def dt(self):
        """Time step."""
        return self.mpc.dt

    @property
    def vehicle_length(self):
        """Vehicle length [m]."""
        return self.mpc.vehicle_length

    @property
    def vehicle_width(self):
        """Vehicle width [m]."""
        return self.mpc.vehicle_width

    # =====================================================================
    # RL DECISION-LAYER OVERRIDES (set per step by the RL decision policy)
    # =====================================================================

    def set_target_speed_override(self, v_target):
        """Override the MPC reference speed for the next solve.

        Used by the decision-level RL policy to set the cruise-speed
        target that the MPC tracker follows. Passing None clears the
        override and restores the default TARGET_SPEED.
        """
        self.mpc.set_target_speed_override(v_target)

    # =====================================================================
    # RISK FIELD UPDATE
    # =====================================================================

    def update_risk_field(self, vehicles_dict, ego_state, dt=0.1):
        """
        Update the DRIFT risk field based on current traffic state.

        Should be called each timestep BEFORE decision making and MPC solving.

        Args:
            vehicles_dict: Dict of vehicle arrays in IDEAM format
                {
                    'left': vehicle_left array,
                    'center': vehicle_centre array,
                    'right': vehicle_right array
                }
                Each array row: [s, ey, epsi, x, y, psi, vx, a]
            ego_state: Ego global state [x, y, psi] or full state
            dt: Time step [s]

        Returns:
            risk_field: Updated risk field (2D array)
        """
        # Extract ego Cartesian state
        if len(ego_state) >= 3:
            if len(ego_state) == 3:
                # [x, y, psi] format
                ego_x, ego_y, ego_psi = ego_state
                ego_vx = 15.0  # Default velocity
            else:
                # Assume [vx, vy, w, s, ey, epsi] + separate [x, y, psi]
                # or similar - try to extract
                ego_x = ego_state[0] if len(ego_state) > 6 else ego_state[0]
                ego_y = ego_state[1] if len(ego_state) > 6 else ego_state[1]
                ego_psi = ego_state[2] if len(ego_state) > 6 else 0.0
                ego_vx = ego_state[0] if len(ego_state) <= 6 else 15.0

        # Extract vehicle arrays
        vehicle_left = vehicles_dict.get('left')
        vehicle_centre = vehicles_dict.get('center')
        if vehicle_centre is None:
            vehicle_centre = vehicles_dict.get('centre')
        vehicle_right = vehicles_dict.get('right')

        # Update DRIFT
        self.last_risk_field = self.drift.step_with_ideam_vehicles(
            vehicle_left, vehicle_centre, vehicle_right,
            ego_x, ego_y, ego_vx, ego_psi, dt
        )

        self.stats['risk_field_updates'] += 1

        return self.last_risk_field

    def update_risk_field_direct(self, vehicles, ego, dt=0.1):
        """
        Update risk field using DRIFT-format vehicles directly.

        Args:
            vehicles: List of vehicle dicts in DRIFT format
            ego: Ego vehicle dict
            dt: Time step

        Returns:
            risk_field: Updated risk field
        """
        self.last_risk_field = self.drift.step(vehicles, ego, dt)
        self.stats['risk_field_updates'] += 1
        return self.last_risk_field

    # =====================================================================
    # DECISION-LEVEL INTEGRATION
    # =====================================================================

    def evaluate_decision_risk(self, ego_state, current_path, target_path,
                                lane_width=3.5):
        """
        Evaluate risk for a potential lane change decision.

        Used to gate lane changes: if risk is too high, reject the transition.

        Args:
            ego_state: Current ego state [vx, vy, w, s, ey, epsi] or similar
            current_path: Current path index (0=left, 1=center, 2=right)
            target_path: Target path index
            lane_width: Lane width [m]

        Returns:
            risk_score: Combined risk score (higher = more dangerous)
            allow_transition: Boolean, whether lane change should be allowed
            details: Dict with risk breakdown
        """
        self.stats['total_lane_change_attempts'] += 1

        # Get risk score from DRIFT
        risk_score, details = self.drift.get_decision_risk_score(
            ego_state, current_path, target_path, lane_width
        )

        # Decision based on threshold
        allow_transition = risk_score < self.weights.decision_threshold

        if not allow_transition:
            self.stats['lane_changes_blocked'] += 1

        self.last_decision_risk = {
            'score': risk_score,
            'allowed': allow_transition,
            'threshold': self.weights.decision_threshold,
            'details': details,
        }

        return risk_score, allow_transition, details

    def gate_lane_change(self, C_label, ego_state, path_now, path_dindex):
        """
        Gate a lane change decision based on risk.

        Convenience method that returns updated C_label if lane change is blocked.

        Args:
            C_label: Current decision label ("K", "L", "R")
            ego_state: Ego state
            path_now: Current path index
            path_dindex: Desired path index

        Returns:
            C_label: Possibly modified to "K" if lane change blocked
            was_blocked: Boolean indicating if lane change was blocked
        """
        if C_label == "K":
            return C_label, False

        # Evaluate risk
        risk_score, allow, _ = self.evaluate_decision_risk(
            ego_state, path_now, path_dindex
        )

        if not allow:
            return "K", True

        return C_label, False

    # =====================================================================
    # CONSTRAINT-LEVEL INTEGRATION
    # =====================================================================

    def get_risk_modulated_params(self, x, y, path_index=None, s=None, ey=None):
        """
        Get risk-modulated CBF and headway parameters.

        Can be called with either Cartesian (x, y) or Frenet (s, ey) coordinates.

        Args:
            x, y: Cartesian position (used if s, ey not provided)
            path_index: Path index for Frenet lookup
            s, ey: Frenet coordinates (optional, preferred if available)

        Returns:
            params: Dict with modulated parameters:
                {
                    'a_l', 'b_l': Leader ellipse semi-axes
                    'a_f', 'b_f': Follower ellipse semi-axes
                    'Th': Time headway
                    'd0': Minimum distance
                    'risk_value': Local risk value used for modulation
                }
        """
        # Get local risk
        if s is not None and ey is not None and path_index is not None:
            risk = self.drift.get_risk_frenet(s, ey, path_index)
        else:
            risk = self.drift.get_risk_cartesian(x, y)

        risk = float(risk) if hasattr(risk, '__len__') else risk

        # Modulate CBF parameters
        a_l, b_l = self.drift.get_cbf_margin_modulation(
            x, y, self._orig_a_l, self._orig_b_l,
            alpha=self.weights.cbf_modulation,
            max_scale=self.weights.max_cbf_scale,
            risk_norm=self.weights.cbf_risk_normalization,
        )
        a_f, b_f = self.drift.get_cbf_margin_modulation(
            x, y, self._orig_a_f, self._orig_b_f,
            alpha=self.weights.cbf_modulation,
            max_scale=self.weights.max_cbf_scale,
            risk_norm=self.weights.cbf_risk_normalization,
        )

        # Modulate headway parameters
        base_Th = 0.3  # From decision_params
        base_d0 = 5.0  # From decision_params
        Th, d0 = self.drift.get_headway_modulation(
            x, y, base_Th, base_d0,
            beta=self.weights.headway_modulation,
            max_scale=self.weights.max_headway_scale,
            risk_norm=self.weights.cbf_risk_normalization,
        )

        params = {
            'a_l': a_l, 'b_l': b_l,
            'a_f': a_f, 'b_f': b_f,
            'Th': Th, 'd0': d0,
            'risk_value': risk,
        }

        self.last_modulated_params = params
        return params

    def apply_risk_modulation(self, x0_g):
        """
        Apply risk-based modulation to MPC parameters.

        Temporarily modifies self.mpc.a_l, b_l, a_f, b_f based on local risk.

        Args:
            x0_g: Ego global state [x, y, psi]
        """
        x, y = x0_g[0], x0_g[1]
        params = self.get_risk_modulated_params(x, y)

        # Only apply modulation if there's meaningful risk
        # Skip modulation when risk is very low to avoid MPC instability
        if params['risk_value'] > 0.05:
            self.mpc.a_l = params['a_l']
            self.mpc.b_l = params['b_l']
            self.mpc.a_f = params['a_f']
            self.mpc.b_f = params['b_f']

    def restore_original_params(self):
        """Restore original MPC parameters after risk modulation."""
        self.mpc.a_l = self._orig_a_l
        self.mpc.b_l = self._orig_b_l
        self.mpc.a_f = self._orig_a_f
        self.mpc.b_f = self._orig_b_f

    # =====================================================================
    # MPC SOLVING WITH RISK
    # =====================================================================

    def _get_reference_trajectory(self, X0, path_d):
        """
        Generate reference trajectory for risk query when no warm-start available.

        Uses constant-velocity prediction along path centerline.

        Args:
            X0: Current state [vx, vy, w, s, ey, epsi]
            path_d: Path object

        Returns:
            predicted_s: Longitudinal position array [T+1]
            predicted_ey: Lateral position array [T+1]
        """
        T = self.mpc.T
        dt = self.mpc.dt
        s_current = X0[3]
        ey_current = X0[4]
        v_current = X0[0]

        # Constant velocity prediction
        predicted_s = np.array([s_current + i * dt * v_current for i in range(T+1)])
        predicted_ey = np.array([ey_current] * (T+1))  # Assume centerline tracking

        return predicted_s, predicted_ey

    def solve_with_risk(self, X0, oa, od, dt, GPR_vy, GPR_w, C_label, X0_g,
                        path_d, last_X, path_now, ego_group, path_ego,
                        target_group, vehicle_left, vehicle_centre, vehicle_right,
                        path_dindex, C_label_additive, C_label_virtual):
        """
        Solve MPC with risk-aware modifications.

        This is the main entry point that replaces iterative_linear_mpc_control.
        It applies:
        1. MPC cost integration: Add risk penalty to cost function
        2. CBF modulation: Scale safety ellipses based on risk

        Args:
            (All args match IDEAM's iterative_linear_mpc_control signature)

        Returns:
            oa, od, ovx, ovy, owz, oS, oey, oepsi: MPC solution
        """
        # 1. COMPUTE PREDICTED TRAJECTORY for risk query
        # Use last solution as warm-start if available
        if last_X and last_X[3] is not None:
            predicted_s = last_X[3]  # oS from previous solve
            predicted_ey = last_X[4]  # oey from previous solve
        else:
            # Fallback: use reference trajectory
            predicted_s, predicted_ey = self._get_reference_trajectory(X0, path_d)

        # 2. QUERY RISK along MPC horizon and inject into MPC cost
        if self.weights.mpc_cost > 0:
            try:
                risk_costs = self.drift.get_risk_cost_vector(
                    predicted_s, predicted_ey, path_dindex, weight=1.0
                )
                self.mpc.risk_cost_vector = risk_costs
                self.mpc.risk_weight = self.weights.mpc_cost
            except Exception as e:
                print(f"[PRIDEAM] Warning: Risk cost computation failed: {e}")
                self.mpc.risk_cost_vector = None
                self.mpc.risk_weight = 0.0

        # 3. APPLY CBF MODULATION (risk-based ellipse scaling)
        if self.weights.cbf_modulation > 0:
            self.apply_risk_modulation(X0_g)

        try:
            # Call original MPC solver
            result = self.mpc.iterative_linear_mpc_control(
                X0, oa, od, dt, GPR_vy, GPR_w, C_label, X0_g,
                path_d, last_X, path_now, ego_group, path_ego,
                target_group, vehicle_left, vehicle_centre, vehicle_right,
                path_dindex, C_label_additive, C_label_virtual
            )

            # Extract solution
            oa, od, ovx, ovy, owz, oS, oey, oepsi = result

            # Handle MPC failure - use fallback from last_X if available
            if oa is None:
                print("[PRIDEAM] MPC failed, using fallback trajectory")
                T = self.mpc.T
                if last_X is not None and last_X[0] is not None:
                    ovx, ovy, owz, oS, oey, oepsi = last_X
                    oa = [0.0] * T
                    od = [0.0] * T
                else:
                    # Emergency fallback - zero control
                    oa = [0.0] * T
                    od = [0.0] * T
                    ovx = [X0[0]] * (T + 1)
                    ovy = [X0[1]] * (T + 1)
                    owz = [X0[2]] * (T + 1)
                    oS = [X0[3]] * (T + 1)
                    oey = [X0[4]] * (T + 1)
                    oepsi = [X0[5]] * (T + 1)

            # Store risk along solution horizon for visualization
            if oS is not None and oey is not None:
                try:
                    self.last_risk_along_horizon = self.drift.get_risk_along_horizon(
                        oS, oey, path_dindex
                    )
                except Exception:
                    self.last_risk_along_horizon = None

            self.stats['mpc_solves'] += 1

        finally:
            # Always restore original parameters
            if self.weights.cbf_modulation > 0:
                self.restore_original_params()

            # Clear risk vector from MPC
            self.mpc.risk_cost_vector = None

        return oa, od, ovx, ovy, owz, oS, oey, oepsi

    def solve_mpc_basic(self, path, x0, ou, dt, GPR_vy, GPR_w, label):
        """
        Solve basic MPC (no iterative, no risk modulation).

        Wraps mpc.MPC_solve for simple cases.

        Args:
            path: Path object
            x0: Initial state
            ou: Initial control
            dt: Time step
            GPR_vy, GPR_w: GPR models (can be None)
            label: Lane label

        Returns:
            MPC solution tuple
        """
        return self.mpc.MPC_solve(path, x0, ou, dt, GPR_vy, GPR_w, label)

    # =====================================================================
    # UTILITY METHODS
    # =====================================================================

    def set_util(self, utils):
        """Pass utility object to underlying MPC."""
        self.mpc.set_util(utils)

    def get_path_curvature(self, path):
        """Compute path curvature for MPC."""
        return self.mpc.get_path_curvature(path)

    def register_path(self, path_index, path_obj):
        """Register a path for DRIFT Frenet conversion."""
        self.drift.register_path(path_index, path_obj)

    def register_paths(self, paths_dict):
        """Register multiple paths."""
        self.drift.register_paths(paths_dict)

    # =====================================================================
    # PROPERTIES
    # =====================================================================

    @property
    def T(self):
        """MPC horizon length."""
        return self.mpc.T

    @property
    def dt(self):
        """MPC time step."""
        return self.mpc.dt

    @property
    def risk_field(self):
        """Current DRIFT risk field."""
        return self.drift.risk_field

    @property
    def vehicle_length(self):
        """Vehicle length from MPC."""
        return self.mpc.vehicle_length

    @property
    def vehicle_width(self):
        """Vehicle width from MPC."""
        return self.mpc.vehicle_width

    def get_stats(self):
        """Get controller statistics."""
        stats = self.stats.copy()
        if stats['total_lane_change_attempts'] > 0:
            stats['block_rate'] = (
                stats['lane_changes_blocked'] / stats['total_lane_change_attempts']
            )
        else:
            stats['block_rate'] = 0.0
        return stats

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'risk_field_updates': 0,
            'mpc_solves': 0,
            'lane_changes_blocked': 0,
            'total_lane_change_attempts': 0,
        }


# =========================================================================
# FACTORY FUNCTION
# =========================================================================

def create_prideam_controller(paths=None, risk_weights=None, **mpc_overrides):
    """
    Factory function to create a fully configured PRIDEAM controller.

    Args:
        paths: Dict of path objects {0: path_left, 1: path_center, 2: path_right}
        risk_weights: Dict of risk weights or RiskWeights instance
        **mpc_overrides: Override values for MPC parameters

    Returns:
        PRIDEAMController: Configured controller instance

    Example:
        from Path.path import path1c, path2c, path3c

        controller = create_prideam_controller(
            paths={0: path1c, 1: path2c, 2: path3c},
            risk_weights={'cbf_modulation': 0.4, 'decision_threshold': 1.2},
        )
    """
    controller = PRIDEAMController(
        mpc_params=mpc_overrides if mpc_overrides else None,
        risk_weights=risk_weights,
        paths=paths,
    )

    return controller
