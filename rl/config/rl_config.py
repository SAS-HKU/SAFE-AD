"""
RL Configuration
================
Central hyperparameter store for the DREAM RL extension.
All numeric constants that appear in env/obs/reward modules live here.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import json


@dataclass
class RLConfig:
    """
    Configuration for the DREAM highway RL environment.

    Observation / action sizes, reward weights, safety thresholds, and
    DRIFT query parameters are all gathered here so that the environment,
    observation builder, and reward function share a single source of truth.
    """

    # ---- Observation -------------------------------------------------------
    OBS_DIM: int = 20
    # Breakdown:
    #   ego kinematics  : 5  (v_x, e_y, e_psi, last_a, last_delta)
    #   current-lane lead: 3  (ds, dv, a_lead)
    #   left-lane nearest: 2  (ds_left, dv_left)
    #   right-lane nearest:2  (ds_right, dv_right)
    #   DRIFT risk       : 6  (r_ego, r_fwd, r_left, r_right, grad_x, grad_y)
    #   context          : 2  (in_merge_zone, steps_since_lc / 50)

    # ---- Action ------------------------------------------------------------
    N_ACTIONS: int = 9
    # Action = 3 lane decisions × 3 speed modes
    # index: 0-2 KEEP, 3-5 LC_DOWN, 6-8 LC_UP
    # within each group: 0=MAINTAIN, 1=SLOWER, 2=FASTER
    # Lane delta: 0=keep(0), 1..2=keep(0), 3..5=down(-1), 6..8=up(+1)
    LANE_DELTAS: List[int] = field(default_factory=lambda: [0, 0, 0, -1, -1, -1, 1, 1, 1])
    # Speed offset added to TARGET_SPEED [m/s]
    SPEED_OFFSETS: List[float] = field(default_factory=lambda: [0.0, -2.0, 2.0,
                                                                  0.0, -2.0, 2.0,
                                                                  0.0, -2.0, 2.0])

    # ---- Simulation --------------------------------------------------------
    DT: float = 0.1           # timestep [s]
    MAX_STEPS: int = 400      # max episode length
    TARGET_SPEED: float = 10.0  # nominal target speed [m/s]

    # ---- Ego kinematic limits ----------------------------------------------
    MAX_ACCEL: float = 1.0     # max positive acceleration [m/s²]
    MIN_ACCEL: float = -4.0    # max braking [m/s²]
    MAX_STEER: float = 0.35    # max steering angle [rad]
    MAX_SPEED: float = 18.0    # speed clamp [m/s]
    MIN_SPEED: float = 1.0     # min speed (stop detection) [m/s]
    WHEELBASE: float = 4.4     # vehicle wheelbase [m]

    # ---- Lane geometry (highway scenario) ----------------------------------
    # Lane 0 = lower (D), Lane 1 = ego (E), Lane 2 = upper (U)
    LANE_CENTERS: List[float] = field(default_factory=lambda: [1.75, 5.25, 8.75])
    LANE_BOUNDARY_LOW: float = 3.5   # D/E boundary [m]
    LANE_BOUNDARY_HIGH: float = 7.0  # E/U boundary [m]
    LANE_WIDTH: float = 3.5

    # ---- Simple tracking controller gains ----------------------------------
    K_LAT: float = 0.6   # lateral P-gain (steering = K_LAT * lateral_error / v_x)
    K_LON: float = 2.0   # longitudinal P-gain
    TAU_LC: int = 15     # steps to complete a lane change (softens lateral command)

    # ---- Safety thresholds -------------------------------------------------
    COLLISION_DIST: float = 4.5   # rear-bumper gap below which collision is declared [m]
    NEAR_MISS_DIST: float = 6.0   # gap for near-miss penalty [m]  (was 8.0; reduced so normal LCs don't trigger)
    TTC_THRESHOLD: float = 2.0    # TTC below which near-miss is counted [s]
    OFFROAD_LATERAL: float = 1.8  # lateral distance from lane centre for off-road [m]
    STALL_SPEED: float = 0.8      # below this speed for STALL_STEPS → stall termination
    STALL_STEPS: int = 30

    # ---- Reward weights ----------------------------------------------------
    W_PROGRESS: float = 1.0    # forward progress (normalised by v_target * dt)
    W_SPEED: float = 0.5       # speed tracking penalty
    W_SPEED_CRUISE: float = 0.3  # cruise-speed penalty (vs nominal TARGET_SPEED, not action-conditioned)
    W_COMFORT: float = 0.2     # control effort penalty
    W_LANE_KEEP: float = 0.3   # lateral deviation penalty
    W_RISK: float = 0.1        # DRIFT risk cost penalty (logged separately as 'cost')
    # NOTE: W_RISK reduced from 0.5→0.1 because numerical PDE solver produces c_risk≈1.0-2.0/step
    # (vs near-zero from flat PINN). At 0.5 the risk penalty dominated reward and destabilised training.
    W_NEAR_MISS: float = 3.0   # near-miss per-step penalty  (was 10.0; reduced to stop drowning LC reward)
    W_LANE_ADV: float = 1.5    # lane-advantage reward (choosing a better lane)  (was 0.6)
    W_INACTION: float = 0.8    # blocked-by-leader penalty (staying behind a slow leader)  (was 0.3)
    W_COMMIT: float = 0.4      # lane-change commitment bonus (finishing started LC)  (was 0.2)
    W_YAW: float = 0.3         # heading stabilisation penalty
    W_CBF: float = 0.5         # CBF intervention penalty
    LANE_ADV_GAP_D0: float = 30.0   # normaliser for gap advantage [m]
    LANE_ADV_DV_V0: float = 5.0     # normaliser for speed advantage [m/s]
    LANE_ADV_RISK_R0: float = 2.0   # normaliser for risk advantage
    INACTION_LEADER_DIST: float = 25.0  # slow-leader detection range [m]
    INACTION_SPEED_FRAC: float = 0.7    # leader < 70% of ego speed → "slow"
    R_COLLISION: float = -100.0  # terminal collision reward
    R_OFFROAD: float = -50.0     # terminal off-road reward
    R_STALL: float = -20.0       # terminal stall reward

    # ---- DRIFT risk query parameters ---------------------------------------
    DRIFT_SUBSTEPS: int = 3
    DRIFT_WARMUP: float = 5.0     # cold-start warm-up duration [s]
    CORRIDOR_LENGTH: float = 25.0  # forward look-ahead for corridor risk [m]
    CORRIDOR_SAMPLES: int = 6      # sampling points along corridor
    RISK_CLIP: float = 10.0        # max raw risk value from PDESolver

    # ---- Observation normalisation denominators ----------------------------
    NORM_V: float = 15.0     # speed normaliser
    NORM_EY: float = 2.0     # lateral error normaliser
    NORM_EPSI: float = 0.4   # heading error normaliser
    NORM_A: float = 4.0      # acceleration normaliser
    NORM_DS: float = 30.0    # gap normaliser (ds/NORM_DS − 1)
    NORM_DV: float = 10.0    # relative speed normaliser
    NORM_RISK: float = 5.0   # risk normaliser
    NORM_GRAD: float = 2.0   # risk gradient normaliser

    # ---- Logging -----------------------------------------------------------
    LOG_EVERY: int = 50      # print episode summary every N episodes

    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RLConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Default instance — import and use directly, or construct with overrides.
DEFAULT_CONFIG = RLConfig()
