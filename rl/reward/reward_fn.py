"""
Reward and Safety-Cost Functions  (v2 — opportunity-aware)
============================================================
Implements the per-step reward and the separate safety cost term used by
the DREAM RL agent.

Design principles (v2)
----------------------
* reward  = task performance (progress + speed + comfort)
          + opportunity  (lane-advantage + inaction penalty + commitment)
* cost    = risk / safety exposure (DRIFT field + near-miss)

v2 changes (vs v1 "safe-cruising" reward):
  1. r_lane_adv   — positive when action moves into a *better* lane
                    (larger gap, faster relative speed, lower risk).
  2. r_inaction   — penalty when ego stays behind a slow same-lane leader
                    while a materially better adjacent lane exists.
  3. r_commit     — bonus for staying consistent once a lane change starts.
  4. r_speed_cruise — partial penalty vs nominal cruise speed (TARGET_SPEED)
                    so the policy cannot trivially redefine success by
                    choosing the SLOWER action and matching it.
  5. r_yaw / r_cbf weights pulled from config instead of hard-coded.

The safety cost `c` is still kept as a *separate* signal to support
constrained RL.  For basic PPO the caller mixes them:
    r_total = reward - config.W_RISK * cost
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.config.rl_config import RLConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Lane-advantage scoring
# ---------------------------------------------------------------------------

def _lane_advantage(ds: float, dv: float, risk: float,
                    cfg: RLConfig) -> float:
    """
    Score how good a lane is from the ego's perspective.

    Higher = better lane.  Combines three normalised terms:
      * +gap  : larger ds to front leader → more room
      * +dv   : slower closing speed → safer
      * -risk : lower DRIFT risk → safer corridor

    Args:
        ds   : gap to same-lane front leader [m]  (inf if no leader)
        dv   : relative speed to that leader [m/s] (negative = closing)
        risk : DRIFT risk evaluated at the lane centre ahead of ego
    """
    # Gap contribution: clip to avoid extreme scores from inf gaps
    gap_score = min(float(ds), 80.0) / cfg.LANE_ADV_GAP_D0
    # Closing-speed contribution: -dv (positive when we approach slower)
    dv_score  = -float(dv) / cfg.LANE_ADV_DV_V0
    # Risk contribution: lower is better → negate
    risk_score = -float(risk) / cfg.LANE_ADV_RISK_R0
    return gap_score + dv_score + risk_score


# ---------------------------------------------------------------------------
# Per-step reward
# ---------------------------------------------------------------------------

def compute_reward(
    ego_x: float,
    ego_y: float,
    ego_v: float,
    ego_yaw: float,
    prev_ego_x: float,
    target_v: float,
    current_lane: int,
    action: int,
    last_a: float,
    last_delta: float,
    prev_a: float,
    prev_delta: float,
    min_gap_all_lanes: float,
    config: RLConfig = DEFAULT_CONFIG,
    cbf_active: float = 0.0,
    # --- v2 new inputs (optional, backward-compatible) ---
    lane_delta: int = 0,
    ds_curr: float = 80.0,
    dv_curr: float = 0.0,
    ds_left: float = 80.0,
    dv_left: float = 0.0,
    ds_right: float = 80.0,
    dv_right: float = 0.0,
    risk_curr: float = 0.0,
    risk_left: float = 0.0,
    risk_right: float = 0.0,
    steps_in_lc: int = 0,
    steps_since_lc: int = 50,
) -> tuple:
    """
    Compute the per-step task reward.

    Args  (v1, unchanged):
        ego_x, ego_y      : Current ego Cartesian position [m]
        ego_v             : Current speed [m/s]
        ego_yaw           : Current heading [rad]
        prev_ego_x        : Ego x at previous step [m]
        target_v          : RL-requested target speed [m/s]
        current_lane      : Current lane index (0/1/2)
        action            : Chosen action integer (0-8)
        last_a            : Applied acceleration [m/s^2]
        last_delta        : Applied steering [rad]
        prev_a            : Previous acceleration (for jerk) [m/s^2]
        prev_delta        : Previous steering (for jerk) [rad]
        min_gap_all_lanes : Min bumper-to-bumper gap to any vehicle [m]
        config            : RLConfig
        cbf_active        : 1.0 if CBF intervened this step, else 0.0

    Args  (v2, new — default to neutral so old call-sites still work):
        lane_delta        : action's lane-change command (-1/0/+1)
        ds_curr / dv_curr : gap & relative speed to same-lane front leader
        ds_left / dv_left : gap & relative speed to left-lane front leader
        ds_right / dv_right: idem for right lane
        risk_curr / risk_left / risk_right : DRIFT risk per lane corridor
        steps_in_lc       : how many steps into an active lane change (0 = none)
        steps_since_lc    : steps since last completed lane change

    Returns:
        reward  : Scalar task reward for this step
        terms   : Dict of individual components (for logging)
    """
    cfg = config
    dt = cfg.DT

    # ------------------------------------------------------------------
    # 1. Progress reward: normalised by expected coverage at target speed
    # ------------------------------------------------------------------
    delta_x = ego_x - prev_ego_x
    r_progress = cfg.W_PROGRESS * (delta_x / (cfg.TARGET_SPEED * dt))
    r_progress = float(np.clip(r_progress, -1.0, 2.0))

    # ------------------------------------------------------------------
    # 2. Speed tracking (action-conditioned + cruise baseline)
    # ------------------------------------------------------------------
    # 2a. Action-conditioned: reward tracking the chosen target_v
    v_err = (ego_v - target_v) / cfg.TARGET_SPEED
    r_speed = -cfg.W_SPEED * v_err ** 2

    # 2b. Cruise baseline: also penalise deviation from nominal cruise speed.
    #     This closes the loophole where choosing SLOWER trivially reduces
    #     the action-conditioned penalty.
    v_cruise_err = (ego_v - cfg.TARGET_SPEED) / cfg.TARGET_SPEED
    r_speed_cruise = -cfg.W_SPEED_CRUISE * v_cruise_err ** 2

    # ------------------------------------------------------------------
    # 3. Comfort: penalise control effort (normalised)
    # ------------------------------------------------------------------
    a_norm = last_a / cfg.NORM_A
    d_norm = last_delta / 0.4
    r_comfort = -cfg.W_COMFORT * (a_norm ** 2 + d_norm ** 2)

    # ------------------------------------------------------------------
    # 4. Lane keeping: penalise lateral deviation from lane centre
    # ------------------------------------------------------------------
    lane_y = cfg.LANE_CENTERS[current_lane]
    e_y = ego_y - lane_y
    r_lane = -cfg.W_LANE_KEEP * (e_y / cfg.OFFROAD_LATERAL) ** 2

    # ------------------------------------------------------------------
    # 4b. Heading stabilisation
    # ------------------------------------------------------------------
    r_yaw = -cfg.W_YAW * (ego_yaw / 0.4) ** 2

    # ------------------------------------------------------------------
    # 5. Near-miss penalty (per step)
    # ------------------------------------------------------------------
    r_near_miss = 0.0
    if min_gap_all_lanes < cfg.NEAR_MISS_DIST:
        severity = 1.0 - (min_gap_all_lanes / cfg.NEAR_MISS_DIST)
        r_near_miss = -cfg.W_NEAR_MISS * severity

    # ------------------------------------------------------------------
    # 6. CBF intervention penalty
    # ------------------------------------------------------------------
    r_cbf = -cfg.W_CBF * float(cbf_active)

    # ==================================================================
    # v2 OPPORTUNITY-AWARE TERMS
    # ==================================================================

    # ------------------------------------------------------------------
    # 7. Lane-advantage reward
    # ------------------------------------------------------------------
    A_curr  = _lane_advantage(ds_curr,  dv_curr,  risk_curr,  cfg)
    A_left  = _lane_advantage(ds_left,  dv_left,  risk_left,  cfg) if current_lane < 2 else -999.0
    A_right = _lane_advantage(ds_right, dv_right, risk_right, cfg) if current_lane > 0 else -999.0

    r_lane_adv = 0.0
    if lane_delta == +1 and current_lane < 2:
        # Chose to go left — reward if left is better
        r_lane_adv = cfg.W_LANE_ADV * float(np.clip(A_left - A_curr, -1.0, 1.5))
    elif lane_delta == -1 and current_lane > 0:
        # Chose to go right — reward if right is better
        r_lane_adv = cfg.W_LANE_ADV * float(np.clip(A_right - A_curr, -1.0, 1.5))
    elif lane_delta == 0:
        # Chose to stay — small penalty if a *clearly* better lane exists
        best_adj = max(A_left, A_right)
        adv_margin = best_adj - A_curr
        if adv_margin > 0.5:
            # Scale: stronger penalty when the advantage is larger
            r_lane_adv = -cfg.W_LANE_ADV * 0.3 * float(np.clip(adv_margin, 0.0, 1.5))

    # ------------------------------------------------------------------
    # 8. Blocked-by-leader (inaction) penalty
    # ------------------------------------------------------------------
    r_inaction = 0.0
    leader_close = ds_curr < cfg.INACTION_LEADER_DIST
    leader_slow  = (dv_curr < 0) and (ego_v + dv_curr) < cfg.INACTION_SPEED_FRAC * ego_v if ego_v > 2.0 else False
    if leader_close and leader_slow and lane_delta == 0:
        # Adjacent lane materially better?
        best_adj = max(A_left, A_right)
        if best_adj - A_curr > 0.3:
            r_inaction = -cfg.W_INACTION * float(np.clip(
                1.0 - ds_curr / cfg.INACTION_LEADER_DIST, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 9. Lane-change commitment bonus
    # ------------------------------------------------------------------
    r_commit = 0.0
    if steps_in_lc > 0:
        # Mid lane-change: reward if action is consistent (not reversing)
        if lane_delta == 0 or lane_delta != 0:
            # Any non-reversal during active LC gets a small bonus.
            # A "reversal" is detected by the env (rejected LC) — here
            # we just reward continued commitment.
            r_commit = cfg.W_COMMIT * 0.5
    elif steps_since_lc < 5 and lane_delta == 0:
        # Just completed a LC — small bonus for settling
        r_commit = cfg.W_COMMIT * 0.3

    # ------------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------------
    reward = (r_progress + r_speed + r_speed_cruise + r_comfort + r_lane
              + r_yaw + r_near_miss + r_cbf
              + r_lane_adv + r_inaction + r_commit)

    terms = {
        'r_progress': r_progress,
        'r_speed': r_speed,
        'r_speed_cruise': r_speed_cruise,
        'r_comfort': r_comfort,
        'r_lane': r_lane,
        'r_yaw': r_yaw,
        'r_near_miss': r_near_miss,
        'r_cbf': r_cbf,
        'r_lane_adv': r_lane_adv,
        'r_inaction': r_inaction,
        'r_commit': r_commit,
        'reward': reward,
    }
    return float(reward), terms


# ---------------------------------------------------------------------------
# Per-step safety cost (separate from reward)
# ---------------------------------------------------------------------------

def compute_safety_cost(
    ego_x: float,
    ego_y: float,
    drift_interface,
    min_gap_all_lanes: float,
    config: RLConfig = DEFAULT_CONFIG,
) -> tuple:
    """
    Compute the per-step safety cost from DRIFT risk exposure.

    This is logged independently of the task reward so constrained RL
    algorithms can use it as a budget constraint.

    Args:
        ego_x, ego_y   : Ego Cartesian position [m]
        drift_interface: DRIFTInterface (already stepped this tick)
        min_gap_all_lanes: Minimum bumper-to-bumper gap [m]
        config         : RLConfig

    Returns:
        cost   : Scalar safety cost for this step
        terms  : Dict of individual components
    """
    cfg = config

    r_ego = 0.0
    r_fwd = 0.0

    if drift_interface is not None:
        try:
            r_ego = float(drift_interface.get_risk_cartesian(ego_x, ego_y))
            # Also check a short forward corridor for approaching hazards
            r_fwd = float(drift_interface.get_risk_corridor(
                ego_x, ego_y, 0.0,
                length=15.0, n_samples=4
            ))
        except Exception:
            pass

    # Raw cost = mean of local + forward risk, scaled
    c_risk = (r_ego + r_fwd) / (2.0 * cfg.RISK_CLIP)   # normalised to [0, 1]

    # Additional cost for very close approaches
    c_gap = 0.0
    if min_gap_all_lanes < cfg.NEAR_MISS_DIST:
        c_gap = 1.0 - (min_gap_all_lanes / cfg.NEAR_MISS_DIST)

    cost = c_risk + c_gap

    terms = {
        'c_risk_ego': r_ego,
        'c_risk_fwd': r_fwd,
        'c_risk': c_risk,
        'c_gap': c_gap,
        'cost': cost,
    }
    return float(cost), terms


# ---------------------------------------------------------------------------
# Terminal reward
# ---------------------------------------------------------------------------

def terminal_reward(reason: str, config: RLConfig = DEFAULT_CONFIG) -> float:
    """
    Return the one-time reward added at episode termination.

    Args:
        reason: One of 'collision', 'offroad', 'stall', 'timeout'
        config: RLConfig

    Returns:
        reward: Terminal reward (negative for failures, 0 for timeout)
    """
    R_TIMEOUT = 20.0  # survival bonus = 20 steps of good driving
    mapping = {
        'collision': config.R_COLLISION,
        'offroad':   config.R_OFFROAD,
        'stall':     config.R_STALL,
        'timeout':   R_TIMEOUT,
    }
    return float(mapping.get(reason, 0.0))
