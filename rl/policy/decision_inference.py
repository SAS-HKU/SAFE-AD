"""
Decision-policy inference adapter for the live simulator.
==========================================================

Bridges the 17-dim :mod:`rl.policy.decision_policy` observation schema to
the uncertainty_merger_DREAM.py benchmark state representation, so that a
BC-pretrained (or PPO-fine-tuned) DecisionPolicy can be dropped in as a
planner alongside DREAM / IDEAM / ADA / APF / OA-CMPC.

Two pieces:

* :func:`build_simulator_obs` — builds the 17-dim vector from the
  (X0, X0_g, lane_left, lane_center, lane_right) state that the benchmark
  already maintains.

* :func:`decide_and_setup` — runs the policy on one obs, decodes to
  (lane_delta, speed_mode), and returns the `force_target_lane` and
  `target_speed` the benchmark should feed into ideam_agent_step.
  Also calls `mpc_ctrl.set_target_speed_override(v_target)` so the MPC
  reference tracks the commanded cruise speed.

The benchmark code should use these helpers as:

    obs = build_simulator_obs(X0, X0_g, ll, lc, lr, lane_now, v_ref)
    ft_lane, v_target = decide_and_setup(obs, policy, mpc_ctrl,
                                         lane_now, v_ref)
    out = ideam_agent_step(..., force_target_lane=ft_lane, ...)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from rl.policy.decision_policy import (
    DEC_OBS_DIM,
    build_decision_obs,
    decode_action,
    target_speed_from_action,
)


# ---------------------------------------------------------------------------
# Gap queries — same convention as uncertainty_merger_DREAM._lane_gap_and_dv
# but returning both front and rear nearest neighbours.
# ---------------------------------------------------------------------------

def _front_rear_gap(ego_s: float, ego_v: float, lane_rows) -> Tuple[
        Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Given the benchmark's `lane_*` vehicle-rows array (columns = state+ctx,
    col 0 is s, col 6 is vx — matching build_rl_observation's usage), find
    the nearest front and nearest rear vehicle. Returns two (ds, dvx)
    tuples or None when no vehicle is present in that half-corridor.
    """
    if lane_rows is None or len(lane_rows) == 0:
        return None, None

    best_f: Optional[Tuple[float, float]] = None
    best_r: Optional[Tuple[float, float]] = None
    best_f_ds = np.inf
    best_r_ds = -np.inf

    for row in np.asarray(lane_rows):
        ds = float(row[0]) - float(ego_s)
        v_other = float(row[6])
        dvx = v_other - float(ego_v)
        if ds >= 0.0 and ds < best_f_ds:
            best_f_ds = ds
            best_f = (ds, dvx)
        elif ds < 0.0 and ds > best_r_ds:
            best_r_ds = ds
            best_r = (ds, dvx)
    return best_f, best_r


def build_simulator_obs(
    X0,
    X0_g,
    lane_left,
    lane_center,
    lane_right,
    lane_now: int,
    lane_rel_start: int = 0,
    ey: float = 0.0,
    epsi: float = 0.0,
) -> np.ndarray:
    """
    Build the 17-dim decision observation from the benchmark simulator
    state. Mirrors the structure produced by
    rl.data.historical_extractor so train and deploy see the same schema.

    Args:
        X0, X0_g : current Frenet / global ego state (only X0[0] = v_x used).
        lane_left, lane_center, lane_right : the benchmark's per-lane
            vehicle arrays (rows of [s, ey, epsi, x, y, psi, vx, a]).
        lane_now : current ego lane index (0..2).
        lane_rel_start : ego lane index relative to episode start.
        ey, epsi : lateral and heading errors w.r.t. current lane centre.

    Returns:
        obs : np.ndarray shape (17,) float32.
    """
    ego_vx = float(X0[0])
    ego_vy = float(X0[1]) if len(X0) > 1 else 0.0
    ego_s  = float(X0[3]) if len(X0) > 3 else float(X0_g[0])

    lane_map = {0: lane_left, 1: lane_center, 2: lane_right}

    fs, rs = _front_rear_gap(ego_s, ego_vx, lane_map.get(lane_now))

    if lane_now - 1 >= 0:
        fl, rl = _front_rear_gap(ego_s, ego_vx, lane_map.get(lane_now - 1))
    else:
        fl, rl = None, None
    if lane_now + 1 <= 2:
        fr, rr = _front_rear_gap(ego_s, ego_vx, lane_map.get(lane_now + 1))
    else:
        fr, rr = None, None

    neighbours = {
        "front_same":  fs,
        "front_left":  fl,
        "front_right": fr,
        "rear_same":   rs,
        "rear_left":   rl,
        "rear_right":  rr,
    }

    obs = build_decision_obs(
        ego_vx=ego_vx,
        ego_vy=ego_vy,
        lane_rel=int(lane_rel_start),
        ey=float(ey),
        epsi=float(epsi),
        neighbours=neighbours,
    )
    return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# Policy inference + controller setup
# ---------------------------------------------------------------------------

def decide_and_setup(
    obs: np.ndarray,
    policy,
    mpc_ctrl,
    lane_now: int,
    v_ref: float,
    deterministic: bool = True,
) -> Tuple[int, float]:
    """
    Run the decision policy on one observation and wire up the MPC for
    the next solve.

    Args:
        obs           : 17-dim float32 vector from :func:`build_simulator_obs`.
        policy        : a :class:`rl.policy.decision_policy.DecisionPolicy`
                        instance (already loaded).
        mpc_ctrl      : the LMPC instance (or PRIDEAMController, which
                        exposes set_target_speed_override via passthrough).
        lane_now      : current ego lane index (0..2).
        v_ref         : the baseline reference speed [m/s] around which
                        speed offsets are applied.
        deterministic : if True, take argmax; otherwise sample.

    Returns:
        force_target_lane : int in [0, 2] to pass into ideam_agent_step.
        v_target          : float commanded cruise speed.
    """
    if obs.shape != (DEC_OBS_DIM,):
        raise ValueError(f"obs must be shape ({DEC_OBS_DIM},), got {obs.shape}")

    a = int(policy.act(obs, deterministic=deterministic))
    lane_delta, _speed_mode = decode_action(a)
    v_target = target_speed_from_action(a, v_ref)

    force_target_lane = int(np.clip(lane_now + lane_delta, 0, 2))
    mpc_ctrl.set_target_speed_override(float(v_target))
    return force_target_lane, float(v_target)
