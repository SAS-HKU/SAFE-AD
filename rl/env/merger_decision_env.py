"""
Decision-layer RL environment for DREAM.
=========================================

A gym.Env that exposes the 17-dim decision observation + 9-way discrete
action schema defined in :mod:`rl.policy.decision_policy`, so that a
behaviour-cloned or PPO-trained ``DecisionPolicy`` can be fine-tuned
against live simulated traffic using the same obs/action layout that is
deployed in ``uncertainty_merger_DREAM.py``.

Design
------
We reuse :class:`rl.env.dream_env.DREAMHighwayEnv` as the underlying
simulator because it already provides:

* a kinematic-bicycle ego in a 3-lane scene,
* IDM-controlled surrounding traffic,
* the DRIFT risk field warmed up and stepped every frame,
* collision / offroad / stall / time-limit termination,
* a shaped reward in :mod:`rl.reward.reward_fn`,
* a ``Discrete(9)`` action space with the same lane-delta × speed-mode
  decomposition the BC policy was trained on.

What this wrapper adds
~~~~~~~~~~~~~~~~~~~~~~
1.  A 17-dim observation vector built via
    :func:`rl.policy.decision_policy.build_decision_obs`, so the policy
    sees the *same* layout it was pretrained on instead of the 22-dim
    legacy PPO obs.
2.  Neighbour slots (``front_{same,left,right}`` / ``rear_{...}``)
    synthesised from the IDM surrounding-vehicle state in the wrapped
    env, matching the ``_front_rear_gap`` convention used by
    :func:`rl.policy.decision_inference.build_simulator_obs`.
3.  A pass-through of the underlying env's reward / termination so PPO
    fine-tuning is meaningful without having to reimplement the reward.

The curved-road geometry of ``uncertainty_merger_DREAM.py`` is *not*
required for training: the decision policy operates on lane-relative
quantities that the wrapped env already produces for a straight
highway. A later version can re-point this env at a merger-style
scenario without changing the obs/action schema, which is the whole
reason the schema is defined centrally in :mod:`rl.policy.decision_policy`.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_MODULE = "gymnasium"
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
        _GYM_MODULE = "gym"
    except ImportError:
        gym = None
        spaces = None
        _GYM_MODULE = None

from rl.policy.decision_policy import (
    DEC_OBS_DIM,
    DEC_N_ACTIONS,
    build_decision_obs,
    PERCEPTION_RANGE,
)


# ---------------------------------------------------------------------------
# Neighbour synthesis from the DREAMHighwayEnv IDM state
# ---------------------------------------------------------------------------

def _collect_idm_vehicles(idm) -> Tuple[list, dict]:
    """
    Return ``(vehicles, lane_of_vid)`` where ``vehicles`` is a list of
    ``(x, y, v, lane_idx)`` and ``lane_of_vid`` maps a vehicle id to its
    lane index. Uses the fixed DREAMHighwayEnv slot naming (U=up, E=ego,
    D=down) → lane 2 / 1 / 0 respectively.
    """
    slots = [
        ("D1", idm.D1_X, idm.D1_V, 0),
        ("D2", idm.D2_X, idm.D2_V, 0),
        ("D3", idm.D3_X, idm.D3_V, 0),
        ("E0", idm.E0_X, idm.E0_V, 1),
        ("E1", idm.E1_X, idm.E1_V, 1),
        ("E2", idm.E2_X, idm.E2_V, 1),
        ("U1", idm.U1_X, idm.U1_V, 2),
        ("U2", idm.U2_X, idm.U2_V, 2),
        ("U3", idm.U3_X, idm.U3_V, 2),
    ]
    # DREAMHighwayEnv uses a fixed y per lane — pull lane centres from
    # the env's RLConfig via IDM-owning env. We just encode the lane
    # index; the decision obs does not use absolute y.
    vehicles = [(name, x, v, lane) for (name, x, v, lane) in slots]
    return vehicles, {name: lane for (name, _x, _v, lane) in slots}


def _neighbours_from_idm(idm, ego_lane: int) -> Dict[str, Optional[Tuple[float, float]]]:
    """
    Build the 6-slot neighbour dict expected by ``build_decision_obs``
    from the IDM surrounding-vehicle snapshot in the wrapped env.

    Slots are keyed by (front/rear) × (same/left/right), where "left"
    means ``ego_lane + 1`` and "right" means ``ego_lane - 1`` (IDM's
    U/E/D → lane 2/1/0 convention: U is leftward / up, D is rightward
    / down).
    """
    vehicles, _ = _collect_idm_vehicles(idm)
    ego_x = float(idm.dyn.x)
    ego_v = float(idm.dyn.v)

    slots = {k: None for k in (
        "front_same", "front_left", "front_right",
        "rear_same",  "rear_left",  "rear_right",
    )}
    best = {k: float("inf") for k in slots}

    for (_name, x, v, lane) in vehicles:
        ds = float(x) - ego_x
        if abs(ds) > PERCEPTION_RANGE:
            continue
        dvx = float(v) - ego_v

        if lane == ego_lane:
            side = "same"
        elif lane == ego_lane + 1:
            side = "left"
        elif lane == ego_lane - 1:
            side = "right"
        else:
            continue  # more than one lane away — no slot

        fr = "front" if ds >= 0.0 else "rear"
        key = f"{fr}_{side}"
        if abs(ds) < best[key]:
            best[key] = abs(ds)
            slots[key] = (ds, dvx)
    return slots


# ---------------------------------------------------------------------------
# Env class
# ---------------------------------------------------------------------------

def _make_base():
    return object if gym is None else gym.Env


class MergerDecisionEnv(_make_base()):
    """
    gym.Env exposing the 17-dim decision obs / 9-way discrete action
    schema, backed by :class:`DREAMHighwayEnv` for dynamics and reward.

    Parameters
    ----------
    config       : optional :class:`rl.config.rl_config.RLConfig`.
    scenario     : scenario key for the underlying DREAMHighwayEnv,
                   default ``"random"`` rotates through its presets.
    warmup       : whether to warm up the DRIFT field at reset.
    render_mode  : forwarded to the underlying env.
    """

    metadata = {"render_modes": ["rgb_array", "none"]}

    def __init__(self,
                 config=None,
                 scenario: str = "random",
                 warmup: bool = True,
                 render_mode: str = "none"):
        super().__init__()
        # Import here so the module can be imported even if pde_solver /
        # DRIFT dependencies are missing in a minimal env.
        from rl.env.dream_env import DREAMHighwayEnv  # noqa: WPS433
        self._inner = DREAMHighwayEnv(
            config=config, scenario=scenario,
            warmup=warmup, render_mode=render_mode,
        )
        self.config = self._inner.config
        self.render_mode = render_mode

        if spaces is not None:
            # 17-dim obs — clipped to ±3 like the other DREAM RL envs.
            self.observation_space = spaces.Box(
                low=-3.0, high=3.0, shape=(DEC_OBS_DIM,), dtype=np.float32,
            )
            self.action_space = spaces.Discrete(DEC_N_ACTIONS)

        self._lane_rel_start = 0  # ego start lane is treated as relative 0

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        _legacy_obs, info = self._inner.reset(seed=seed, options=options)
        self._lane_rel_start = int(self._inner._current_lane)
        obs = self._build_decision_obs()
        info = dict(info)
        info["legacy_obs"] = _legacy_obs
        return obs, info

    def step(self, action: int):
        _legacy_obs, reward, terminated, truncated, info = self._inner.step(int(action))
        obs = self._build_decision_obs()
        info = dict(info)
        info["legacy_obs"] = _legacy_obs
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._inner.render()

    def close(self):
        self._inner.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_decision_obs(self) -> np.ndarray:
        idm = self._inner._idm
        cfg = self._inner.config
        ego_lane = int(self._inner._current_lane)
        lane_rel = ego_lane - int(self._lane_rel_start)

        ego_vx = float(idm.dyn.v)
        ego_vy = 0.0  # straight-highway simplification

        # Lateral / heading error relative to current lane centre.
        lane_y = float(cfg.LANE_CENTERS[ego_lane])
        ey = float(idm.dyn.y) - lane_y
        epsi = float(idm.dyn.yaw)

        nb = _neighbours_from_idm(idm, ego_lane)

        obs = build_decision_obs(
            ego_vx=ego_vx,
            ego_vy=ego_vy,
            lane_rel=lane_rel,
            ey=ey,
            epsi=epsi,
            neighbours=nb,
        )
        return obs.astype(np.float32)
