"""
DREAM Highway RL Environment
==============================
gym.Env-compatible wrapper around the straight-highway DRIFT + IDM scenario.

This environment reuses all existing DREAM components without modification:
- IDM_general.IDM : surrounding vehicle simulation (9 IDM vehicles)
- KinematicModel  : ego vehicle kinematic bicycle model
- DRIFTInterface  : PDE risk field (DRIFT) with per-step evolution
- pde_solver      : compute_total_Q, compute_velocity_field, etc.

The RL agent provides a high-level tactical action (lane choice + speed mode).
A simple P-controller translates this into (acceleration, steering) commands
that are applied directly to the kinematic model.  The existing DRIFT risk field
runs in the background and is queried for observations and reward.

Action space (Discrete 9)
--------------------------
Index  Lane   Speed
  0    KEEP   MAINTAIN  (+0 m/s from target)
  1    KEEP   SLOWER    (−2 m/s)
  2    KEEP   FASTER    (+2 m/s)
  3    DOWN   MAINTAIN
  4    DOWN   SLOWER
  5    DOWN   FASTER
  6    UP     MAINTAIN
  7    UP     SLOWER
  8    UP     FASTER

Observation space (Box 20, float32, ~[-3, 3])
----------------------------------------------
See rl/obs/observation_builder.py for the full layout.

Episode termination
--------------------
- Collision (min gap < COLLISION_DIST)
- Off-road (|e_y| > OFFROAD_LATERAL)
- Stall (speed < STALL_SPEED for STALL_STEPS consecutive steps)
- MAX_STEPS reached
"""

import sys
import os
import math
import numpy as np

# Resolve paths
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Optional gymnasium / gym import
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_MODULE = "gymnasium"
except ImportError:
    try:
        import gym
        from gym import spaces
        _GYM_MODULE = "gym"
    except ImportError:
        # Provide a minimal stub so the file can be imported even without gym
        gym = None
        spaces = None
        _GYM_MODULE = None

# ---------------------------------------------------------------------------
# DREAM imports
# ---------------------------------------------------------------------------
import config_highway
import config as _cfg_module
# Monkey-patch so pde_solver picks up the highway grid (straight road, large domain)
_cfg_module.Config = config_highway.Config

from config_highway import Config as _cfg
from pde_solver import create_vehicle as drift_create_vehicle
from Integration.drift_interface import DRIFTInterface
from KinematicModel import KinematicModel
from IDM_general import IDM

from rl.config.rl_config import RLConfig, DEFAULT_CONFIG
from rl.obs.observation_builder import build_observation
from rl.reward.reward_fn import compute_reward, compute_safety_cost, terminal_reward


# ---------------------------------------------------------------------------
# Predefined initial conditions (from DREAM_run_simulation.py)
# ---------------------------------------------------------------------------

# Two scenario variants for training diversity
_SCENARIOS = {
    "dangerous": {
        "position": [
            18.0, 5.6, 7.0, 0.0,   # E0
            100.0, 5.2, 7.0, 0.0,  # E1
            130.0, 5.0, 7.0, 0.0,  # E2
            35.0,  9.0, 7.0, 0.0,  # U1
            75.0,  9.0, 7.0, 0.0,  # U2
            120.0, 9.0, 7.0, 0.0,  # U3
            34.0,  1.8, 7.0, 0.0,  # D1
            70.0,  2.0, 7.0, 0.0,  # D2
            140.0, 1.6, 7.0, 0.0,  # D3
            60.0, 5.3, 10.0        # ego x, y, v
        ],
        "initial_V": {
            "U1": 12.0, "U2": 10.0, "U3": 9.0,
            "D1":  9.0, "D2": 12.0, "D3": 9.0,
            "E0":  8.5, "E1": 11.5, "E2": 12.0,
        },
    },
    "faster": {
        "position": [
            16.8, 5.4, 6.9, 0.0,
            71.6, 4.9, 6.7, 0.0,
            146.5, 5.0, 7.3, 0.0,
            36.5,  8.8, 6.4, 0.0,
            70.4,  9.0, 6.8, 0.0,
            148.9, 9.0, 7.6, 0.0,
            37.5,  1.9, 7.9, 0.0,
            73.0,  1.6, 7.0, 0.0,
            153.9, 1.7, 6.5, 0.0,
            45.0,  5.1, 6.5,
        ],
        "initial_V": {
            "U1": 16.6, "U2": 16.1, "U3": 8.3,
            "D1": 12.8, "D2": 16.7, "D3": 12.1,
            "E0": 16.0, "E1":  8.6, "E2": 8.4,
        },
    },
}


def _make_base_env_class():
    """Return the environment class using whichever gym variant is available."""

    base = object if gym is None else gym.Env

    class DREAMHighwayEnv(base):
        """
        Straight 3-lane highway RL environment with DRIFT risk field.

        Scenarios rotate across reset() calls so the agent sees diverse traffic.
        """

        metadata = {'render_modes': ['rgb_array', 'none']}

        def __init__(self, config: RLConfig = None, scenario: str = "dangerous",
                     warmup: bool = True, render_mode: str = 'none'):
            """
            Args:
                config     : RLConfig instance (uses DEFAULT_CONFIG if None)
                scenario   : Key into _SCENARIOS dict, or 'random' to alternate
                warmup     : Whether to warm up the DRIFT field on reset()
                render_mode: 'none' or 'rgb_array'
            """
            super().__init__()
            self.config = config or DEFAULT_CONFIG
            self._scenario_key = scenario
            self._do_warmup = warmup
            self.render_mode = render_mode

            cfg = self.config

            # --- Gym spaces ---
            if spaces is not None:
                self.observation_space = spaces.Box(
                    low=-3.0, high=3.0,
                    shape=(cfg.OBS_DIM,), dtype=np.float32
                )
                self.action_space = spaces.Discrete(cfg.N_ACTIONS)

            # --- Internal state (initialised in reset()) ---
            self._idm: IDM = None
            self._ego: KinematicModel = None
            self._drift: DRIFTInterface = None
            self._step_count: int = 0
            self._current_lane: int = 1           # start in ego/E lane
            self._target_lane: int = 1
            self._steps_in_lc: int = 0            # steps into current lane change
            self._steps_since_lc: int = 50        # saturate to 50
            self._last_a: float = 0.0
            self._last_delta: float = 0.0
            self._prev_a: float = 0.0
            self._prev_delta: float = 0.0
            self._prev_x: float = 0.0
            self._stall_count: int = 0
            self._episode_logs: list = []         # per-step log dicts
            self._episode_return: float = 0.0
            self._episode_cost: float = 0.0
            self._n_episodes: int = 0

            # Scenario index for rotation
            self._scenario_idx: int = 0

        # ------------------------------------------------------------------
        # gym.Env interface
        # ------------------------------------------------------------------

        def reset(self, seed=None, options=None):
            """
            Reset environment to a new episode.

            Returns:
                obs  : initial observation (np.float32 array, shape [OBS_DIM])
                info : dict with initial diagnostics
            """
            cfg = self.config

            # Pick scenario
            if self._scenario_key == 'random':
                keys = list(_SCENARIOS.keys())
                key = keys[self._n_episodes % len(keys)]
            else:
                key = self._scenario_key
            scenario = _SCENARIOS[key]

            pos = list(scenario["position"])
            V0 = dict(scenario["initial_V"])

            # Add small random perturbations for training diversity
            rng = np.random.default_rng(seed if seed is not None else None)
            pos[-3] += rng.uniform(-5.0, 5.0)   # ego x ± 5 m
            pos[-1] += rng.uniform(-1.5, 1.5)   # ego v ± 1.5 m/s

            # Ensure position list has exactly 39 entries
            # IDM expects: E0x,E0y,E0v,E0a, E1..., U1..., D1..., ego_x, ego_y, ego_v
            # DREAM_run_simulation uses a different order — reorder to match IDM.__init__
            # Format: [E0x,E0y,E0v,E0a, E1x..., E2x...,
            #           U1x,U1y,U1v,U1a, U2..., U3...,
            #           D1x,D1y,D1v,D1a, D2..., D3...,
            #           ego_x, ego_y, ego_v]
            # The _SCENARIOS dicts are already in the IDM init order.
            idm_pos = pos

            self._idm = IDM(idm_pos, cfg.DT, cfg.WHEELBASE)

            # Override initial velocities
            for name, v in V0.items():
                setattr(self._idm, f"{name}_V", v)

            # Ego kinematic model
            ego_x = pos[-3]
            ego_y = pos[-2]
            ego_v = pos[-1]
            self._ego = KinematicModel(cfg.DT, cfg.WHEELBASE, x=ego_x, y=ego_y, v=ego_v)
            # Sync IDM's internal dynamics model position
            self._idm.dyn.x = ego_x
            self._idm.dyn.y = ego_y
            self._idm.dyn.v = ego_v
            self._idm.dyn.yaw = 0.0

            # Determine initial lane
            self._current_lane = _y_to_lane(ego_y, cfg)
            self._target_lane = self._current_lane

            # DRIFT interface (highway config already active via monkey-patch)
            self._drift = DRIFTInterface()

            # Warm-up
            if self._do_warmup:
                vehicles_init = _idm_to_drift_vehicles(self._idm)
                ego_dict = drift_create_vehicle(
                    vid=0, x=ego_x, y=ego_y, vx=ego_v, vy=0.0
                )
                ego_dict['heading'] = 0.0
                self._drift.warmup(vehicles_init, ego_dict, dt=cfg.DT,
                                   duration=cfg.DRIFT_WARMUP, substeps=cfg.DRIFT_SUBSTEPS)

            # Reset tracking state
            self._step_count = 0
            self._steps_since_lc = 50
            self._steps_in_lc = 0
            self._last_a = 0.0
            self._last_delta = 0.0
            self._prev_a = 0.0
            self._prev_delta = 0.0
            self._prev_x = ego_x
            self._stall_count = 0
            self._episode_logs = []
            self._episode_return = 0.0
            self._episode_cost = 0.0

            obs, obs_raw = self._build_obs()
            info = {
                'scenario': key,
                'ego_x': ego_x, 'ego_y': ego_y, 'ego_v': ego_v,
                'current_lane': self._current_lane,
                'obs_raw': obs_raw,
            }
            return obs, info

        def step(self, action: int):
            """
            Apply high-level tactical action and advance one timestep.

            Args:
                action: int in [0, N_ACTIONS)

            Returns:
                obs    : next observation (np.float32, shape [OBS_DIM])
                reward : scalar task reward
                terminated: bool — episode ended by failure or success
                truncated : bool — episode ended by time limit
                info   : dict with diagnostics
            """
            cfg = self.config
            action = int(action)

            # --- Decode action ---
            lane_delta   = cfg.LANE_DELTAS[action]
            speed_offset = cfg.SPEED_OFFSETS[action]
            target_v     = float(np.clip(
                cfg.TARGET_SPEED + speed_offset,
                cfg.MIN_SPEED, cfg.MAX_SPEED
            ))

            # --- Request lane change ---
            requested_lane = int(np.clip(self._current_lane + lane_delta, 0, 2))
            lc_requested = (self._steps_in_lc == 0 and requested_lane != self._current_lane)
            lc_rejected  = False
            if lc_requested:
                # Check safety before initiating lane change
                if self._is_lc_safe(requested_lane):
                    self._target_lane = requested_lane
                    self._steps_in_lc = cfg.TAU_LC
                    self._steps_since_lc = 0
                else:
                    # LC rejected by safety gate — flag it so PPO can learn
                    lc_rejected = True
                    lane_delta = 0   # reward sees effective action = lane-keep

            # --- Compute P-controller commands ---
            a_cmd, delta_cmd = self._compute_control(target_v)

            # --- Store previous actions for jerk computation ---
            self._prev_a = self._last_a
            self._prev_delta = self._last_delta
            self._prev_x = self._idm.dyn.x

            # --- Apply control to kinematic model ---
            self._idm.dyn.update_state(a_cmd, delta_cmd, cfg.MAX_STEER)
            self._last_a = a_cmd
            self._last_delta = delta_cmd

            # --- Update surrounding vehicles (IDM) ---
            _idm_update_surrounding(self._idm, cfg)

            # --- Advance lane change progress ---
            if self._steps_in_lc > 0:
                self._steps_in_lc -= 1
                if self._steps_in_lc == 0:
                    self._current_lane = self._target_lane
                    self._steps_since_lc = 0

            self._steps_since_lc = min(self._steps_since_lc + 1, 50)

            # --- Update DRIFT risk field ---
            ego_x = self._idm.dyn.x
            ego_y = self._idm.dyn.y
            ego_v = self._idm.dyn.v
            ego_yaw = self._idm.dyn.yaw

            vehicles_drift = _idm_to_drift_vehicles(self._idm)
            ego_dict = drift_create_vehicle(vid=0, x=ego_x, y=ego_y, vx=ego_v, vy=0.0)
            ego_dict['heading'] = ego_yaw
            self._drift.step(vehicles_drift, ego_dict,
                             dt=cfg.DT, substeps=cfg.DRIFT_SUBSTEPS)

            # --- Compute min gap to all surrounding vehicles ---
            min_gap = _compute_min_gap(self._idm, ego_x, ego_y)

            # --- Per-lane front-leader info for reward v2 ---
            _lane_info = _per_lane_front_leader(self._idm, ego_x, ego_v)
            _risk_per_lane = _per_lane_risk(self._drift, ego_x, cfg)

            # --- Observation ---
            obs, obs_raw = self._build_obs()

            # --- Reward ---
            reward, r_terms = compute_reward(
                ego_x=ego_x, ego_y=ego_y, ego_v=ego_v, ego_yaw=ego_yaw,
                prev_ego_x=self._prev_x,
                target_v=target_v,
                current_lane=self._current_lane,
                action=action,
                last_a=a_cmd, last_delta=delta_cmd,
                prev_a=self._prev_a, prev_delta=self._prev_delta,
                min_gap_all_lanes=min_gap,
                config=cfg,
                lane_delta=lane_delta,
                ds_curr=_lane_info[self._current_lane][0],
                dv_curr=_lane_info[self._current_lane][1],
                ds_left=_lane_info[min(self._current_lane + 1, 2)][0],
                dv_left=_lane_info[min(self._current_lane + 1, 2)][1],
                ds_right=_lane_info[max(self._current_lane - 1, 0)][0],
                dv_right=_lane_info[max(self._current_lane - 1, 0)][1],
                risk_curr=_risk_per_lane[self._current_lane],
                risk_left=_risk_per_lane[min(self._current_lane + 1, 2)],
                risk_right=_risk_per_lane[max(self._current_lane - 1, 0)],
                steps_in_lc=self._steps_in_lc,
                steps_since_lc=self._steps_since_lc,
            )

            # --- Safety cost ---
            cost, c_terms = compute_safety_cost(
                ego_x=ego_x, ego_y=ego_y,
                drift_interface=self._drift,
                min_gap_all_lanes=min_gap,
                config=cfg,
            )

            # Small penalty when the agent chose a LC that got rejected by
            # the safety gate — teaches the policy which situations allow LCs.
            if lc_rejected:
                reward -= 0.3
                r_terms['r_lc_rejected'] = -0.3
            else:
                r_terms['r_lc_rejected'] = 0.0

            # Fold cost into reward for basic PPO (weighted penalty)
            total_reward = reward - cfg.W_RISK * cost

            # --- Termination checks ---
            terminated = False
            truncated = False
            term_reason = None

            # Collision
            if min_gap < cfg.COLLISION_DIST:
                terminated = True
                term_reason = 'collision'
                total_reward += terminal_reward('collision', cfg)

            # Off-road
            lane_y = cfg.LANE_CENTERS[self._current_lane]
            e_y = ego_y - lane_y
            if abs(e_y) > cfg.OFFROAD_LATERAL and not terminated:
                # Off-road if ego has drifted more than 1 lane-width from centre
                if abs(ego_y - 5.25) > 5.0:  # >5 m from highway mid = off-road
                    terminated = True
                    term_reason = 'offroad'
                    total_reward += terminal_reward('offroad', cfg)

            # Stall
            if ego_v < cfg.STALL_SPEED:
                self._stall_count += 1
            else:
                self._stall_count = 0
            if self._stall_count >= cfg.STALL_STEPS and not terminated:
                terminated = True
                term_reason = 'stall'
                total_reward += terminal_reward('stall', cfg)

            # Time limit
            self._step_count += 1
            if self._step_count >= cfg.MAX_STEPS and not terminated:
                truncated = True
                term_reason = 'timeout'

            # --- Logging ---
            self._episode_return += total_reward
            self._episode_cost += cost
            log = {
                'step': self._step_count,
                'action': action, 'lane_delta': lane_delta, 'speed_offset': speed_offset,
                'ego_x': ego_x, 'ego_y': ego_y, 'ego_v': ego_v,
                'current_lane': self._current_lane, 'target_lane': self._target_lane,
                'a_cmd': a_cmd, 'delta_cmd': delta_cmd,
                'min_gap': min_gap,
                'reward': total_reward, 'cost': cost,
                **r_terms, **c_terms,
                'obs_raw': obs_raw,
                'terminated': terminated, 'truncated': truncated,
                'term_reason': term_reason,
            }
            self._episode_logs.append(log)

            if terminated or truncated:
                self._n_episodes += 1
                if self._n_episodes % cfg.LOG_EVERY == 0:
                    print(f"[DREAMEnv] ep={self._n_episodes:4d} "
                          f"steps={self._step_count:3d} "
                          f"return={self._episode_return:+.1f} "
                          f"cost={self._episode_cost:.2f} "
                          f"reason={term_reason}")

            info = {
                'step': self._step_count,
                'action': action,
                'ego_v': ego_v, 'ego_x': ego_x, 'ego_y': ego_y,
                'current_lane': self._current_lane,
                'min_gap': min_gap,
                'reward': reward, 'cost': cost,
                'r_terms': r_terms, 'c_terms': c_terms,
                'term_reason': term_reason,
                'cbf_blocked': False,   # placeholder for future MPC-CBF integration
                'mpc_failure': False,
                'obs_raw': obs_raw,
                'lc_requested': lc_requested,
                'lc_rejected': lc_rejected,
            }

            return obs, float(total_reward), terminated, truncated, info

        def render(self):
            if self.render_mode == 'rgb_array':
                return self._render_rgb()
            return None

        def close(self):
            self._drift = None
            self._idm = None

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------

        def _build_obs(self):
            """Assemble observation from current simulation state."""
            cfg = self.config
            idm = self._idm
            # Build vehicle state dict in the format expected by observation_builder
            idm_veh = {
                'U': [(idm.U1_X, idm.U1_V), (idm.U2_X, idm.U2_V), (idm.U3_X, idm.U3_V)],
                'E': [(idm.E0_X, idm.E0_V), (idm.E1_X, idm.E1_V), (idm.E2_X, idm.E2_V)],
                'D': [(idm.D1_X, idm.D1_V), (idm.D2_X, idm.D2_V), (idm.D3_X, idm.D3_V)],
            }

            # Current-lane leader acceleration (heuristic: use closest leader)
            ego_x = idm.dyn.x
            lane_key = {0: 'D', 1: 'E', 2: 'U'}
            curr_veh = idm_veh[lane_key[self._current_lane]]
            lead_a = 0.0
            best_ds = 9999.0
            for (vx, vv) in curr_veh:
                ds = vx - ego_x
                if 0 < ds < best_ds:
                    best_ds = ds
                    # Acceleration stored in IDM model attributes
                    a_map = {
                        'D': [idm.D1_a, idm.D2_a, idm.D3_a],
                        'E': [idm.E0_a, idm.E1_a, idm.E2_a],
                        'U': [idm.U1_a, idm.U2_a, idm.U3_a],
                    }
                    accs = a_map[lane_key[self._current_lane]]
                    vxs  = curr_veh
                    for i, (vxi, _) in enumerate(vxs):
                        if abs(vxi - vx) < 1e-3:
                            lead_a = accs[i]
                            break
            idm_veh['a_lead_curr'] = lead_a

            return build_observation(
                ego_x=idm.dyn.x,
                ego_y=idm.dyn.y,
                ego_v=idm.dyn.v,
                ego_yaw=idm.dyn.yaw,
                last_a=self._last_a,
                last_delta=self._last_delta,
                current_lane=self._current_lane,
                steps_since_lc=self._steps_since_lc,
                idm_vehicles=idm_veh,
                drift_interface=self._drift,
                config=cfg,
            )

        def _compute_control(self, target_v: float) -> tuple:
            """
            Simple P-controller tracking target speed and target lane centre.

            Returns (a_cmd, delta_cmd) clipped to actuator limits.
            """
            cfg = self.config
            idm = self._idm
            ego_y = idm.dyn.y
            ego_v = idm.dyn.v
            ego_yaw = idm.dyn.yaw

            # Target lateral position: interpolate toward target lane centre
            src_y = cfg.LANE_CENTERS[self._current_lane]
            tgt_y = cfg.LANE_CENTERS[self._target_lane]
            # During lane change (steps_in_lc > 0), progress linearly
            if self._steps_in_lc > 0:
                frac = 1.0 - (self._steps_in_lc / cfg.TAU_LC)
                desired_y = src_y + frac * (tgt_y - src_y)
            else:
                desired_y = tgt_y

            # Lateral P control → steering angle
            lat_err = desired_y - ego_y
            v_safe = max(ego_v, 0.5)  # avoid division by zero
            delta_cmd = cfg.K_LAT * lat_err / v_safe - cfg.K_LAT * 0.5 * ego_yaw
            delta_cmd = float(np.clip(delta_cmd, -cfg.MAX_STEER, cfg.MAX_STEER))

            # Longitudinal P control → acceleration
            v_err = target_v - ego_v
            a_cmd = cfg.K_LON * v_err
            a_cmd = float(np.clip(a_cmd, cfg.MIN_ACCEL, cfg.MAX_ACCEL))

            return a_cmd, delta_cmd

        def _is_lc_safe(self, target_lane: int) -> bool:
            """
            Lightweight safety check before initiating a lane change.

            Returns True if there is enough space in the target lane.
            """
            cfg = self.config
            idm = self._idm
            ego_x = idm.dyn.x
            ego_v = idm.dyn.v

            lane_key = {0: 'D', 1: 'E', 2: 'U'}
            tgt_key = lane_key[target_lane]
            veh_in_lane = {
                'D': [(idm.D1_X, idm.D1_V), (idm.D2_X, idm.D2_V), (idm.D3_X, idm.D3_V)],
                'E': [(idm.E0_X, idm.E0_V), (idm.E1_X, idm.E1_V), (idm.E2_X, idm.E2_V)],
                'U': [(idm.U1_X, idm.U1_V), (idm.U2_X, idm.U2_V), (idm.U3_X, idm.U3_V)],
            }[tgt_key]

            # Gap thresholds — tighter than before so LC is not trivially
            # blocked.  The reward's near-miss penalty already discourages
            # unsafe LCs; the safety gate only needs to prevent physical
            # overlap (collision).
            min_ahead = cfg.COLLISION_DIST * 2.0   # was NEAR_MISS_DIST*1.5 = 12m → now ~9m
            min_behind = cfg.COLLISION_DIST * 1.5  # was NEAR_MISS_DIST = 8m  → now ~6.75m

            for (vx, vv) in veh_in_lane:
                gap = vx - ego_x
                if 0 < gap < min_ahead:
                    return False  # vehicle too close ahead in target lane
                if gap < 0 and abs(gap) < min_behind:
                    return False  # vehicle too close behind in target lane

            # DRIFT risk check disabled for RL training — the reward function's
            # risk cost and near-miss penalty already discourage unsafe LCs.
            # Keeping this gate caused 91-100% LC rejection in training, which
            # prevents the agent from ever learning when LCs are appropriate.
            # The gap-based check above is sufficient for collision avoidance.

            return True

        def _render_rgb(self) -> np.ndarray:
            """Minimal RGB render (placeholder — not needed for training)."""
            return np.zeros((200, 400, 3), dtype=np.uint8)

        # ------------------------------------------------------------------
        # Accessors for smoke tests / analysis
        # ------------------------------------------------------------------

        def get_episode_logs(self) -> list:
            """Return per-step log dicts for the current episode."""
            return list(self._episode_logs)

        def get_risk_field(self) -> np.ndarray:
            """Return the current DRIFT risk field (2D array)."""
            if self._drift is not None:
                return self._drift.risk_field
            return np.zeros((_cfg.ny, _cfg.nx))

    return DREAMHighwayEnv


# Instantiate class (works with or without gym)
DREAMHighwayEnv = _make_base_env_class()


# ---------------------------------------------------------------------------
# Helper functions (module-level, reusable)
# ---------------------------------------------------------------------------

def _y_to_lane(y: float, cfg: RLConfig) -> int:
    """Map Cartesian y to lane index (0/1/2)."""
    if y < cfg.LANE_BOUNDARY_LOW:
        return 0
    elif y < cfg.LANE_BOUNDARY_HIGH:
        return 1
    else:
        return 2


def _idm_to_drift_vehicles(idm: IDM) -> list:
    """
    Convert IDM_general vehicle state to a list of DRIFT vehicle dicts.
    All vehicles have heading = 0.0 (straight highway).
    """
    vehicles = []
    vid = 1
    data = [
        (idm.U1_X, idm.U1_Y, idm.U1_V),
        (idm.U2_X, idm.U2_Y, idm.U2_V),
        (idm.U3_X, idm.U3_Y, idm.U3_V),
        (idm.E0_X, idm.E0_Y, idm.E0_V),
        (idm.E1_X, idm.E1_Y, idm.E1_V),
        (idm.E2_X, idm.E2_Y, idm.E2_V),
        (idm.D1_X, idm.D1_Y, idm.D1_V),
        (idm.D2_X, idm.D2_Y, idm.D2_V),
        (idm.D3_X, idm.D3_Y, idm.D3_V),
    ]
    for (x, y, v) in data:
        vd = drift_create_vehicle(vid=vid, x=x, y=y, vx=v, vy=0.0, vclass='car')
        vd['heading'] = 0.0
        vehicles.append(vd)
        vid += 1
    return vehicles


def _idm_update_surrounding(idm: IDM, cfg: RLConfig):
    """
    Advance all surrounding IDM vehicles by one timestep.
    Replicates the update logic from DREAM_run_simulation.py.
    """
    ude = idm.Judge_Location(cfg.LANE_BOUNDARY_HIGH, cfg.LANE_BOUNDARY_LOW)
    # IDM model parameters (defaults matching DREAM_run_simulation.py)
    L = cfg.WHEELBASE
    S0 = 2.0
    T_head = 1.5
    a_max = 5.0
    b = 1.67
    V0 = {
        "U1": 12.0, "U2": 10.0, "U3": 9.0,
        "D1":  9.0, "D2": 12.0, "D3": 9.0,
        "E0":  8.5, "E1": 11.5, "E2": 12.0,
    }

    if ude == "U":
        idm.update_state_onlane(cfg.LANE_BOUNDARY_HIGH, cfg.LANE_BOUNDARY_LOW,
                                L, S0, T_head, a_max, b, V0)
        idm.update_state_E(L, S0, T_head, a_max, b, V0)
        idm.update_state_D(L, S0, T_head, a_max, b, V0)
    elif ude == "D":
        idm.update_state_onlane(cfg.LANE_BOUNDARY_HIGH, cfg.LANE_BOUNDARY_LOW,
                                L, S0, T_head, a_max, b, V0)
        idm.update_state_E(L, S0, T_head, a_max, b, V0)
        idm.update_state_U(L, S0, T_head, a_max, b, V0)
    else:  # E
        idm.update_state_onlane(cfg.LANE_BOUNDARY_HIGH, cfg.LANE_BOUNDARY_LOW,
                                L, S0, T_head, a_max, b, V0)
        idm.update_state_U(L, S0, T_head, a_max, b, V0)
        idm.update_state_D(L, S0, T_head, a_max, b, V0)


def _compute_min_gap(idm: IDM, ego_x: float, ego_y: float = None,
                     car_length: float = 4.4, car_width: float = 2.0,
                     lateral_margin: float = 1.0) -> float:
    """
    Compute the minimum bumper-to-bumper longitudinal gap between the ego and
    surrounding vehicles, considering only vehicles within lateral_margin of
    the ego in the y-direction.

    Vehicles in different lanes (laterally separated by more than
    car_width + lateral_margin) are excluded — two vehicles in adjacent lanes
    that are longitudinally aligned do NOT constitute a collision.

    Args:
        idm           : IDM model with vehicle state attributes
        ego_x         : Ego x position [m]
        ego_y         : Ego y position [m] (None → use IDM dyn.y)
        car_length    : Vehicle length for bumper-to-bumper gap [m]
        car_width     : Vehicle width [m]
        lateral_margin: Extra lateral clearance beyond car_width [m]

    Returns:
        min_gap : Minimum bumper-to-bumper gap [m], ≥ 0
    """
    if ego_y is None:
        ego_y = idm.dyn.y

    # (x, y) pairs for all 9 surrounding vehicles
    veh_states = [
        (idm.U1_X, idm.U1_Y), (idm.U2_X, idm.U2_Y), (idm.U3_X, idm.U3_Y),
        (idm.E0_X, idm.E0_Y), (idm.E1_X, idm.E1_Y), (idm.E2_X, idm.E2_Y),
        (idm.D1_X, idm.D1_Y), (idm.D2_X, idm.D2_Y), (idm.D3_X, idm.D3_Y),
    ]

    # Only consider vehicles whose lateral centre is within (car_width + lateral_margin)
    # of the ego's lateral centre — i.e., same lane or adjacent lane during a lane change
    lat_thresh = car_width + lateral_margin  # default ~3.0 m

    min_gap = float('inf')
    for (vx, vy) in veh_states:
        if abs(vy - ego_y) > lat_thresh:
            continue   # different lane, no collision risk
        gap = abs(vx - ego_x) - car_length
        if gap < min_gap:
            min_gap = gap
    return max(0.0, min_gap)


def _per_lane_front_leader(idm: IDM, ego_x: float,
                           ego_v: float) -> dict:
    """
    For each lane (0/1/2), find the *nearest front leader* and return
    (ds, dv) where ds is bumper-to-bumper gap and dv is v_leader - v_ego
    (negative = closing).

    Returns dict {0: (ds, dv), 1: ..., 2: ...}.
    """
    # lane_key mapping: 0=D, 1=E, 2=U
    lane_veh = {
        0: [(idm.D1_X, idm.D1_V), (idm.D2_X, idm.D2_V), (idm.D3_X, idm.D3_V)],
        1: [(idm.E0_X, idm.E0_V), (idm.E1_X, idm.E1_V), (idm.E2_X, idm.E2_V)],
        2: [(idm.U1_X, idm.U1_V), (idm.U2_X, idm.U2_V), (idm.U3_X, idm.U3_V)],
    }
    result = {}
    for lane_id in (0, 1, 2):
        best_ds = 80.0   # default: no leader seen within 80m
        best_dv = 0.0
        for (vx, vv) in lane_veh[lane_id]:
            gap = vx - ego_x
            if gap > 0 and gap < best_ds:
                best_ds = gap
                best_dv = vv - ego_v
        result[lane_id] = (best_ds, best_dv)
    return result


def _per_lane_risk(drift_interface, ego_x: float,
                   cfg: RLConfig) -> dict:
    """
    Query DRIFT risk at the forward corridor of each lane centre.

    Returns dict {0: risk, 1: risk, 2: risk}.
    """
    result = {}
    for lane_id in (0, 1, 2):
        lane_y = cfg.LANE_CENTERS[lane_id]
        r = 0.0
        if drift_interface is not None:
            try:
                r = float(drift_interface.get_risk_corridor(
                    ego_x, lane_y, 0.0,
                    length=20.0, n_samples=4
                ))
            except Exception:
                pass
        result[lane_id] = r
    return result
