"""
Decision-policy wrapper for local highway-env scenarios.
========================================================

Expose selected ``HighwayEnv-master`` scenarios through the same
17-dim observation / 9-action decision interface used by
``rl.policy.decision_policy.DecisionPolicy``.

Why this wrapper exists
-----------------------
The upstream highway-env baselines use their own observation spaces and a
5-action meta-action interface. Our policy checkpoint is neither an SB3
policy nor trained on that schema; it expects the 17-dim handcrafted
decision observation and emits one of 9 combined
``lane_delta x speed_mode`` actions.

This wrapper keeps the policy contract fixed and adapts the environment
instead:

* observations are rebuilt from lane-local geometry and nearest
  neighbours,
* actions directly set the ego vehicle's target lane and target speed,
* rewards and terminations are still sourced from highway-env so the
  scenario remains meaningful.

Recommended scenarios
---------------------
``merge-v0`` is the best fit for the current policy because it is the
closest analogue to the project's merger / lane-change setting.
``highway-v0`` and ``highway-fast-v0`` also fit reasonably well.
``roundabout-v0`` and ``u-turn-v0`` are supported but induce a larger
domain shift because the local lane geometry is curved.
``lane-keeping-v0`` is intentionally unsupported because it is a
continuous steering task rather than a tactical lane/speed decision task.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIGHWAY_ENV_ROOT = os.path.join(REPO_ROOT, "HighwayEnv-master")
if HIGHWAY_ENV_ROOT not in sys.path:
    sys.path.insert(0, HIGHWAY_ENV_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
from gymnasium import spaces

import highway_env  # noqa: F401
from rl.env.highwayenv_scenarios import register_highwayenv_scenarios

from rl.policy.decision_policy import (
    DEC_N_ACTIONS,
    DEC_OBS_DIM,
    SPEED_OFFSETS,
    build_decision_obs,
    decode_action,
)


register_highwayenv_scenarios()

RECOMMENDED_ENVS = {"merge-v0", "highway-fast-v0", "occluded-merger-v0"}
SUPPORTED_ENVS = RECOMMENDED_ENVS | {"highway-v0", "roundabout-v0", "u-turn-v0", "two-way-v0", "two-threat-v0"}
UNSUPPORTED_ENVS = {"lane-keeping-v0"}


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _lane_count(env, lane_index) -> int:
    if lane_index is None:
        return 0
    _from, _to, _id = lane_index
    return len(env.road.network.graph[_from][_to])


def _lane_speed(vehicle, lane_index) -> float:
    lane = vehicle.road.network.get_lane(lane_index)
    s, _lat = lane.local_coordinates(vehicle.position)
    lane_heading = lane.heading_at(s)
    return float(vehicle.speed * np.cos(_wrap_to_pi(vehicle.heading - lane_heading)))


def _lane_error_state(vehicle, lane_index) -> tuple[float, float, float, float]:
    lane = vehicle.road.network.get_lane(lane_index)
    s, ey = lane.local_coordinates(vehicle.position)
    lane_heading = lane.heading_at(s)
    epsi = _wrap_to_pi(vehicle.heading - lane_heading)
    ego_vx = float(vehicle.speed * np.cos(epsi))
    ego_vy = float(vehicle.speed * np.sin(epsi))
    return ego_vx, ego_vy, float(ey), float(epsi)


def _lane_neighbour(road, ego_vehicle, lane_index, ego_s: float, ego_vx: float):
    if lane_index is None:
        return None, None

    front, rear = road.neighbour_vehicles(ego_vehicle, lane_index)
    lane = road.network.get_lane(lane_index)

    def _encode(other):
        if other is None:
            return None
        other_s, _lat = lane.local_coordinates(other.position)
        ds = float(other_s - ego_s)
        dvx = float(_lane_speed(other, lane_index) - ego_vx)
        return ds, dvx

    return _encode(front), _encode(rear)


class HighwayDecisionEnv(gym.Env):
    """
    Wrap one highway-env scenario with the project's decision policy API.

    Parameters
    ----------
    env_id:
        Scenario id registered by local ``HighwayEnv-master``.
    env_config:
        Optional config updates forwarded to highway-env at reset/build.
    render_mode:
        Passed through to the inner env.
    speed_bounds:
        Optional ``(min_speed, max_speed)`` override for commanded target
        speeds. When omitted, reward speed range or vehicle target speeds
        are used.
    """

    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        env_id: str = "merge-v0",
        env_config: Optional[dict[str, Any]] = None,
        render_mode: str | None = None,
        speed_bounds: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        if env_id in UNSUPPORTED_ENVS:
            raise ValueError(
                f"{env_id} is not compatible with the decision policy; it is a continuous-control task."
            )
        if env_id not in SUPPORTED_ENVS:
            raise ValueError(
                f"Unsupported env_id '{env_id}'. Supported: {sorted(SUPPORTED_ENVS)}"
            )

        self.env_id = env_id
        self.render_mode = render_mode
        self._env_config = dict(env_config or {})
        self._outer = gym.make(env_id, render_mode=render_mode)
        self._inner = self._outer.unwrapped
        if self._env_config:
            self._inner.configure(self._env_config)

        self._max_episode_steps = None
        spec = getattr(self._outer, "spec", None)
        if spec is not None:
            self._max_episode_steps = getattr(spec, "max_episode_steps", None)

        self._manual_steps = 0
        self._start_lane_semantic = 0
        self._speed_bounds_override = speed_bounds

        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(DEC_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(DEC_N_ACTIONS)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._manual_steps = 0
        merged_options = dict(options or {})
        if self._env_config:
            merged_options["config"] = {
                **merged_options.get("config", {}),
                **self._env_config,
            }
        _obs, info = self._outer.reset(seed=seed, options=merged_options or None)
        self._start_lane_semantic = self._semantic_lane_index(self._inner.vehicle.lane_index)
        obs = self._build_decision_obs()
        info = dict(info)
        info["legacy_obs"] = _obs
        info["decision_env_id"] = self.env_id
        return obs, info

    def step(self, action: int):
        self._manual_steps += 1
        action = int(action)

        lane_delta, _speed_mode = decode_action(action)
        lane_changed = False
        lc_rejected = False

        ego = self._inner.vehicle
        target_lane = ego.target_lane_index or ego.lane_index
        env_lane_step = -lane_delta

        if env_lane_step != 0:
            candidate = self._candidate_lane(target_lane, env_lane_step)
            if candidate is not None:
                ego.target_lane_index = candidate
                lane_changed = candidate != target_lane
            else:
                lc_rejected = True

        speed_lo, speed_hi = self._speed_bounds(ego)
        base_speed = float(getattr(ego, "target_speed", ego.speed))
        ego.target_speed = float(
            np.clip(base_speed + float(SPEED_OFFSETS[action]), speed_lo, speed_hi)
        )

        self._inner.time += 1 / self._inner.config["policy_frequency"]
        self._inner._simulate(None)

        obs = self._build_decision_obs()
        mapped_action = self._reward_action_for_decision(lane_delta, action)
        reward = float(self._inner._reward(mapped_action))
        terminated = bool(self._inner._is_terminated())
        truncated = bool(self._inner._is_truncated())
        if self._max_episode_steps is not None and self._manual_steps >= self._max_episode_steps:
            truncated = True

        info = dict(self._inner._info(obs, mapped_action))
        info["decision_action"] = action
        info["decision_env_id"] = self.env_id
        info["lane_changed_commanded"] = lane_changed
        info["lc_rejected"] = lc_rejected
        info["target_speed"] = float(ego.target_speed)

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._outer.render()

    def close(self):
        self._outer.close()

    def _candidate_lane(self, lane_index, env_lane_step: int):
        if lane_index is None:
            return None
        _from, _to, lane_id = lane_index
        lane_count = len(self._inner.road.network.graph[_from][_to])
        candidate = (_from, _to, int(np.clip(lane_id + env_lane_step, 0, lane_count - 1)))
        if candidate == lane_index:
            return None
        lane = self._inner.road.network.get_lane(candidate)
        if not lane.is_reachable_from(self._inner.vehicle.position):
            return None
        return candidate

    def _speed_bounds(self, ego) -> tuple[float, float]:
        if self._speed_bounds_override is not None:
            return self._speed_bounds_override
        reward_range = self._inner.config.get("reward_speed_range")
        if reward_range and len(reward_range) == 2:
            return float(reward_range[0]), float(reward_range[1])
        if hasattr(ego, "target_speeds") and ego.target_speeds is not None:
            target_speeds = np.asarray(ego.target_speeds, dtype=np.float32)
            return float(target_speeds.min()), float(target_speeds.max())
        return 0.0, max(30.0, float(getattr(ego, "speed", 0.0)) + 10.0)

    def _reward_action_for_decision(self, lane_delta: int, action: int) -> int:
        if lane_delta > 0:
            return 0  # highway-env LANE_LEFT
        if lane_delta < 0:
            return 2  # highway-env LANE_RIGHT
        speed_mode = int(action % 3)
        if speed_mode == 1:
            return 4  # SLOWER
        if speed_mode == 2:
            return 3  # FASTER
        return 1      # IDLE

    def _semantic_lane_index(self, lane_index) -> int:
        if lane_index is None:
            return 0
        lane_count = _lane_count(self._inner, lane_index)
        return int(lane_count - 1 - lane_index[2])

    def _build_decision_obs(self) -> np.ndarray:
        ego = self._inner.vehicle
        lane_index = ego.lane_index
        lane = self._inner.road.network.get_lane(lane_index)
        ego_s, _lat = lane.local_coordinates(ego.position)
        ego_vx, ego_vy, ey, epsi = _lane_error_state(ego, lane_index)

        semantic_lane = self._semantic_lane_index(lane_index)
        lane_rel = int(semantic_lane - self._start_lane_semantic)

        left_lane = self._candidate_lane(lane_index, -1)
        right_lane = self._candidate_lane(lane_index, +1)

        fs, rs = _lane_neighbour(self._inner.road, ego, lane_index, ego_s, ego_vx)
        fl, rl = _lane_neighbour(self._inner.road, ego, left_lane, ego_s, ego_vx)
        fr, rr = _lane_neighbour(self._inner.road, ego, right_lane, ego_s, ego_vx)

        obs = build_decision_obs(
            ego_vx=ego_vx,
            ego_vy=ego_vy,
            lane_rel=lane_rel,
            ey=ey,
            epsi=epsi,
            neighbours={
                "front_same": fs,
                "front_left": fl,
                "front_right": fr,
                "rear_same": rs,
                "rear_left": rl,
                "rear_right": rr,
            },
        )
        return obs.astype(np.float32)


def make_highwayenv_decision_env(
    env_id: str = "merge-v0",
    env_config: Optional[dict[str, Any]] = None,
    render_mode: str | None = None,
    speed_bounds: Optional[tuple[float, float]] = None,
    *,
    use_drift: bool = False,
    drift_warmup_s: float = 2.0,
    reward_gate_scale_r0: float = 1.5,
    risk_clip: float = 5.0,
    record_risk_metrics: bool = False,
):
    env: gym.Env = HighwayDecisionEnv(
        env_id=env_id,
        env_config=env_config,
        render_mode=render_mode,
        speed_bounds=speed_bounds,
    )
    if use_drift:
        from rl.env.highwayenv_drift_wrapper import DriftOverlayWrapper

        env = DriftOverlayWrapper(
            env,
            use_drift=True,
            drift_warmup_s=drift_warmup_s,
            reward_gate_scale_r0=reward_gate_scale_r0,
            risk_clip=risk_clip,
            record_risk_metrics=record_risk_metrics,
        )
    return env
