from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIGHWAY_ENV_ROOT = os.path.join(REPO_ROOT, "HighwayEnv-master")
if HIGHWAY_ENV_ROOT not in sys.path:
    sys.path.insert(0, HIGHWAY_ENV_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import highway_env  # noqa: F401
from rl.data.social_features import field_metrics
from rl.env.highwayenv_decision_env import make_highwayenv_decision_env
from rl.env.highwayenv_drift_wrapper import DriftOverlayWrapper
from rl.env.highwayenv_scenarios import register_highwayenv_scenarios
from rl.reward.social_reward import (
    DEFAULT_SOCIAL_REWARD_CONFIG,
    SocialRewardConfig,
    compose_reward,
    decode_lane_delta,
    drac,
    hard_brake_imposed,
    lane_utility,
    rear_bad_cut_in_flag,
    safe_thw,
    safe_ttc,
)


register_highwayenv_scenarios()


@dataclass
class TrafficConfig:
    preset: str = "medium"
    vehicles_count: int | None = None
    vehicles_density: float | None = None
    ego_spacing: float | None = None
    duration: float | None = None
    sv_speed_min: float | None = 18.0
    sv_speed_max: float | None = 30.0
    sv_speed_noise: float = 1.5
    lane_speed_bias: tuple[float, ...] = ()
    resample_speeds: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRAFFIC_PRESETS: dict[str, dict[str, Any]] = {
    "native": {
        "vehicles_count": None,
        "vehicles_density": None,
        "ego_spacing": None,
        "duration": None,
        "sv_speed_min": None,
        "sv_speed_max": None,
        "sv_speed_noise": 0.0,
        "resample_speeds": False,
    },
    "sparse": {
        "vehicles_count": 10,
        "vehicles_density": 0.7,
        "ego_spacing": 2.0,
        "duration": 30.0,
        "sv_speed_min": 20.0,
        "sv_speed_max": 26.0,
        "sv_speed_noise": 1.0,
    },
    "medium": {
        "vehicles_count": 20,
        "vehicles_density": 1.0,
        "ego_spacing": 1.5,
        "duration": 30.0,
        "sv_speed_min": 18.0,
        "sv_speed_max": 30.0,
        "sv_speed_noise": 1.5,
    },
    "dense": {
        "vehicles_count": 35,
        "vehicles_density": 1.3,
        "ego_spacing": 1.2,
        "duration": 35.0,
        "sv_speed_min": 15.0,
        "sv_speed_max": 24.0,
        "sv_speed_noise": 2.0,
    },
    "slow": {
        "vehicles_count": 20,
        "vehicles_density": 1.0,
        "ego_spacing": 1.5,
        "duration": 30.0,
        "sv_speed_min": 12.0,
        "sv_speed_max": 18.0,
        "sv_speed_noise": 1.0,
    },
    "mixed": {
        "vehicles_count": 24,
        "vehicles_density": 1.0,
        "ego_spacing": 1.5,
        "duration": 30.0,
        "sv_speed_min": 15.0,
        "sv_speed_max": 30.0,
        "sv_speed_noise": 3.0,
    },
    "fast": {
        "vehicles_count": 20,
        "vehicles_density": 0.8,
        "ego_spacing": 1.5,
        "duration": 30.0,
        "sv_speed_min": 24.0,
        "sv_speed_max": 32.0,
        "sv_speed_noise": 2.0,
    },
}


def resolve_traffic_config(
    *,
    preset: str = "medium",
    vehicles_count: int | None = None,
    vehicles_density: float | None = None,
    ego_spacing: float | None = None,
    duration: float | None = None,
    sv_speed_min: float | None = None,
    sv_speed_max: float | None = None,
    sv_speed_noise: float | None = None,
    lane_speed_bias: tuple[float, ...] = (),
    resample_speeds: bool | None = None,
) -> TrafficConfig:
    base = dict(TRAFFIC_PRESETS.get(str(preset), TRAFFIC_PRESETS["medium"]))
    if vehicles_count is not None:
        base["vehicles_count"] = int(vehicles_count)
    if vehicles_density is not None:
        base["vehicles_density"] = float(vehicles_density)
    if ego_spacing is not None:
        base["ego_spacing"] = float(ego_spacing)
    if duration is not None:
        base["duration"] = float(duration)
    if sv_speed_min is not None:
        base["sv_speed_min"] = float(sv_speed_min)
    if sv_speed_max is not None:
        base["sv_speed_max"] = float(sv_speed_max)
    if sv_speed_noise is not None:
        base["sv_speed_noise"] = float(sv_speed_noise)
    if resample_speeds is not None:
        base["resample_speeds"] = bool(resample_speeds)
    return TrafficConfig(
        preset=str(preset),
        vehicles_count=int(base["vehicles_count"]) if base.get("vehicles_count") is not None else None,
        vehicles_density=float(base["vehicles_density"]) if base.get("vehicles_density") is not None else None,
        ego_spacing=float(base["ego_spacing"]) if base.get("ego_spacing") is not None else None,
        duration=float(base["duration"]) if base.get("duration") is not None else None,
        sv_speed_min=float(base["sv_speed_min"]) if base.get("sv_speed_min") is not None else None,
        sv_speed_max=float(base["sv_speed_max"]) if base.get("sv_speed_max") is not None else None,
        sv_speed_noise=float(base["sv_speed_noise"]),
        lane_speed_bias=tuple(float(x) for x in lane_speed_bias),
        resample_speeds=bool(base.get("resample_speeds", True)),
    )


def stock_env_config(traffic: TrafficConfig, *, action_mode: str = "default") -> dict[str, Any]:
    config = {"show_trajectories": False}
    if traffic.vehicles_count is not None:
        config["vehicles_count"] = int(traffic.vehicles_count)
    if traffic.vehicles_density is not None:
        config["vehicles_density"] = float(traffic.vehicles_density)
    if traffic.ego_spacing is not None:
        config["ego_spacing"] = float(traffic.ego_spacing)
    if traffic.duration is not None:
        config["duration"] = float(traffic.duration)
    if action_mode == "continuous":
        config["action"] = {
            "type": "ContinuousAction",
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-0.7853981633974483, 0.7853981633974483],
        }
    elif action_mode == "discrete_meta":
        config["action"] = {"type": "DiscreteMetaAction"}
    elif action_mode == "discrete_kinematic":
        config["action"] = {
            "type": "DiscreteAction",
            "actions_per_axis": 3,
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-0.7853981633974483, 0.7853981633974483],
        }
    elif action_mode != "default":
        raise ValueError(f"Unsupported HighwayEnv action_mode '{action_mode}'")
    return config


def _raw_env_from(env) -> Any:
    current = env
    while current is not None:
        if hasattr(current, "raw_env"):
            return current.raw_env
        if hasattr(current, "_inner"):
            return current._inner
        if not hasattr(current, "env"):
            break
        current = current.env
    return env.unwrapped


def _find_wrapper_attr(env, attr: str):
    current = env
    while current is not None:
        if hasattr(current, attr):
            return getattr(current, attr)
        if not hasattr(current, "env"):
            break
        current = current.env
    return None


def _build_current_observation(env) -> np.ndarray:
    if hasattr(env, "_build_decision_obs"):
        return env._build_decision_obs().astype(np.float32)
    raw = _raw_env_from(env)
    return np.asarray(raw.observation_type.observe(), dtype=np.float32)


class StockTrafficWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, traffic: TrafficConfig) -> None:
        super().__init__(env)
        self.traffic = traffic

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        raw = _raw_env_from(self.env)
        if (
            not bool(self.traffic.resample_speeds)
            or self.traffic.sv_speed_min is None
            or self.traffic.sv_speed_max is None
        ):
            obs = _build_current_observation(self.env)
            info = dict(info)
            info["traffic_config"] = self.traffic.to_dict()
            return obs, info
        rng = getattr(raw, "np_random", None)
        if rng is None:
            rng = np.random.default_rng(seed)
        for vehicle in list(getattr(raw.road, "vehicles", [])):
            if vehicle is getattr(raw, "vehicle", None):
                continue
            if vehicle in getattr(raw, "controlled_vehicles", []):
                continue
            lane_bias = 0.0
            lane_index = getattr(vehicle, "lane_index", None)
            if lane_index is not None and self.traffic.lane_speed_bias:
                lane_id = int(lane_index[2])
                if 0 <= lane_id < len(self.traffic.lane_speed_bias):
                    lane_bias = float(self.traffic.lane_speed_bias[lane_id])
            sampled = float(rng.uniform(self.traffic.sv_speed_min, self.traffic.sv_speed_max))
            sampled += float(rng.normal(0.0, self.traffic.sv_speed_noise))
            sampled += lane_bias
            sampled = float(np.clip(sampled, self.traffic.sv_speed_min, self.traffic.sv_speed_max))
            vehicle.speed = sampled
            if hasattr(vehicle, "target_speed"):
                vehicle.target_speed = sampled
        obs = _build_current_observation(self.env)
        info = dict(info)
        info["traffic_config"] = self.traffic.to_dict()
        return obs, info


class HighwaySocialRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        reward_config: SocialRewardConfig | None = None,
        ablation: str = "full",
    ) -> None:
        super().__init__(env)
        self.reward_config = reward_config or DEFAULT_SOCIAL_REWARD_CONFIG
        self.ablation = str(ablation)
        self._prev_snapshot: dict[str, Any] | None = None
        self._episode_steps: list[dict[str, float]] = []
        self._prev_action: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_steps = []
        self._prev_action = None
        self._prev_snapshot = self._snapshot()
        info = dict(info)
        info["social_reward_config"] = self.reward_config.to_dict()
        info["social_ablation"] = self.ablation
        return obs, info

    def step(self, action):
        obs, stock_reward, terminated, truncated, info = self.env.step(action)
        prev = self._prev_snapshot or self._snapshot()
        curr = self._snapshot()
        lane_delta = self._lane_delta(action, prev, curr)
        action_vec = np.asarray(action, dtype=np.float64).reshape(-1)
        if self._prev_action is None:
            action_delta = 0.0
        else:
            n = min(int(action_vec.size), int(self._prev_action.size))
            action_delta = float(np.mean(np.abs(action_vec[:n] - self._prev_action[:n]))) if n > 0 else 0.0
        self._prev_action = action_vec
        progress_delta = self._progress_delta(prev, curr)

        if lane_delta > 0:
            prev_rear_ttc = float(prev["rear_ttc_left"])
            prev_rear_accel = float(prev["rear_accel_left"])
        elif lane_delta < 0:
            prev_rear_ttc = float(prev["rear_ttc_right"])
            prev_rear_accel = float(prev["rear_accel_right"])
        else:
            prev_rear_ttc = float(prev["rear_ttc_same"])
            prev_rear_accel = float(prev["rear_accel_same"])

        if lane_delta > 0:
            curr_rear_ttc = float(curr["rear_ttc_left"])
            curr_rear_accel = float(curr["rear_accel_left"])
        elif lane_delta < 0:
            curr_rear_ttc = float(curr["rear_ttc_right"])
            curr_rear_accel = float(curr["rear_accel_right"])
        else:
            curr_rear_ttc = float(curr["rear_ttc_same"])
            curr_rear_accel = float(curr["rear_accel_same"])

        imposed_rear_decel = max(0.0, -curr_rear_accel)
        bad_cut_in = bool(lane_delta != 0 and rear_bad_cut_in_flag(curr_rear_ttc, prev_rear_ttc))

        reward, terms = compose_reward(
            stock_reward=float(stock_reward),
            delta_x=progress_delta,
            dt=max(1e-6, float(curr["dt"])),
            speed=float(curr["speed"]),
            cruise_speed=float(curr["cruise_speed"]),
            lane_delta=lane_delta,
            adv_left=float(prev["adv_left"]),
            adv_right=float(prev["adv_right"]),
            blocked=bool(prev["blocked"]),
            ttc_min=float(curr["ttc_min"]),
            thw=float(curr["thw_same"]),
            drac_value=float(curr["drac_same"]),
            imposed_rear_decel=imposed_rear_decel,
            rear_ttc_now=curr_rear_ttc,
            rear_ttc_prev=prev_rear_ttc,
            bad_cut_in=bad_cut_in,
            accel=float(curr["accel"]),
            steer=float(curr["steer"]),
            jerk=float(curr["jerk"]),
            r_corr=float(curr["r_corr"]),
            flux_back=float(curr["risk_flux_backward"]),
            cfg=self.reward_config,
            ablation=self.ablation,
        )

        social_step = {
            "reward_total": float(reward),
            "stock_reward": float(stock_reward),
            "ttc_min": float(curr["ttc_min"]),
            "thw_same": float(curr["thw_same"]),
            "drac_same": float(curr["drac_same"]),
            "progress_dx": progress_delta,
            "speed": float(curr["speed"]),
            "accel": float(curr["accel"]),
            "steer": float(curr["steer"]),
            "jerk": float(curr["jerk"]),
            "action_delta": float(action_delta),
            "lane_change_cmd": float(abs(lane_delta) > 0),
            "best_adv": float(prev["best_adv"]),
            "blocked": float(bool(prev["blocked"])),
            "missed_opportunity": float(bool(lane_delta == 0 and prev["blocked"] and prev["best_adv"] > self.reward_config.missed_opportunity_adv)),
            "imposed_rear_decel": float(imposed_rear_decel),
            "hard_brake_imposed": float(hard_brake_imposed(-imposed_rear_decel)),
            "bad_cut_in": float(bad_cut_in),
            "r_corr": float(curr["r_corr"]),
            "r_left": float(curr["r_left"]),
            "r_right": float(curr["r_right"]),
            "risk_flux_backward": float(curr["risk_flux_backward"]),
            "risk_mass_total": float(curr["risk_mass_total"]),
            "collision": float(bool(curr["crashed"])),
        }
        social_step.update({k: float(v) for k, v in terms.items()})
        self._episode_steps.append(social_step)
        self._prev_snapshot = curr

        info = dict(info)
        info["social_reward_terms"] = dict(terms)
        info["social_step"] = dict(social_step)
        info["social_ablation"] = self.ablation
        info["social_reward_config"] = self.reward_config.to_dict()
        if terminated or truncated:
            info["social_episode_summary"] = self._episode_summary()
        return obs, float(reward), terminated, truncated, info

    def _lane_delta(self, action, prev: dict[str, Any], curr: dict[str, Any]) -> int:
        if hasattr(self.action_space, "n"):
            try:
                return decode_lane_delta(int(action), int(self.action_space.n))
            except (TypeError, ValueError):
                return 0
        prev_lane = prev.get("lane_index")
        curr_lane = curr.get("lane_index")
        try:
            prev_id = int(prev_lane[2])
            curr_id = int(curr_lane[2])
        except (TypeError, ValueError, IndexError):
            return 0
        delta = curr_id - prev_id
        if delta > 0:
            return +1
        if delta < 0:
            return -1
        return 0

    def _episode_summary(self) -> dict[str, float]:
        if not self._episode_steps:
            return {}
        keys = sorted(self._episode_steps[0].keys())
        return {
            key: float(np.nanmean([row[key] for row in self._episode_steps]))
            for key in keys
        } | {
            "episode_length": float(len(self._episode_steps)),
            "collision_any": float(max(row["collision"] for row in self._episode_steps)),
            "bad_cut_in_any": float(max(row["bad_cut_in"] for row in self._episode_steps)),
        }

    @staticmethod
    def _progress_delta(prev: dict[str, Any], curr: dict[str, Any]) -> float:
        prev_pos = np.asarray(prev["position"], dtype=np.float64)
        curr_pos = np.asarray(curr["position"], dtype=np.float64)
        return float(np.linalg.norm(curr_pos - prev_pos))

    @staticmethod
    def _lane_speed(vehicle, lane_index) -> float:
        lane = vehicle.road.network.get_lane(lane_index)
        s, _lat = lane.local_coordinates(vehicle.position)
        lane_heading = lane.heading_at(s)
        return float(vehicle.speed * math.cos(float(vehicle.heading - lane_heading)))

    def _lane_roles(self, raw, current_lane):
        current_y = raw.road.network.get_lane(current_lane).position(0.0, 0.0)[1]
        left_lane = None
        right_lane = None
        for candidate in raw.road.network.side_lanes(current_lane):
            candidate_y = raw.road.network.get_lane(candidate).position(0.0, 0.0)[1]
            if candidate_y < current_y:
                left_lane = candidate
            elif candidate_y > current_y:
                right_lane = candidate
        return left_lane, right_lane

    def _neighbor_metrics(self, raw, ego, lane_index, ego_s: float, ego_vx: float) -> tuple[dict[str, float], dict[str, float]]:
        front, rear = raw.road.neighbour_vehicles(ego, lane_index)
        lane = raw.road.network.get_lane(lane_index)

        def _encode(other, *, is_rear: bool) -> dict[str, float]:
            if other is None:
                return {
                    "gap": 80.0,
                    "dv": 0.0,
                    "speed": 0.0,
                    "ttc": 60.0,
                    "thw": 60.0,
                    "accel": 0.0,
                }
            other_s, _lat = lane.local_coordinates(other.position)
            ds = float(other_s - ego_s)
            gap = max(0.0, abs(ds))
            other_vx = self._lane_speed(other, lane_index)
            dv = float(other_vx - ego_vx)
            if is_rear:
                closing = max(0.0, other_vx - ego_vx)
                ttc = safe_ttc(gap, closing)
                thw = safe_thw(gap, max(0.0, other_vx))
            else:
                closing = max(0.0, ego_vx - other_vx)
                ttc = safe_ttc(gap, closing)
                thw = safe_thw(gap, max(0.0, ego_vx))
            accel = 0.0
            action = getattr(other, "action", None)
            if isinstance(action, dict):
                accel = float(action.get("acceleration", 0.0))
            return {
                "gap": gap,
                "dv": dv,
                "speed": float(other_vx),
                "ttc": float(ttc),
                "thw": float(thw),
                "accel": float(accel),
            }

        return _encode(front, is_rear=False), _encode(rear, is_rear=True)

    def _snapshot(self) -> dict[str, Any]:
        raw = _raw_env_from(self.env)
        ego = raw.vehicle
        lane_index = ego.lane_index
        lane = raw.road.network.get_lane(lane_index)
        ego_s, _lat = lane.local_coordinates(ego.position)
        lane_heading = lane.heading_at(ego_s)
        ego_vx = float(ego.speed * math.cos(float(ego.heading - lane_heading)))
        dt = 1.0 / float(raw.config["policy_frequency"])
        prev_speed = None if self._prev_snapshot is None else float(self._prev_snapshot["speed"])
        accel = 0.0
        steer = 0.0
        action = getattr(ego, "action", None)
        if isinstance(action, dict):
            accel = float(action.get("acceleration", 0.0))
            steer = float(action.get("steering", 0.0))
        elif prev_speed is not None:
            accel = float((ego_vx - prev_speed) / max(1e-6, dt))
        prev_accel = 0.0 if self._prev_snapshot is None else float(self._prev_snapshot["accel"])
        jerk = float((accel - prev_accel) / max(1e-6, dt))

        drift_metrics_fn = _find_wrapper_attr(self.env, "current_drift_metrics")
        drift_metrics = dict(drift_metrics_fn()) if callable(drift_metrics_fn) else {}
        r_corr = float(drift_metrics.get("r_fwd", 0.0))
        r_left = float(drift_metrics.get("r_left", 0.0))
        r_right = float(drift_metrics.get("r_right", 0.0))

        left_lane, right_lane = self._lane_roles(raw, lane_index)
        front_same, rear_same = self._neighbor_metrics(raw, ego, lane_index, ego_s, ego_vx)
        if left_lane is not None:
            front_left, rear_left = self._neighbor_metrics(raw, ego, left_lane, ego_s, ego_vx)
        else:
            front_left = rear_left = {"gap": 80.0, "dv": 0.0, "speed": 0.0, "ttc": 60.0, "thw": 60.0, "accel": 0.0}
        if right_lane is not None:
            front_right, rear_right = self._neighbor_metrics(raw, ego, right_lane, ego_s, ego_vx)
        else:
            front_right = rear_right = {"gap": 80.0, "dv": 0.0, "speed": 0.0, "ttc": 60.0, "thw": 60.0, "accel": 0.0}

        utility_curr = lane_utility(front_same["gap"], front_same["dv"], r_corr, self.reward_config)
        utility_left = lane_utility(front_left["gap"], front_left["dv"], r_left, self.reward_config)
        utility_right = lane_utility(front_right["gap"], front_right["dv"], r_right, self.reward_config)
        adv_left = float(utility_left - utility_curr)
        adv_right = float(utility_right - utility_curr)
        best_adv = float(max(adv_left, adv_right))

        leader_speed = ego_vx + float(front_same["dv"]) if np.isfinite(front_same["dv"]) else 0.0
        blocked = bool(
            front_same["gap"] < float(self.reward_config.blocked_gap_thr)
            and leader_speed < float(self.reward_config.blocked_speed_frac) * max(float(ego_vx), 2.0)
        )

        risk_field_fn = _find_wrapper_attr(self.env, "get_masked_risk_field")
        drift_grid_fn = _find_wrapper_attr(self.env, "get_drift_grid")
        risk_field = risk_field_fn() if callable(risk_field_fn) else None
        grid_X, grid_Y = drift_grid_fn() if callable(drift_grid_fn) else (None, None)
        nbr_xs = []
        nbr_ys = []
        nbr_closing = []
        world_to_drift = _find_wrapper_attr(self.env, "world_to_drift")
        for other in raw.road.vehicles:
            if other is ego:
                continue
            x = float(other.position[0]) - float(ego.position[0])
            y = float(other.position[1]) - float(ego.position[1])
            if callable(world_to_drift):
                dx, dy = world_to_drift(float(other.position[0]), float(other.position[1]))
                ex, ey = world_to_drift(float(ego.position[0]), float(ego.position[1]))
                x = float(dx - ex)
                y = float(dy - ey)
            nbr_xs.append(x)
            nbr_ys.append(y)
            other_vx = float(other.velocity[0])
            nbr_closing.append(float(other_vx - ego.velocity[0]))
        if risk_field is not None and grid_X is not None and grid_Y is not None:
            externality = field_metrics(
                np.asarray(np.nan_to_num(risk_field, nan=0.0), dtype=np.float64),
                np.asarray(grid_X, dtype=np.float64),
                np.asarray(grid_Y, dtype=np.float64),
                np.asarray(nbr_xs, dtype=np.float64),
                np.asarray(nbr_ys, dtype=np.float64),
                np.asarray(nbr_closing, dtype=np.float64),
            )
        else:
            externality = {
                "risk_mass_total": 0.0,
                "risk_mass_others": 0.0,
                "risk_gradient_peak": 0.0,
                "risk_flux_backward": 0.0,
                "risk_field_entropy": 0.0,
            }

        cruise_speed = float(self.reward_config.progress_ref_speed)
        reward_speed_range = raw.config.get("reward_speed_range")
        if reward_speed_range and len(reward_speed_range) == 2:
            cruise_speed = 0.5 * (float(reward_speed_range[0]) + float(reward_speed_range[1]))

        ttc_min = float(min(front_same["ttc"], front_left["ttc"], front_right["ttc"]))
        return {
            "x": float(ego.position[0]),
            "position": (float(ego.position[0]), float(ego.position[1])),
            "lane_index": lane_index,
            "speed": float(ego_vx),
            "accel": float(accel),
            "steer": float(steer),
            "jerk": float(jerk),
            "dt": float(dt),
            "cruise_speed": cruise_speed,
            "adv_left": adv_left,
            "adv_right": adv_right,
            "best_adv": best_adv,
            "blocked": blocked,
            "ttc_min": ttc_min,
            "thw_same": float(front_same["thw"]),
            "drac_same": float(drac(front_same["gap"], max(0.0, ego_vx - front_same["speed"]))),
            "rear_ttc_same": float(rear_same["ttc"]),
            "rear_ttc_left": float(rear_left["ttc"]),
            "rear_ttc_right": float(rear_right["ttc"]),
            "rear_accel_same": float(rear_same["accel"]),
            "rear_accel_left": float(rear_left["accel"]),
            "rear_accel_right": float(rear_right["accel"]),
            "r_corr": r_corr,
            "r_left": r_left,
            "r_right": r_right,
            "risk_flux_backward": float(externality["risk_flux_backward"]),
            "risk_mass_total": float(externality["risk_mass_total"]),
            "crashed": bool(ego.crashed),
        }


def load_reward_config(path: str | None) -> SocialRewardConfig:
    if not path:
        return DEFAULT_SOCIAL_REWARD_CONFIG
    return SocialRewardConfig.load(path)


def make_social_highwayenv_env(
    *,
    env_id: str = "highway-fast-v0",
    interface: str = "stock",
    render_mode: str | None = None,
    traffic: TrafficConfig | None = None,
    reward_config: SocialRewardConfig | None = None,
    ablation: str = "full",
    use_drift: bool = True,
    drift_warmup_s: float = 1.0,
    reward_gate_scale_r0: float | None = None,
    risk_clip: float | None = None,
    record_risk_metrics: bool = False,
    action_mode: str = "default",
) -> gym.Env:
    traffic = traffic or resolve_traffic_config()
    reward_config = reward_config or DEFAULT_SOCIAL_REWARD_CONFIG
    reward_gate_scale_r0 = (
        float(reward_gate_scale_r0)
        if reward_gate_scale_r0 is not None
        else float(reward_config.risk_gate_r0)
    )
    risk_clip = float(risk_clip) if risk_clip is not None else float(reward_config.risk_clip)

    if interface == "stock":
        base_env = gym.make(
            env_id,
            render_mode=render_mode,
            config=stock_env_config(traffic, action_mode=action_mode),
        )
    elif interface == "decision":
        if action_mode not in {"default", "discrete_meta"}:
            raise ValueError("The HighwayEnv decision interface only supports the discrete decision action protocol.")
        base_env = make_highwayenv_decision_env(
            env_id=env_id,
            env_config=stock_env_config(traffic, action_mode="default"),
            render_mode=render_mode,
            use_drift=False,
        )
    else:
        raise ValueError(f"Unsupported interface '{interface}'")

    env: gym.Env = StockTrafficWrapper(base_env, traffic)
    if use_drift:
        env = DriftOverlayWrapper(
            env,
            use_drift=True,
            drift_warmup_s=drift_warmup_s,
            reward_gate_scale_r0=reward_gate_scale_r0,
            risk_clip=risk_clip,
            record_risk_metrics=record_risk_metrics,
            gate_reward=False,
        )
    env = HighwaySocialRewardWrapper(
        env,
        reward_config=reward_config,
        ablation=ablation,
    )
    setattr(env, "env_id", env_id)
    setattr(env, "interface", interface)
    setattr(env, "action_mode", action_mode)
    setattr(env, "traffic_config", traffic.to_dict())
    return env
