from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401

    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False


REPO_ROOT = Path(__file__).resolve().parents[1]
HIGHWAYENV_ROOT = REPO_ROOT / "HighwayEnv-master"
for _path in (str(HIGHWAYENV_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gymnasium as gym
import highway_env  # noqa: F401
from rl.utils.typing_compat import ensure_typing_extensions_compat

ensure_typing_extensions_compat()
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

from highway_env.vehicle.behavior import IDMVehicle

from rl.data.risk_proxy import (
    risk_corridor_tau,
    risk_max_along_segment,
)
from rl.data.social_features import (
    BAD_CUT_TTC_ABS,
    BAD_CUT_TTC_DROP_FRAC,
    MISSED_OPP_BEST_ADV,
    RiskFieldQuery,
    SOCIAL_CLASS_NAMES,
    composite_scores,
    courtesy_block,
    decision_block,
    display_name,
    frame_interaction_block,
    field_metrics,
    social_traffic_efficiency_index,
    surrogate_safety_block,
    temporal_propagation_block,
)
from rl.reward.social_reward import DEFAULT_SOCIAL_REWARD_CONFIG, lane_utility


TTC_CAP = 60.0
NEAR_COLLISION_DIST = 8.0
COLLISION_DIST = 3.0
CRITICAL_TTC = 3.0
DEFAULT_POLICY_LABELS = {"ppo": "stock-ppo", "dqn": "stock-dqn"}
DEFAULT_BASELINE_LABEL = "IDM/MOBIL"
SUMMARY_TABLE_FIELDS = [
    ("return_mean", "Return", "higher"),
    ("collision_rate", "Collision Rate", "lower"),
    ("ttc_min_mean", "TTC Min", "higher"),
    ("cmr_mean", "Criticality Rate", "lower"),
    ("min_spacing_mean", "Min Spacing", "higher"),
    ("mean_speed", "Mean Speed", "higher"),
    ("mean_abs_jerk", "Mean |Jerk|", "lower"),
    ("final_progress_mean", "Final Progress", "higher"),
    ("risk_mass_others_mean", "Imposed Risk Potential", "lower"),
    ("risk_flux_backward_mean", "Backward Risk Flux", "lower"),
    ("interaction_density_mean", "Interaction Density", "lower"),
    ("risk_mass_per_agent_mean", "Mean Risk per Vehicle", "lower"),
    ("backward_risk_flux_ratio_mean", "Backward Flux Ratio", "lower"),
    ("mean_speed_frame_mean", "Frame Mean Speed", "higher"),
    ("speed_variance_frame_mean", "Frame Speed Variance", "lower"),
    ("total_progress_rate_frame_mean", "Frame Total Progress Rate", "higher"),
    ("efficiency_index_ei_mean", "Efficiency Index (EI)", "higher"),
    ("safety_efficiency_index_sei_mean", "Safety-Efficiency Index (SEI)", "higher"),
    ("social_traffic_efficiency_index_mean", "Social Traffic Efficiency Index", "higher"),
    ("shockwave_onset_flag_mean", "Shockwave Onset Rate", "lower"),
    ("ttc_min_s_mean", "Frame Min TTC", "higher"),
    ("frac_critical_ttc_mean", "Frame Frac TTC < 1.5s", "lower"),
    ("max_drac_mps2_mean", "Frame Max DRAC", "lower"),
    ("safety_score_mean", "Safety Score", "higher"),
    ("courtesy_score_mean", "Courtesy Score", "higher"),
    ("social_friendliness_score_mean", "Social-Friendliness Score", "higher"),
]


@dataclass
class EpisodeResult:
    planner: str
    algo: str
    seed: int
    episode_return: float
    episode_length: int
    crashed: bool
    truncated: bool
    ttc_min: float
    cmr: float
    min_spacing: float
    mean_speed: float
    mean_reward: float
    mean_abs_jerk: float
    final_progress: float
    right_lane_score: float
    near_collision_rate: float
    near_collision_any: bool
    social_summary: dict[str, float]
    log: dict[str, list[float]]


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _apply_style() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "grid", "no-latex"])
    else:
        plt.style.use("default")


def _nanmean_or_nan(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanstd_or_nan(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr))


def _checkpoint_stem(path: str) -> str:
    return path[:-4] if str(path).endswith(".zip") else str(path)


def _checkpoint_file(path: str) -> str:
    return str(path) if str(path).endswith(".zip") else f"{path}.zip"


def _make_env_kwargs(duration: float | None = None) -> dict:
    config = {"show_trajectories": False}
    if duration is not None:
        config["duration"] = float(duration)
    return {"config": config}


def train_or_load_sb3_policy(
    algo: str,
    checkpoint_path: str,
    *,
    train_env_id: str = "highway-fast-v0",
    total_timesteps: int = 20_000,
    n_envs: int = 4,
    force_train: bool = False,
    duration: float | None = None,
) -> str:
    checkpoint_zip = _checkpoint_file(checkpoint_path)
    if os.path.exists(checkpoint_zip) and not force_train:
        return checkpoint_zip

    os.makedirs(os.path.dirname(checkpoint_zip) or ".", exist_ok=True)
    env_kwargs = _make_env_kwargs(duration=duration)
    algo = str(algo).strip().lower()
    if algo == "ppo":
        batch_size = 64
        vec_env = make_vec_env(train_env_id, n_envs=max(1, int(n_envs)), env_kwargs=env_kwargs)
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            n_steps=max(8, batch_size * 12 // max(1, int(n_envs))),
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=1,
            tensorboard_log=None,
        )
        model.learn(total_timesteps=int(total_timesteps))
        model.save(_checkpoint_stem(checkpoint_path))
        vec_env.close()
        return checkpoint_zip

    if algo == "dqn":
        env = gym.make(train_env_id, **env_kwargs)
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log=None,
        )
        model.learn(total_timesteps=int(total_timesteps))
        model.save(_checkpoint_stem(checkpoint_path))
        env.close()
        return checkpoint_zip

    raise ValueError(f"Unsupported algo '{algo}'")


def load_sb3_policy(algo: str, checkpoint_path: str):
    checkpoint_zip = _checkpoint_file(checkpoint_path)
    algo = str(algo).strip().lower()
    if algo == "ppo":
        return PPO.load(checkpoint_zip)
    if algo == "dqn":
        return DQN.load(checkpoint_zip)
    raise ValueError(f"Unsupported algo '{algo}'")


def _make_eval_env(env_id: str, *, duration: float | None = None, render_mode: str | None = None):
    env_kwargs = _make_env_kwargs(duration=duration)
    return gym.make(env_id, render_mode=render_mode, **env_kwargs)


def _swap_ego_to_idm(env) -> None:
    raw = env.unwrapped
    old_vehicle = raw.vehicle
    idm_vehicle = IDMVehicle.create_from(old_vehicle)
    idx = raw.road.vehicles.index(old_vehicle)
    raw.road.vehicles[idx] = idm_vehicle
    raw.controlled_vehicles[0] = idm_vehicle
    raw.vehicle = idm_vehicle
    raw.action_type.controlled_vehicle = idm_vehicle


def _forward_speed(vehicle) -> float:
    return float(vehicle.speed * math.cos(float(vehicle.heading)))


def _lane_score(raw) -> float:
    lane = int(raw.vehicle.lane_index[2]) if raw.vehicle.lane_index is not None else 0
    neighbours = raw.road.network.all_side_lanes(raw.vehicle.lane_index)
    denom = max(len(neighbours) - 1, 1)
    return float(lane / denom)


def _pair_bumper_distance(ego, other) -> float:
    delta = np.asarray(other.position, dtype=float) - np.asarray(ego.position, dtype=float)
    center_dist = float(np.linalg.norm(delta))
    bumper = center_dist - 0.5 * (
        float(getattr(ego, "LENGTH", 5.0)) + float(getattr(other, "LENGTH", 5.0))
    )
    return max(0.0, bumper)


def _pair_ttc(ego, other) -> float:
    delta = np.asarray(other.position, dtype=float) - np.asarray(ego.position, dtype=float)
    dist = float(np.linalg.norm(delta))
    bumper_dist = _pair_bumper_distance(ego, other)
    if bumper_dist <= 0.0 or dist < 1e-6:
        return 0.0
    rel_vel = np.asarray(ego.velocity, dtype=float) - np.asarray(other.velocity, dtype=float)
    closing = float(np.dot(rel_vel, delta / dist))
    if closing <= 1e-6:
        return TTC_CAP
    return float(min(bumper_dist / closing, TTC_CAP))


def _scene_metrics(raw) -> tuple[float, float]:
    ego = raw.vehicle
    min_spacing = float("inf")
    min_ttc = TTC_CAP
    for other in raw.road.vehicles:
        if other is ego:
            continue
        min_spacing = min(min_spacing, _pair_bumper_distance(ego, other))
        min_ttc = min(min_ttc, _pair_ttc(ego, other))
    if not np.isfinite(min_spacing):
        min_spacing = float("nan")
    if not np.isfinite(min_ttc):
        min_ttc = TTC_CAP
    return float(min_spacing), float(min_ttc)


def _current_accel(raw, prev_speed: float | None, dt: float) -> float:
    action = getattr(raw.vehicle, "action", None)
    if isinstance(action, dict) and "acceleration" in action:
        return float(action["acceleration"])
    if prev_speed is None or dt <= 0.0:
        return 0.0
    return float((_forward_speed(raw.vehicle) - prev_speed) / dt)


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _lane_long_speed(vehicle, lane_index) -> float:
    lane = vehicle.road.network.get_lane(lane_index)
    s, _lat = lane.local_coordinates(vehicle.position)
    lane_heading = lane.heading_at(s)
    return float(vehicle.speed * np.cos(_wrap_to_pi(vehicle.heading - lane_heading)))


def _lane_roles(raw, current_lane):
    current_y = raw.road.network.get_lane(current_lane).position(0.0, 0.0)[1]
    left_lane = None
    right_lane = None
    for candidate in raw.road.network.side_lanes(current_lane):
        candidate_y = raw.road.network.get_lane(candidate).position(0.0, 0.0)[1]
        # HighwayEnv convention in this repo: higher y is rightward.
        if candidate_y < current_y:
            left_lane = candidate
        elif candidate_y > current_y:
            right_lane = candidate
    return left_lane, right_lane


def _world_to_ego_frame(raw, point_xy: np.ndarray) -> tuple[float, float]:
    ego = raw.vehicle
    dx = float(point_xy[0] - ego.position[0])
    dy = float(point_xy[1] - ego.position[1])
    c = float(np.cos(ego.heading))
    s = float(np.sin(ego.heading))
    x_ego = dx * c + dy * s
    y_ego = -dx * s + dy * c
    return x_ego, y_ego


def _lane_neighbor_state(raw, lane_index, ego_s: float, ego_vx_lane: float):
    ego = raw.vehicle
    front, rear = raw.road.neighbour_vehicles(ego, lane_index)
    lane = raw.road.network.get_lane(lane_index)

    def _encode(other):
        if other is None:
            return None
        other_s, _lat = lane.local_coordinates(other.position)
        ds = float(other_s - ego_s)
        vx_long = _lane_long_speed(other, lane_index)
        dvx = float(vx_long - ego_vx_lane)
        x_ego, y_ego = _world_to_ego_frame(raw, np.asarray(other.position, dtype=float))
        accel = 0.0
        action = getattr(other, "action", None)
        if isinstance(action, dict):
            accel = float(action.get("acceleration", 0.0))
        return {
            "ds": ds,
            "dvx": dvx,
            "vx_long": float(vx_long),
            "x": float(x_ego),
            "y": float(y_ego),
            "accel": float(accel),
        }

    return _encode(front), _encode(rear)


def _current_lane_direction(raw, prev_lane_index, current_lane_index) -> str:
    if prev_lane_index is None or current_lane_index is None or prev_lane_index == current_lane_index:
        return "same"
    prev_y = raw.road.network.get_lane(prev_lane_index).position(0.0, 0.0)[1]
    curr_y = raw.road.network.get_lane(current_lane_index).position(0.0, 0.0)[1]
    if curr_y > prev_y:
        return "right"
    return "left"


def _action_direction(action: int) -> str:
    if int(action) == 0:
        return "left"
    if int(action) == 2:
        return "right"
    return "same"


def _neighbor_arrays(raw, ego_vx_world: float):
    ego = raw.vehicle
    xs = []
    ys = []
    closing = []
    for other in raw.road.vehicles:
        if other is ego:
            continue
        x_ego, y_ego = _world_to_ego_frame(raw, np.asarray(other.position, dtype=float))
        c = float(np.cos(ego.heading))
        s = float(np.sin(ego.heading))
        v_long_other = float(other.velocity[0] * c + other.velocity[1] * s)
        xs.append(float(x_ego))
        ys.append(float(y_ego))
        closing.append(float(ego_vx_world - v_long_other))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(closing, dtype=np.float32),
    )


def _frame_entries(raw):
    entries = []
    for idx, vehicle in enumerate(raw.road.vehicles):
        entries.append(
            (
                int(idx),
                float(vehicle.position[0]),
                float(vehicle.position[1]),
                float(vehicle.velocity[0]),
                float(vehicle.velocity[1]),
                float(vehicle.heading),
            )
        )
    return entries


def _frame_macro_snapshot(raw, prev_macro: dict[str, float] | None, *, dt: float, hard_brake_rate: float = 0.0) -> dict[str, float]:
    entries = _frame_entries(raw)
    interaction = frame_interaction_block(entries)
    temporal = temporal_propagation_block(
        prev_risk_mass=None if prev_macro is None else prev_macro.get("risk_mass_frame"),
        curr_risk_mass=float(interaction["risk_mass_frame"]),
        prev_speed_var=None if prev_macro is None else prev_macro.get("speed_variance_frame"),
        curr_speed_var=float(interaction["speed_variance_frame"]),
        dt=float(dt),
    )
    ssm = surrogate_safety_block(entries)
    risk_adjusted_progress = float(
        interaction["total_progress_rate_frame"] / (1.0 + interaction["risk_mass_per_agent"])
    )
    stei = float(
        social_traffic_efficiency_index(
            progress_rate=float(interaction["mean_speed_frame"]),
            risk_mass_per_agent=float(interaction["risk_mass_per_agent"]),
            risk_others_per_agent=0.0,
            backward_risk_flux_ratio=float(interaction["backward_risk_flux_ratio"]),
            speed_variance=float(interaction["speed_variance_frame"]),
            hard_brake_rate=float(hard_brake_rate),
        )
    )
    return {
        **interaction,
        **temporal,
        **ssm,
        "risk_adjusted_progress": risk_adjusted_progress,
        "social_traffic_efficiency_index": stei,
    }


def _social_snapshot(raw) -> dict[str, object]:
    ego = raw.vehicle
    lane_index = ego.lane_index
    lane = raw.road.network.get_lane(lane_index)
    ego_s, _lat = lane.local_coordinates(ego.position)
    lane_heading = lane.heading_at(ego_s)
    ego_vx_lane = float(ego.speed * np.cos(_wrap_to_pi(ego.heading - lane_heading)))
    ego_vx_world = float(ego.velocity[0] * np.cos(ego.heading) + ego.velocity[1] * np.sin(ego.heading))
    left_lane, right_lane = _lane_roles(raw, lane_index)
    front_same, rear_same = _lane_neighbor_state(raw, lane_index, ego_s, ego_vx_lane)
    front_left, rear_left = (_lane_neighbor_state(raw, left_lane, ego_s, ego_vx_lane) if left_lane is not None else (None, None))
    front_right, rear_right = (_lane_neighbor_state(raw, right_lane, ego_s, ego_vx_lane) if right_lane is not None else (None, None))

    nbr_xs_ego, nbr_ys_ego, nbr_closing = _neighbor_arrays(raw, ego_vx_world)
    risk_query = RiskFieldQuery(mode="analytic")
    R, X, Y = risk_query.field_grid(nbr_xs_ego, nbr_ys_ego, nbr_closing)
    ext = field_metrics(R, X, Y, nbr_xs_ego, nbr_ys_ego, nbr_closing)

    def _front_gap_dv(slot):
        if slot is None:
            return 80.0, 0.0
        return float(abs(slot["ds"])), float(slot["dvx"])

    g_curr, dv_curr = _front_gap_dv(front_same)
    g_left, dv_left = _front_gap_dv(front_left)
    g_right, dv_right = _front_gap_dv(front_right)

    r_curr = float(np.mean([risk_corridor_tau(ego_vx_lane, 0.0, tau, 6, nbr_xs_ego, nbr_ys_ego, nbr_closing) for tau in (1.0, 2.0, 3.0)]))
    r_left = float(np.mean([risk_corridor_tau(ego_vx_lane, 4.0, tau, 6, nbr_xs_ego, nbr_ys_ego, nbr_closing) for tau in (1.0, 2.0, 3.0)])) if left_lane is not None else r_curr
    r_right = float(np.mean([risk_corridor_tau(ego_vx_lane, -4.0, tau, 6, nbr_xs_ego, nbr_ys_ego, nbr_closing) for tau in (1.0, 2.0, 3.0)])) if right_lane is not None else r_curr
    utility_curr = lane_utility(g_curr, dv_curr, r_curr, DEFAULT_SOCIAL_REWARD_CONFIG)
    utility_left = lane_utility(g_left, dv_left, r_left, DEFAULT_SOCIAL_REWARD_CONFIG) if left_lane is not None else utility_curr - 999.0
    utility_right = lane_utility(g_right, dv_right, r_right, DEFAULT_SOCIAL_REWARD_CONFIG) if right_lane is not None else utility_curr - 999.0
    adv_left = float(utility_left - utility_curr) if left_lane is not None else float("-inf")
    adv_right = float(utility_right - utility_curr) if right_lane is not None else float("-inf")
    best_adv = float(max(adv_left, adv_right))
    leader_v_abs_curr = ego_vx_lane + dv_curr if front_same is not None else 0.0
    blocked = int((g_curr < 25.0) and (ego_vx_lane > 2.0) and (leader_v_abs_curr < 0.7 * ego_vx_lane))
    corridor_risk = float(risk_max_along_segment(0.0, 0.0, max(5.0, ego_vx_lane * 2.0), 8, nbr_xs_ego, nbr_ys_ego, nbr_closing))

    return {
        "lane_index": lane_index,
        "ego_speed": float(ego_vx_lane),
        "best_adv": best_adv,
        "blocked_by_leader_flag": int(blocked),
        "rear_same": rear_same,
        "rear_left": rear_left,
        "rear_right": rear_right,
        "corridor_risk": corridor_risk,
        "risk_mass_total": float(ext["risk_mass_total"]),
        "risk_mass_others": float(ext["risk_mass_others"]),
        "risk_gradient_peak": float(ext["risk_gradient_peak"]),
        "risk_flux_backward": float(ext["risk_flux_backward"]),
        "risk_field_entropy": float(ext["risk_field_entropy"]),
    }


def _select_rear(snapshot: dict[str, object], direction: str):
    if direction == "left":
        return snapshot.get("rear_left")
    if direction == "right":
        return snapshot.get("rear_right")
    return snapshot.get("rear_same")


def _compute_social_step(prev_snap: dict[str, object], curr_snap: dict[str, object], lane_direction: str, min_spacing: float, min_ttc: float, crashed: bool) -> dict[str, float]:
    lane_delta_label = 0 if lane_direction == "same" else (1 if lane_direction == "left" else -1)
    prev_rear = _select_rear(prev_snap, lane_direction)
    curr_rear = _select_rear(curr_snap, lane_direction)
    traj = None
    if curr_rear is not None:
        traj = np.asarray(
            [[
                float(curr_rear["x"]),
                float(curr_rear["y"]),
                float(curr_rear["vx_long"]),
                float(curr_rear["accel"]),
            ]],
            dtype=np.float32,
        )
    courtesy = courtesy_block(
        prev_rear,
        traj,
        ego_speed_now=float(prev_snap["ego_speed"]),
        ego_speed_after=float(curr_snap["ego_speed"]),
        dt=1.0,
    )
    decision = decision_block(
        best_adv=float(prev_snap["best_adv"]),
        lane_delta_label=int(lane_delta_label),
        blocked_by_leader_flag=int(prev_snap["blocked_by_leader_flag"]),
    )
    near_miss_future = int(
        (np.isfinite(min_spacing) and float(min_spacing) < NEAR_COLLISION_DIST)
        or (np.isfinite(min_ttc) and float(min_ttc) < CRITICAL_TTC)
    )
    future_risk_change = float(curr_snap["corridor_risk"]) - float(prev_snap["corridor_risk"])
    escape_success_flag = int(
        int(prev_snap["blocked_by_leader_flag"]) == 1
        and int(curr_snap["blocked_by_leader_flag"]) == 0
        and not bool(crashed)
    )
    composite = composite_scores(
        future_risk_change=future_risk_change,
        near_miss_future=near_miss_future,
        collision_future=int(bool(crashed)),
        future_speed_gain=float(curr_snap["ego_speed"]) - float(prev_snap["ego_speed"]),
        escape_success_flag=escape_success_flag,
        blocked_by_leader_flag=int(prev_snap["blocked_by_leader_flag"]),
        rear_decel_peak_3s=float(courtesy["rear_decel_peak_3s"]),
        rear_ttc_delta=float(courtesy["rear_ttc_delta"]),
        hard_brake_imposed_flag=int(courtesy["hard_brake_imposed_flag"]),
        bad_cut_in_flag=int(courtesy["bad_cut_in_flag"]),
        missed_opportunity_flag=int(decision["missed_opportunity_flag"]),
        bad_lane_change_flag=int(decision["bad_lane_change_flag"]),
    )
    out = {
        "future_risk_change": future_risk_change,
        "near_miss_future": float(near_miss_future),
        "collision_future": float(int(bool(crashed))),
        "escape_success_flag": float(escape_success_flag),
        "blocked_by_leader_flag": float(prev_snap["blocked_by_leader_flag"]),
        "best_adv": float(prev_snap["best_adv"]),
        "risk_mass_total": float(curr_snap["risk_mass_total"]),
        "risk_mass_others": float(curr_snap["risk_mass_others"]),
        "risk_gradient_peak": float(curr_snap["risk_gradient_peak"]),
        "risk_flux_backward": float(curr_snap["risk_flux_backward"]),
        "risk_field_entropy": float(curr_snap["risk_field_entropy"]),
    }
    out.update({k: float(v) if np.isfinite(v) else float("nan") for k, v in courtesy.items()})
    out.update({k: float(v) for k, v in decision.items()})
    out.update({k: float(v) for k, v in composite.items()})
    return out


def _initial_log() -> dict[str, list[float]]:
    keys = [
        "step", "reward", "progress", "speed", "accel", "ttc", "min_spacing", "lane_score",
        "future_risk_change", "near_miss_future", "collision_future", "escape_success_flag",
        "blocked_by_leader_flag", "best_adv",
        "rear_decel_peak_3s", "rear_ttc_now", "rear_ttc_after", "rear_ttc_delta",
        "rear_thw_now", "rear_thw_after", "rear_thw_delta",
        "hard_brake_imposed_flag", "bad_cut_in_flag",
        "missed_opportunity_flag", "bad_lane_change_flag",
        "risk_mass_total", "risk_mass_others", "risk_gradient_peak",
        "risk_flux_backward", "risk_field_entropy",
        "safety_score", "progress_score", "courtesy_score",
        "social_friendliness_score", "social_class",
        "num_agents_frame", "close_pair_count", "closing_pair_count",
        "interaction_density", "closing_interaction_density",
        "risk_mass_frame", "risk_mass_per_agent",
        "risk_per_close_pair", "risk_per_closing_pair",
        "risk_flux_backward_frame", "backward_risk_flux_ratio",
        "mean_speed_frame", "speed_variance_frame",
        "total_progress_rate_frame", "risk_mass_delta_frame",
        "risk_mass_growth_rate_frame", "risk_adjusted_progress",
        "social_traffic_efficiency_index", "shockwave_onset_flag",
        "ttc_min_s", "ttc_p10_s", "frac_critical_ttc",
        "mean_thw_s", "frac_tailgate_thw", "max_drac_mps2",
        "frac_critical_drac", "min_distance_m",
        "efficiency_index_ei", "safety_efficiency_index_sei",
    ]
    return {k: [] for k in keys}


def run_episode(planner_name: str, algo: str, planner, env_id: str, seed: int, *, duration: float | None = None) -> EpisodeResult:
    env = _make_eval_env(env_id, duration=duration)
    obs, _info = env.reset(seed=seed)
    if planner_name == DEFAULT_BASELINE_LABEL:
        _swap_ego_to_idm(env)
        obs = env.unwrapped.observation_type.observe()

    raw = env.unwrapped
    dt = 1.0 / float(raw.config["policy_frequency"])
    x0 = float(raw.vehicle.position[0])
    prev_speed = None
    prev_lane_index = raw.vehicle.lane_index
    prev_snapshot = _social_snapshot(raw)
    prev_macro = _frame_macro_snapshot(raw, None, dt=dt, hard_brake_rate=0.0)

    log = _initial_log()
    ep_return = 0.0
    crashed = False
    truncated = False

    max_steps = int(round(float(raw.config["duration"]) * float(raw.config["policy_frequency"])))
    for step in range(max_steps):
        if planner_name == DEFAULT_BASELINE_LABEL:
            action = 1
            lane_direction = _current_lane_direction(raw, prev_lane_index, raw.vehicle.lane_index)
        else:
            action = int(planner.predict(obs, deterministic=True)[0])
            lane_direction = _action_direction(action)
        obs, reward, terminated, trunc, _info = env.step(action)
        raw = env.unwrapped
        speed = _forward_speed(raw.vehicle)
        accel = _current_accel(raw, prev_speed=prev_speed, dt=dt)
        prev_speed = speed
        min_spacing, min_ttc = _scene_metrics(raw)
        progress = float(raw.vehicle.position[0] - x0)
        lane_score = _lane_score(raw)
        crashed = bool(raw.vehicle.crashed)
        truncated = bool(trunc)
        curr_snapshot = _social_snapshot(raw)
        if planner_name == DEFAULT_BASELINE_LABEL:
            lane_direction = _current_lane_direction(raw, prev_lane_index, raw.vehicle.lane_index)
        social_step = _compute_social_step(prev_snapshot, curr_snapshot, lane_direction, min_spacing, min_ttc, crashed)
        if raw.road.vehicles:
            n_hard = sum(
                1
                for vehicle in raw.road.vehicles
                if isinstance(getattr(vehicle, "action", None), dict)
                and float(vehicle.action.get("acceleration", 0.0)) <= -3.0
            )
            hard_brake_rate = float(n_hard / max(1, len(raw.road.vehicles)))
        else:
            hard_brake_rate = 0.0
        macro_step = _frame_macro_snapshot(raw, prev_macro, dt=dt, hard_brake_rate=hard_brake_rate)

        log["step"].append(float(step))
        log["reward"].append(float(reward))
        log["progress"].append(progress)
        log["speed"].append(speed)
        log["accel"].append(accel)
        log["ttc"].append(min_ttc)
        log["min_spacing"].append(min_spacing)
        log["lane_score"].append(lane_score)
        for key, value in social_step.items():
            log[key].append(float(value))
        for key, value in macro_step.items():
            log[key].append(float(value))
        ep_return += float(reward)

        prev_snapshot = curr_snapshot
        prev_macro = macro_step
        prev_lane_index = raw.vehicle.lane_index
        if bool(terminated or trunc):
            break

    env.close()
    speeds = np.asarray(log["speed"], dtype=float)
    accels = np.asarray(log["accel"], dtype=float)
    spacing = np.asarray(log["min_spacing"], dtype=float)
    ttc = np.asarray(log["ttc"], dtype=float)
    rewards = np.asarray(log["reward"], dtype=float)
    lane_scores = np.asarray(log["lane_score"], dtype=float)
    near_mask = np.isfinite(spacing) & (spacing < NEAR_COLLISION_DIST)
    jerk = np.abs(np.diff(accels) / max(dt, 1e-6)) if accels.size > 1 else np.asarray([], dtype=float)

    social_summary = {
        key: _nanmean_or_nan(values)
        for key, values in log.items()
        if key not in {"step", "reward", "progress", "speed", "accel", "ttc", "min_spacing", "lane_score"}
    }
    for idx, name in enumerate(SOCIAL_CLASS_NAMES):
        cls_arr = np.asarray(log["social_class"], dtype=float)
        social_summary[f"{name}_frac"] = float(np.mean(cls_arr == float(idx))) if cls_arr.size else 0.0

    return EpisodeResult(
        planner=planner_name,
        algo=algo,
        seed=int(seed),
        episode_return=float(ep_return),
        episode_length=int(len(log["step"])),
        crashed=bool(crashed),
        truncated=bool(truncated),
        ttc_min=float(np.nanmin(ttc)) if ttc.size else float("nan"),
        cmr=float(np.mean(ttc < CRITICAL_TTC)) if ttc.size else float("nan"),
        min_spacing=float(np.nanmin(spacing)) if spacing.size else float("nan"),
        mean_speed=float(np.nanmean(speeds)) if speeds.size else float("nan"),
        mean_reward=float(np.nanmean(rewards)) if rewards.size else float("nan"),
        mean_abs_jerk=float(np.nanmean(jerk)) if jerk.size else 0.0,
        final_progress=float(log["progress"][-1]) if log["progress"] else 0.0,
        right_lane_score=float(np.nanmean(lane_scores)) if lane_scores.size else float("nan"),
        near_collision_rate=float(np.mean(near_mask)) if near_mask.size else 0.0,
        near_collision_any=bool(np.any(near_mask)),
        social_summary=social_summary,
        log=log,
    )


def _stack_episode_metric(episodes: list[EpisodeResult], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = max((len(ep.log[key]) for ep in episodes), default=0)
    t = np.arange(max_len, dtype=float)
    arr = np.full((len(episodes), max_len), np.nan, dtype=float)
    for i, ep in enumerate(episodes):
        vals = np.asarray(ep.log[key], dtype=float)
        arr[i, : vals.size] = vals
    mean = np.full((max_len,), np.nan, dtype=float)
    std = np.full((max_len,), np.nan, dtype=float)
    for idx in range(max_len):
        col = arr[:, idx]
        valid = col[np.isfinite(col)]
        if valid.size:
            mean[idx] = float(np.mean(valid))
            std[idx] = float(np.std(valid))
    return t, mean, std


def _planner_summary(episodes: list[EpisodeResult]) -> dict[str, float]:
    summary = {
        "episodes": int(len(episodes)),
        "return_mean": float(np.mean([ep.episode_return for ep in episodes])),
        "return_std": float(np.std([ep.episode_return for ep in episodes])),
        "collision_rate": float(np.mean([ep.crashed for ep in episodes])),
        "near_collision_episode_rate": float(np.mean([ep.near_collision_any for ep in episodes])),
        "near_collision_step_rate": float(np.mean([ep.near_collision_rate for ep in episodes])),
        "ttc_min_mean": _nanmean_or_nan([ep.ttc_min for ep in episodes]),
        "cmr_mean": _nanmean_or_nan([ep.cmr for ep in episodes]),
        "min_spacing_mean": _nanmean_or_nan([ep.min_spacing for ep in episodes]),
        "mean_speed": _nanmean_or_nan([ep.mean_speed for ep in episodes]),
        "mean_reward": _nanmean_or_nan([ep.mean_reward for ep in episodes]),
        "mean_abs_jerk": _nanmean_or_nan([ep.mean_abs_jerk for ep in episodes]),
        "final_progress_mean": _nanmean_or_nan([ep.final_progress for ep in episodes]),
        "right_lane_score_mean": _nanmean_or_nan([ep.right_lane_score for ep in episodes]),
        "episode_length_mean": _nanmean_or_nan([ep.episode_length for ep in episodes]),
    }
    social_keys = sorted(episodes[0].social_summary.keys()) if episodes else []
    for key in social_keys:
        summary[f"{key}_mean"] = _nanmean_or_nan([ep.social_summary.get(key, np.nan) for ep in episodes])
    return summary


def _bar_panel(ax, values: list[float], labels: list[str], title: str, ylabel: str, *, lower_is_better: bool, ymin: float | None = None):
    colors = ["#1f77b4", "#2ca02c"]
    bars = ax.bar(labels, values, color=colors[: len(values)], alpha=0.88)
    finite = [i for i, v in enumerate(values) if np.isfinite(v)]
    if finite:
        best_idx = min(finite, key=lambda i: values[i]) if lower_is_better else max(finite, key=lambda i: values[i])
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.2)
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if ymin is not None:
        ax.set_ylim(bottom=float(ymin))


def _save_core_summary_plot(save_path: str, policy_summary: dict[str, float], idm_summary: dict[str, float], *, policy_label: str, baseline_label: str, env_id: str) -> None:
    _apply_style()
    labels = [policy_label, baseline_label]
    with plt.style.context(["science", "no-latex"] if _HAS_SCIENCEPLOTS else ["default"]):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        fig.suptitle(f"{env_id}: {policy_label} vs {baseline_label}", fontsize=12)
        _bar_panel(axes[0, 0], [policy_summary["ttc_min_mean"], idm_summary["ttc_min_mean"]], labels, "TTC_min", "s", lower_is_better=False, ymin=0)
        _bar_panel(axes[0, 1], [policy_summary["cmr_mean"], idm_summary["cmr_mean"]], labels, "CMR (TTC < 3s)", "fraction", lower_is_better=True, ymin=0)
        _bar_panel(axes[0, 2], [policy_summary["min_spacing_mean"], idm_summary["min_spacing_mean"]], labels, "Min spacing", "m", lower_is_better=False, ymin=0)
        _bar_panel(axes[1, 0], [policy_summary["mean_abs_jerk"], idm_summary["mean_abs_jerk"]], labels, "Mean |jerk|", r"m/s$^3$", lower_is_better=True, ymin=0)
        _bar_panel(axes[1, 1], [policy_summary["near_collision_step_rate"], idm_summary["near_collision_step_rate"]], labels, "Near-collision rate", "fraction", lower_is_better=True, ymin=0)
        _bar_panel(axes[1, 2], [policy_summary["final_progress_mean"], idm_summary["final_progress_mean"]], labels, "Final progress", "m", lower_is_better=False, ymin=0)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _save_social_summary_plot(save_path: str, policy_summary: dict[str, float], idm_summary: dict[str, float], *, policy_label: str, baseline_label: str, env_id: str) -> None:
    _apply_style()
    labels = [policy_label, baseline_label]
    panels = [
        ("rear_decel_peak_3s_mean", "rear_decel_peak_3s", r"m/s$^2$", True),
        ("rear_ttc_delta_mean", "rear_ttc_delta", "s", False),
        ("rear_thw_delta_mean", "rear_thw_delta", "s", False),
        ("risk_mass_others_mean", "risk_mass_others", "risk", True),
        ("risk_flux_backward_mean", "risk_flux_backward", "risk flux", True),
        ("social_friendliness_score_mean", "social_friendliness_score", "score", False),
    ]
    with plt.style.context(["science", "no-latex"] if _HAS_SCIENCEPLOTS else ["default"]):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
        fig.suptitle(f"{env_id}: Social / Externality Metrics", fontsize=12)
        for ax, (key, label_key, ylabel, lower_is_better) in zip(axes.flat, panels):
            values = [policy_summary.get(key, np.nan), idm_summary.get(key, np.nan)]
            _bar_panel(ax, values, labels, display_name(label_key), ylabel, lower_is_better=lower_is_better, ymin=0 if lower_is_better else None)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _save_timeseries_plot(save_path: str, policy_eps: list[EpisodeResult], idm_eps: list[EpisodeResult], *, policy_label: str, baseline_label: str, env_id: str) -> None:
    _apply_style()
    with plt.style.context(["science", "no-latex"] if _HAS_SCIENCEPLOTS else ["default"]):
        fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
        fig.suptitle(f"{env_id} Time-Series: {policy_label} vs {baseline_label}", fontsize=12)
        panels = [
            ("progress", "Progress s(t)", "m"),
            ("speed", "Speed v_x(t)", "m/s"),
            ("ttc", "Critical TTC(t)", "s"),
            ("rear_ttc_after", display_name("rear_ttc_delta"), "s"),
            ("risk_mass_others", display_name("risk_mass_others"), "risk"),
            ("social_friendliness_score", display_name("social_friendliness_score"), "score"),
        ]
        for ax, (key, title, ylabel) in zip(axes.flat, panels):
            t_p, mean_p, std_p = _stack_episode_metric(policy_eps, key)
            t_i, mean_i, std_i = _stack_episode_metric(idm_eps, key)
            ax.plot(t_p, mean_p, color="#1f77b4", lw=1.8, label=policy_label)
            ax.fill_between(t_p, mean_p - std_p, mean_p + std_p, color="#1f77b4", alpha=0.18)
            ax.plot(t_i, mean_i, color="#2ca02c", lw=1.8, label=baseline_label)
            ax.fill_between(t_i, mean_i - std_i, mean_i + std_i, color="#2ca02c", alpha=0.18)
            if key == "ttc":
                ax.axhline(CRITICAL_TTC, color="red", lw=0.9, ls=":", label=f"TTC_crit={CRITICAL_TTC:.0f}s")
                ax.set_ylim(0, TTC_CAP)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("step")
            ax.legend(fontsize=7)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _write_summary_md(path: str, policy_summary: dict[str, float], idm_summary: dict[str, float], *, policy_label: str, baseline_label: str, env_id: str) -> None:
    lines = [
        f"# {env_id}: {policy_label} vs {baseline_label}",
        "",
        f"| metric | {policy_label} | {baseline_label} | better |",
        "|---|---:|---:|---|",
        f"| return_mean | {policy_summary['return_mean']:.3f} | {idm_summary['return_mean']:.3f} | higher |",
        f"| collision_rate | {policy_summary['collision_rate']:.3f} | {idm_summary['collision_rate']:.3f} | lower |",
        f"| TTC_min_mean | {policy_summary['ttc_min_mean']:.3f} | {idm_summary['ttc_min_mean']:.3f} | higher |",
        f"| rear_decel_peak_3s_mean | {policy_summary.get('rear_decel_peak_3s_mean', float('nan')):.3f} | {idm_summary.get('rear_decel_peak_3s_mean', float('nan')):.3f} | less negative / higher |",
        f"| risk_mass_others_mean | {policy_summary.get('risk_mass_others_mean', float('nan')):.3f} | {idm_summary.get('risk_mass_others_mean', float('nan')):.3f} | lower |",
        f"| risk_flux_backward_mean | {policy_summary.get('risk_flux_backward_mean', float('nan')):.3f} | {idm_summary.get('risk_flux_backward_mean', float('nan')):.3f} | lower |",
        f"| social_friendliness_score_mean | {policy_summary.get('social_friendliness_score_mean', float('nan')):.3f} | {idm_summary.get('social_friendliness_score_mean', float('nan')):.3f} | higher |",
        "",
        "Notes:",
        "- Social metrics follow the naming convention in `rl/data/social_features.py`.",
        "- They are computed online from rollout steps, so `future_*` quantities are one-step rollout surrogates rather than offline horizon labels.",
        "- `risk_mass_others` and `risk_flux_backward` are analytic ego-frame risk-proxy externality metrics over the surrounding traffic.",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _format_metric(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{float(value):.3f}"


def write_summary_table(
    save_dir: str,
    planner_summaries: dict[str, dict[str, float]],
    *,
    env_id: str,
    table_fields: list[tuple[str, str, str]] | None = None,
) -> tuple[str, str]:
    csv_path = os.path.join(save_dir, "summary_table.csv")
    md_path = os.path.join(save_dir, "summary_table.md")
    planner_names = list(planner_summaries.keys())
    fields = table_fields or SUMMARY_TABLE_FIELDS

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "label", "better"] + planner_names)
        for key, label, better in fields:
            writer.writerow(
                [key, label, better]
                + [_format_metric(planner_summaries[name].get(key, float("nan"))) for name in planner_names]
            )

    lines = [f"# {env_id} Summary Table", ""]
    lines.append("| metric | better | " + " | ".join(planner_names) + " |")
    lines.append("|---|---|" + "---|" * len(planner_names))
    for key, label, better in fields:
        row_vals = [_format_metric(planner_summaries[name].get(key, float("nan"))) for name in planner_names]
        lines.append("| " + " | ".join([label, better] + row_vals) + " |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return csv_path, md_path


def _episode_records(episodes: list[EpisodeResult]) -> list[dict]:
    records = []
    for ep in episodes:
        records.append(
            {
                "planner": ep.planner,
                "algo": ep.algo,
                "seed": ep.seed,
                "episode_return": ep.episode_return,
                "episode_length": ep.episode_length,
                "crashed": ep.crashed,
                "truncated": ep.truncated,
                "ttc_min": ep.ttc_min,
                "cmr": ep.cmr,
                "min_spacing": ep.min_spacing,
                "mean_speed": ep.mean_speed,
                "mean_reward": ep.mean_reward,
                "mean_abs_jerk": ep.mean_abs_jerk,
                "final_progress": ep.final_progress,
                "right_lane_score": ep.right_lane_score,
                "near_collision_rate": ep.near_collision_rate,
                "near_collision_any": ep.near_collision_any,
                "social_summary": ep.social_summary,
            }
        )
    return records


def compare_policy_vs_idm(
    *,
    algo: str,
    checkpoint: str,
    eval_env_id: str = "highway-v0",
    episodes: int = 10,
    save_dir: str = "rl/logs/highway_stock_policy_vs_idm_social",
    duration: float | None = None,
    policy_label: str,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
) -> dict:
    model = load_sb3_policy(algo, checkpoint)
    policy_eps: list[EpisodeResult] = []
    idm_eps: list[EpisodeResult] = []
    for ep in range(int(episodes)):
        policy_eps.append(run_episode(policy_label, algo, model, eval_env_id, ep, duration=duration))
        idm_eps.append(run_episode(baseline_label, algo, None, eval_env_id, ep, duration=duration))

    policy_summary = _planner_summary(policy_eps)
    idm_summary = _planner_summary(idm_eps)

    os.makedirs(save_dir, exist_ok=True)
    core_plot = os.path.join(save_dir, "metrics_summary.png")
    social_plot = os.path.join(save_dir, "social_metrics_summary.png")
    timeseries_plot = os.path.join(save_dir, "metrics_timeseries.png")
    summary_json = os.path.join(save_dir, "summary.json")
    summary_md = os.path.join(save_dir, "summary.md")
    episodes_json = os.path.join(save_dir, "episodes.json")
    table_csv, table_md = write_summary_table(
        save_dir,
        {policy_label: policy_summary, baseline_label: idm_summary},
        env_id=eval_env_id,
    )

    _save_core_summary_plot(core_plot, policy_summary, idm_summary, policy_label=policy_label, baseline_label=baseline_label, env_id=eval_env_id)
    _save_social_summary_plot(social_plot, policy_summary, idm_summary, policy_label=policy_label, baseline_label=baseline_label, env_id=eval_env_id)
    _save_timeseries_plot(timeseries_plot, policy_eps, idm_eps, policy_label=policy_label, baseline_label=baseline_label, env_id=eval_env_id)
    _write_summary_md(summary_md, policy_summary, idm_summary, policy_label=policy_label, baseline_label=baseline_label, env_id=eval_env_id)

    summary = {
        "algo": algo,
        "eval_env_id": eval_env_id,
        "checkpoint": _checkpoint_file(checkpoint),
        "episodes": int(episodes),
        "metrics": {
            policy_label: policy_summary,
            baseline_label: idm_summary,
        },
        "plots": {
            "metrics_summary": core_plot,
            "social_metrics_summary": social_plot,
            "metrics_timeseries": timeseries_plot,
        },
        "summary_table_csv": table_csv,
        "summary_table_md": table_md,
        "episode_records": episodes_json,
        "summary_md": summary_md,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(episodes_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                policy_label: _episode_records(policy_eps),
                baseline_label: _episode_records(idm_eps),
            },
            f,
            indent=2,
        )
    return summary


def _parse_args():
    p = argparse.ArgumentParser(description="Compare a stock trained SB3 highway policy against pure IDM/MOBIL using social_features-style online metrics.")
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--checkpoint", default="", help="Checkpoint stem or .zip path. Defaults to rl/checkpoints/sb3_highway_<algo>.")
    p.add_argument("--train-env-id", default="highway-fast-v0")
    p.add_argument("--eval-env-id", default="highway-v0")
    p.add_argument("--train-steps", type=int, default=20_000)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--save-dir", default="")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--train-if-missing", type=_str2bool, default=False)
    p.add_argument("--force-train", type=_str2bool, default=False)
    p.add_argument("--policy-label", default="")
    p.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    return p.parse_args()


def main():
    args = _parse_args()
    checkpoint = args.checkpoint or f"rl/checkpoints/sb3_highway_{args.algo}"
    policy_label = args.policy_label or DEFAULT_POLICY_LABELS[args.algo]
    save_dir = args.save_dir or f"rl/logs/highway_{args.algo}_vs_idm_social"
    checkpoint_zip = _checkpoint_file(checkpoint)
    if args.train_if_missing or args.force_train:
        checkpoint_zip = train_or_load_sb3_policy(
            args.algo,
            checkpoint,
            train_env_id=args.train_env_id,
            total_timesteps=args.train_steps,
            n_envs=args.n_envs,
            force_train=args.force_train,
            duration=args.duration,
        )
    elif not os.path.exists(checkpoint_zip):
        raise SystemExit(
            f"Missing checkpoint: {checkpoint_zip}\n"
            "Run with --train-if-missing true or provide an existing --checkpoint."
        )

    summary = compare_policy_vs_idm(
        algo=args.algo,
        checkpoint=checkpoint_zip,
        eval_env_id=args.eval_env_id,
        episodes=args.episodes,
        save_dir=save_dir,
        duration=args.duration,
        policy_label=policy_label,
        baseline_label=args.baseline_label,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
