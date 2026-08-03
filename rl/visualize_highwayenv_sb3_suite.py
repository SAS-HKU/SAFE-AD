from __future__ import annotations

import argparse
from collections import deque
import io
import importlib
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
from config import Config as cfg
from rl.utils.typing_compat import ensure_typing_extensions_compat

ensure_typing_extensions_compat()
from stable_baselines3 import DQN, PPO

from rl.compare_stock_policy_vs_idm_social import (
    DEFAULT_BASELINE_LABEL,
    EpisodeResult,
    SUMMARY_TABLE_FIELDS,
    _action_direction,
    _apply_style,
    _compute_social_step,
    _current_accel,
    _episode_records,
    _forward_speed,
    _frame_macro_snapshot,
    _initial_log,
    _lane_score,
    _nanmean_or_nan,
    _planner_summary,
    _scene_metrics,
    _social_snapshot,
    _stack_episode_metric,
    _str2bool,
    _swap_ego_to_idm,
    write_summary_table,
)
from rl.env.highwayenv_drift_wrapper import DriftOverlayWrapper
from rl.env.highwayenv_social_env import (
    StockTrafficWrapper,
    _find_wrapper_attr,
    load_reward_config,
    resolve_traffic_config,
    stock_env_config,
)
from rl.risk.pinn_adapter import PINNRiskAdapter, load_best_available
from rl.utils.timing import get_timer, CATEGORY_RL_INFERENCE


DEFAULT_SOCIAL_PPO_LABEL = "social-ppo"
DEFAULT_SOCIAL_DQN_LABEL = "social-dqn"
DEFAULT_STOCK_PPO_LABEL = "stock-ppo"
DEFAULT_STOCK_DQN_LABEL = "stock-dqn"
DEFAULT_OVERLAY_MODE = "pinn"
FIELD_CMAP = "turbo"
FIELD_ALPHA = 0.72
FIELD_MIN_ALPHA = 0.16
FIELD_SMOOTH_SIGMA = 0.75
FIELD_VMIN_QUANTILE = 0.05
FIELD_VMAX_QUANTILE = 0.995


SOCIAL_CKPT_MAP = {
    "highway-v0": {
        "ppo": "rl/logs/social_ppo_a5/checkpoints/best_model.zip",
        "dqn": "rl/logs/social_dqn_a5/checkpoints/best_model.zip",
    },
    "merge-v0": {
        "ppo": "rl/logs/social_ppo_a5/checkpoints/best_model.zip",
        "dqn": "rl/logs/social_dqn_a5/checkpoints/best_model.zip",
    },
    "highway-fast-v0": {
        "ppo": "rl/logs/social_ppo_a5/checkpoints/best_model.zip",
        "dqn": "rl/logs/social_dqn_a5/checkpoints/best_model.zip",
    },
    "roundabout-v0": {
        "ppo": "rl/logs/social_ppo_a5_roundabout/checkpoints/best_model.zip",
        "dqn": "rl/logs/social_dqn_a5_roundabout/checkpoints/best_model.zip",
    },
    "intersection-v0": {
        "ppo": "rl/logs/social_ppo_a5_intersection/checkpoints/best_model.zip",
        "dqn": "rl/logs/social_dqn_a5_intersection/checkpoints/best_model.zip",
    },
}

STOCK_CKPT_MAP = {
    "highway-v0": {
        "ppo": "rl/logs/highway_v0_curve_compare/ppo_highway_v0.zip",
        "dqn": "rl/logs/highway_v0_curve_compare/dqn_highway_v0.zip",
    },
    "merge-v0": {
        "ppo": "rl/logs/highway_v0_curve_compare/ppo_highway_v0.zip",
        "dqn": "rl/logs/highway_v0_curve_compare/dqn_highway_v0.zip",
    },
    "highway-fast-v0": {
        "ppo": "rl/logs/highway_v0_curve_compare/ppo_highway_v0.zip",
        "dqn": "rl/logs/highway_v0_curve_compare/dqn_highway_v0.zip",
    },
    "roundabout-v0": {
        "ppo": "rl/logs/roundabout_stock_curve_compare/ppo_highway_v0.zip",
        "dqn": "rl/logs/roundabout_stock_curve_compare/dqn_highway_v0.zip",
    },
    "intersection-v0": {
        "ppo": "rl/logs/intersection_stock_curve_compare/ppo_highway_v0.zip",
        "dqn": "rl/logs/intersection_stock_curve_compare/dqn_highway_v0.zip",
    },
}


@dataclass
class PlannerSpec:
    label: str
    kind: str
    algo: str
    checkpoint: str | None


@dataclass
class RenderStep:
    planner: str
    step: int
    action: int
    reward: float
    total_return: float
    speed: float
    ttc: float
    min_spacing: float
    social_score: float
    corridor_risk: float
    crashed: bool
    frame: np.ndarray
    # Legacy single-overlay fields (populated with the first requested overlay mode)
    overlay_rgba: np.ndarray | None = None
    overlay_extent: tuple[float, float, float, float] | None = None
    overlay_mode: str | None = None
    # Multi-overlay storage:  {resolved_mode: (rgba, extent)}
    overlays: dict = field(default_factory=dict)


def _checkpoint_file(path: str) -> str:
    return str(path) if str(path).endswith(".zip") else f"{path}.zip"


def _load_sb3_model(algo: str, checkpoint: str):
    # Some checkpoints were serialized with NumPy 2.x's private module path.
    # NumPy 1.26 exposes the same implementation at the public legacy path.
    try:
        importlib.import_module("numpy._core.numeric")
    except ModuleNotFoundError:
        sys.modules.setdefault(
            "numpy._core.numeric", importlib.import_module("numpy.core.numeric")
        )
    path = _checkpoint_file(checkpoint)
    algo = str(algo).strip().lower()
    custom_objects = None
    try:
        with zipfile.ZipFile(path) as archive:
            state = torch.load(
                io.BytesIO(archive.read("policy.pth")),
                map_location="cpu",
                weights_only=False,
            )
        if algo == "ppo":
            obs_dim = int(state["mlp_extractor.policy_net.0.weight"].shape[1])
            action_dim = int(state["action_net.weight"].shape[0])
        elif algo == "dqn":
            obs_dim = int(state["q_net.q_net.0.weight"].shape[1])
            q_weights = [
                value
                for key, value in state.items()
                if key.startswith("q_net.q_net.") and key.endswith(".weight")
            ]
            action_dim = int(q_weights[-1].shape[0])
        else:
            obs_dim = action_dim = 0
        if obs_dim and action_dim:
            if algo == "ppo" and "log_std" in state:
                action_space = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(action_dim,),
                    dtype=np.float32,
                )
            else:
                action_space = gym.spaces.Discrete(action_dim)
            custom_objects = {
                "observation_space": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                ),
                "action_space": action_space,
                "_last_obs": None,
                "_last_episode_starts": None,
                "ep_info_buffer": deque(maxlen=100),
                "ep_success_buffer": deque(maxlen=100),
            }
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile):
        custom_objects = None
    if algo == "ppo":
        return PPO.load(path, custom_objects=custom_objects, device="cpu")
    if algo == "dqn":
        return DQN.load(path, custom_objects=custom_objects, device="cpu")
    raise ValueError(f"Unsupported algo '{algo}'")


def _env_title(env_id: str) -> str:
    return str(env_id).replace("-v0", "").replace("-", " ").title()


def _make_eval_env(
    env_id: str,
    *,
    traffic,
    render_mode: str | None = None,
    enable_field_wrapper: bool = False,
    drift_warmup_s: float = 1.0,
    env_config_override: dict[str, Any] | None = None,
):
    env_config = stock_env_config(traffic)
    if env_config_override:
        env_config = _deep_update(env_config, env_config_override)
    base = gym.make(
        env_id,
        render_mode=render_mode,
        config=env_config,
    )
    env = StockTrafficWrapper(base, traffic)
    if enable_field_wrapper:
        env = DriftOverlayWrapper(
            env,
            use_drift=True,
            drift_warmup_s=float(drift_warmup_s),
            gate_reward=False,
            record_risk_metrics=False,
        )
    return env


def _find_wrapper_instance(env, attr: str):
    current = env
    while current is not None:
        if hasattr(current, attr):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    return None


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_env_config_override(
    *,
    json_text: str | None = None,
    json_file: str | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if json_file:
        with open(json_file, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        if not isinstance(file_cfg, dict):
            raise ValueError("Env config file must contain a JSON object.")
        merged = _deep_update(merged, file_cfg)
    if json_text:
        cli_cfg = json.loads(json_text)
        if not isinstance(cli_cfg, dict):
            raise ValueError("Env config JSON must be a JSON object.")
        merged = _deep_update(merged, cli_cfg)
    return merged


def _init_pinn_adapter(env, checkpoint: str | None, device: str = "cpu") -> PINNRiskAdapter | None:
    drift_wrapper = _find_wrapper_instance(env, "get_drift_grid")
    if drift_wrapper is None:
        return None
    grid = drift_wrapper.get_drift_grid()
    if not isinstance(grid, tuple) or grid[0] is None or grid[1] is None:
        return None
    grid_x, grid_y = grid
    x_range = (float(np.nanmin(grid_x)), float(np.nanmax(grid_x)))
    y_range = (float(np.nanmin(grid_y)), float(np.nanmax(grid_y)))
    ckpt = str(checkpoint).strip() if checkpoint else ""
    if ckpt:
        adapter = PINNRiskAdapter(
            checkpoint_path=ckpt,
            device=device,
            inference_x_range=x_range,
            inference_y_range=y_range,
        )
    else:
        adapter = load_best_available(
            repo_root=str(REPO_ROOT),
            device=device,
            inference_x_range=x_range,
            inference_y_range=y_range,
        )
    return adapter if adapter.available else None


def _build_pinn_field(env, adapter: PINNRiskAdapter | None, sim_time: float) -> np.ndarray | None:
    if adapter is None or not adapter.available:
        return None
    drift_wrapper = _find_wrapper_instance(env, "get_drift_grid")
    if drift_wrapper is None:
        return None
    grid_x, grid_y = drift_wrapper.get_drift_grid()
    if grid_x is None or grid_y is None:
        return None
    drift = getattr(drift_wrapper, "drift", None)
    if drift is None or drift.last_Q is None or drift.last_vx is None or drift.last_vy is None or drift.last_D is None:
        return None
    collect_fn = getattr(drift_wrapper, "_collect_drift_state", None)
    ego_vehicle = None
    vehicles = None
    if callable(collect_fn):
        try:
            ego_vehicle, vehicles = collect_fn()
        except Exception:
            ego_vehicle, vehicles = None, None
    field = adapter.query_grid(
        X=np.asarray(grid_x, dtype=np.float32),
        Y=np.asarray(grid_y, dtype=np.float32),
        t=float(sim_time),
        Q=np.asarray(drift.last_Q, dtype=np.float32),
        vx=np.asarray(drift.last_vx, dtype=np.float32),
        vy=np.asarray(drift.last_vy, dtype=np.float32),
        D=np.asarray(drift.last_D, dtype=np.float32),
        vehicles=vehicles,
        ego_vehicle=ego_vehicle,
    )
    field = np.asarray(field, dtype=np.float32)
    if field.size != grid_x.size:
        return None
    field = field.reshape(grid_x.shape)
    road_mask = drift_wrapper.get_road_mask() if hasattr(drift_wrapper, "get_road_mask") else None
    if road_mask is not None:
        mask = np.asarray(road_mask, dtype=np.float32)
        field = np.where(mask > 0.05, field, np.nan)
    return field


def _build_drift_field(env) -> np.ndarray | None:
    masked_fn = _find_wrapper_attr(env, "get_masked_risk_field")
    if not callable(masked_fn):
        return None
    field = masked_fn()
    if field is None:
        return None
    return np.asarray(field, dtype=np.float32)


def _overlay_payload_from_field(env, field: np.ndarray | None) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None]:
    if field is None:
        return None, None
    drift_wrapper = _find_wrapper_instance(env, "get_drift_grid")
    if drift_wrapper is None:
        return None, None
    grid_x, grid_y = drift_wrapper.get_drift_grid()
    if grid_x is None or grid_y is None:
        return None, None
    raw_env = env.unwrapped
    viewer = getattr(raw_env, "viewer", None)
    sim_surface = getattr(viewer, "sim_surface", None)
    if sim_surface is None:
        return None, None

    field_np = np.asarray(field, dtype=np.float32)
    finite = field_np[np.isfinite(field_np) & (field_np > 0)]
    if finite.size < 8:
        return None, None
    vmin = float(np.percentile(finite, FIELD_VMIN_QUANTILE * 100.0))
    vmax = float(np.percentile(finite, FIELD_VMAX_QUANTILE * 100.0))
    vmax = max(vmax, vmin + 1e-3)
    field_sm = np.asarray(field_np, dtype=np.float32)
    if FIELD_SMOOTH_SIGMA > 1e-6:
        from scipy.ndimage import gaussian_filter

        mask = np.isfinite(field_sm).astype(np.float32)
        smooth_num = gaussian_filter(np.nan_to_num(field_sm, nan=0.0), sigma=FIELD_SMOOTH_SIGMA)
        smooth_den = gaussian_filter(mask, sigma=FIELD_SMOOTH_SIGMA)
        field_sm = np.where(smooth_den > 1e-6, smooth_num / np.maximum(smooth_den, 1e-6), np.nan)
        field_sm = np.where(mask > 0.05, field_sm, np.nan)

    scaled = np.clip((field_sm - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    support = np.isfinite(field_sm) & (field_sm > 0.0)
    alpha = FIELD_MIN_ALPHA + (FIELD_ALPHA - FIELD_MIN_ALPHA) * np.sqrt(scaled)
    alpha = np.where(support, alpha, 0.0)
    rgba = plt.get_cmap(FIELD_CMAP)(scaled)
    rgba[..., 3] = alpha

    x_origin = float(getattr(drift_wrapper, "_x_origin", 0.0))
    world_x = np.asarray(grid_x, dtype=np.float32) + x_origin
    world_y = np.asarray(grid_y, dtype=np.float32)
    origin = np.asarray(sim_surface.origin, dtype=np.float32)
    scaling = float(sim_surface.scaling)
    xpix_min = scaling * (float(np.nanmin(world_x)) - float(origin[0]))
    xpix_max = scaling * (float(np.nanmax(world_x)) - float(origin[0]))
    ypix_min = scaling * (float(np.nanmin(world_y)) - float(origin[1]))
    ypix_max = scaling * (float(np.nanmax(world_y)) - float(origin[1]))
    extent = (xpix_min, xpix_max, ypix_max, ypix_min)
    return np.asarray(rgba, dtype=np.float32), extent


def _capture_overlay_payload(
    env,
    *,
    overlay_mode: str,
    adapter: PINNRiskAdapter | None,
    sim_time: float,
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None, str | None]:
    mode = str(overlay_mode).strip().lower()
    if mode in {"", "none", "off"}:
        return None, None, None
    if mode == "pinn":
        field = _build_pinn_field(env, adapter, sim_time)
        if field is None:
            field = _build_drift_field(env)
            mode = "drift-fallback"
    elif mode == "drift":
        field = _build_drift_field(env)
    else:
        field = None
    rgba, extent = _overlay_payload_from_field(env, field)
    return rgba, extent, (mode if rgba is not None else None)


def _planner_specs_for_env(
    env_id: str,
    *,
    include_social_ppo: bool,
    include_social_dqn: bool,
    include_stock_ppo: bool,
    include_stock_dqn: bool,
    include_idm: bool,
    social_ppo_checkpoint: str | None,
    social_dqn_checkpoint: str | None,
    stock_ppo_checkpoint: str | None,
    stock_dqn_checkpoint: str | None,
) -> list[PlannerSpec]:
    social_defaults = SOCIAL_CKPT_MAP.get(env_id, {})
    stock_defaults = STOCK_CKPT_MAP.get(env_id, {})
    specs: list[PlannerSpec] = []
    if include_social_ppo:
        specs.append(
            PlannerSpec(
                label=DEFAULT_SOCIAL_PPO_LABEL,
                kind="model",
                algo="ppo",
                checkpoint=social_ppo_checkpoint or social_defaults.get("ppo"),
            )
        )
    if include_social_dqn:
        specs.append(
            PlannerSpec(
                label=DEFAULT_SOCIAL_DQN_LABEL,
                kind="model",
                algo="dqn",
                checkpoint=social_dqn_checkpoint or social_defaults.get("dqn"),
            )
        )
    if include_stock_ppo:
        specs.append(
            PlannerSpec(
                label=DEFAULT_STOCK_PPO_LABEL,
                kind="model",
                algo="ppo",
                checkpoint=stock_ppo_checkpoint or stock_defaults.get("ppo"),
            )
        )
    if include_stock_dqn:
        specs.append(
            PlannerSpec(
                label=DEFAULT_STOCK_DQN_LABEL,
                kind="model",
                algo="dqn",
                checkpoint=stock_dqn_checkpoint or stock_defaults.get("dqn"),
            )
        )
    if include_idm:
        specs.append(
            PlannerSpec(
                label=DEFAULT_BASELINE_LABEL,
                kind="idm",
                algo="idm",
                checkpoint=None,
            )
        )
    return specs


def _resolve_available_specs(specs: list[PlannerSpec]) -> tuple[list[PlannerSpec], list[dict[str, str]]]:
    available: list[PlannerSpec] = []
    missing: list[dict[str, str]] = []
    for spec in specs:
        if spec.kind == "idm":
            available.append(spec)
            continue
        ckpt = spec.checkpoint
        if ckpt and os.path.exists(_checkpoint_file(ckpt)):
            available.append(spec)
        else:
            missing.append(
                {
                    "planner": spec.label,
                    "algo": spec.algo,
                    "checkpoint": ckpt or "",
                }
            )
    return available, missing


def _path_progress_delta(prev_position: np.ndarray, curr_position: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(curr_position, dtype=float) - np.asarray(prev_position, dtype=float)))


def run_episode(
    spec: PlannerSpec,
    *,
    env_id: str,
    traffic,
    seed: int,
    render_mode: str | None = None,
    save_frames: bool = False,
    max_frame_steps: int = 80,
    overlay_modes: list[str] | tuple[str, ...] | str = (DEFAULT_OVERLAY_MODE,),
    pinn_checkpoint: str | None = None,
    pinn_device: str = "cpu",
    env_config_override: dict[str, Any] | None = None,
) -> tuple[EpisodeResult, list[RenderStep]]:
    # Normalise overlay_modes to a deduplicated, lowercased list
    if isinstance(overlay_modes, str):
        overlay_modes = [overlay_modes]
    overlay_modes_norm: list[str] = []
    for mode in overlay_modes:
        m = str(mode).strip().lower()
        if m and m not in {"none", "off"} and m not in overlay_modes_norm:
            overlay_modes_norm.append(m)

    env = _make_eval_env(
        env_id,
        traffic=traffic,
        render_mode=render_mode,
        enable_field_wrapper=bool(overlay_modes_norm),
        env_config_override=env_config_override,
    )
    model = None if spec.kind == "idm" else _load_sb3_model(spec.algo, str(spec.checkpoint))
    obs, _info = env.reset(seed=seed)
    if spec.kind == "idm":
        _swap_ego_to_idm(env)
        obs = env.unwrapped.observation_type.observe()
    pinn_adapter = (
        _init_pinn_adapter(env, pinn_checkpoint, device=pinn_device)
        if "pinn" in overlay_modes_norm else None
    )
    drift_wrapper = _find_wrapper_instance(env, "get_drift_grid")
    field_warmup_s = float(getattr(drift_wrapper, "drift_warmup_s", 0.0))

    raw = env.unwrapped
    dt = 1.0 / float(raw.config["policy_frequency"])
    prev_speed = None
    prev_lane_index = raw.vehicle.lane_index
    prev_snapshot = _social_snapshot(raw)
    prev_macro = _frame_macro_snapshot(raw, None, dt=dt, hard_brake_rate=0.0)
    prev_position = np.array(raw.vehicle.position, dtype=float, copy=True)
    path_progress = 0.0

    log = _initial_log()
    render_steps: list[RenderStep] = []
    ep_return = 0.0
    crashed = False
    truncated = False

    max_steps = int(round(float(raw.config["duration"]) * float(raw.config["policy_frequency"])))
    timer = get_timer()
    for step in range(max_steps):
        if spec.kind == "idm":
            action = 1
            lane_direction = prev_lane_index
        else:
            with timer.measure(CATEGORY_RL_INFERENCE):
                model_obs = np.asarray(obs, dtype=np.float32)
                expected_shape = tuple(model.observation_space.shape)
                if model_obs.shape != expected_shape:
                    if model_obs.size != int(np.prod(expected_shape)):
                        raise ValueError(
                            f"Checkpoint expects observation {expected_shape}, "
                            f"but environment returned {model_obs.shape}"
                        )
                    model_obs = model_obs.reshape(expected_shape)
                action = int(model.predict(model_obs, deterministic=True)[0])
            lane_direction = _action_direction(action)

        obs, reward, terminated, trunc, _info = env.step(action)
        raw = env.unwrapped
        speed = _forward_speed(raw.vehicle)
        accel = _current_accel(raw, prev_speed=prev_speed, dt=dt)
        prev_speed = speed
        min_spacing, min_ttc = _scene_metrics(raw)
        curr_position = np.array(raw.vehicle.position, dtype=float, copy=True)
        path_progress += _path_progress_delta(prev_position, curr_position)
        prev_position = curr_position
        lane_score = _lane_score(raw)
        crashed = bool(raw.vehicle.crashed)
        truncated = bool(trunc)
        curr_snapshot = _social_snapshot(raw)
        if spec.kind == "idm":
            lane_direction = _action_direction(1)
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
        log["progress"].append(path_progress)
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

        if save_frames and step < int(max_frame_steps):
            frame = env.render()
            if frame is not None:
                sim_time = field_warmup_s + float(step + 1) * dt
                overlays_per_step: dict = {}
                first_rgba = None
                first_extent = None
                first_kind = None
                for mode in (overlay_modes_norm or [""]):
                    if not mode:
                        continue
                    rgba, extent, kind = _capture_overlay_payload(
                        env,
                        overlay_mode=mode,
                        adapter=pinn_adapter,
                        sim_time=sim_time,
                    )
                    if rgba is not None and kind:
                        overlays_per_step[mode] = (rgba, extent, kind)
                    if first_rgba is None and rgba is not None:
                        first_rgba, first_extent, first_kind = rgba, extent, kind
                render_steps.append(
                    RenderStep(
                        planner=spec.label,
                        step=int(step),
                        action=int(action),
                        reward=float(reward),
                        total_return=float(ep_return),
                        speed=float(speed),
                        ttc=float(min_ttc),
                        min_spacing=float(min_spacing),
                        social_score=float(social_step.get("social_friendliness_score", float("nan"))),
                        corridor_risk=float(curr_snapshot.get("corridor_risk", 0.0)),
                        crashed=bool(crashed),
                        frame=np.asarray(frame),
                        overlay_rgba=first_rgba,
                        overlay_extent=first_extent,
                        overlay_mode=first_kind,
                        overlays=overlays_per_step,
                    )
                )

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
    near_mask = np.isfinite(spacing) & (spacing < 8.0)
    jerk = np.abs(np.diff(accels) / max(dt, 1e-6)) if accels.size > 1 else np.asarray([], dtype=float)

    social_summary = {
        key: _nanmean_or_nan(values)
        for key, values in log.items()
        if key not in {"step", "reward", "progress", "speed", "accel", "ttc", "min_spacing", "lane_score"}
    }
    cls_arr = np.asarray(log["social_class"], dtype=float)
    for idx, name in enumerate(("social_good", "social_defensive", "social_aggressive", "social_passive", "social_harmful")):
        social_summary[f"{name}_frac"] = float(np.mean(cls_arr == float(idx))) if cls_arr.size else 0.0

    result = EpisodeResult(
        planner=spec.label,
        algo=spec.algo,
        seed=int(seed),
        episode_return=float(ep_return),
        episode_length=int(len(log["step"])),
        crashed=bool(crashed),
        truncated=bool(truncated),
        ttc_min=float(np.nanmin(ttc)) if ttc.size else float("nan"),
        cmr=float(np.mean(ttc < 3.0)) if ttc.size else float("nan"),
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
    return result, render_steps


def _resolve_overlay_for_mode(frame_step: RenderStep, mode: str | None):
    """Return (rgba, extent, label) for the requested overlay mode, with legacy fallback."""
    if mode and frame_step.overlays:
        payload = frame_step.overlays.get(mode)
        if payload is not None:
            rgba, extent, kind = payload
            return rgba, extent, kind
    # Legacy single-overlay fallback
    return frame_step.overlay_rgba, frame_step.overlay_extent, frame_step.overlay_mode


def _save_multiplanner_frame(
    save_path: str,
    frame_steps: list[RenderStep],
    *,
    env_id: str,
    traffic_label: str,
    step: int,
    overlay_mode: str | None = None,
    layout: str = "grid",
) -> None:
    _apply_style()
    n = len(frame_steps)
    if layout not in {"grid", "vertical"}:
        raise ValueError(f"Unsupported frame layout: {layout}")
    ncols = 1 if layout == "vertical" else (2 if n > 1 else 1)
    nrows = int(math.ceil(n / ncols))
    if n == 1:
        figsize = (10.0, 3.4)
    elif layout == "vertical":
        figsize = (8.0, 3.05 * nrows)
    elif n == 2:
        figsize = (8.0 * ncols, 3.35)
    else:
        figsize = (8.0 * ncols, 4.7 * nrows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False
    )
    axes_flat = axes.ravel()
    for ax, frame_step in zip(axes_flat, frame_steps):
        height, width = frame_step.frame.shape[:2]
        ax.imshow(frame_step.frame)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        rgba, extent, kind = _resolve_overlay_for_mode(frame_step, overlay_mode)
        if rgba is not None and extent is not None:
            ax.imshow(
                rgba,
                origin="upper",
                extent=extent,
                interpolation="bilinear",
                zorder=2,
            )
        ax.axis("off")
        ax.set_title(frame_step.planner, fontsize=11, fontweight="bold")
        annotation_lines = [
            f"step={frame_step.step}  a={frame_step.action}  r={frame_step.reward:.3f}",
            f"ret={frame_step.total_return:.3f}  v={frame_step.speed:.2f} m/s",
        ]
        if np.isfinite(frame_step.ttc) or np.isfinite(frame_step.min_spacing):
            parts = []
            if np.isfinite(frame_step.ttc):
                parts.append(f"TTC={frame_step.ttc:.2f} s")
            if np.isfinite(frame_step.min_spacing):
                parts.append(f"gap={frame_step.min_spacing:.2f} m")
            annotation_lines.append("  ".join(parts))
        if np.isfinite(frame_step.social_score) or np.isfinite(frame_step.corridor_risk):
            parts = []
            if np.isfinite(frame_step.social_score):
                parts.append(f"social={frame_step.social_score:.2f}")
            if np.isfinite(frame_step.corridor_risk):
                parts.append(f"risk={frame_step.corridor_risk:.2f}")
            annotation_lines.append("  ".join(parts))
        annotation_lines.append(f"crash={frame_step.crashed}")
        if kind:
            annotation_lines.append(f"overlay={kind}")
        overlay_text = "\n".join(annotation_lines)
        ax.text(
            0.01,
            0.01,
            overlay_text,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.70, edgecolor="0.35"),
        )
    for ax in axes_flat[n:]:
        ax.axis("off")
    if overlay_mode and any(
        _resolve_overlay_for_mode(frame_step, overlay_mode)[0] is not None
        for frame_step in frame_steps
    ):
        scalar_map = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap=FIELD_CMAP
        )
        scalar_map.set_array([])
        colorbar = fig.colorbar(
            scalar_map,
            ax=list(axes_flat[:n]),
            orientation="horizontal",
            fraction=0.045,
            pad=0.02,
            aspect=40,
        )
        colorbar.set_label("Normalized risk intensity", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    suptitle = f"{_env_title(env_id)} | traffic={traffic_label} | step={step}"
    if overlay_mode:
        suptitle += f" | overlay={overlay_mode}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_rollout_frames(
    frames_dir: str,
    planner_frames: dict[str, list[RenderStep]],
    *,
    env_id: str,
    traffic_label: str,
    overlay_mode: str | None = None,
) -> None:
    os.makedirs(frames_dir, exist_ok=True)
    max_steps = max((len(v) for v in planner_frames.values()), default=0)
    for step in range(max_steps):
        frame_steps = [frames[step] for _name, frames in planner_frames.items() if step < len(frames)]
        if not frame_steps:
            continue
        _save_multiplanner_frame(
            os.path.join(frames_dir, f"step_{step:04d}.png"),
            frame_steps,
            env_id=env_id,
            traffic_label=traffic_label,
            step=step,
            overlay_mode=overlay_mode,
        )


def _save_timeseries_plot_multi(save_path: str, planner_episodes: dict[str, list[EpisodeResult]], *, env_id: str, traffic_label: str) -> None:
    _apply_style()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    panels = [
        ("progress", "Path Progress", "m"),
        ("speed", "Speed", "m/s"),
        ("ttc", "Critical TTC", "s"),
        ("risk_mass_others", "Imposed Risk Potential", "risk"),
        ("social_friendliness_score", "Social-Friendliness Score", "score"),
        ("safety_score", "Safety Score", "score"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"{_env_title(env_id)} | traffic={traffic_label}: Multi-Planner Time Series", fontsize=12)
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        for color, (planner_name, episodes) in zip(colors, planner_episodes.items()):
            t, mean, std = _stack_episode_metric(episodes, key)
            ax.plot(t, mean, color=color, lw=1.8, label=planner_name)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_core_summary_plot_multi(save_path: str, planner_summaries: dict[str, dict[str, float]], *, env_id: str, traffic_label: str) -> None:
    _apply_style()
    labels = list(planner_summaries.keys())
    values_map = [
        ("ttc_min_mean", "TTC_min", "s", False, 0.0),
        ("cmr_mean", "CMR (TTC < 3s)", "fraction", True, 0.0),
        ("min_spacing_mean", "Min spacing", "m", False, 0.0),
        ("mean_abs_jerk", "Mean |jerk|", r"m/s$^3$", True, 0.0),
        ("near_collision_step_rate", "Near-collision rate", "fraction", True, 0.0),
        ("final_progress_mean", "Path progress", "m", False, 0.0),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    fig.suptitle(f"{_env_title(env_id)} | traffic={traffic_label}: Core Metrics", fontsize=12)
    for ax, (key, title, ylabel, lower_is_better, ymin) in zip(axes.flat, values_map):
        values = [planner_summaries[name].get(key, float("nan")) for name in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors[: len(labels)], alpha=0.88)
        finite = [i for i, v in enumerate(values) if np.isfinite(v)]
        if finite:
            best_idx = min(finite, key=lambda i: values[i]) if lower_is_better else max(finite, key=lambda i: values[i])
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(2.2)
        for xi, val in zip(x, values):
            if np.isfinite(val):
                ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=float(ymin))
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_social_summary_plot_multi(save_path: str, planner_summaries: dict[str, dict[str, float]], *, env_id: str, traffic_label: str) -> None:
    _apply_style()
    panels = [
        ("risk_mass_others_mean", "Imposed Risk Potential", "risk", True),
        ("risk_flux_backward_mean", "Backward Risk Flux", "risk flux", True),
        ("safety_score_mean", "Safety Score", "score", False),
        ("courtesy_score_mean", "Courtesy Score", "score", False),
        ("social_friendliness_score_mean", "Social-Friendliness Score", "score", False),
        ("social_harmful_frac_mean", "Social-Harmful Fraction", "fraction", True),
    ]
    labels = list(planner_summaries.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)
    fig.suptitle(f"{_env_title(env_id)} | traffic={traffic_label}: Social / Externality Metrics", fontsize=12)
    for ax, (key, title, ylabel, lower_is_better) in zip(axes.flat, panels):
        values = [planner_summaries[name].get(key, float("nan")) for name in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors[: len(labels)], alpha=0.88)
        finite = [i for i, v in enumerate(values) if np.isfinite(v)]
        if finite:
            best_idx = min(finite, key=lambda i: values[i]) if lower_is_better else max(finite, key=lambda i: values[i])
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(2.2)
        for xi, val in zip(x, values):
            if np.isfinite(val):
                ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        if lower_is_better:
            ax.set_ylim(bottom=0.0)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_unified_table(
    save_dir: str,
    rows: list[dict[str, Any]],
) -> tuple[str, str]:
    csv_path = os.path.join(save_dir, "unified_suite_table.csv")
    md_path = os.path.join(save_dir, "unified_suite_table.md")
    base_fields = [
        "env_id",
        "traffic_label",
        "planner",
        "episodes",
        "return_mean",
        "collision_rate",
        "ttc_min_mean",
        "min_spacing_mean",
        "mean_speed",
        "final_progress_mean",
        "risk_mass_others_mean",
        "risk_flux_backward_mean",
        "safety_score_mean",
        "courtesy_score_mean",
        "social_friendliness_score_mean",
        "efficiency_index_ei_mean",
        "safety_efficiency_index_sei_mean",
        "social_traffic_efficiency_index_mean",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=base_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in base_fields})

    lines = [
        "# HighwayEnv SB3 Suite Summary",
        "",
        "| env_id | traffic | planner | return | collision | TTC_min | mean_speed | progress | imposed_risk | backward_flux | social_score | EI | SEI | STEI |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("env_id", "")),
                    str(row.get("traffic_label", "")),
                    str(row.get("planner", "")),
                    f"{float(row.get('return_mean', float('nan'))):.3f}" if np.isfinite(row.get("return_mean", np.nan)) else "nan",
                    f"{float(row.get('collision_rate', float('nan'))):.3f}" if np.isfinite(row.get("collision_rate", np.nan)) else "nan",
                    f"{float(row.get('ttc_min_mean', float('nan'))):.3f}" if np.isfinite(row.get("ttc_min_mean", np.nan)) else "nan",
                    f"{float(row.get('mean_speed', float('nan'))):.3f}" if np.isfinite(row.get("mean_speed", np.nan)) else "nan",
                    f"{float(row.get('final_progress_mean', float('nan'))):.3f}" if np.isfinite(row.get("final_progress_mean", np.nan)) else "nan",
                    f"{float(row.get('risk_mass_others_mean', float('nan'))):.3f}" if np.isfinite(row.get("risk_mass_others_mean", np.nan)) else "nan",
                    f"{float(row.get('risk_flux_backward_mean', float('nan'))):.3f}" if np.isfinite(row.get("risk_flux_backward_mean", np.nan)) else "nan",
                    f"{float(row.get('social_friendliness_score_mean', float('nan'))):.3f}" if np.isfinite(row.get("social_friendliness_score_mean", np.nan)) else "nan",
                    f"{float(row.get('efficiency_index_ei_mean', float('nan'))):.3f}" if np.isfinite(row.get("efficiency_index_ei_mean", np.nan)) else "nan",
                    f"{float(row.get('safety_efficiency_index_sei_mean', float('nan'))):.3f}" if np.isfinite(row.get("safety_efficiency_index_sei_mean", np.nan)) else "nan",
                    f"{float(row.get('social_traffic_efficiency_index_mean', float('nan'))):.3f}" if np.isfinite(row.get("social_traffic_efficiency_index_mean", np.nan)) else "nan",
                ]
            )
            + " |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return csv_path, md_path


def compare_suite(
    *,
    env_id: str,
    traffic_labels: list[str],
    episodes: int,
    frame_seed: int,
    max_frame_steps: int,
    save_dir: str,
    include_social_ppo: bool,
    include_social_dqn: bool,
    include_stock_ppo: bool,
    include_stock_dqn: bool,
    include_idm: bool,
    social_ppo_checkpoint: str | None,
    social_dqn_checkpoint: str | None,
    stock_ppo_checkpoint: str | None,
    stock_dqn_checkpoint: str | None,
    overlay_modes: list[str] | tuple[str, ...] | str,
    pinn_checkpoint: str | None,
    pinn_device: str,
    vehicles_count: int | None,
    vehicles_density: float | None,
    ego_spacing: float | None,
    duration: float | None,
    sv_speed_min: float | None,
    sv_speed_max: float | None,
    sv_speed_noise: float | None,
    lane_speed_bias: tuple[float, ...],
    env_config_override: dict[str, Any] | None,
) -> dict[str, Any]:
    # Normalise overlay_modes to a deduplicated, lowercased list
    if isinstance(overlay_modes, str):
        overlay_modes = [overlay_modes]
    overlay_modes_norm: list[str] = []
    for mode in overlay_modes:
        m = str(mode).strip().lower()
        if m and m not in {"none", "off"} and m not in overlay_modes_norm:
            overlay_modes_norm.append(m)
    os.makedirs(save_dir, exist_ok=True)
    suite_rows: list[dict[str, Any]] = []
    suite_missing: list[dict[str, str]] = []
    scenario_summaries: dict[str, Any] = {}

    for traffic_label in traffic_labels:
        traffic = resolve_traffic_config(
            preset=traffic_label,
            vehicles_count=vehicles_count,
            vehicles_density=vehicles_density,
            ego_spacing=ego_spacing,
            duration=duration,
            sv_speed_min=sv_speed_min,
            sv_speed_max=sv_speed_max,
            sv_speed_noise=sv_speed_noise,
            lane_speed_bias=lane_speed_bias,
        )
        specs = _planner_specs_for_env(
            env_id,
            include_social_ppo=include_social_ppo,
            include_social_dqn=include_social_dqn,
            include_stock_ppo=include_stock_ppo,
            include_stock_dqn=include_stock_dqn,
            include_idm=include_idm,
            social_ppo_checkpoint=social_ppo_checkpoint,
            social_dqn_checkpoint=social_dqn_checkpoint,
            stock_ppo_checkpoint=stock_ppo_checkpoint,
            stock_dqn_checkpoint=stock_dqn_checkpoint,
        )
        available_specs, missing = _resolve_available_specs(specs)
        suite_missing.extend([{**row, "traffic_label": traffic_label, "env_id": env_id} for row in missing])
        if not available_specs:
            continue

        planner_episodes: dict[str, list[EpisodeResult]] = {}
        planner_frame_steps: dict[str, list[RenderStep]] = {}
        for spec in available_specs:
            episodes_list: list[EpisodeResult] = []
            frame_steps: list[RenderStep] = []
            for seed in range(int(episodes)):
                collect_frames = seed == int(frame_seed)
                ep, render_steps = run_episode(
                    spec,
                    env_id=env_id,
                    traffic=traffic,
                    seed=seed,
                    render_mode="rgb_array" if collect_frames else None,
                    save_frames=collect_frames,
                    max_frame_steps=max_frame_steps,
                    overlay_modes=overlay_modes_norm,
                    pinn_checkpoint=pinn_checkpoint,
                    pinn_device=pinn_device,
                    env_config_override=env_config_override,
                )
                episodes_list.append(ep)
                if collect_frames:
                    frame_steps = render_steps
            planner_episodes[spec.label] = episodes_list
            planner_frame_steps[spec.label] = frame_steps

        planner_summaries = {name: _planner_summary(eps) for name, eps in planner_episodes.items()}
        subdir = os.path.join(save_dir, f"{env_id.replace('-', '_')}_{traffic_label}")
        os.makedirs(subdir, exist_ok=True)
        core_plot = os.path.join(subdir, "metrics_summary.png")
        social_plot = os.path.join(subdir, "social_metrics_summary.png")
        timeseries_plot = os.path.join(subdir, "metrics_timeseries.png")
        episodes_json = os.path.join(subdir, "episodes.json")
        summary_json = os.path.join(subdir, "summary.json")
        table_csv, table_md = write_summary_table(subdir, planner_summaries, env_id=f"{env_id} [{traffic_label}]")

        _save_core_summary_plot_multi(core_plot, planner_summaries, env_id=env_id, traffic_label=traffic_label)
        _save_social_summary_plot_multi(social_plot, planner_summaries, env_id=env_id, traffic_label=traffic_label)
        _save_timeseries_plot_multi(timeseries_plot, planner_episodes, env_id=env_id, traffic_label=traffic_label)

        # One frames_<mode>/ subdir per requested overlay mode (or just frames/ if none)
        frames_dirs: dict[str, str] = {}
        if overlay_modes_norm:
            for mode in overlay_modes_norm:
                frames_dir_mode = os.path.join(subdir, f"frames_{mode}")
                _save_rollout_frames(
                    frames_dir_mode,
                    planner_frame_steps,
                    env_id=env_id,
                    traffic_label=traffic_label,
                    overlay_mode=mode,
                )
                frames_dirs[mode] = frames_dir_mode
        else:
            frames_dir_plain = os.path.join(subdir, "frames")
            _save_rollout_frames(
                frames_dir_plain,
                planner_frame_steps,
                env_id=env_id,
                traffic_label=traffic_label,
                overlay_mode=None,
            )
            frames_dirs["none"] = frames_dir_plain

        with open(episodes_json, "w", encoding="utf-8") as f:
            json.dump({planner_name: _episode_records(eps) for planner_name, eps in planner_episodes.items()}, f, indent=2)

        scenario_summary = {
            "env_id": env_id,
            "traffic_label": traffic_label,
            "traffic_config": traffic.to_dict(),
            "env_config_override": env_config_override or {},
            "metrics": planner_summaries,
            "plots": {
                "metrics_summary": core_plot,
                "social_metrics_summary": social_plot,
                "metrics_timeseries": timeseries_plot,
                "frames_dirs": frames_dirs,
            },
            "overlay_modes": overlay_modes_norm,
            "summary_table_csv": table_csv,
            "summary_table_md": table_md,
            "episodes_json": episodes_json,
            "missing_planners": missing,
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(scenario_summary, f, indent=2)
        scenario_summaries[traffic_label] = scenario_summary

        for planner_name, summary in planner_summaries.items():
            suite_rows.append(
                {
                    "env_id": env_id,
                    "traffic_label": traffic_label,
                    "planner": planner_name,
                    **summary,
                }
            )

    unified_csv, unified_md = _write_unified_table(save_dir, suite_rows)
    timing_csv = os.path.join(save_dir, "compute_time_eval.csv")
    get_timer().write_csv(timing_csv)
    suite_summary = {
        "env_id": env_id,
        "traffic_labels": traffic_labels,
        "episodes": int(episodes),
        "frame_seed": int(frame_seed),
        "max_frame_steps": int(max_frame_steps),
        "overlay_modes": overlay_modes_norm,
        "pinn_checkpoint": pinn_checkpoint,
        "pinn_device": pinn_device,
        "env_config_override": env_config_override or {},
        "unified_table_csv": unified_csv,
        "unified_table_md": unified_md,
        "compute_time_csv": timing_csv,
        "missing_planners": suite_missing,
        "scenarios": scenario_summaries,
    }
    with open(os.path.join(save_dir, "suite_summary.json"), "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)
    return suite_summary


def _parse_args():
    p = argparse.ArgumentParser(description="Consistent HighwayEnv suite for stock RL, social RL, and IDM with native frame rendering, traffic presets, and unified statistics.")
    p.add_argument("--env-id", default="highway-v0", choices=["highway-v0", "merge-v0", "roundabout-v0", "intersection-v0"])
    p.add_argument("--traffic-preset", nargs="+", default=["medium", "dense"], help="Traffic presets to compare, e.g. medium dense or native medium.")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--frame-seed", type=int, default=0)
    p.add_argument("--max-frame-steps", type=int, default=60)
    p.add_argument("--save-dir", default="rl/logs/highwayenv_sb3_suite")
    p.add_argument("--include-social-ppo", type=_str2bool, default=True)
    p.add_argument("--include-social-dqn", type=_str2bool, default=True)
    p.add_argument("--include-stock-ppo", type=_str2bool, default=True)
    p.add_argument("--include-stock-dqn", type=_str2bool, default=True)
    p.add_argument("--include-idm", type=_str2bool, default=True)
    p.add_argument("--social-ppo-checkpoint", default="")
    p.add_argument("--social-dqn-checkpoint", default="")
    p.add_argument("--stock-ppo-checkpoint", default="")
    p.add_argument("--stock-dqn-checkpoint", default="")
    p.add_argument("--overlay-mode", nargs="+", default=[DEFAULT_OVERLAY_MODE],
                   choices=["none", "drift", "pinn"],
                   help="One or more risk overlays to render per frame. Pass space-separated values (e.g. 'pinn drift') to emit frames_pinn/ and frames_drift/ side-by-side from the same simulation.")
    p.add_argument("--pinn-checkpoint", default="", help="Optional .pt checkpoint for PINN field overlay. Defaults to the best available PINN checkpoint.")
    p.add_argument("--pinn-device", default="cpu",
                   help="Torch device for the PINN risk-field adapter ('cpu', 'cuda', 'cuda:0'). GPU helps for the full-grid overlay (~30k pts).")
    p.add_argument("--vehicles-count", type=int, default=None)
    p.add_argument("--vehicles-density", type=float, default=None)
    p.add_argument("--ego-spacing", type=float, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--sv-speed-min", type=float, default=None)
    p.add_argument("--sv-speed-max", type=float, default=None)
    p.add_argument("--sv-speed-noise", type=float, default=None)
    p.add_argument("--lane-speed-bias", default="", help="Comma-separated lane speed biases.")
    p.add_argument("--env-config-json", default="", help="Raw JSON object merged into the HighwayEnv config before env creation.")
    p.add_argument("--env-config-file", default="", help="Path to a JSON file merged into the HighwayEnv config before env creation.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    lane_bias = tuple(float(x) for x in args.lane_speed_bias.split(",") if str(x).strip())
    env_config_override = _load_env_config_override(
        json_text=str(args.env_config_json).strip() or None,
        json_file=str(args.env_config_file).strip() or None,
    )
    summary = compare_suite(
        env_id=args.env_id,
        traffic_labels=[str(x) for x in args.traffic_preset],
        episodes=int(args.episodes),
        frame_seed=int(args.frame_seed),
        max_frame_steps=int(args.max_frame_steps),
        save_dir=args.save_dir,
        include_social_ppo=bool(args.include_social_ppo),
        include_social_dqn=bool(args.include_social_dqn),
        include_stock_ppo=bool(args.include_stock_ppo),
        include_stock_dqn=bool(args.include_stock_dqn),
        include_idm=bool(args.include_idm),
        social_ppo_checkpoint=str(args.social_ppo_checkpoint).strip() or None,
        social_dqn_checkpoint=str(args.social_dqn_checkpoint).strip() or None,
        stock_ppo_checkpoint=str(args.stock_ppo_checkpoint).strip() or None,
        stock_dqn_checkpoint=str(args.stock_dqn_checkpoint).strip() or None,
        overlay_modes=[str(m).strip().lower() for m in args.overlay_mode],
        pinn_checkpoint=str(args.pinn_checkpoint).strip() or None,
        pinn_device=str(args.pinn_device).strip() or "cpu",
        vehicles_count=args.vehicles_count,
        vehicles_density=args.vehicles_density,
        ego_spacing=args.ego_spacing,
        duration=args.duration,
        sv_speed_min=args.sv_speed_min,
        sv_speed_max=args.sv_speed_max,
        sv_speed_noise=args.sv_speed_noise,
        lane_speed_bias=lane_bias,
        env_config_override=env_config_override,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
