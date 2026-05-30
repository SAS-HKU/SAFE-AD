"""
MetaDrive DRIFT Wrapper
=======================
gym.Wrapper that augments MetaDrive's stock single-agent envs with the
DRIFT social-risk field. Mirrors the pattern of
[highwayenv_drift_wrapper.py](highwayenv_drift_wrapper.py).

Key differences vs the HighwayEnv wrapper:
- Uses an **ego-relative** body-frame grid that is re-centred each step
  (MetaDrive maps are procedural / curved). The PDE field is reset and
  warmed-up briefly each step to reach quasi-steady response to current
  vehicle layout. This loses cross-step temporal memory of the field but
  matches the spatial-feature semantics needed by the RL policy.
- Augments the observation with an 8-D risk vector
  ``(r_ego, r_5m, r_10m, r_20m, r_left, r_right, grad_x, grad_y)``.
- Shapes reward additively: ``shaped = stock - λ_risk * r_ego``.
- For Safe-RL envs, augments the cost: ``cost = stock + w_risk_cost * 𝟙[r_ego > τ]``.

The wrapper is **passive**: it never edits MetaDrive sources, never blocks
actions. It only wraps observation, reward, cost, and info dict.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import Config as cfg
from pde_solver import compute_Q_vehicle, compute_Q_occlusion
from Integration.drift_interface import DRIFTInterface

from rl.config.metadrive_config import MetaDriveRLConfig, DEFAULT_METADRIVE_CONFIG
from rl.risk.metadrive_scene_adapter import (
    collect_drift_state,
    enumerate_traffic_vehicles,
)


_RISK_FEATURE_NAMES = (
    "r_ego",
    "r_5m",
    "r_10m",
    "r_20m",
    "r_left",
    "r_right",
    "grad_x",
    "grad_y",
)
RISK_FEATURE_DIM = len(_RISK_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Calibrated reward-cost primitives.
# These are inlined copies of the pure functions in
# `rl/reward/social_reward.py` (quadratic_upper_cost, courtesy_brake_cost,
# backward_flux_cost). We replicate rather than import to keep the training
# hot path free of the `rl.reward` -> `rl.data.historical_extractor` ->
# `rl.policy.decision_policy` import chain. Constants live in MetaDriveRLConfig
# and match SocialRewardConfig, so offline analysis and online shaping agree.
# ---------------------------------------------------------------------------

def _quadratic_upper_cost(value: float, start: float, max_value: float) -> float:
    """0 below ``start``, ramps quadratically to 1 at ``max_value``."""
    if float(value) <= float(start):
        return 0.0
    denom = max(1e-6, float(max_value) - float(start))
    scaled = float(np.clip((float(value) - float(start)) / denom, 0.0, 1.0))
    return scaled * scaled


def _courtesy_brake_cost(imposed_decel: float, courtesy_decel: float, hard_brake: float) -> float:
    """Penalty for forcing a follower to brake harder than ``courtesy_decel``."""
    if float(imposed_decel) <= float(courtesy_decel):
        return 0.0
    denom = max(1e-6, float(hard_brake) - float(courtesy_decel))
    scaled = float(np.clip((float(imposed_decel) - float(courtesy_decel)) / denom, 0.0, 1.0))
    return scaled * scaled


def _backward_flux_cost(flux: float, b0: float) -> float:
    """Saturating cost on backward-propagating risk pressure (∈ [0, 1))."""
    flux = max(0.0, float(flux))
    return flux / (flux + max(1e-6, float(b0)))


_GRID_READY_FOR: tuple = ()


def _configure_metadrive_grid(mdcfg: MetaDriveRLConfig) -> None:
    """Set the shared `config.Config` grid to the ego-relative extents.

    DRIFT's `cfg` module is a global; HighwayEnv wraps it the same way. We
    only reconfigure when the (extents, resolution) tuple changes.
    """
    global _GRID_READY_FOR
    extents = (
        float(mdcfg.GRID_X_MIN), float(mdcfg.GRID_X_MAX),
        float(mdcfg.GRID_Y_MIN), float(mdcfg.GRID_Y_MAX),
        float(mdcfg.GRID_DX), float(mdcfg.GRID_DY),
    )
    if _GRID_READY_FOR == extents:
        return
    cfg.x_min, cfg.x_max = extents[0], extents[1]
    cfg.y_min, cfg.y_max = extents[2], extents[3]
    cfg.nx = int(round((cfg.x_max - cfg.x_min) / extents[4])) + 1
    cfg.ny = int(round((cfg.y_max - cfg.y_min) / extents[5])) + 1
    cfg.dx = (cfg.x_max - cfg.x_min) / (cfg.nx - 1)
    cfg.dy = (cfg.y_max - cfg.y_min) / (cfg.ny - 1)
    cfg.x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    cfg.y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    cfg.X, cfg.Y = np.meshgrid(cfg.x, cfg.y)
    _GRID_READY_FOR = extents


@dataclass
class RiskFeatures:
    r_ego: float
    r_5m: float
    r_10m: float
    r_20m: float
    r_left: float
    r_right: float
    grad_x: float
    grad_y: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.r_ego, self.r_5m, self.r_10m, self.r_20m,
             self.r_left, self.r_right, self.grad_x, self.grad_y],
            dtype=np.float32,
        )

    def as_dict(self) -> dict:
        return {name: float(getattr(self, name)) for name in _RISK_FEATURE_NAMES}


class MetaDriveDriftWrapper(gym.Wrapper):
    """Wraps a MetaDrive single-agent env with DRIFT risk-field augmentation.

    Args:
        env: A MetaDrive env (MetaDriveEnv, SafeMetaDriveEnv, etc.)
        config: MetaDriveRLConfig instance.
        append_risk_obs: If True, append normalised 8-D risk features to Box observations.
        shape_risk_reward: If True, use DRIFT risk for reward and SafeMetaDrive cost shaping.
        compute_risk_metrics: If True, compute DRIFT features for metrics in ``info``.
        is_safe_env: If True, adds risk-threshold cost to ``info["cost"]``.
        record_grid: If True, store the risk grid in info each step
            (or every ``RECORD_GRID_EVERY`` steps to save memory).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        config: Optional[MetaDriveRLConfig] = None,
        use_risk: Optional[bool] = None,
        append_risk_obs: Optional[bool] = None,
        shape_risk_reward: Optional[bool] = None,
        compute_risk_metrics: Optional[bool] = None,
        is_safe_env: bool = False,
        record_grid: bool = False,
    ) -> None:
        super().__init__(env)
        self.mdcfg: MetaDriveRLConfig = config or DEFAULT_METADRIVE_CONFIG
        if use_risk is not None:
            if append_risk_obs is None:
                append_risk_obs = bool(use_risk)
            if shape_risk_reward is None:
                shape_risk_reward = bool(use_risk)
            if compute_risk_metrics is None:
                compute_risk_metrics = bool(use_risk)
        self.append_risk_obs = bool(True if append_risk_obs is None else append_risk_obs)
        self.shape_risk_reward = bool(True if shape_risk_reward is None else shape_risk_reward)
        requested_compute = bool(True if compute_risk_metrics is None else compute_risk_metrics)
        self.record_grid = bool(record_grid)
        self.compute_risk_metrics = bool(
            requested_compute or self.append_risk_obs or self.shape_risk_reward or self.record_grid
        )
        # Backward-compatible alias for older utilities.
        self.use_risk = bool(self.compute_risk_metrics)
        self.is_safe_env = bool(is_safe_env)

        _configure_metadrive_grid(self.mdcfg)
        self.drift: Optional[DRIFTInterface] = None
        self._episode_steps = 0
        self._last_features: Optional[RiskFeatures] = None
        self._last_grid_snapshot: Optional[np.ndarray] = None
        self._last_action_vec: Optional[np.ndarray] = None
        self._last_speed_mps: Optional[float] = None
        self._last_accel_mps2: float = 0.0
        # Overtake tracking: ids of traffic vehicles that have been observed
        # ahead of ego AND are now behind (mirrors MetaDrive's native overtake
        # logic; that one is broken in 0.4.3 due to a lidar-arg bug).
        self._was_in_front: set = set()
        self._overtaken_ids: set = set()

        # Social-externality tracking: the most recent ego-body neighbour dicts
        # (from the DRIFT scene adapter), ego longitudinal speed, and the
        # previous-step rear follower TTC (for the rear-TTC-erosion penalty).
        self._last_neighbors: Optional[list] = None
        self._last_ego_long_speed: float = 0.0
        self._prev_rear_ttc: Optional[float] = None

        # Expand observation space only when the policy receives risk features.
        self.observation_space = self._make_augmented_space(env.observation_space)

    # ------------------------------------------------------------------ space

    def _make_augmented_space(self, base_space: spaces.Space) -> spaces.Space:
        if not self.append_risk_obs:
            return base_space
        if not isinstance(base_space, spaces.Box):
            # Non-Box obs spaces (Dict/Tuple) are not in v1 scope.
            return base_space
        base_low = base_space.low.flatten().astype(np.float32)
        base_high = base_space.high.flatten().astype(np.float32)
        # Risk features are normalised to ~[-2, 2] but allow a generous margin
        risk_low = np.full((RISK_FEATURE_DIM,), -5.0, dtype=np.float32)
        risk_high = np.full((RISK_FEATURE_DIM,), 5.0, dtype=np.float32)
        new_low = np.concatenate([base_low, risk_low])
        new_high = np.concatenate([base_high, risk_high])
        return spaces.Box(low=new_low, high=new_high, dtype=np.float32)

    # ----------------------------------------------------------------- gym API

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        try:
            out = self.env.reset(seed=seed, options=options)
        except TypeError:
            # MetaDrive's BaseEnv.reset() doesn't accept `options`
            out = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        if isinstance(out, tuple):
            obs, info = out
        else:  # legacy gym fallback
            obs, info = out, {}
        info = dict(info) if isinstance(info, dict) else {}
        self._episode_steps = 0
        self._last_action_vec = None
        self._last_speed_mps = self._ego_speed_mps()
        self._last_accel_mps2 = 0.0
        self._was_in_front = set()
        self._overtaken_ids = set()
        self._last_neighbors = None
        self._last_ego_long_speed = self._ego_speed_mps() or 0.0
        self._prev_rear_ttc = None

        _configure_metadrive_grid(self.mdcfg)
        if self.compute_risk_metrics:
            self.drift = DRIFTInterface()
            self.drift.reset()
            features = self._compute_risk_features_for_current_state()
        else:
            self.drift = None
            features = RiskFeatures(0, 0, 0, 0, 0, 0, 0, 0)

        self._last_features = features
        obs_aug = self._augment_obs(obs, features)
        info.update(self._format_info(features, base_reward=None, base_cost=None))
        if self.record_grid and self.drift is not None:
            info["risk_grid"] = self._snapshot_grid()
        return obs_aug, info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, base_reward, terminated, truncated, info = out
        else:  # legacy gym fallback
            obs, base_reward, done, info = out
            terminated, truncated = bool(done), False
        info = dict(info) if isinstance(info, dict) else {}
        self._episode_steps += 1

        base_reward = float(base_reward)
        base_cost = float(info.get("cost", 0.0))
        action_vec = self._applied_control_vector(action)
        action_metrics = self._control_change_metrics(action_vec)
        motion_metrics = self._motion_metrics_after_step()

        if self.compute_risk_metrics and self.drift is not None:
            features = self._compute_risk_features_for_current_state()
        else:
            features = RiskFeatures(0, 0, 0, 0, 0, 0, 0, 0)
        self._last_features = features

        comfort_metrics = self._comfort_metrics(action_metrics, motion_metrics)
        social_metrics = self._social_metrics()
        shaped_reward, penalties = self._shape_reward(
            base_reward, features, comfort_metrics, social_metrics
        )
        shaped_cost, risk_cost = self._shape_cost(base_cost, features)
        self._commit_step_state(action_vec, motion_metrics)

        # Overtake counting (MetaDrive 0.4.3's built-in is buggy)
        self._update_overtake_count()

        info["cost"] = shaped_cost
        info["overtake_vehicle_num"] = int(len(self._overtaken_ids))
        info.update(self._format_info(
            features,
            base_reward=base_reward,
            base_cost=base_cost,
            penalties=penalties,
            comfort_metrics=comfort_metrics,
            social_metrics=social_metrics,
            risk_cost=risk_cost,
        ))
        if self.record_grid and self.drift is not None:
            if self._episode_steps % max(1, int(self.mdcfg.RECORD_GRID_EVERY)) == 0:
                info["risk_grid"] = self._snapshot_grid()

        obs_aug = self._augment_obs(obs, features)
        return obs_aug, shaped_reward, terminated, truncated, info

    # ------------------------------------------------------------------ shaping

    def _env_dt(self) -> float:
        raw = self.env.unwrapped
        config = getattr(raw, "config", {}) or {}
        physics_dt = float(config.get("physics_world_step_size", 0.02))
        decision_repeat = float(config.get("decision_repeat", 5))
        return max(1e-6, physics_dt * decision_repeat)

    def _ego_speed_mps(self) -> Optional[float]:
        ego = self._get_ego_vehicle()
        if ego is None:
            return None
        return float(getattr(ego, "speed_km_h", 0.0)) / 3.6

    def _action_to_control_vector(self, action) -> np.ndarray:
        """Return normalised [steer, throttle/brake] in MetaDrive's action convention."""
        if isinstance(self.action_space, spaces.Discrete):
            raw = self.env.unwrapped
            config = getattr(raw, "config", {}) or {}
            steer_dim = int(config.get("discrete_steering_dim", 3))
            throttle_dim = int(config.get("discrete_throttle_dim", 3))
            idx = int(np.asarray(action).item())
            steer_idx = idx % max(1, steer_dim)
            throttle_idx = idx // max(1, steer_dim)
            steer_unit = 2.0 / max(1, steer_dim - 1)
            throttle_unit = 2.0 / max(1, throttle_dim - 1)
            steer = float(steer_idx) * steer_unit - 1.0
            throttle = float(throttle_idx) * throttle_unit - 1.0
            return np.asarray([steer, throttle], dtype=np.float32)
        if hasattr(spaces, "MultiDiscrete") and isinstance(self.action_space, spaces.MultiDiscrete):
            arr = np.asarray(action, dtype=float).flatten()
            raw = self.env.unwrapped
            config = getattr(raw, "config", {}) or {}
            steer_dim = int(config.get("discrete_steering_dim", 3))
            throttle_dim = int(config.get("discrete_throttle_dim", 3))
            steer_unit = 2.0 / max(1, steer_dim - 1)
            throttle_unit = 2.0 / max(1, throttle_dim - 1)
            steer = float(arr[0]) * steer_unit - 1.0 if arr.size >= 1 else 0.0
            throttle = float(arr[1]) * throttle_unit - 1.0 if arr.size >= 2 else 0.0
            return np.asarray([steer, throttle], dtype=np.float32)
        arr = np.asarray(action, dtype=np.float32).flatten()
        if arr.size == 0:
            return np.zeros(2, dtype=np.float32)
        if arr.size == 1:
            return np.asarray([float(arr[0]), 0.0], dtype=np.float32)
        return np.asarray([float(arr[0]), float(arr[1])], dtype=np.float32)

    def _applied_control_vector(self, fallback_action=None) -> np.ndarray:
        ego = self._get_ego_vehicle()
        if ego is not None and hasattr(ego, "steering") and hasattr(ego, "throttle_brake"):
            return np.asarray(
                [float(getattr(ego, "steering", 0.0)), float(getattr(ego, "throttle_brake", 0.0))],
                dtype=np.float32,
            )
        if fallback_action is None:
            return np.zeros(2, dtype=np.float32)
        return self._action_to_control_vector(fallback_action)

    def _control_change_metrics(self, action_vec: np.ndarray) -> dict[str, float]:
        action_vec = np.clip(np.asarray(action_vec, dtype=np.float32), -1.0, 1.0)
        if self._last_action_vec is None:
            delta = np.zeros_like(action_vec, dtype=np.float32)
        else:
            delta = action_vec - self._last_action_vec
        steer_delta = float(abs(delta[0])) if delta.size >= 1 else 0.0
        throttle_delta = float(abs(delta[1])) if delta.size >= 2 else 0.0
        return {
            "control_steer": float(action_vec[0]) if action_vec.size >= 1 else 0.0,
            "control_throttle": float(action_vec[1]) if action_vec.size >= 2 else 0.0,
            "action_delta": float(np.sum(np.abs(delta))),
            "steer_delta": steer_delta,
            "throttle_delta": throttle_delta,
        }

    def _motion_metrics_after_step(self) -> dict[str, float]:
        dt = self._env_dt()
        speed = self._ego_speed_mps()
        if speed is None:
            return {"speed_mps": 0.0, "accel": 0.0, "jerk": 0.0, "dt": dt}
        if self._last_speed_mps is None:
            accel = 0.0
        else:
            accel = (speed - float(self._last_speed_mps)) / dt
        jerk = (accel - float(self._last_accel_mps2)) / dt
        return {
            "speed_mps": float(speed),
            "accel": float(accel),
            "jerk": float(jerk),
            "dt": float(dt),
        }

    def _comfort_metrics(
        self,
        action_metrics: dict[str, float],
        motion_metrics: dict[str, float],
    ) -> dict[str, float]:
        steer_abs = abs(float(action_metrics.get("control_steer", 0.0)))
        action_delta_cost = float(self.mdcfg.LAMBDA_ACTION_DELTA) * float(action_metrics["action_delta"])
        jerk_cost = float(self.mdcfg.LAMBDA_JERK) * abs(float(motion_metrics["jerk"]))
        steer_abs_cost = float(self.mdcfg.LAMBDA_STEER_ABS) * steer_abs
        steer_delta_cost = float(self.mdcfg.LAMBDA_STEER_DELTA) * float(action_metrics["steer_delta"])
        throttle_delta_cost = float(self.mdcfg.LAMBDA_THROTTLE_DELTA) * float(action_metrics["throttle_delta"])
        comfort_cost = action_delta_cost + jerk_cost + steer_abs_cost + steer_delta_cost + throttle_delta_cost
        return {
            **action_metrics,
            **motion_metrics,
            "accel_abs": abs(float(motion_metrics["accel"])),
            "jerk_abs": abs(float(motion_metrics["jerk"])),
            "steer_abs": steer_abs,
            "action_delta_penalty": action_delta_cost,
            "jerk_penalty_cost": jerk_cost,
            "steer_abs_penalty_cost": steer_abs_cost,
            "steer_delta_penalty_cost": steer_delta_cost,
            "throttle_delta_penalty_cost": throttle_delta_cost,
            "comfort_cost": float(comfort_cost),
        }

    def _commit_step_state(self, action_vec: np.ndarray, motion_metrics: dict[str, float]) -> None:
        self._last_action_vec = np.asarray(action_vec, dtype=np.float32)
        self._last_speed_mps = float(motion_metrics.get("speed_mps", 0.0))
        self._last_accel_mps2 = float(motion_metrics.get("accel", 0.0))

    def _shape_reward(
        self,
        base_reward: float,
        features: RiskFeatures,
        comfort_metrics: dict[str, float],
        social_metrics: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Compose the shaped reward and return a per-term penalty breakdown.

        Penalty terms (all <= 0) sum exactly to ``shaped - base``:
          risk + steer + jerk + throttle + hard_brake + courtesy + rear_ttc
          + backward_flux.  ``comfort_penalty`` is the steer+jerk+throttle
          aggregate (kept for backward-compatible logging; not double-counted).
        """
        penalties = {
            "risk_penalty": 0.0,
            "comfort_penalty": 0.0,
            "steer_penalty": 0.0,
            "jerk_penalty": 0.0,
            "throttle_penalty": 0.0,
            "hard_brake_penalty": 0.0,
            "courtesy_penalty": 0.0,
            "rear_ttc_penalty": 0.0,
            "backward_flux_penalty": 0.0,
        }
        if not self.shape_risk_reward:
            return float(base_reward), penalties

        # --- risk ---
        r_ego = float(np.clip(features.r_ego, 0.0, self.mdcfg.RISK_CLIP))
        penalties["risk_penalty"] = -float(self.mdcfg.LAMBDA_RISK) * r_ego

        # --- comfort sub-terms (already weighted inside _comfort_metrics) ---
        steer_cost = (float(comfort_metrics.get("steer_abs_penalty_cost", 0.0))
                      + float(comfort_metrics.get("steer_delta_penalty_cost", 0.0)))
        jerk_cost = float(comfort_metrics.get("jerk_penalty_cost", 0.0))
        throttle_cost = (float(comfort_metrics.get("throttle_delta_penalty_cost", 0.0))
                         + float(comfort_metrics.get("action_delta_penalty", 0.0)))
        penalties["steer_penalty"] = -steer_cost
        penalties["jerk_penalty"] = -jerk_cost
        penalties["throttle_penalty"] = -throttle_cost
        penalties["comfort_penalty"] = -(steer_cost + jerk_cost + throttle_cost)

        # --- ego hard braking ---
        decel_ego = max(0.0, -float(comfort_metrics.get("accel", 0.0)))
        hb = _quadratic_upper_cost(
            decel_ego, self.mdcfg.EGO_DECEL_COMFORT, self.mdcfg.EGO_DECEL_HARD)
        penalties["hard_brake_penalty"] = -float(self.mdcfg.W_HARD_BRAKE) * hb

        # --- social courtesy to the rear follower ---
        cb = _courtesy_brake_cost(
            float(social_metrics.get("decel_follower", 0.0)),
            self.mdcfg.COURTESY_DECEL, self.mdcfg.FOLLOWER_HARD_BRAKE)
        penalties["courtesy_penalty"] = -float(self.mdcfg.W_COURTESY) * cb
        penalties["rear_ttc_penalty"] = (
            -float(self.mdcfg.W_REAR_TTC) * float(social_metrics.get("rear_ttc_loss", 0.0)))

        # --- backward risk flux ---
        bf = _backward_flux_cost(
            float(social_metrics.get("backward_pressure", 0.0)), self.mdcfg.FLUX_B0)
        penalties["backward_flux_penalty"] = -float(self.mdcfg.W_BACK_FLUX) * bf

        shaped = (float(base_reward)
                  + penalties["risk_penalty"]
                  + penalties["comfort_penalty"]
                  + penalties["hard_brake_penalty"]
                  + penalties["courtesy_penalty"]
                  + penalties["rear_ttc_penalty"]
                  + penalties["backward_flux_penalty"])
        return float(shaped), penalties

    def _shape_cost(
        self, base_cost: float, features: RiskFeatures
    ) -> tuple[float, float]:
        if not self.is_safe_env or not self.shape_risk_reward:
            return float(base_cost), 0.0
        risk_cost = float(self.mdcfg.W_RISK_COST) * float(
            features.r_ego > self.mdcfg.TAU_RISK
        )
        return float(base_cost) + risk_cost, risk_cost

    # -------------------------------------------------------------- obs helper

    def _augment_obs(self, base_obs: np.ndarray, features: RiskFeatures) -> np.ndarray:
        if not self.append_risk_obs:
            return base_obs
        if not isinstance(self.observation_space, spaces.Box):
            return base_obs  # non-Box: pass through unchanged
        base_arr = np.asarray(base_obs, dtype=np.float32).flatten()
        risk_arr = features.as_array() / np.array([
            self.mdcfg.NORM_RISK, self.mdcfg.NORM_RISK, self.mdcfg.NORM_RISK,
            self.mdcfg.NORM_RISK, self.mdcfg.NORM_RISK, self.mdcfg.NORM_RISK,
            self.mdcfg.NORM_GRAD, self.mdcfg.NORM_GRAD,
        ], dtype=np.float32)
        risk_arr = np.clip(risk_arr, -5.0, 5.0)
        return np.concatenate([base_arr, risk_arr]).astype(np.float32)

    # -------------------------------------------------------------- PDE driver

    def _compute_risk_features_for_current_state(self) -> RiskFeatures:
        ego_vehicle = self._get_ego_vehicle()
        if ego_vehicle is None:
            return RiskFeatures(0, 0, 0, 0, 0, 0, 0, 0)
        traffic = enumerate_traffic_vehicles(self.env)
        ego_dict, vehicles = collect_drift_state(
            ego_vehicle,
            traffic,
            ahead_m=self.mdcfg.NEIGHBOR_AHEAD_M,
            behind_m=self.mdcfg.NEIGHBOR_BEHIND_M,
            lateral_m=self.mdcfg.NEIGHBOR_LATERAL_M,
            radius_m=self.mdcfg.NEIGHBOR_RADIUS_M,
            max_neighbors=self.mdcfg.MAX_NEIGHBORS,
        )

        # Stash ego-body neighbour dicts + ego speed for the social-externality
        # metrics (_social_metrics consumes them after the PDE solve below).
        self._last_neighbors = vehicles
        self._last_ego_long_speed = self._ego_speed_mps() or 0.0

        # Reset PDE state — ego-relative grid is non-inertial, so propagating
        # the field across steps would advect risk in the wrong frame. Instead,
        # we run a brief intra-step warmup to reach quasi-steady response.
        self.drift.reset()
        for _ in range(int(self.mdcfg.PDE_INTRA_STEP_WARMUP_STEPS)):
            self.drift.step(
                vehicles,
                ego_dict,
                dt=float(self.mdcfg.PDE_INTRA_STEP_WARMUP_S)
                   / max(1, int(self.mdcfg.PDE_INTRA_STEP_WARMUP_STEPS)),
                substeps=int(self.mdcfg.PDE_SUBSTEPS),
                source_fn=self._source_fn,
            )

        return self._sample_features()

    def _source_fn(self, vehicles, ego, X, Y):
        q_veh = compute_Q_vehicle(vehicles, ego, X, Y)
        q_occ, occ_mask = compute_Q_occlusion(vehicles, ego, X, Y)
        q_veh = self.mdcfg.Q_SCALE * q_veh
        q_occ = self.mdcfg.Q_SCALE * q_occ
        q_total = np.clip(q_veh + q_occ, 0.0, self.mdcfg.Q_CAP)
        # Diagnostic mode: set MD_WRAPPER_DEBUG=1 to inspect Q magnitudes
        if os.environ.get("MD_WRAPPER_DEBUG"):
            print(
                f"  [drift] n_vehicles={len(vehicles)} "
                f"Q_veh.max={float(np.max(q_veh)):.4f} "
                f"Q_occ.max={float(np.max(q_occ)):.4f} "
                f"Q_total.max={float(np.max(q_total)):.4f}"
            )
        return q_total, q_veh, q_occ, occ_mask

    def _sample_features(self) -> RiskFeatures:
        # All queries are in ego-body frame (ego at origin)
        r_ego = float(self.drift.get_risk_cartesian(0.0, 0.0))
        r_5 = float(self.drift.get_risk_cartesian(5.0, 0.0))
        r_10 = float(self.drift.get_risk_cartesian(10.0, 0.0))
        r_20 = float(self.drift.get_risk_cartesian(20.0, 0.0))
        # Left/right lane corridors: max over a short lookahead at ±lane offset.
        # +y = ego's left (body frame).
        r_left = self._corridor_risk(+float(self.mdcfg.QUERY_LATERAL_LANE_M))
        r_right = self._corridor_risk(-float(self.mdcfg.QUERY_LATERAL_LANE_M))
        grad_x, grad_y = self.drift.get_risk_gradient_cartesian(0.0, 0.0)
        return RiskFeatures(
            r_ego=r_ego,
            r_5m=r_5,
            r_10m=r_10,
            r_20m=r_20,
            r_left=r_left,
            r_right=r_right,
            grad_x=float(grad_x),
            grad_y=float(grad_y),
        )

    def _corridor_risk(self, y_lane: float, length: float = 25.0, n_samples: int = 6) -> float:
        xs = np.linspace(0.0, length, n_samples)
        ys = np.full_like(xs, fill_value=float(y_lane))
        risks = self.drift.get_risk_cartesian(xs, ys)
        if hasattr(risks, "__len__") and len(risks) > 0:
            return float(np.max(np.asarray(risks)))
        return float(risks)

    # -------------------------------------------------------- social metrics

    def _social_metrics(self) -> dict[str, float]:
        """Externality the ego imposes on surrounding vehicles, in ego-body frame.

        Returns raw quantities (not yet weighted):
          - ``decel_follower``    : |min(0, a)| of the rear follower [m/s^2]
          - ``rear_ttc_now``      : follower time-to-collision toward ego [s]
          - ``rear_ttc_loss``     : normalised drop in rear TTC vs previous step
          - ``backward_pressure`` : Σ risk(x,y)·max(0, closing) over behind cars
          - ``follower_present``  : 1 if a rear follower was found
        These feed the courtesy / rear-TTC / backward-flux reward terms and the
        eval social-friendliness metrics. Always computed when DRIFT is active
        (independent of reward weights) so stock/risk-only policies are still
        measured for the social comparison.
        """
        out = {
            "decel_follower": 0.0,
            "rear_ttc_now": float("inf"),
            "rear_ttc_loss": 0.0,
            "backward_pressure": 0.0,
            "follower_present": 0.0,
        }
        if self.drift is None or not self._last_neighbors:
            self._prev_rear_ttc = None
            return out

        ego_v = float(self._last_ego_long_speed or 0.0)
        lane_half = float(self.mdcfg.FOLLOWER_LANE_HALF_W)
        max_dist = float(self.mdcfg.FOLLOWER_MAX_DIST)

        # Nearest follower: behind ego (x<0), within the lane band and range.
        follower = None
        best_x = -float("inf")
        for veh in self._last_neighbors:
            x = float(veh.get("x", 0.0))
            y = float(veh.get("y", 0.0))
            if x < 0.0 and abs(y) <= lane_half and abs(x) <= max_dist:
                if x > best_x:  # closest behind = largest x (least negative)
                    best_x = x
                    follower = veh

        rear_ttc_now = float("inf")
        if follower is not None:
            out["follower_present"] = 1.0
            out["decel_follower"] = max(0.0, -float(follower.get("a", 0.0)))
            gap = max(0.0, abs(float(follower.get("x", 0.0)))
                      - 0.5 * float(follower.get("length", 4.5)) - 2.25)
            closing = float(follower.get("vx", 0.0)) - ego_v  # >0: catching up
            if gap > 0.0 and closing > 1e-3:
                rear_ttc_now = gap / closing

        # Rear-TTC erosion vs previous step (only when the follower is closing).
        if (self._prev_rear_ttc is not None and math.isfinite(self._prev_rear_ttc)
                and math.isfinite(rear_ttc_now) and rear_ttc_now < self._prev_rear_ttc):
            out["rear_ttc_loss"] = min(
                1.0,
                (self._prev_rear_ttc - rear_ttc_now) / max(1e-6, float(self.mdcfg.REAR_TTC_SAFE)),
            )
        self._prev_rear_ttc = rear_ttc_now if math.isfinite(rear_ttc_now) else None
        out["rear_ttc_now"] = rear_ttc_now

        # Backward pressure: risk dumped onto closing followers behind the ego.
        bp = 0.0
        for veh in self._last_neighbors:
            x = float(veh.get("x", 0.0))
            if x >= 0.0:
                continue
            closing = float(veh.get("vx", 0.0)) - ego_v
            if closing <= 0.0:
                continue
            r = float(self.drift.get_risk_cartesian(x, float(veh.get("y", 0.0))))
            bp += r * closing
        out["backward_pressure"] = bp
        return out

    # ---------------------------------------------------------------- helpers

    def _update_overtake_count(self) -> None:
        """Count traffic vehicles that transitioned from ahead-of-ego to behind.

        We use Python `id(vehicle)` as a stable key across steps. A vehicle V
        counts as overtaken once it's:
          (a) been observed in front of ego (x_body > +ego_length/2), AND
          (b) currently behind ego (x_body < -ego_length/2).
        Each id is only counted once per episode.
        """
        ego_vehicle = self._get_ego_vehicle()
        if ego_vehicle is None:
            return
        try:
            from rl.risk.metadrive_scene_adapter import (
                enumerate_traffic_vehicles, world_to_ego_body,
            )
            traffic = enumerate_traffic_vehicles(self.env)
        except Exception:
            return
        ego_pos = np.asarray(ego_vehicle.position, dtype=float)
        ego_heading = float(getattr(ego_vehicle, "heading_theta", 0.0))
        ego_half = float(getattr(ego_vehicle, "LENGTH", 4.5)) * 0.5
        for v in traffic:
            try:
                vp = np.asarray(v.position, dtype=float)
            except Exception:
                continue
            x_body, _y_body = world_to_ego_body(
                vp[0], vp[1], ego_pos[0], ego_pos[1], ego_heading
            )
            vid = id(v)
            if x_body > ego_half:
                self._was_in_front.add(vid)
            elif x_body < -ego_half and vid in self._was_in_front:
                self._overtaken_ids.add(vid)

    def _get_ego_vehicle(self):
        env = self.env.unwrapped
        agent = getattr(env, "agent", None)
        if agent is not None:
            return agent
        # Fallback: multi-agent envs not in v1 scope, but pick first agent
        agents = getattr(env, "agents", None)
        if isinstance(agents, dict) and agents:
            return next(iter(agents.values()))
        return None

    def _snapshot_grid(self) -> Optional[np.ndarray]:
        if self.drift is None:
            return None
        return self.drift.risk_field.astype(np.float32)

    def _format_info(
        self,
        features: RiskFeatures,
        *,
        base_reward: Optional[float],
        base_cost: Optional[float],
        penalties: Optional[dict[str, float]] = None,
        comfort_metrics: Optional[dict[str, float]] = None,
        social_metrics: Optional[dict[str, float]] = None,
        risk_cost: float = 0.0,
    ) -> dict:
        comfort_metrics = dict(comfort_metrics or {})
        penalties = dict(penalties or {})
        social_metrics = dict(social_metrics or {})
        info_block = {
            "use_drift": bool(self.compute_risk_metrics),
            "append_risk_obs": bool(self.append_risk_obs),
            "shape_risk_reward": bool(self.shape_risk_reward),
            "compute_risk_metrics": bool(self.compute_risk_metrics),
            "drift_step": int(self._episode_steps),
            "base_reward": None if base_reward is None else float(base_reward),
            "base_cost": None if base_cost is None else float(base_cost),
            "risk_cost": float(risk_cost),
            # Per-term reward decomposition (all <= 0); keys default to 0.0 so
            # downstream loggers/plotters see a stable schema every step.
            "risk_penalty": float(penalties.get("risk_penalty", 0.0)),
            "comfort_penalty": float(penalties.get("comfort_penalty", 0.0)),
            "steer_penalty": float(penalties.get("steer_penalty", 0.0)),
            "jerk_penalty": float(penalties.get("jerk_penalty", 0.0)),
            "throttle_penalty": float(penalties.get("throttle_penalty", 0.0)),
            "hard_brake_penalty": float(penalties.get("hard_brake_penalty", 0.0)),
            "courtesy_penalty": float(penalties.get("courtesy_penalty", 0.0)),
            "rear_ttc_penalty": float(penalties.get("rear_ttc_penalty", 0.0)),
            "backward_flux_penalty": float(penalties.get("backward_flux_penalty", 0.0)),
            # Raw social-externality quantities (for eval metrics).
            "decel_follower": float(social_metrics.get("decel_follower", 0.0)),
            "rear_ttc_now": float(social_metrics.get("rear_ttc_now", float("inf"))),
            "rear_ttc_loss": float(social_metrics.get("rear_ttc_loss", 0.0)),
            "backward_pressure": float(social_metrics.get("backward_pressure", 0.0)),
            "follower_present": float(social_metrics.get("follower_present", 0.0)),
        }
        info_block.update({k: float(v) for k, v in comfort_metrics.items()})
        info_block.update(features.as_dict())
        return info_block

    # --------------------------------------------------------------- convenience

    @property
    def last_risk_features(self) -> Optional[RiskFeatures]:
        return self._last_features

    def get_risk_grid(self) -> Optional[np.ndarray]:
        return self._snapshot_grid()

    def get_grid_axes(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.compute_risk_metrics:
            return None, None
        return np.array(cfg.X, copy=True), np.array(cfg.Y, copy=True)


# ---------------------------------------------------------------------------
# Smoke test (Verification step 1 from the plan)
# ---------------------------------------------------------------------------

def _smoke_test() -> int:
    """Instantiate the wrapper, run 100 random-action steps, check invariants.

    Returns:
        0 on success, non-zero on failure (so the script is usable as a CI gate).
    """
    print("=" * 60)
    print("MetaDriveDriftWrapper smoke test")
    print("=" * 60)
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
    except ImportError as exc:
        print(f"FAIL: cannot import MetaDriveEnv: {exc}")
        return 1

    base_env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=2,
        start_seed=0,
        # Use a denser traffic setting for the smoke test so we reliably
        # observe non-zero risk features even on the first ~100 steps.
        traffic_density=0.3,
        horizon=400,
    ))
    cfg_obj = MetaDriveRLConfig()
    wrapper = MetaDriveDriftWrapper(
        base_env,
        config=cfg_obj,
        use_risk=True,
        is_safe_env=False,
        record_grid=False,
    )

    try:
        obs, info = wrapper.reset(seed=0)
        stock_dim = int(np.prod(base_env.observation_space.shape))
        expected_dim = stock_dim + RISK_FEATURE_DIM
        assert obs.shape == (expected_dim,), (
            f"obs shape {obs.shape} != expected {(expected_dim,)}"
        )
        print(
            f"reset: obs shape OK ({obs.shape}, stock={stock_dim}, risk={RISK_FEATURE_DIM})"
        )

        # Drive forward at low throttle, zero steering — so the ego stays on
        # road long enough for the DRIFT field to respond to traffic.
        action = np.array([0.0, 0.4], dtype=np.float32)
        n_steps = 100
        finite_obs = True
        finite_reward = True
        risk_seen_nonzero = False
        max_r_ego = 0.0
        for i in range(n_steps):
            obs, reward, terminated, truncated, info = wrapper.step(action)
            if not np.all(np.isfinite(obs)):
                finite_obs = False
                print(f"step {i}: non-finite obs detected!")
                break
            if not np.isfinite(reward):
                finite_reward = False
                print(f"step {i}: non-finite reward!")
                break
            r_ego = float(info.get("r_ego", 0.0))
            max_r_ego = max(max_r_ego, r_ego)
            if r_ego > 1e-3:
                risk_seen_nonzero = True
            if terminated or truncated:
                obs, info = wrapper.reset()
        print(
            f"after {n_steps} steps: obs_finite={finite_obs} "
            f"reward_finite={finite_reward} risk_nonzero_seen={risk_seen_nonzero} "
            f"max_r_ego={max_r_ego:.3f}"
        )
        last = wrapper.last_risk_features
        if last is not None:
            print(
                f"last features: r_ego={last.r_ego:.3f} r_5m={last.r_5m:.3f} "
                f"r_10m={last.r_10m:.3f} r_20m={last.r_20m:.3f} "
                f"r_left={last.r_left:.3f} r_right={last.r_right:.3f}"
            )
        ok = finite_obs and finite_reward and risk_seen_nonzero
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 2
    finally:
        wrapper.close()


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
