"""
MetaDrive environment factories.

The factories are intentionally protocol-driven. A protocol owns the simulator
contract that must match between training, evaluation, and visualization:
action space, map, horizon, scenario split, traffic density, and risk flags.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import gymnasium as gym

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rl.config.metadrive_config import (
    DEFAULT_METADRIVE_CONFIG,
    MetaDriveProtocol,
    MetaDriveRLConfig,
    get_metadrive_protocol,
)
from rl.env.metadrive_drift_wrapper import MetaDriveDriftWrapper


def _base_config(
    protocol: MetaDriveProtocol,
    *,
    start_seed: int,
    num_scenarios: int,
    traffic_density: float,
    traffic_mode: Optional[str] = None,
    accident_prob: Optional[float] = None,
    agent_policy=None,
    discrete_action: Optional[bool] = None,
    use_render: bool = False,
) -> dict:
    """Construct the MetaDrive config shared by train/eval factories."""
    use_discrete_action = bool(protocol.discrete_action if discrete_action is None else discrete_action)
    resolved_traffic_mode = protocol.traffic_mode if traffic_mode is None else str(traffic_mode)
    cfg = dict(
        use_render=bool(use_render),
        num_scenarios=int(num_scenarios),
        start_seed=int(start_seed),
        traffic_density=float(traffic_density),
        traffic_mode=str(resolved_traffic_mode),
        random_traffic=bool(protocol.random_traffic),
        random_lane_width=bool(protocol.random_lane_width),
        random_lane_num=bool(protocol.random_lane_num),
        random_agent_model=bool(protocol.random_agent_model),
        accident_prob=float(protocol.accident_prob if accident_prob is None else accident_prob),
        horizon=int(protocol.horizon),
        discrete_action=use_discrete_action,
        use_multi_discrete=bool(protocol.use_multi_discrete),
        random_spawn_lane_index=bool(protocol.random_spawn_lane_index),
        manual_control=False,
        log_level=50,
        # Do not set vehicle_config={"overtake_stat": True}: MetaDrive 0.4.3
        # calls a lidar method with a missing argument. The wrapper counts
        # overtakes from vehicle poses instead.
    )
    if agent_policy is not None:
        cfg["agent_policy"] = agent_policy
    if use_discrete_action:
        cfg["discrete_throttle_dim"] = int(protocol.discrete_throttle_dim)
        cfg["discrete_steering_dim"] = int(protocol.discrete_steering_dim)
    if protocol.map is not None:
        cfg["map"] = str(protocol.map)
    return cfg


def _risk_flags(
    protocol: MetaDriveProtocol,
    *,
    append_risk_obs: Optional[bool] = None,
    shape_risk_reward: Optional[bool] = None,
    compute_risk_metrics: Optional[bool] = None,
    use_risk: Optional[bool] = None,
) -> dict:
    """Resolve wrapper risk flags.

    `use_risk` is kept only for backward compatibility with older scripts. New
    code should pass the three explicit flags.
    """
    if use_risk is not None:
        if append_risk_obs is None:
            append_risk_obs = bool(use_risk)
        if shape_risk_reward is None:
            shape_risk_reward = bool(use_risk)
        if compute_risk_metrics is None:
            compute_risk_metrics = bool(use_risk)
    return dict(
        append_risk_obs=protocol.append_risk_obs if append_risk_obs is None else bool(append_risk_obs),
        shape_risk_reward=protocol.shape_risk_reward if shape_risk_reward is None else bool(shape_risk_reward),
        compute_risk_metrics=(
            protocol.compute_risk_metrics if compute_risk_metrics is None else bool(compute_risk_metrics)
        ),
    )


def make_metadrive_train_env(
    *,
    config: Optional[MetaDriveRLConfig] = None,
    protocol: str | MetaDriveProtocol = "matched_social_risk",
    seed_offset: int = 0,
    append_risk_obs: Optional[bool] = None,
    shape_risk_reward: Optional[bool] = None,
    compute_risk_metrics: Optional[bool] = None,
    use_risk: Optional[bool] = None,
    traffic_density: Optional[float] = None,
    traffic_mode: Optional[str] = None,
    record_grid: bool = False,
    use_render: bool = False,
) -> gym.Env:
    """Build a single-agent MetaDrive training environment."""
    from metadrive.envs.metadrive_env import MetaDriveEnv

    mdcfg = config or DEFAULT_METADRIVE_CONFIG
    proto = get_metadrive_protocol(protocol)
    base_env = MetaDriveEnv(
        _base_config(
            proto,
            start_seed=int(proto.train_start_seed) + int(seed_offset),
            num_scenarios=int(proto.train_num_scenarios),
            traffic_density=float(proto.traffic_density if traffic_density is None else traffic_density),
            traffic_mode=traffic_mode,
            use_render=use_render,
        )
    )
    return MetaDriveDriftWrapper(
        base_env,
        config=mdcfg,
        is_safe_env=False,
        record_grid=record_grid,
        **_risk_flags(
            proto,
            append_risk_obs=append_risk_obs,
            shape_risk_reward=shape_risk_reward,
            compute_risk_metrics=compute_risk_metrics,
            use_risk=use_risk,
        ),
    )


def make_metadrive_eval_env(
    *,
    config: Optional[MetaDriveRLConfig] = None,
    protocol: str | MetaDriveProtocol = "matched_social_risk",
    append_risk_obs: Optional[bool] = None,
    shape_risk_reward: Optional[bool] = None,
    compute_risk_metrics: Optional[bool] = None,
    use_risk: Optional[bool] = None,
    traffic_density: Optional[float] = None,
    traffic_mode: Optional[str] = None,
    record_grid: bool = False,
    use_render: bool = False,
    num_scenarios: Optional[int] = None,
    start_seed: Optional[int] = None,
    agent_policy=None,
    discrete_action: Optional[bool] = None,
) -> gym.Env:
    """Build a paired-seed MetaDrive evaluation environment."""
    mdcfg = config or DEFAULT_METADRIVE_CONFIG
    proto = get_metadrive_protocol(protocol)
    if proto.env_name == "safe-metadrive":
        from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
        env_cls = SafeMetaDriveEnv
        is_safe_env = True
    else:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        env_cls = MetaDriveEnv
        is_safe_env = False
    base_env = env_cls(
        _base_config(
            proto,
            start_seed=int(proto.eval_start_seed if start_seed is None else start_seed),
            num_scenarios=int(proto.eval_num_scenarios if num_scenarios is None else num_scenarios),
            traffic_density=float(proto.traffic_density if traffic_density is None else traffic_density),
            traffic_mode=traffic_mode,
            agent_policy=agent_policy,
            discrete_action=discrete_action,
            use_render=use_render,
        )
    )
    return MetaDriveDriftWrapper(
        base_env,
        config=mdcfg,
        is_safe_env=is_safe_env,
        record_grid=record_grid,
        **_risk_flags(
            proto,
            append_risk_obs=append_risk_obs,
            shape_risk_reward=shape_risk_reward,
            compute_risk_metrics=compute_risk_metrics,
            use_risk=use_risk,
        ),
    )


def make_safe_metadrive_env(
    *,
    config: Optional[MetaDriveRLConfig] = None,
    protocol: str | MetaDriveProtocol = "safe_metadrive_risk",
    seed_offset: int = 0,
    append_risk_obs: Optional[bool] = None,
    shape_risk_reward: Optional[bool] = None,
    compute_risk_metrics: Optional[bool] = None,
    use_risk: Optional[bool] = None,
    accident_prob: Optional[float] = None,
    traffic_density: Optional[float] = None,
    traffic_mode: Optional[str] = None,
    record_grid: bool = False,
    use_render: bool = False,
) -> gym.Env:
    """Build a SafeMetaDrive environment for later safety-cost runs."""
    from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

    mdcfg = config or DEFAULT_METADRIVE_CONFIG
    proto = get_metadrive_protocol(protocol)
    base_env = SafeMetaDriveEnv(
        _base_config(
            proto,
            start_seed=int(proto.train_start_seed) + int(seed_offset),
            num_scenarios=int(proto.train_num_scenarios),
            traffic_density=float(proto.traffic_density if traffic_density is None else traffic_density),
            traffic_mode=traffic_mode,
            accident_prob=proto.accident_prob if accident_prob is None else float(accident_prob),
            use_render=use_render,
        )
    )
    return MetaDriveDriftWrapper(
        base_env,
        config=mdcfg,
        is_safe_env=True,
        record_grid=record_grid,
        **_risk_flags(
            proto,
            append_risk_obs=append_risk_obs,
            shape_risk_reward=shape_risk_reward,
            compute_risk_metrics=compute_risk_metrics,
            use_risk=use_risk,
        ),
    )
