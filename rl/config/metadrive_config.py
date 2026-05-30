"""
MetaDrive RL Configuration
==========================
Central hyperparameter store for the MetaDrive social-risk RL integration
(plan Set 2). Mirrors `rl_config.RLConfig` but only carries the constants
that the MetaDrive wrapper, trainer, and eval actually consume.

All values are documented in [the plan](../../../Users/ymshu/.claude/plans/this-is-the-experiment-flickering-sutton.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass(frozen=True)
class MetaDriveProtocol:
    """Executable MetaDrive experiment protocol.

    The protocol owns simulator settings that must match between training and
    evaluation. Risk flags describe what the policy was trained with; evaluators
    may still enable metric-only risk computation without changing observations.
    """

    name: str
    description: str
    env_name: str = "metadrive"
    map: Optional[str] = "C"
    traffic_mode: str = "trigger"
    discrete_action: bool = True
    discrete_steering_dim: int = 3
    discrete_throttle_dim: int = 3
    use_multi_discrete: bool = False
    horizon: int = 500
    random_spawn_lane_index: bool = False
    random_traffic: bool = False
    random_lane_width: bool = False
    random_lane_num: bool = False
    random_agent_model: bool = False
    train_num_scenarios: int = 200
    train_start_seed: int = 0
    eval_num_scenarios: int = 100
    eval_start_seed: int = 10_000
    traffic_density: float = 0.3
    accident_prob: float = 0.0
    append_risk_obs: bool = False
    shape_risk_reward: bool = False
    compute_risk_metrics: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetaDriveRLConfig:
    # ---- Ego-relative DRIFT grid -------------------------------------------
    # Built in ego-body frame. x_min < 0 covers vehicles behind ego.
    GRID_X_MIN: float = -30.0
    GRID_X_MAX: float = 90.0
    GRID_Y_MIN: float = -15.0
    GRID_Y_MAX: float = 15.0
    GRID_DX: float = 1.5
    GRID_DY: float = 1.5

    # ---- PDE integration ---------------------------------------------------
    PDE_DT: float = 0.1                # one wrapper step advances PDE by this in ego-relative grid
    PDE_SUBSTEPS: int = 4              # CFL-safe with D_occ=6, dx=dy=1.5 → sub_dt < ~0.09 s
    PDE_INTRA_STEP_WARMUP_S: float = 0.4  # quasi-steady warmup each step in non-inertial frame
    PDE_INTRA_STEP_WARMUP_STEPS: int = 4  # 4*4=16 PDE iters/step (was 6*6=36, ~55% faster)

    # ---- Source calibration (MetaDrive vehicles travel 15-30 m/s like HighwayEnv) ----
    Q_SCALE: float = 0.22              # same as highwayenv_drift_wrapper
    Q_CAP: float = 0.45

    # ---- Neighbor selection (ego-body frame, before re-centering) ----------
    NEIGHBOR_AHEAD_M: float = 90.0
    NEIGHBOR_BEHIND_M: float = 30.0
    NEIGHBOR_LATERAL_M: float = 14.0
    NEIGHBOR_RADIUS_M: float = 95.0
    MAX_NEIGHBORS: int = 20

    # ---- Risk feature query offsets in ego-body frame ----------------------
    QUERY_LOOKAHEAD_M: List[float] = field(default_factory=lambda: [0.0, 5.0, 10.0, 20.0])
    QUERY_LATERAL_LANE_M: float = 3.5  # left/right lane center offset (one lane)

    # ---- Observation normalisation -----------------------------------------
    RISK_CLIP: float = 5.0             # max raw risk before clipping into features
    NORM_RISK: float = 5.0             # divisor for risk features
    NORM_GRAD: float = 2.0             # divisor for risk gradient features

    # ---- Reward shaping ----------------------------------------------------
    LAMBDA_RISK: float = 0.10          # weight of -r_ego subtracted from stock reward
    LAMBDA_ACTION_DELTA: float = 0.02  # weight for total low-level command changes
    LAMBDA_JERK: float = 0.01          # weight for absolute jerk [m/s^3]
    LAMBDA_STEER_ABS: float = 0.0      # optional weight for absolute steering command magnitude
    LAMBDA_STEER_DELTA: float = 0.02   # extra weight for steering command changes
    LAMBDA_THROTTLE_DELTA: float = 0.02  # extra weight for throttle/brake command changes
    # SafeMetaDrive additive cost
    W_RISK_COST: float = 1.0           # extra cost weight when r_ego > τ
    TAU_RISK: float = 1.5              # threshold for risk-cost indicator

    # ---- social_full reward terms (default 0 → only active in the social_full
    #      profile; constants calibrated to match rl/reward/social_reward.py so
    #      offline analysis and online shaping stay consistent) ----------------
    W_HARD_BRAKE: float = 0.0          # ego hard-brake penalty weight
    EGO_DECEL_COMFORT: float = 2.0     # m/s²: |decel| below this is comfortable (no penalty)
    EGO_DECEL_HARD: float = 4.0        # m/s²: |decel| at/above this saturates the penalty
    W_COURTESY: float = 0.0            # imposed follower-braking penalty weight
    COURTESY_DECEL: float = 2.0        # m/s²: follower decel below this is courteous
    FOLLOWER_HARD_BRAKE: float = 3.0   # m/s²: follower decel saturating the courtesy penalty
    W_REAR_TTC: float = 0.0            # follower TTC-erosion penalty weight
    REAR_TTC_SAFE: float = 3.0         # s: normaliser for rear-TTC loss
    W_BACK_FLUX: float = 0.0           # backward risk-flux penalty weight
    FLUX_B0: float = 1.0               # saturating constant for backward_flux_cost
    FOLLOWER_LANE_HALF_W: float = 2.0  # m: |y_body| band to count a vehicle as ego's follower
    FOLLOWER_MAX_DIST: float = 40.0    # m: max |x_body| behind ego to consider a follower

    # ---- Scenario / seeding ------------------------------------------------
    TRAIN_NUM_SCENARIOS: int = 200
    TRAIN_START_SEED: int = 0
    EVAL_NUM_SCENARIOS: int = 100
    EVAL_START_SEED: int = 10_000      # paired seeds, disjoint from train
    TRAFFIC_DENSITY: float = 0.1       # MetaDrive default; sweep {0.1, 0.3}
    HORIZON: int = 1000                # max env steps per episode

    # ---- Metrics thresholds (matches RLConfig where applicable) ------------
    NEAR_MISS_DIST: float = 6.0        # gap below which a step counts as near-miss
    TTC_THRESHOLD: float = 1.5         # TTC below which a step counts as TTC violation
    HIGH_RISK_TAU: float = 1.5         # risk threshold for high_risk_duration metric

    # ---- Logging -----------------------------------------------------------
    RECORD_GRID_EVERY: int = 5         # log a risk-grid snapshot every N steps (eval only)

    # ------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetaDriveRLConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


DEFAULT_METADRIVE_CONFIG = MetaDriveRLConfig()


def _matched_protocol(
    *,
    name: str,
    map_id: str,
    scenario_label: str,
    risk: bool,
    discrete_action: bool = True,
    traffic_mode: str = "trigger",
) -> MetaDriveProtocol:
    """Create matched stock/risk protocol variants for paper scenario tables."""
    action_label = "discrete 3x3" if discrete_action else "continuous"
    traffic_label = str(traffic_mode).lower()
    return MetaDriveProtocol(
        name=name,
        description=(
            f"Matched MetaDrive {scenario_label} benchmark with {action_label} actions "
            f"and {traffic_label} traffic. "
            + (
                "Risk-aware PPO protocol with DRIFT observation, reward shaping, "
                "comfort regularization, and risk metrics."
                if risk else
                "Stock PPO baseline with the same simulator, map, traffic, "
                "horizon, seeds, and action protocol as the risk-aware agent."
            )
        ),
        map=map_id,
        traffic_mode=traffic_label,
        discrete_action=bool(discrete_action),
        discrete_steering_dim=3,
        discrete_throttle_dim=3,
        horizon=500,
        random_spawn_lane_index=False,
        random_traffic=False,
        random_lane_width=False,
        random_lane_num=False,
        random_agent_model=False,
        train_num_scenarios=200,
        train_start_seed=0,
        eval_num_scenarios=100,
        eval_start_seed=10_000,
        traffic_density=0.3,
        accident_prob=0.0,
        append_risk_obs=bool(risk),
        shape_risk_reward=bool(risk),
        compute_risk_metrics=bool(risk),
    )


METADRIVE_PROTOCOLS: Dict[str, MetaDriveProtocol] = {
    "official_notebook_reference": MetaDriveProtocol(
        name="official_notebook_reference",
        description=(
            "Exact Stable-Baselines3 PPO setup from MetaDrive training.ipynb: "
            "fixed C map, discrete 3x3 action, horizon 500, no traffic."
        ),
        map="C",
        discrete_action=True,
        discrete_steering_dim=3,
        discrete_throttle_dim=3,
        horizon=500,
        random_spawn_lane_index=False,
        train_num_scenarios=1,
        train_start_seed=5,
        eval_num_scenarios=1,
        eval_start_seed=5,
        traffic_density=0.0,
        accident_prob=0.0,
        append_risk_obs=False,
        shape_risk_reward=False,
        compute_risk_metrics=False,
    ),
    "matched_stock": MetaDriveProtocol(
        name="matched_stock",
        description=(
            "Fair stock baseline for the social-risk benchmark. It keeps the "
            "notebook PPO action protocol but uses the same traffic and "
            "scenario split as the risk-aware agent."
        ),
        map="C",
        discrete_action=True,
        discrete_steering_dim=3,
        discrete_throttle_dim=3,
        horizon=500,
        random_spawn_lane_index=False,
        train_num_scenarios=200,
        train_start_seed=0,
        eval_num_scenarios=100,
        eval_start_seed=10_000,
        traffic_density=0.3,
        accident_prob=0.0,
        append_risk_obs=False,
        shape_risk_reward=False,
        compute_risk_metrics=False,
    ),
    "matched_social_risk": MetaDriveProtocol(
        name="matched_social_risk",
        description=(
            "Risk-aware PPO protocol matched to the stock baseline except for "
            "DRIFT risk observation, reward shaping, and risk metrics."
        ),
        map="C",
        discrete_action=True,
        discrete_steering_dim=3,
        discrete_throttle_dim=3,
        horizon=500,
        random_spawn_lane_index=False,
        train_num_scenarios=200,
        train_start_seed=0,
        eval_num_scenarios=100,
        eval_start_seed=10_000,
        traffic_density=0.3,
        accident_prob=0.0,
        append_risk_obs=True,
        shape_risk_reward=True,
        compute_risk_metrics=True,
    ),
    "safe_metadrive_risk": MetaDriveProtocol(
        name="safe_metadrive_risk",
        description=(
            "SafeMetaDrive extension for later constrained/safety-cost runs."
        ),
        env_name="safe-metadrive",
        map="C",
        discrete_action=True,
        discrete_steering_dim=3,
        discrete_throttle_dim=3,
        horizon=500,
        random_spawn_lane_index=False,
        train_num_scenarios=200,
        train_start_seed=0,
        eval_num_scenarios=100,
        eval_start_seed=10_000,
        traffic_density=0.3,
        accident_prob=0.8,
        append_risk_obs=True,
        shape_risk_reward=True,
        compute_risk_metrics=True,
    ),
}


METADRIVE_PROTOCOLS.update({
    # Specialist policies for main quantitative tables.
    "matched_stock_straight": _matched_protocol(
        name="matched_stock_straight",
        map_id="S",
        scenario_label="straight-road",
        risk=False,
    ),
    "matched_social_risk_straight": _matched_protocol(
        name="matched_social_risk_straight",
        map_id="S",
        scenario_label="straight-road",
        risk=True,
    ),
    "matched_stock_curve": _matched_protocol(
        name="matched_stock_curve",
        map_id="C",
        scenario_label="curved-road",
        risk=False,
    ),
    "matched_social_risk_curve": _matched_protocol(
        name="matched_social_risk_curve",
        map_id="C",
        scenario_label="curved-road",
        risk=True,
    ),
    "matched_stock_merge": _matched_protocol(
        name="matched_stock_merge",
        map_id="r",
        scenario_label="merge/ramp",
        risk=False,
    ),
    "matched_social_risk_merge": _matched_protocol(
        name="matched_social_risk_merge",
        map_id="r",
        scenario_label="merge/ramp",
        risk=True,
    ),
    "matched_stock_intersection": _matched_protocol(
        name="matched_stock_intersection",
        map_id="X",
        scenario_label="intersection",
        risk=False,
    ),
    "matched_social_risk_intersection": _matched_protocol(
        name="matched_social_risk_intersection",
        map_id="X",
        scenario_label="intersection",
        risk=True,
    ),
    "matched_stock_roundabout": _matched_protocol(
        name="matched_stock_roundabout",
        map_id="O",
        scenario_label="roundabout",
        risk=False,
    ),
    "matched_social_risk_roundabout": _matched_protocol(
        name="matched_social_risk_roundabout",
        map_id="O",
        scenario_label="roundabout",
        risk=True,
    ),
    # Mixed-map policies for held-out/generalization analysis.
    "matched_stock_mixed": _matched_protocol(
        name="matched_stock_mixed",
        map_id="SCrXO",
        scenario_label="mixed straight-curve-merge-intersection-roundabout",
        risk=False,
    ),
    "matched_social_risk_mixed": _matched_protocol(
        name="matched_social_risk_mixed",
        map_id="SCrXO",
        scenario_label="mixed straight-curve-merge-intersection-roundabout",
        risk=True,
    ),
})


_CONTINUOUS_PROTOCOL_SPECS = (
    ("straight", "S", "straight-road"),
    ("curve", "C", "curved-road"),
    ("merge", "r", "merge/ramp"),
    ("intersection", "X", "intersection"),
    ("roundabout", "O", "roundabout"),
    ("mixed", "SCrXO", "mixed straight-curve-merge-intersection-roundabout"),
)
for _suffix, _map_id, _label in _CONTINUOUS_PROTOCOL_SPECS:
    METADRIVE_PROTOCOLS[f"matched_stock_{_suffix}_continuous"] = _matched_protocol(
        name=f"matched_stock_{_suffix}_continuous",
        map_id=_map_id,
        scenario_label=_label,
        risk=False,
        discrete_action=False,
    )
    METADRIVE_PROTOCOLS[f"matched_social_risk_{_suffix}_continuous"] = _matched_protocol(
        name=f"matched_social_risk_{_suffix}_continuous",
        map_id=_map_id,
        scenario_label=_label,
        risk=True,
        discrete_action=False,
    )


for _suffix, _map_id, _label in _CONTINUOUS_PROTOCOL_SPECS:
    METADRIVE_PROTOCOLS[f"matched_stock_{_suffix}_respawn"] = _matched_protocol(
        name=f"matched_stock_{_suffix}_respawn",
        map_id=_map_id,
        scenario_label=_label,
        risk=False,
        discrete_action=True,
        traffic_mode="respawn",
    )
    METADRIVE_PROTOCOLS[f"matched_social_risk_{_suffix}_respawn"] = _matched_protocol(
        name=f"matched_social_risk_{_suffix}_respawn",
        map_id=_map_id,
        scenario_label=_label,
        risk=True,
        discrete_action=True,
        traffic_mode="respawn",
    )
    METADRIVE_PROTOCOLS[f"matched_stock_{_suffix}_respawn_continuous"] = _matched_protocol(
        name=f"matched_stock_{_suffix}_respawn_continuous",
        map_id=_map_id,
        scenario_label=_label,
        risk=False,
        discrete_action=False,
        traffic_mode="respawn",
    )
    METADRIVE_PROTOCOLS[f"matched_social_risk_{_suffix}_respawn_continuous"] = _matched_protocol(
        name=f"matched_social_risk_{_suffix}_respawn_continuous",
        map_id=_map_id,
        scenario_label=_label,
        risk=True,
        discrete_action=False,
        traffic_mode="respawn",
    )


METADRIVE_PROTOCOLS.update({
    "generalization_stock_continuous": MetaDriveProtocol(
        name="generalization_stock_continuous",
        description=(
            "Notebook-derived MetaDrive generalization environment: 1000 randomized "
            "training scenarios, 200 held-out scenarios, randomized lane width, lane "
            "count, and ego/traffic vehicle models."
        ),
        map=None,  # keep MetaDrive default random block-count map generation
        traffic_mode="trigger",
        discrete_action=False,
        horizon=1000,
        random_spawn_lane_index=True,
        random_traffic=False,
        random_lane_width=True,
        random_lane_num=True,
        random_agent_model=True,
        train_num_scenarios=1000,
        train_start_seed=1000,
        eval_num_scenarios=200,
        eval_start_seed=0,
        traffic_density=0.1,
        accident_prob=0.0,
        append_risk_obs=False,
        shape_risk_reward=False,
        compute_risk_metrics=False,
    ),
    "generalization_social_risk_continuous": MetaDriveProtocol(
        name="generalization_social_risk_continuous",
        description=(
            "Risk-aware counterpart to the notebook-derived MetaDrive generalization "
            "environment, with DRIFT observation/reward/metrics enabled."
        ),
        map=None,
        traffic_mode="trigger",
        discrete_action=False,
        horizon=1000,
        random_spawn_lane_index=True,
        random_traffic=False,
        random_lane_width=True,
        random_lane_num=True,
        random_agent_model=True,
        train_num_scenarios=1000,
        train_start_seed=1000,
        eval_num_scenarios=200,
        eval_start_seed=0,
        traffic_density=0.1,
        accident_prob=0.0,
        append_risk_obs=True,
        shape_risk_reward=True,
        compute_risk_metrics=True,
    ),
    "safe_metadrive_stock": MetaDriveProtocol(
        name="safe_metadrive_stock",
        description=(
            "Matched SafeMetaDrive stock baseline using the notebook-style static "
            "accident-object stress setting."
        ),
        env_name="safe-metadrive",
        map="CCCCC",
        traffic_mode="trigger",
        discrete_action=True,
        discrete_steering_dim=3,
        discrete_throttle_dim=3,
        horizon=500,
        random_spawn_lane_index=False,
        train_num_scenarios=200,
        train_start_seed=0,
        eval_num_scenarios=100,
        eval_start_seed=10_000,
        traffic_density=0.3,
        accident_prob=0.8,
        append_risk_obs=False,
        shape_risk_reward=False,
        compute_risk_metrics=False,
    ),
    "safe_metadrive_social_risk": MetaDriveProtocol(
        name="safe_metadrive_social_risk",
        description=(
            "Matched SafeMetaDrive social-risk protocol with DRIFT observation, "
            "reward shaping, risk cost, and comfort regularization."
        ),
        env_name="safe-metadrive",
        map="CCCCC",
        traffic_mode="trigger",
        discrete_action=True,
        discrete_steering_dim=3,
        discrete_throttle_dim=3,
        horizon=500,
        random_spawn_lane_index=False,
        train_num_scenarios=200,
        train_start_seed=0,
        eval_num_scenarios=100,
        eval_start_seed=10_000,
        traffic_density=0.3,
        accident_prob=0.8,
        append_risk_obs=True,
        shape_risk_reward=True,
        compute_risk_metrics=True,
    ),
    "safe_metadrive_stock_continuous": MetaDriveProtocol(
        name="safe_metadrive_stock_continuous",
        description="Continuous-action counterpart to safe_metadrive_stock.",
        env_name="safe-metadrive",
        map="CCCCC",
        traffic_mode="trigger",
        discrete_action=False,
        horizon=500,
        random_spawn_lane_index=False,
        train_num_scenarios=200,
        train_start_seed=0,
        eval_num_scenarios=100,
        eval_start_seed=10_000,
        traffic_density=0.3,
        accident_prob=0.8,
        append_risk_obs=False,
        shape_risk_reward=False,
        compute_risk_metrics=False,
    ),
    "safe_metadrive_social_risk_continuous": MetaDriveProtocol(
        name="safe_metadrive_social_risk_continuous",
        description="Continuous-action counterpart to safe_metadrive_social_risk.",
        env_name="safe-metadrive",
        map="CCCCC",
        traffic_mode="trigger",
        discrete_action=False,
        horizon=500,
        random_spawn_lane_index=False,
        train_num_scenarios=200,
        train_start_seed=0,
        eval_num_scenarios=100,
        eval_start_seed=10_000,
        traffic_density=0.3,
        accident_prob=0.8,
        append_risk_obs=True,
        shape_risk_reward=True,
        compute_risk_metrics=True,
    ),
})


def get_metadrive_protocol(name: str | MetaDriveProtocol) -> MetaDriveProtocol:
    if isinstance(name, MetaDriveProtocol):
        return name
    key = str(name)
    try:
        return METADRIVE_PROTOCOLS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(METADRIVE_PROTOCOLS))
        raise KeyError(f"Unknown MetaDrive protocol '{key}'. Valid protocols: {valid}") from exc
