from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from rl.data.historical_extractor import (
    BLOCKED_GAP_THR,
    BLOCKED_SPEED_FRAC,
    LANE_ADV_D0,
    LANE_ADV_R0,
    LANE_ADV_V0,
)
from rl.data.social_features import (
    BAD_CUT_TTC_ABS,
    BAD_CUT_TTC_DROP_FRAC,
    HARD_BRAKE_DECEL_MPS2,
    MISSED_OPP_BEST_ADV,
    field_metrics,
)


@dataclass
class SocialRewardConfig:
    schema_version: str = "social_reward_v1"
    description: str = (
        "Stock-HighwayEnv SB3 reward calibrated from dataset-derived social features."
    )
    ttc_safe: float = 3.0
    thw_des: float = 1.8
    drac_comfort: float = 2.0
    drac_max: float = 6.0
    courtesy_decel: float = 2.0
    hard_brake: float = 3.0
    missed_opportunity_adv: float = MISSED_OPP_BEST_ADV
    bad_lane_change_adv: float = -0.3
    risk_gate_r0: float = 1.5
    risk_clip: float = 5.0
    lane_adv_gap_d0: float = LANE_ADV_D0
    lane_adv_dv_v0: float = LANE_ADV_V0
    lane_adv_risk_r0: float = LANE_ADV_R0
    blocked_gap_thr: float = BLOCKED_GAP_THR
    blocked_speed_frac: float = BLOCKED_SPEED_FRAC
    progress_ref_speed: float = 25.0
    comfort_accel_norm: float = 4.0
    comfort_steer_norm: float = 0.35
    comfort_jerk_norm: float = 5.0
    flux_b0: float = 1.0
    w_p: float = 0.3
    w_v: float = 0.2
    w_adv: float = 1.0
    w_bad: float = 1.5
    w_miss: float = 0.8
    w_ttc: float = 2.0
    w_thw: float = 0.5
    w_drac: float = 1.0
    w_soc: float = 1.0
    w_c: float = 0.2
    w_R: float = 0.5
    w_B: float = 0.05
    calibration: dict[str, Any] = field(default_factory=dict)
    term_descriptions: dict[str, str] = field(
        default_factory=lambda: {
            "r_prog_safe": "Forward progress gated by corridor risk.",
            "r_speed": "Cruise-speed tracking reward.",
            "r_lane_adv": "Reward for moving to a tactically better lane.",
            "c_bad_lane": "Penalty for changing into a worse lane.",
            "c_missed": "Penalty for staying blocked when a better lane exists.",
            "c_ttc": "Dense safety penalty from minimum TTC.",
            "c_thw": "Dense car-following headway penalty.",
            "c_drac": "Required-braking penalty.",
            "c_social": "Courtesy penalty from imposed follower braking / bad cut-ins.",
            "c_comfort": "Comfort penalty from acceleration, steering, and jerk.",
            "c_field": "Bounded corridor-risk penalty from DRIFT.",
            "c_flux_weak": "Weak backward risk-flux regularizer.",
            "stock_reward": "Original HighwayEnv reward before social shaping.",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SocialRewardConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


DEFAULT_SOCIAL_REWARD_CONFIG = SocialRewardConfig()


ABLATION_PROFILES = {
    "A0": {
        "use_stock_reward": True,
        "use_progress": False,
        "use_speed": False,
        "use_lane": False,
        "use_missed": False,
        "use_safety": False,
        "use_social": False,
        "use_comfort": False,
        "use_field": False,
        "use_flux": False,
    },
    "A1": {
        "use_stock_reward": False,
        "use_progress": True,
        "use_speed": True,
        "use_lane": False,
        "use_missed": False,
        "use_safety": False,
        "use_social": False,
        "use_comfort": True,
        "use_field": True,
        "use_flux": False,
    },
    "A2": {
        "use_stock_reward": False,
        "use_progress": True,
        "use_speed": True,
        "use_lane": False,
        "use_missed": False,
        "use_safety": True,
        "use_social": False,
        "use_comfort": True,
        "use_field": True,
        "use_flux": False,
    },
    "A3": {
        "use_stock_reward": False,
        "use_progress": True,
        "use_speed": True,
        "use_lane": True,
        "use_missed": True,
        "use_safety": True,
        "use_social": False,
        "use_comfort": True,
        "use_field": True,
        "use_flux": False,
    },
    "A4": {
        "use_stock_reward": False,
        "use_progress": True,
        "use_speed": True,
        "use_lane": True,
        "use_missed": True,
        "use_safety": True,
        "use_social": True,
        "use_comfort": True,
        "use_field": True,
        "use_flux": False,
    },
    "A5": {
        "use_stock_reward": False,
        "use_progress": True,
        "use_speed": True,
        "use_lane": True,
        "use_missed": True,
        "use_safety": True,
        "use_social": True,
        "use_comfort": True,
        "use_field": True,
        "use_flux": True,
    },
}
ABLATION_PROFILES["full"] = dict(ABLATION_PROFILES["A5"])


def ablation_profile(name: str) -> dict[str, bool]:
    key = str(name).strip()
    if key not in ABLATION_PROFILES:
        raise ValueError(f"Unsupported ablation profile '{name}'")
    return dict(ABLATION_PROFILES[key])


def lane_utility(gap: float, dv: float, lane_risk: float, cfg: SocialRewardConfig) -> float:
    gap_score = min(float(gap), 80.0) / max(1e-6, float(cfg.lane_adv_gap_d0))
    dv_score = -float(dv) / max(1e-6, float(cfg.lane_adv_dv_v0))
    risk_score = -float(lane_risk) / max(1e-6, float(cfg.lane_adv_risk_r0))
    return float(gap_score + dv_score + risk_score)


def safe_ttc(gap_m: float, closing_mps: float, cap: float = 60.0) -> float:
    if gap_m <= 0.0:
        return 0.0
    if closing_mps <= 1e-3:
        return float(cap)
    return float(min(float(gap_m) / float(closing_mps), float(cap)))


def safe_thw(gap_m: float, speed_mps: float, cap: float = 60.0) -> float:
    if gap_m <= 0.0:
        return 0.0
    if speed_mps <= 1e-3:
        return float(cap)
    return float(min(float(gap_m) / float(speed_mps), float(cap)))


def drac(gap_m: float, closing_mps: float) -> float:
    if gap_m <= 1e-3 or closing_mps <= 0.0:
        return 0.0
    return float((closing_mps ** 2) / max(1e-3, 2.0 * float(gap_m)))


def quadratic_margin_cost(value: float, safe_value: float) -> float:
    margin = max(0.0, (float(safe_value) - float(value)) / max(1e-6, float(safe_value)))
    return float(margin * margin)


def quadratic_upper_cost(value: float, start: float, max_value: float) -> float:
    if float(value) <= float(start):
        return 0.0
    denom = max(1e-6, float(max_value) - float(start))
    scaled = np.clip((float(value) - float(start)) / denom, 0.0, 1.0)
    return float(scaled * scaled)


def courtesy_brake_cost(imposed_decel: float, cfg: SocialRewardConfig) -> float:
    if float(imposed_decel) <= float(cfg.courtesy_decel):
        return 0.0
    denom = max(1e-6, float(cfg.hard_brake) - float(cfg.courtesy_decel))
    scaled = np.clip((float(imposed_decel) - float(cfg.courtesy_decel)) / denom, 0.0, 1.0)
    return float(scaled * scaled)


def bounded_field_cost(r_corr: float, cfg: SocialRewardConfig) -> float:
    return float(1.0 - math.exp(-max(0.0, float(r_corr)) / max(1e-6, float(cfg.risk_gate_r0))))


def backward_flux_cost(flux: float, cfg: SocialRewardConfig) -> float:
    flux = max(0.0, float(flux))
    return float(flux / (flux + max(1e-6, float(cfg.flux_b0))))


def speed_reward(speed: float, cruise_speed: float, cfg: SocialRewardConfig) -> float:
    cruise_speed = max(1e-3, float(cruise_speed))
    norm_err = (float(speed) - cruise_speed) / cruise_speed
    return float(cfg.w_v * (1.0 - norm_err * norm_err))


def progress_reward(delta_x: float, dt: float, r_corr: float, cfg: SocialRewardConfig) -> float:
    denom = max(1e-6, float(cfg.progress_ref_speed) * float(dt))
    progress_norm = float(delta_x) / denom
    return float(cfg.w_p * progress_norm * math.exp(-max(0.0, float(r_corr)) / max(1e-6, float(cfg.risk_gate_r0))))


def comfort_cost(accel: float, steer: float, jerk: float, cfg: SocialRewardConfig) -> float:
    a_term = (float(accel) / max(1e-6, float(cfg.comfort_accel_norm))) ** 2
    s_term = (float(steer) / max(1e-6, float(cfg.comfort_steer_norm))) ** 2
    j_term = (float(jerk) / max(1e-6, float(cfg.comfort_jerk_norm))) ** 2
    return float(cfg.w_c * (a_term + s_term + j_term))


def social_field_externality(
    risk_field: np.ndarray | None,
    X: np.ndarray | None,
    Y: np.ndarray | None,
    nbr_xs_ego: np.ndarray,
    nbr_ys_ego: np.ndarray,
    nbr_closing: np.ndarray,
) -> dict[str, float]:
    if risk_field is None or X is None or Y is None:
        return {
            "risk_mass_total": 0.0,
            "risk_mass_others": 0.0,
            "risk_gradient_peak": 0.0,
            "risk_flux_backward": 0.0,
            "risk_field_entropy": 0.0,
        }
    return field_metrics(
        np.asarray(risk_field, dtype=np.float64),
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        np.asarray(nbr_xs_ego, dtype=np.float64),
        np.asarray(nbr_ys_ego, dtype=np.float64),
        np.asarray(nbr_closing, dtype=np.float64),
    )


def compose_reward(
    *,
    stock_reward: float,
    delta_x: float,
    dt: float,
    speed: float,
    cruise_speed: float,
    lane_delta: int,
    adv_left: float,
    adv_right: float,
    blocked: bool,
    ttc_min: float,
    thw: float,
    drac_value: float,
    imposed_rear_decel: float,
    rear_ttc_now: float,
    rear_ttc_prev: float,
    bad_cut_in: bool,
    accel: float,
    steer: float,
    jerk: float,
    r_corr: float,
    flux_back: float,
    cfg: SocialRewardConfig,
    ablation: str = "full",
) -> tuple[float, dict[str, float]]:
    profile = ablation_profile(ablation)
    components: dict[str, float] = {
        "stock_reward": float(stock_reward),
        "r_prog_safe": 0.0,
        "r_speed": 0.0,
        "r_lane_adv": 0.0,
        "c_bad_lane": 0.0,
        "c_missed": 0.0,
        "c_ttc": 0.0,
        "c_thw": 0.0,
        "c_drac": 0.0,
        "c_social": 0.0,
        "c_comfort": 0.0,
        "c_field": 0.0,
        "c_flux_weak": 0.0,
    }

    if profile["use_stock_reward"]:
        total = float(stock_reward)
        components["reward_total"] = total
        return total, components

    best_adv = max(float(adv_left), float(adv_right))
    chosen_adv = 0.0
    if lane_delta > 0:
        chosen_adv = float(adv_left)
    elif lane_delta < 0:
        chosen_adv = float(adv_right)

    if profile["use_progress"]:
        components["r_prog_safe"] = progress_reward(delta_x, dt, r_corr, cfg)
    if profile["use_speed"]:
        components["r_speed"] = speed_reward(speed, cruise_speed, cfg)
    if profile["use_lane"]:
        if lane_delta != 0 and chosen_adv > 0.0:
            components["r_lane_adv"] = float(cfg.w_adv * chosen_adv)
        if lane_delta != 0 and chosen_adv < 0.0:
            components["c_bad_lane"] = float(cfg.w_bad * abs(chosen_adv))
    if profile["use_missed"] and lane_delta == 0 and blocked and best_adv > float(cfg.missed_opportunity_adv):
        components["c_missed"] = float(cfg.w_miss * (best_adv - float(cfg.missed_opportunity_adv)))
    if profile["use_safety"]:
        components["c_ttc"] = float(cfg.w_ttc * quadratic_margin_cost(ttc_min, cfg.ttc_safe))
        components["c_thw"] = float(cfg.w_thw * quadratic_margin_cost(thw, cfg.thw_des))
        components["c_drac"] = float(cfg.w_drac * quadratic_upper_cost(drac_value, cfg.drac_comfort, cfg.drac_max))
    if profile["use_social"]:
        social_cost = courtesy_brake_cost(imposed_rear_decel, cfg)
        if np.isfinite(rear_ttc_prev) and np.isfinite(rear_ttc_now) and rear_ttc_now < rear_ttc_prev:
            ttc_loss = max(0.0, float(rear_ttc_prev) - float(rear_ttc_now))
            social_cost += min(1.0, ttc_loss / max(1e-6, float(cfg.ttc_safe)))
        if bad_cut_in:
            social_cost += 0.5
        components["c_social"] = float(cfg.w_soc * social_cost)
    if profile["use_comfort"]:
        components["c_comfort"] = comfort_cost(accel, steer, jerk, cfg)
    if profile["use_field"]:
        components["c_field"] = float(cfg.w_R * bounded_field_cost(r_corr, cfg))
    if profile["use_flux"]:
        components["c_flux_weak"] = float(cfg.w_B * backward_flux_cost(flux_back, cfg))

    total = (
        components["r_prog_safe"]
        + components["r_speed"]
        + components["r_lane_adv"]
        - components["c_bad_lane"]
        - components["c_missed"]
        - components["c_ttc"]
        - components["c_thw"]
        - components["c_drac"]
        - components["c_social"]
        - components["c_comfort"]
        - components["c_field"]
        - components["c_flux_weak"]
    )
    components["reward_total"] = float(total)
    return float(total), components


def decode_lane_delta(action: int, action_space_n: int) -> int:
    if int(action_space_n) == 5:
        if int(action) == 0:
            return +1
        if int(action) == 2:
            return -1
        return 0
    if int(action_space_n) == 9:
        band = int(action) // 3
        if band == 1:
            return -1
        if band == 2:
            return +1
        return 0
    return 0


def rear_bad_cut_in_flag(rear_ttc_now: float, rear_ttc_prev: float) -> bool:
    if np.isfinite(rear_ttc_now) and float(rear_ttc_now) < BAD_CUT_TTC_ABS:
        return True
    if np.isfinite(rear_ttc_prev) and rear_ttc_prev > 1e-6 and np.isfinite(rear_ttc_now):
        frac_drop = (float(rear_ttc_prev) - float(rear_ttc_now)) / float(rear_ttc_prev)
        if frac_drop > BAD_CUT_TTC_DROP_FRAC:
            return True
    return False


def hard_brake_imposed(accel: float) -> bool:
    return float(accel) <= float(HARD_BRAKE_DECEL_MPS2)
