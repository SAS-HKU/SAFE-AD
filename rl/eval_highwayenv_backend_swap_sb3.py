"""Paired closed-loop prospective-teacher to PINN swap for SB3 policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rl.env.highwayenv_social_env import (
    load_reward_config,
    make_social_highwayenv_env,
    resolve_traffic_config,
)
from rl.eval_highwayenv_social_sb3 import (
    _model_observation,
    evaluate_episode,
    load_sb3_model,
)


def _seeds(value: str) -> list[int]:
    value = value.strip()
    if ":" in value:
        start, stop = (int(part) for part in value.split(":", 1))
        return list(range(start, stop))
    return [int(part) for part in value.split(",") if part.strip()]


def _bootstrap(values, *, repetitions: int, seed: int):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, (int(repetitions), len(values)), replace=True).mean(1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _counterfactual_trace(
    model,
    *,
    env_id: str,
    seed: int,
    traffic,
    reward_config,
    ablation: str,
    action_mode: str,
    pinn_checkpoint: str,
    pinn_device: str,
    max_steps: int,
) -> dict:
    common = dict(
        env_id=env_id,
        interface="stock",
        render_mode=None,
        traffic=traffic,
        reward_config=reward_config,
        ablation=ablation,
        use_drift=True,
        action_mode=action_mode,
        append_risk_obs=True,
        record_risk_metrics=False,
        pinn_checkpoint=pinn_checkpoint,
        pinn_device=pinn_device,
    )
    teacher = make_social_highwayenv_env(field_backend="prospective", **common)
    surrogate = make_social_highwayenv_env(field_backend="pinn", **common)
    teacher_obs, _ = teacher.reset(seed=seed)
    surrogate_obs, _ = surrogate.reset(seed=seed)
    descriptor_errors = []
    action_errors = []
    normalized_action_errors = []
    action_mismatch = []
    stock_state_errors = []
    steps = 0
    try:
        for _ in range(int(max_steps)):
            teacher_value = _model_observation(model, teacher_obs)
            surrogate_value = _model_observation(model, surrogate_obs)
            teacher_action, _ = model.predict(teacher_value, deterministic=True)
            surrogate_action, _ = model.predict(surrogate_value, deterministic=True)
            teacher_action_array = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
            surrogate_action_array = np.asarray(surrogate_action, dtype=np.float32).reshape(-1)
            descriptor_errors.append(
                np.asarray(surrogate_value[-8:] - teacher_value[-8:], dtype=np.float32)
            )
            delta = surrogate_action_array - teacher_action_array
            action_errors.append(delta)
            if hasattr(model.action_space, "low") and hasattr(model.action_space, "high"):
                action_low = np.asarray(model.action_space.low, dtype=np.float32).reshape(-1)
                action_high = np.asarray(model.action_space.high, dtype=np.float32).reshape(-1)
                action_span = np.maximum(action_high - action_low, 1e-6)
            else:
                action_span = np.full_like(delta, max(int(model.action_space.n) - 1, 1))
            normalized_action_errors.append(delta / action_span)
            action_mismatch.append(float(not np.array_equal(surrogate_action_array, teacher_action_array)))
            stock_state_errors.append(
                float(np.max(np.abs(surrogate_value[:-8] - teacher_value[:-8])))
            )
            # Both simulators receive the teacher action; therefore any policy
            # difference above is attributable to the field backend, while the
            # physical state trajectory remains synchronized.
            teacher_obs, _rt, term_t, trunc_t, _it = teacher.step(teacher_action)
            surrogate_obs, _rp, term_p, trunc_p, _ip = surrogate.step(teacher_action)
            steps += 1
            if term_t or trunc_t or term_p or trunc_p:
                break
    finally:
        teacher.close()
        surrogate.close()
    descriptor = np.asarray(descriptor_errors, dtype=np.float32)
    action = np.asarray(action_errors, dtype=np.float32)
    normalized_action = np.asarray(normalized_action_errors, dtype=np.float32)
    action_norm = np.linalg.norm(action, axis=1) if action.size else np.zeros(0)
    normalized_action_norm = (
        np.linalg.norm(normalized_action, axis=1)
        if normalized_action.size
        else np.zeros(0)
    )
    return {
        "environment": env_id,
        "seed": int(seed),
        "steps": int(steps),
        "descriptor_rmse": float(np.sqrt(np.mean(descriptor**2))) if descriptor.size else 0.0,
        "descriptor_l2_mean": float(np.mean(np.linalg.norm(descriptor, axis=1))) if descriptor.size else 0.0,
        "steering_abs_error_mean": float(np.mean(np.abs(action[:, 0]))) if action.size else 0.0,
        "throttle_abs_error_mean": (
            float(np.mean(np.abs(action[:, 1])))
            if action.ndim == 2 and action.shape[1] > 1
            else 0.0
        ),
        "action_l2_mean": float(np.mean(action_norm)) if action_norm.size else 0.0,
        "action_l2_p95": float(np.percentile(action_norm, 95)) if action_norm.size else 0.0,
        "normalized_action_l2_mean": (
            float(np.mean(normalized_action_norm))
            if normalized_action_norm.size
            else 0.0
        ),
        "action_l2_gt_0_05_rate": (
            float(np.mean(action_norm > 0.05)) if action_norm.size else 0.0
        ),
        "action_l2_gt_0_10_rate": (
            float(np.mean(action_norm > 0.10)) if action_norm.size else 0.0
        ),
        "action_mismatch_rate": float(np.mean(action_mismatch)) if action_mismatch else 0.0,
        "stock_state_max_error": float(np.max(stock_state_errors)) if stock_state_errors else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["ppo", "dqn", "sac", "td3", "ddpg"], required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--pinn-checkpoint", required=True)
    parser.add_argument("--env-id", nargs="+", default=["merge-v0"])
    parser.add_argument(
        "--action-mode",
        choices=["default", "discrete_meta", "discrete_kinematic", "continuous"],
        default="default",
    )
    parser.add_argument("--reward-config", default="rl/config/social_reward_v1.json")
    parser.add_argument("--ablation", default="A5")
    parser.add_argument("--traffic-preset", default="medium")
    parser.add_argument("--seeds", type=_seeds, default=_seeds("100:110"))
    parser.add_argument("--pinn-device", default="cpu")
    parser.add_argument("--max-trace-steps", type=int, default=80)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=600,
        help="Finite closed-loop horizon for scenarios without native truncation.",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--output-dir", default="evaluation/highwayenv_prospective_pinn_backend_swap"
    )
    args = parser.parse_args()
    model = load_sb3_model(args.algo, args.policy_checkpoint)
    expected = int(np.prod(model.observation_space.shape))
    if expected != 33:
        raise ValueError(
            f"A true field-observation backend swap requires a 33-D policy; checkpoint expects {expected}"
        )
    traffic = resolve_traffic_config(preset=args.traffic_preset)
    reward_config = load_reward_config(args.reward_config)
    rows = []
    traces = []
    for env_id in args.env_id:
        for seed in args.seeds:
            for backend in ("prospective", "pinn"):
                result = evaluate_episode(
                    model,
                    env_id=env_id,
                    interface="stock",
                    traffic=traffic,
                    reward_config=reward_config,
                    ablation=args.ablation,
                    seed=seed,
                    use_drift=True,
                    action_mode=args.action_mode,
                    append_risk_obs=True,
                    field_backend=backend,
                    pinn_checkpoint=args.pinn_checkpoint,
                    pinn_device=args.pinn_device,
                    max_steps=args.max_episode_steps,
                )
                row = {"backend": backend, **result.__dict__}
                row.pop("reward_terms_mean", None)
                rows.append(row)
                print(
                    f"[{env_id} seed={seed}] {backend}: return={result.episode_return:.2f} "
                    f"collision={int(result.crashed)} progress={result.progress:.1f} m "
                    f"field={result.mean_field_backend_ms:.2f} ms",
                    flush=True,
                )
            traces.append(
                _counterfactual_trace(
                    model,
                    env_id=env_id,
                    seed=seed,
                    traffic=traffic,
                    reward_config=reward_config,
                    ablation=args.ablation,
                    action_mode=args.action_mode,
                    pinn_checkpoint=args.pinn_checkpoint,
                    pinn_device=args.pinn_device,
                    max_steps=args.max_trace_steps,
                )
            )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "backend_swap_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "frozen_state_action_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)

    metrics = (
        "episode_return",
        "episode_length",
        "crashed",
        "progress",
        "mean_speed",
        "mean_jerk_abs",
        "ttc_min",
        "imposed_rear_decel_max",
        "mean_field_backend_ms",
    )
    comparisons = []
    for env_id in args.env_id:
        selected = [row for row in rows if row["env_id"] == env_id]
        for metric in metrics:
            teacher = {
                int(row["seed"]): float(row[metric])
                for row in selected
                if row["backend"] == "prospective"
            }
            surrogate = {
                int(row["seed"]): float(row[metric])
                for row in selected
                if row["backend"] == "pinn"
            }
            common = sorted(set(teacher) & set(surrogate))
            delta = [surrogate[seed] - teacher[seed] for seed in common]
            mean, low, high = _bootstrap(
                delta,
                repetitions=args.bootstrap,
                seed=5200 + len(comparisons),
            )
            comparisons.append(
                {
                    "environment": env_id,
                    "metric": metric,
                    "n_pairs": len(common),
                    "teacher_mean": float(np.mean([teacher[seed] for seed in common])),
                    "pinn_mean": float(np.mean([surrogate[seed] for seed in common])),
                    "paired_delta_pinn_minus_teacher": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    summary = {
        "policy_checkpoint": str(Path(args.policy_checkpoint).resolve()),
        "pinn_checkpoint": str(Path(args.pinn_checkpoint).resolve()),
        "policy_observation_dim": expected,
        "swap_is_actual_closed_loop": True,
        "counterfactual_trace_uses_synchronized_teacher_actions": True,
        "paired_seeds": list(args.seeds),
        "comparisons": comparisons,
        "frozen_state_action_trace": traces,
    }
    (output / "backend_swap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
