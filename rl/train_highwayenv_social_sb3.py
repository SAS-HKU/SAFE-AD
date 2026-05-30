from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
HIGHWAYENV_ROOT = REPO_ROOT / "HighwayEnv-master"
for _path in (str(HIGHWAYENV_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rl.utils.typing_compat import ensure_typing_extensions_compat

ensure_typing_extensions_compat()
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from rl.env.highwayenv_social_env import (
    load_reward_config,
    make_social_highwayenv_env,
    resolve_traffic_config,
)
from rl.eval_highwayenv_social_sb3 import evaluate_model
from rl.utils.timing import get_timer, CATEGORY_RL_TRAIN


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _lane_bias_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(x) for x in str(raw).split(",") if str(x).strip())


def _tensorboard_dir(path: str) -> str | None:
    return path if importlib.util.find_spec("tensorboard") is not None else None


def _ppo_builder(env, *, seed: int, tensorboard_dir: str, total_steps: int, device: str = "auto"):
    batch_size = 64
    n_steps_default = max(8, batch_size * 12 // max(1, getattr(env, "num_envs", 1)))
    n_steps = max(8, min(int(total_steps), n_steps_default))
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.8,
        verbose=1,
        tensorboard_log=_tensorboard_dir(tensorboard_dir),
        seed=seed,
        device=device,
    )


def _dqn_builder(env, *, seed: int, tensorboard_dir: str, device: str = "auto"):
    return DQN(
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
        tensorboard_log=_tensorboard_dir(tensorboard_dir),
        seed=seed,
        device=device,
    )


def _sac_builder(env, *, seed: int, tensorboard_dir: str, device: str = "auto"):
    return SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        buffer_size=200000,
        learning_starts=1000,
        batch_size=256,
        gamma=0.95,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=_tensorboard_dir(tensorboard_dir),
        seed=seed,
        device=device,
    )


def _td3_builder(env, *, seed: int, tensorboard_dir: str, device: str = "auto"):
    return TD3(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        buffer_size=200000,
        learning_starts=1000,
        batch_size=256,
        gamma=0.95,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=_tensorboard_dir(tensorboard_dir),
        seed=seed,
        device=device,
    )


def _ddpg_builder(env, *, seed: int, tensorboard_dir: str, device: str = "auto"):
    return DDPG(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        buffer_size=200000,
        learning_starts=1000,
        batch_size=256,
        gamma=0.95,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=_tensorboard_dir(tensorboard_dir),
        seed=seed,
        device=device,
    )


def _build_model(algo: str, env, *, seed: int, tensorboard_dir: str, total_steps: int, device: str = "auto"):
    if algo == "ppo":
        return _ppo_builder(env, seed=seed, tensorboard_dir=tensorboard_dir, total_steps=total_steps, device=device)
    if algo == "dqn":
        return _dqn_builder(env, seed=seed, tensorboard_dir=tensorboard_dir, device=device)
    if algo == "sac":
        return _sac_builder(env, seed=seed, tensorboard_dir=tensorboard_dir, device=device)
    if algo == "td3":
        return _td3_builder(env, seed=seed, tensorboard_dir=tensorboard_dir, device=device)
    if algo == "ddpg":
        return _ddpg_builder(env, seed=seed, tensorboard_dir=tensorboard_dir, device=device)
    raise ValueError(f"Unsupported algo '{algo}'")


def _resolve_action_mode(algo: str, action_mode: str) -> str:
    if action_mode != "auto":
        return action_mode
    if algo in {"sac", "td3", "ddpg"}:
        return "continuous"
    return "default"


class SocialEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        algo: str,
        eval_env_ids: list[str],
        interface: str,
        traffic,
        reward_config,
        ablation: str,
        eval_freq: int,
        eval_episodes: int,
        use_drift: bool,
        action_mode: str,
        checkpoints_dir: str,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.algo = algo
        self.eval_env_ids = list(eval_env_ids)
        self.interface = interface
        self.traffic = traffic
        self.reward_config = reward_config
        self.ablation = ablation
        self.eval_freq = max(1, int(eval_freq))
        self.eval_episodes = max(1, int(eval_episodes))
        self.use_drift = bool(use_drift)
        self.action_mode = str(action_mode)
        self.checkpoints_dir = checkpoints_dir
        self.records: list[dict[str, float]] = []
        self._step_window: dict[str, list[float]] = defaultdict(list)
        self._train_returns = deque(maxlen=50)
        self._train_lengths = deque(maxlen=50)
        self._best_key = float("-inf")
        self._next_eval = 0

    def _on_training_start(self) -> None:
        self._evaluate()
        self._next_eval = self.eval_freq

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]
        for info in infos:
            step = info.get("social_step")
            if isinstance(step, dict):
                for key, value in step.items():
                    try:
                        self._step_window[key].append(float(value))
                    except (TypeError, ValueError):
                        pass
            episode = info.get("episode")
            if isinstance(episode, dict):
                self._train_returns.append(float(episode.get("r", 0.0)))
                self._train_lengths.append(float(episode.get("l", 0.0)))
        while self.num_timesteps >= self._next_eval:
            self._evaluate()
            self._next_eval += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if not self.records or self.records[-1]["timesteps"] != int(self.num_timesteps):
            self._evaluate()

    def _evaluate(self) -> None:
        eval_results = evaluate_model(
            self.model,
            env_ids=self.eval_env_ids,
            interface=self.interface,
            traffic=self.traffic,
            reward_config=self.reward_config,
            ablation=self.ablation,
            episodes=self.eval_episodes,
            use_drift=self.use_drift,
            action_mode=self.action_mode,
        )
        row: dict[str, float] = {
            "timesteps": float(self.num_timesteps),
            "train_return_mean": float(sum(self._train_returns) / len(self._train_returns)) if self._train_returns else float("nan"),
            "train_episode_length_mean": float(sum(self._train_lengths) / len(self._train_lengths)) if self._train_lengths else float("nan"),
        }
        for key, values in self._step_window.items():
            row[f"train_{key}_mean"] = float(sum(values) / len(values)) if values else float("nan")
        self._step_window.clear()

        for env_id, summary in eval_results.items():
            prefix = env_id.replace("-", "_")
            for key, value in summary.items():
                if key == "episodes_detail" or isinstance(value, dict):
                    continue
                row[f"{prefix}_{key}"] = float(value)
            reward_terms = summary.get("reward_terms_mean", {})
            if isinstance(reward_terms, dict):
                for key, value in reward_terms.items():
                    row[f"{prefix}_{key}"] = float(value)

        self.records.append(row)
        primary_env = self.eval_env_ids[0]
        primary_score = float(eval_results[primary_env]["return_mean"])
        if primary_score > self._best_key:
            self._best_key = primary_score
            self.model.save(os.path.join(self.checkpoints_dir, "best_model"))


def _write_csv(path: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_train_env(algo: str, env_id: str, *, interface: str, traffic, reward_config, ablation: str, use_drift: bool, action_mode: str, seed: int, n_envs: int):
    def _env_factory():
        return Monitor(
            make_social_highwayenv_env(
                env_id=env_id,
                interface=interface,
                traffic=traffic,
                reward_config=reward_config,
                ablation=ablation,
                use_drift=use_drift,
                action_mode=action_mode,
            )
        )

    if algo == "ppo":
        return make_vec_env(_env_factory, n_envs=max(1, int(n_envs)), seed=seed)
    return _env_factory()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SB3 agents on HighwayEnv with matched stock/social reward protocols.")
    parser.add_argument("--algo", choices=["ppo", "dqn", "sac", "td3", "ddpg"], required=True)
    parser.add_argument("--env-id", default="highway-fast-v0")
    parser.add_argument("--eval-env-id", nargs="+", default=["highway-v0", "merge-v0"])
    parser.add_argument("--interface", choices=["stock", "decision"], default="stock")
    parser.add_argument(
        "--action-mode",
        choices=["auto", "default", "discrete_meta", "discrete_kinematic", "continuous"],
        default="auto",
        help="HighwayEnv ego action protocol. Auto keeps existing PPO/DQN discrete defaults and uses continuous for SAC/TD3/DDPG.",
    )
    parser.add_argument("--reward-config", default="rl/config/social_reward_v1.json")
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--use-drift", type=_str2bool, default=True)
    parser.add_argument("--traffic-preset", default="medium")
    parser.add_argument("--vehicles-count", type=int, default=None)
    parser.add_argument("--vehicles-density", type=float, default=None)
    parser.add_argument("--ego-spacing", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--sv-speed-min", type=float, default=None)
    parser.add_argument("--sv-speed-max", type=float, default=None)
    parser.add_argument("--sv-speed-noise", type=float, default=None)
    parser.add_argument("--lane-speed-bias", default="")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--device", default="auto",
                        help="Torch device for policy/value nets: 'auto', 'cpu', 'cuda', or 'cuda:0'.")
    args = parser.parse_args()
    action_mode = _resolve_action_mode(args.algo, args.action_mode)
    if args.algo in {"sac", "td3", "ddpg"} and action_mode != "continuous":
        raise SystemExit(f"{args.algo.upper()} requires --action-mode continuous.")
    if args.algo == "dqn" and action_mode == "continuous":
        raise SystemExit("DQN requires a discrete HighwayEnv action mode.")

    lane_bias = _lane_bias_tuple(args.lane_speed_bias)
    traffic = resolve_traffic_config(
        preset=args.traffic_preset,
        vehicles_count=args.vehicles_count,
        vehicles_density=args.vehicles_density,
        ego_spacing=args.ego_spacing,
        duration=args.duration,
        sv_speed_min=args.sv_speed_min,
        sv_speed_max=args.sv_speed_max,
        sv_speed_noise=args.sv_speed_noise,
        lane_speed_bias=lane_bias,
    )
    reward_config = load_reward_config(args.reward_config)

    run_dir = Path(args.run_dir) if args.run_dir else Path("rl/logs") / f"{args.algo}_{args.env_id}_{args.ablation}"
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "algo": args.algo,
        "env_id": args.env_id,
        "eval_env_id": args.eval_env_id,
        "interface": args.interface,
        "action_mode": action_mode,
        "requested_action_mode": args.action_mode,
        "ablation": args.ablation,
        "use_drift": bool(args.use_drift),
        "total_steps": int(args.total_steps),
        "eval_freq": int(args.eval_freq),
        "eval_episodes": int(args.eval_episodes),
        "n_envs": int(args.n_envs),
        "seed": int(args.seed),
        "device": str(args.device),
        "traffic": traffic.to_dict(),
        "reward_config_path": args.reward_config,
        "reward_config": reward_config.to_dict(),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    train_env = _build_train_env(
        args.algo,
        args.env_id,
        interface=args.interface,
        traffic=traffic,
        reward_config=reward_config,
        ablation=args.ablation,
        use_drift=bool(args.use_drift),
        action_mode=action_mode,
        seed=int(args.seed),
        n_envs=int(args.n_envs),
    )
    model = _build_model(
        args.algo,
        train_env,
        seed=int(args.seed),
        tensorboard_dir=str(tensorboard_dir),
        total_steps=int(args.total_steps),
        device=str(args.device),
    )
    callback = SocialEvalCallback(
        algo=args.algo,
        eval_env_ids=list(args.eval_env_id),
        interface=args.interface,
        traffic=traffic,
        reward_config=reward_config,
        ablation=args.ablation,
        eval_freq=int(args.eval_freq),
        eval_episodes=int(args.eval_episodes),
        use_drift=bool(args.use_drift),
        action_mode=action_mode,
        checkpoints_dir=str(checkpoints_dir),
        verbose=0,
    )
    timer = get_timer()
    sync_cuda = str(args.device).lower().startswith("cuda")
    with timer.measure(CATEGORY_RL_TRAIN, sync_cuda=sync_cuda) as cat:
        model.learn(total_timesteps=int(args.total_steps), callback=callback)
    cat.extras["total_steps"] = int(args.total_steps)
    cat.extras["device"] = str(args.device)
    cat.extras["sps"] = int(args.total_steps) / max(cat.total_s, 1e-9)
    timer.write_csv(str(run_dir / "compute_time_rl_train.csv"))
    model.save(str(checkpoints_dir / "final_model"))
    train_env.close()

    training_log_path = run_dir / "training_log.json"
    with open(training_log_path, "w", encoding="utf-8") as f:
        json.dump(callback.records, f, indent=2)
    _write_csv(str(run_dir / "training_log.csv"), callback.records)

    final_eval = evaluate_model(
        model,
        env_ids=list(args.eval_env_id),
        interface=args.interface,
        traffic=traffic,
        reward_config=reward_config,
        ablation=args.ablation,
        episodes=int(args.eval_episodes),
        use_drift=bool(args.use_drift),
        action_mode=action_mode,
    )
    summary = {
        "run_dir": str(run_dir),
        "config": config_snapshot,
        "best_primary_return": float(callback._best_key),
        "final_eval": final_eval,
        "records": callback.records,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
