"""
MetaDrive SB3 trainer (PPO/DQN/SAC/TD3/DDPG)
============================================
Single-agent training entry for the MetaDrive social-risk RL benchmark.

Usage examples
--------------
    # Smoke train, 50k steps, risk-aware matched protocol
    python rl/train_metadrive_sb3.py --protocol matched_social_risk --steps 50000 --seed 0

    # Matched stock baseline (same env/action protocol, no risk obs/reward)
    python rl/train_metadrive_sb3.py --protocol matched_stock --steps 1000000 --seed 0

    # Continuous-action baselines
    python rl/train_metadrive_sb3.py --protocol matched_stock_merge_continuous --algo sac --steps 1000000

    # Safe-RL stress env (with risk-cost augmentation)
    python rl/train_metadrive_sb3.py --protocol safe_metadrive_social_risk --steps 1000000

Checkpoints land in `rl/checkpoints/metadrive/<run_name>/` and CSV logs in
`rl/logs/metadrive/<run_name>/progress.csv`. We use a tiny CSV callback rather
than tensorboard alone so the same log layout is consumable by the eval
pipeline.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
METADRIVE_ROOT = REPO_ROOT / "metadrive"
if METADRIVE_ROOT.is_dir() and str(METADRIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(METADRIVE_ROOT))

# Propagate to SubprocVecEnv workers (spawn method re-launches Python).
_extra_paths = [str(REPO_ROOT)]
if METADRIVE_ROOT.is_dir():
    _extra_paths.append(str(METADRIVE_ROOT))
_existing_pp = os.environ.get("PYTHONPATH", "")
_pp_parts = [p for p in _existing_pp.split(os.pathsep) if p]
for _p in _extra_paths:
    if _p not in _pp_parts:
        _pp_parts.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(_pp_parts)

try:
    import numpy as np
    from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, SubprocVecEnv, VecMonitor,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"stable-baselines3 is required: pip install stable-baselines3\n  ({exc})"
    )

from rl.config.metadrive_config import (
    METADRIVE_PROTOCOLS,
    MetaDriveRLConfig,
    get_metadrive_protocol,
)
from rl.env.metadrive_envs import (
    make_metadrive_train_env,
    make_safe_metadrive_env,
)

TRAFFIC_MODE_CHOICES = ("protocol", "trigger", "respawn", "basic", "hybrid")
REWARD_PROFILE_CHOICES = ("risk_only", "comfort_light", "risk_comfort", "social_full")


# ---------------------------------------------------------------------- args

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", choices=tuple(sorted(METADRIVE_PROTOCOLS)),
                   default="matched_social_risk",
                   help="Named MetaDrive protocol; owns env/action/risk settings")
    p.add_argument("--env", choices=("metadrive", "safe-metadrive"),
                   default=None,
                   help="Deprecated. Prefer --protocol; kept for older commands.")
    p.add_argument("--algo", choices=("ppo", "dqn", "sac", "td3", "ddpg"), default="ppo")
    p.add_argument("--use-risk", action="store_true", default=None,
                   help="Deprecated override: enable all risk hooks")
    p.add_argument("--no-risk", action="store_false", dest="use_risk",
                   help="Deprecated override: disable all risk hooks")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--traffic-density", type=float, default=None)
    p.add_argument("--traffic-mode", choices=TRAFFIC_MODE_CHOICES, default="protocol",
                   help=(
                       "Traffic mode override. Prefer named *_respawn protocols for "
                       "paper runs; this is mainly for controlled ablations."
                   ))
    p.add_argument("--lambda-risk", type=float, default=None,
                   help="Override MetaDriveRLConfig.LAMBDA_RISK")
    p.add_argument("--reward-profile", choices=REWARD_PROFILE_CHOICES, default="risk_only",
                   help=(
                       "Reward shaping profile. risk_only reproduces the stable old "
                       "social-risk setup; comfort_light/risk_comfort are ablations; "
                       "social_full adds steering, ego hard-brake, follower courtesy, "
                       "rear-TTC, and backward-flux terms."
                   ))
    p.add_argument("--lambda-action-delta", type=float, default=None,
                   help="Override comfort action-delta weight")
    p.add_argument("--lambda-jerk", type=float, default=None,
                   help="Override comfort jerk weight")
    p.add_argument("--lambda-steer-abs", type=float, default=None,
                   help="Override absolute steering command weight")
    p.add_argument("--lambda-steer-delta", type=float, default=None,
                   help="Override comfort steering-change weight")
    p.add_argument("--lambda-throttle-delta", type=float, default=None,
                   help="Override comfort throttle/brake-change weight")
    p.add_argument("--w-hard-brake", type=float, default=None,
                   help="Override ego hard-brake penalty weight (social_full)")
    p.add_argument("--w-courtesy", type=float, default=None,
                   help="Override follower-courtesy penalty weight (social_full)")
    p.add_argument("--w-rear-ttc", type=float, default=None,
                   help="Override rear-TTC erosion penalty weight (social_full)")
    p.add_argument("--w-back-flux", type=float, default=None,
                   help="Override backward risk-flux penalty weight (social_full)")
    p.add_argument("--tau-risk", type=float, default=None,
                   help="Override MetaDriveRLConfig.TAU_RISK")
    p.add_argument("--run-name", type=str, default=None,
                   help="Subdirectory under rl/logs/metadrive (auto-derived if omitted)")
    p.add_argument("--allow-overwrite", action="store_true",
                   help="Allow writing into an existing checkpoint directory with final.zip")
    p.add_argument("--save-freq", type=int, default=25_000,
                   help="Save a checkpoint every N env steps")
    p.add_argument("--device", type=str, default="cpu",
                   help="torch device. SB3 recommends 'cpu' for MlpPolicy.")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Number of parallel MetaDrive envs (SubprocVecEnv when >1)")
    return p.parse_args()


def _resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return f"{args.protocol}_{args.algo}_seed{args.seed}"


def _build_config(args: argparse.Namespace) -> MetaDriveRLConfig:
    mdcfg = MetaDriveRLConfig()
    if args.reward_profile == "risk_only":
        mdcfg.LAMBDA_ACTION_DELTA = 0.0
        mdcfg.LAMBDA_JERK = 0.0
        mdcfg.LAMBDA_STEER_ABS = 0.0
        mdcfg.LAMBDA_STEER_DELTA = 0.0
        mdcfg.LAMBDA_THROTTLE_DELTA = 0.0
    elif args.reward_profile == "comfort_light":
        mdcfg.LAMBDA_ACTION_DELTA = 0.002
        mdcfg.LAMBDA_JERK = 0.001
        mdcfg.LAMBDA_STEER_ABS = 0.001
        mdcfg.LAMBDA_STEER_DELTA = 0.002
        mdcfg.LAMBDA_THROTTLE_DELTA = 0.002
    elif args.reward_profile == "risk_comfort":
        pass
    elif args.reward_profile == "social_full":
        # Full social-friendly model: risk (LAMBDA_RISK) + comfort smoothness
        # + steering magnitude + ego hard-brake + follower courtesy + rear-TTC
        # erosion + backward risk-flux. Weights kept small to avoid collapsing
        # into a static policy; tune via the --w-* / --lambda-* overrides.
        mdcfg.LAMBDA_ACTION_DELTA = 0.0      # avoid double-count with steer/throttle delta
        mdcfg.LAMBDA_JERK = 0.01
        mdcfg.LAMBDA_STEER_ABS = 0.002       # discourage large steering magnitude
        mdcfg.LAMBDA_STEER_DELTA = 0.01      # discourage frequent steering changes
        mdcfg.LAMBDA_THROTTLE_DELTA = 0.01
        mdcfg.W_HARD_BRAKE = 0.05
        mdcfg.W_COURTESY = 0.05
        mdcfg.W_REAR_TTC = 0.02
        mdcfg.W_BACK_FLUX = 0.02
    else:
        raise ValueError(f"Unknown reward profile {args.reward_profile}")

    if args.traffic_density is not None:
        mdcfg.TRAFFIC_DENSITY = float(args.traffic_density)
    if args.lambda_risk is not None:
        mdcfg.LAMBDA_RISK = float(args.lambda_risk)
    if args.lambda_action_delta is not None:
        mdcfg.LAMBDA_ACTION_DELTA = float(args.lambda_action_delta)
    if args.lambda_jerk is not None:
        mdcfg.LAMBDA_JERK = float(args.lambda_jerk)
    if args.lambda_steer_abs is not None:
        mdcfg.LAMBDA_STEER_ABS = float(args.lambda_steer_abs)
    if args.lambda_steer_delta is not None:
        mdcfg.LAMBDA_STEER_DELTA = float(args.lambda_steer_delta)
    if args.lambda_throttle_delta is not None:
        mdcfg.LAMBDA_THROTTLE_DELTA = float(args.lambda_throttle_delta)
    if args.w_hard_brake is not None:
        mdcfg.W_HARD_BRAKE = float(args.w_hard_brake)
    if args.w_courtesy is not None:
        mdcfg.W_COURTESY = float(args.w_courtesy)
    if args.w_rear_ttc is not None:
        mdcfg.W_REAR_TTC = float(args.w_rear_ttc)
    if args.w_back_flux is not None:
        mdcfg.W_BACK_FLUX = float(args.w_back_flux)
    if args.tau_risk is not None:
        mdcfg.TAU_RISK = float(args.tau_risk)
    return mdcfg


# ----------------------------------------------------------- env construction

def _single_env_factory(protocol_name: str, mdcfg: MetaDriveRLConfig,
                        use_risk: Optional[bool], traffic_density: Optional[float],
                        traffic_mode: Optional[str],
                        seed_offset: int):
    """Module-level factory so it's picklable for SubprocVecEnv on Windows."""
    proto = get_metadrive_protocol(protocol_name)
    if proto.env_name == "metadrive":
        env = make_metadrive_train_env(
            config=mdcfg,
            protocol=proto,
            use_risk=use_risk,
            traffic_density=traffic_density,
            traffic_mode=traffic_mode,
            seed_offset=int(seed_offset),
        )
    elif proto.env_name == "safe-metadrive":
        env = make_safe_metadrive_env(
            config=mdcfg,
            protocol=proto,
            use_risk=use_risk,
            traffic_density=traffic_density,
            traffic_mode=traffic_mode,
            seed_offset=int(seed_offset),
        )
    else:
        raise ValueError(f"Unknown env {proto.env_name}")
    return Monitor(env)


def _make_vec_env(args: argparse.Namespace, mdcfg: MetaDriveRLConfig):
    """Build a (Dummy|Subproc)VecEnv depending on --n-envs.

    All envs share the same MetaDrive scenario pool — SB3 forwards the same
    seed to every worker via reset(), so per-env disjoint ranges trigger
    `scenario_index out of range` asserts. Per-env diversity comes from the
    subprocess RNG and the SB3 model's own seeding of `env.action_space`.
    """
    from functools import partial
    n_envs = max(1, int(args.n_envs))
    traffic_mode = None if args.traffic_mode == "protocol" else str(args.traffic_mode)
    factories = [
        partial(_single_env_factory, args.protocol, mdcfg,
                args.use_risk, args.traffic_density, traffic_mode, 0)
        for _ in range(n_envs)
    ]
    if n_envs == 1:
        vec = DummyVecEnv(factories)
    else:
        vec = SubprocVecEnv(factories, start_method="spawn")
    return vec


# ------------------------------------------------------------------ model

def _validate_algo_protocol(args: argparse.Namespace, proto) -> None:
    if proto.discrete_action and args.algo in {"sac", "td3", "ddpg"}:
        raise SystemExit(
            f"{args.algo.upper()} requires a continuous Box action space, "
            f"but protocol '{proto.name}' uses discrete actions. Use a *_continuous protocol."
        )
    if (not proto.discrete_action or proto.use_multi_discrete) and args.algo == "dqn":
        raise SystemExit(
            "DQN requires a single Discrete action space, "
            f"but protocol '{proto.name}' is not single-discrete. Use a discrete matched_* protocol."
        )


def _build_model(args: argparse.Namespace, env, tensorboard_dir: Optional[str]):
    proto = get_metadrive_protocol(args.protocol)
    _validate_algo_protocol(args, proto)
    common = dict(
        verbose=1,
        seed=int(args.seed),
        tensorboard_log=tensorboard_dir,
        device=str(args.device),
    )
    if args.algo == "ppo":
        # n_steps is per-env: with 8 envs and n_steps=256, each PPO update sees
        # 8*256=2048 transitions, matching the 1-env config. Batch size scales
        # similarly so we keep 32 mini-batches per epoch.
        n_envs = max(1, int(args.n_envs))
        per_env_n_steps = max(64, 2048 // n_envs)
        return PPO(
            "MlpPolicy", env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            n_steps=per_env_n_steps, batch_size=64, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            ent_coef=0.0, clip_range=0.2, **common,
        )
    if args.algo == "dqn":
        return DQN(
            "MlpPolicy", env,
            policy_kwargs=dict(net_arch=[256, 256]),
            buffer_size=100_000, learning_starts=5_000,
            batch_size=64, learning_rate=1e-4,
            gamma=0.99, train_freq=4, gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.20,
            exploration_final_eps=0.05,
            **common,
        )
    if args.algo == "sac":
        return SAC(
            "MlpPolicy", env,
            policy_kwargs=dict(net_arch=[256, 256]),
            buffer_size=200_000, learning_starts=1000,
            batch_size=256, learning_rate=3e-4,
            gamma=0.99, tau=0.005, **common,
        )
    if args.algo == "td3":
        n_actions = int(np.prod(env.action_space.shape))
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=0.10 * np.ones(n_actions, dtype=np.float32),
        )
        return TD3(
            "MlpPolicy", env,
            policy_kwargs=dict(net_arch=[256, 256]),
            buffer_size=200_000, learning_starts=1000,
            batch_size=256, learning_rate=3e-4,
            gamma=0.99, tau=0.005, action_noise=action_noise, **common,
        )
    if args.algo == "ddpg":
        n_actions = int(np.prod(env.action_space.shape))
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=0.10 * np.ones(n_actions, dtype=np.float32),
        )
        return DDPG(
            "MlpPolicy", env,
            policy_kwargs=dict(net_arch=[256, 256]),
            buffer_size=200_000, learning_starts=1000,
            batch_size=256, learning_rate=1e-4,
            gamma=0.99, tau=0.005, action_noise=action_noise, **common,
        )
    raise ValueError(f"Unknown --algo {args.algo}")


# ---------------------------------------------------------------- callbacks

class CSVLogCallback(BaseCallback):
    """Append per-rollout metrics to a CSV: timesteps, ep_rew_mean, ep_len_mean,
    ep_cost_mean, ep_risk_penalty_mean, ep_r_ego_mean.

    Reads info dicts that the wrapper attaches each step.
    """

    def __init__(self, csv_path: str, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._csv_path = csv_path
        self._header_written = False
        # Episode accumulators keyed by env index
        self._ep_buf: dict[int, dict] = {}

    # Per-term penalty keys logged for the reward-term decomposition plots.
    _PENALTY_KEYS = (
        "risk_penalty", "comfort_penalty", "steer_penalty", "jerk_penalty",
        "throttle_penalty", "hard_brake_penalty", "courtesy_penalty",
        "rear_ttc_penalty", "backward_flux_penalty",
    )

    @classmethod
    def _new_buf(cls) -> dict:
        buf = {
            "len": 0, "rew": 0.0, "cost": 0.0, "r_ego_sum": 0.0,
            "decel_follower_sum": 0.0, "backward_pressure_sum": 0.0,
            "hard_brake_imposed_steps": 0,
        }
        for k in cls._PENALTY_KEYS:
            buf[k] = 0.0
        return buf

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones", [False] * len(infos))
        rewards = self.locals.get("rewards", [0.0] * len(infos))
        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            buf = self._ep_buf.setdefault(env_idx, self._new_buf())
            buf["len"] += 1
            buf["rew"] += float(rewards[env_idx])
            buf["cost"] += float(info.get("cost", 0.0))
            buf["r_ego_sum"] += float(info.get("r_ego", 0.0))
            for k in self._PENALTY_KEYS:
                buf[k] += float(info.get(k, 0.0))
            decel_follower = float(info.get("decel_follower", 0.0))
            buf["decel_follower_sum"] += decel_follower
            buf["backward_pressure_sum"] += float(info.get("backward_pressure", 0.0))
            if decel_follower >= 3.0:
                buf["hard_brake_imposed_steps"] += 1

            if dones[env_idx]:
                ep_info = info.get("episode", {}) if isinstance(info.get("episode"), dict) else {}
                inv_len = 1.0 / max(1, buf["len"])
                row = {
                    "timesteps": int(self.num_timesteps),
                    "env_idx": int(env_idx),
                    "ep_len": int(buf["len"]),
                    "ep_reward": float(ep_info.get("r", buf["rew"])),
                    "ep_cost": float(buf["cost"]),
                    "ep_r_ego_mean": float(buf["r_ego_sum"] * inv_len),
                    # Per-term penalty episode sums (for the decomposition plot)
                    "ep_risk_penalty": float(buf["risk_penalty"]),
                    "ep_comfort_penalty": float(buf["comfort_penalty"]),
                    "ep_steer_penalty": float(buf["steer_penalty"]),
                    "ep_jerk_penalty": float(buf["jerk_penalty"]),
                    "ep_throttle_penalty": float(buf["throttle_penalty"]),
                    "ep_hard_brake_penalty": float(buf["hard_brake_penalty"]),
                    "ep_courtesy_penalty": float(buf["courtesy_penalty"]),
                    "ep_rear_ttc_penalty": float(buf["rear_ttc_penalty"]),
                    "ep_backward_flux_penalty": float(buf["backward_flux_penalty"]),
                    # Social-externality episode means / rates
                    "ep_decel_follower_mean": float(buf["decel_follower_sum"] * inv_len),
                    "ep_backward_pressure_mean": float(buf["backward_pressure_sum"] * inv_len),
                    "ep_hard_brake_imposed_rate": float(buf["hard_brake_imposed_steps"] * inv_len),
                    "success": int(bool(info.get("arrive_dest", False))),
                    "crash_vehicle": int(bool(info.get("crash_vehicle", False))),
                    "out_of_road": int(bool(info.get("out_of_road", False))),
                    "route_completion": float(info.get("route_completion", 0.0)),
                }
                self._append_row(row)
                self._ep_buf[env_idx] = self._new_buf()
        return True

    def _append_row(self, row: dict) -> None:
        new_file = not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file or not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# -------------------------------------------------------------------- main

def main() -> int:
    args = _parse_args()
    run_name = _resolve_run_name(args)
    proto = get_metadrive_protocol(args.protocol)
    if args.env is not None and args.env != proto.env_name:
        raise SystemExit(
            f"--env {args.env} conflicts with protocol '{proto.name}' "
            f"which uses env '{proto.env_name}'. Drop --env or choose another protocol."
        )
    _validate_algo_protocol(args, proto)

    log_dir = REPO_ROOT / "rl" / "logs" / "metadrive" / run_name
    ckpt_dir = REPO_ROOT / "rl" / "checkpoints" / "metadrive" / run_name
    final_ckpt = ckpt_dir / "final.zip"
    if final_ckpt.exists() and not args.allow_overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing checkpoint: {final_ckpt}\n"
            "Use a new --run-name for new experiments, or pass --allow-overwrite intentionally."
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mdcfg = _build_config(args)
    (log_dir / "config.json").write_text(json.dumps(mdcfg.to_dict(), indent=2))
    (log_dir / "protocol.json").write_text(json.dumps(proto.to_dict(), indent=2))
    effective_traffic_mode = proto.traffic_mode if args.traffic_mode == "protocol" else str(args.traffic_mode)
    (log_dir / "run_overrides.json").write_text(json.dumps({
        "traffic_density": args.traffic_density,
        "traffic_mode": effective_traffic_mode,
        "traffic_mode_source": "protocol" if args.traffic_mode == "protocol" else "cli_override",
        "reward_profile": args.reward_profile,
        "use_risk_override": args.use_risk,
    }, indent=2))

    print("=" * 64)
    print(f"MetaDrive RL training run: {run_name}")
    print(f"  protocol: {proto.name}  env: {proto.env_name}  algo: {args.algo}")
    print(f"  traffic: mode={effective_traffic_mode}  density={args.traffic_density if args.traffic_density is not None else proto.traffic_density}")
    print(
        "  reward: "
        f"profile={args.reward_profile} "
        f"lambda_risk={mdcfg.LAMBDA_RISK} "
        f"lambda_action_delta={mdcfg.LAMBDA_ACTION_DELTA} "
        f"lambda_jerk={mdcfg.LAMBDA_JERK} "
        f"lambda_steer_abs={mdcfg.LAMBDA_STEER_ABS} "
        f"lambda_steer_delta={mdcfg.LAMBDA_STEER_DELTA} "
        f"lambda_throttle_delta={mdcfg.LAMBDA_THROTTLE_DELTA} "
        f"w_hard_brake={mdcfg.W_HARD_BRAKE} "
        f"w_courtesy={mdcfg.W_COURTESY} "
        f"w_rear_ttc={mdcfg.W_REAR_TTC} "
        f"w_back_flux={mdcfg.W_BACK_FLUX}"
    )
    print(
        "  risk flags: "
        f"append_obs={proto.append_risk_obs if args.use_risk is None else bool(args.use_risk)} "
        f"shape_reward={proto.shape_risk_reward if args.use_risk is None else bool(args.use_risk)} "
        f"metrics={proto.compute_risk_metrics if args.use_risk is None else bool(args.use_risk)}"
    )
    print(f"  steps: {args.steps}  seed: {args.seed}  n_envs: {args.n_envs}  device: {args.device}")
    print(f"  log_dir:  {log_dir}")
    print(f"  ckpt_dir: {ckpt_dir}")
    print("=" * 64)

    env = _make_vec_env(args, mdcfg)
    tb_dir = str(log_dir / "tb") if importlib.util.find_spec("tensorboard") else None
    model = _build_model(args, env, tensorboard_dir=tb_dir)

    csv_cb = CSVLogCallback(str(log_dir / "progress.csv"))
    ckpt_cb = CheckpointCallback(
        save_freq=int(args.save_freq),
        save_path=str(ckpt_dir),
        name_prefix=str(args.algo),
    )
    try:
        model.learn(total_timesteps=int(args.steps),
                    callback=[csv_cb, ckpt_cb],
                    progress_bar=False)
    finally:
        model.save(str(ckpt_dir / "final"))
        env.close()

    print(f"Final checkpoint: {ckpt_dir / 'final.zip'}")
    return 0


if __name__ == "__main__":
    _rc = main()
    # MetaDrive/Panda3D + CUDA can segfault during interpreter teardown on
    # Windows (exit 0xC0000005 / 0xC0000374) even after training and
    # model.save() finished. The checkpoint is already on disk by now, so flush
    # and hard-exit to bypass the crashy native static-destructor path. Without
    # this the process returns a nonzero crash code despite a successful run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(_rc) if _rc is not None else 0)
