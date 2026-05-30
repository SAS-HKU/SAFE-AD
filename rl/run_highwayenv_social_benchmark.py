"""
Batch driver for the HighwayEnv social-RL benchmark matrix.

This mirrors the MetaDrive social benchmark at the level that is technically
meaningful for HighwayEnv:

* DQN and discrete-track PPO use the native HighwayEnv discrete meta-action
  protocol.
* SAC, TD3, and DDPG use the HighwayEnv continuous action protocol.
* Stock, risk-only, and social-full agents share the same scene, traffic,
  seeds, and action protocol; only the reward/field shaping arm changes.

Examples
--------
    # Full specialist matrix on CUDA, one million steps per run
    python -m rl.run_highwayenv_social_benchmark --scenarios highway merge intersection roundabout

    # Dry-run the exact commands
    python -m rl.run_highwayenv_social_benchmark --scenarios merge --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_ALGOS = ("ppo", "dqn", "sac", "td3", "ddpg")
CONTINUOUS_ONLY = {"sac", "td3", "ddpg"}
DISCRETE_ONLY = {"dqn"}
REWARD_ARMS = ("stock", "risk_only", "social_full")
SCENARIOS = {
    "highway": {
        "train_env": "highway-fast-v0",
        "eval_envs": ["highway-v0"],
        "traffic": "medium",
    },
    "merge": {
        "train_env": "merge-v0",
        "eval_envs": ["merge-v0"],
        "traffic": "medium",
    },
    "intersection": {
        "train_env": "intersection-v0",
        "eval_envs": ["intersection-v0"],
        "traffic": "medium",
    },
    "roundabout": {
        "train_env": "roundabout-v0",
        "eval_envs": ["roundabout-v0"],
        "traffic": "medium",
    },
}


def _steps_label(steps: int) -> str:
    if steps % 1_000_000 == 0:
        return f"{steps // 1_000_000}m"
    if steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    return str(steps)


def _track_for(algo: str, ppo_track: str) -> list[str]:
    if algo in DISCRETE_ONLY:
        return ["discrete"]
    if algo in CONTINUOUS_ONLY:
        return ["continuous"]
    if ppo_track == "both":
        return ["discrete", "continuous"]
    return [ppo_track]


def _action_mode(track: str) -> str:
    return "continuous" if track == "continuous" else "default"


def _ablation(reward_arm: str) -> str:
    if reward_arm == "stock":
        return "A0"
    if reward_arm == "risk_only":
        return "A2"
    if reward_arm == "social_full":
        return "full"
    raise ValueError(f"Unknown reward arm '{reward_arm}'")


def _run_dir(run_name: str) -> Path:
    return REPO_ROOT / "rl" / "logs" / "highwayenv" / run_name


def _final_checkpoint(run_name: str) -> Path:
    return _run_dir(run_name) / "checkpoints" / "final_model.zip"


def _build_runs(args: argparse.Namespace) -> list[dict]:
    label = _steps_label(int(args.steps))
    runs: list[dict] = []
    seen: set[str] = set()
    for scenario in args.scenarios:
        spec = SCENARIOS[scenario]
        for reward in args.reward_arms:
            for algo in args.algos:
                for track in _track_for(algo, args.ppo_track):
                    reward_tag = {"stock": "stock", "risk_only": "risk", "social_full": "social"}[reward]
                    track_tag = "cont" if track == "continuous" else "disc"
                    run_name = f"{args.tag}_{scenario}_{reward_tag}_{algo}_{track_tag}_{label}"
                    if run_name in seen:
                        continue
                    seen.add(run_name)
                    runs.append(
                        {
                            "scenario": scenario,
                            "train_env": spec["train_env"],
                            "eval_envs": list(spec["eval_envs"]),
                            "traffic": spec["traffic"],
                            "reward": reward,
                            "algo": algo,
                            "track": track,
                            "run_name": run_name,
                        }
                    )
    return runs


def _command(run: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "rl.train_highwayenv_social_sb3",
        "--algo",
        run["algo"],
        "--env-id",
        run["train_env"],
        "--eval-env-id",
        *run["eval_envs"],
        "--action-mode",
        _action_mode(run["track"]),
        "--ablation",
        _ablation(run["reward"]),
        "--traffic-preset",
        run["traffic"],
        "--total-steps",
        str(int(args.steps)),
        "--eval-freq",
        str(int(args.eval_freq)),
        "--eval-episodes",
        str(int(args.eval_episodes)),
        "--n-envs",
        str(int(args.n_envs)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--run-dir",
        str(_run_dir(run["run_name"])),
    ]
    if args.extra:
        cmd += list(args.extra)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=["highway"], choices=sorted(SCENARIOS))
    parser.add_argument("--algos", nargs="+", default=list(ALL_ALGOS), choices=ALL_ALGOS)
    parser.add_argument("--reward-arms", nargs="+", default=list(REWARD_ARMS), choices=REWARD_ARMS)
    parser.add_argument(
        "--ppo-track",
        choices=("discrete", "continuous", "both"),
        default="discrete",
        help="Default matches the paper matrix: PPO/DQN discrete, SAC/TD3/DDPG continuous.",
    )
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="highwaybench")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    runs = _build_runs(args)
    pending, skipped = [], []
    for run in runs:
        if _final_checkpoint(run["run_name"]).exists() and not args.overwrite:
            skipped.append(run)
        else:
            pending.append(run)

    print(
        f"Planned {len(runs)} HighwayEnv runs "
        f"(scenarios={args.scenarios}, algos={args.algos}, rewards={args.reward_arms}, "
        f"steps={args.steps}, device={args.device}, tag={args.tag})\n"
    )
    for run in runs:
        mark = "SKIP (final_model.zip exists)" if run in skipped else "RUN "
        print(
            f"  [{mark}] {run['run_name']:<48} <- "
            f"{run['train_env']} ({run['algo']}/{run['track']}/{run['reward']})"
        )
    print(f"\n{len(pending)} to run, {len(skipped)} already done.\n")

    if args.dry_run:
        print("--- commands (dry run) ---")
        for run in pending:
            print("  " + " ".join(_command(run, args)))
        return 0

    failures: list[tuple[str, int]] = []
    for idx, run in enumerate(pending, 1):
        cmd = _command(run, args)
        print("=" * 72)
        print(f"[{idx}/{len(pending)}] {run['run_name']}")
        print("  " + " ".join(cmd))
        print("=" * 72)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if _final_checkpoint(run["run_name"]).exists():
            print(f"[OK ] {run['run_name']}")
            continue
        failures.append((run["run_name"], proc.returncode))
        print(f"[FAIL] {run['run_name']} exited {proc.returncode} with no final_model.zip")
        if not args.continue_on_error:
            break

    print("\n" + "=" * 72)
    print(f"Done. {len(pending) - len(failures)}/{len(pending)} succeeded, {len(skipped)} skipped, {len(failures)} failed.")
    for name, rc in failures:
        print(f"  FAILED ({rc}): {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
