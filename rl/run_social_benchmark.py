"""
One-step social-friendly benchmark driver for MetaDrive.
========================================================
Trains the whole matrix ``{stock, social_full} x {PPO, DQN, SAC, TD3, DDPG}``
per scenario with a *single* command, on CUDA, into a **fresh checkpoint
namespace** so the existing ``matched_*_respawn_*_1m`` (risk_only) checkpoints
are preserved as the "without steering/braking" ablation.

Why a driver
------------
* Avoids retyping ~10 trainer commands per scenario.
* Routes each algorithm to its required action space automatically:
  DQN -> discrete (`*_respawn`); SAC/TD3/DDPG -> continuous
  (`*_respawn_continuous`); PPO -> continuous by default (``--ppo-track`` to
  also/instead train the discrete PPO that pairs with DQN).
* Resumable: a run whose ``final.zip`` already exists is skipped unless
  ``--overwrite`` is given, so old checkpoints are never clobbered and an
  interrupted sweep can be re-run cheaply.

Run names follow ``<tag>_<scenario>_<stock|social>_<algo>_<stepslabel>`` (default
tag ``socialbench``), e.g. ``socialbench_intersection_social_sac_1m``. These land
in their own ``rl/checkpoints/metadrive/<run_name>/`` folders, disjoint from the
older ``matched_*`` runs.

Examples
--------
    # Full matrix on the intersection scenario, 1M steps, CUDA
    python -m rl.run_social_benchmark --scenarios intersection

    # Three scenarios, also include the discrete PPO that pairs with DQN
    python -m rl.run_social_benchmark --scenarios intersection merge roundabout --ppo-track both

    # Preview the commands without running anything
    python -m rl.run_social_benchmark --scenarios intersection --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.config.metadrive_config import METADRIVE_PROTOCOLS

ALL_ALGOS = ("ppo", "dqn", "sac", "td3", "ddpg")
CONTINUOUS_ONLY = {"sac", "td3", "ddpg"}
DISCRETE_ONLY = {"dqn"}
REWARDS = ("stock", "social_full")
SCENARIOS = ("straight", "curve", "merge", "intersection", "roundabout", "mixed")


def _steps_label(steps: int) -> str:
    if steps % 1_000_000 == 0:
        return f"{steps // 1_000_000}m"
    if steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    return str(steps)


def _track_for(algo: str, ppo_track: str) -> list[str]:
    """Return the action-space track(s) an algorithm should train on."""
    if algo in DISCRETE_ONLY:
        return ["discrete"]
    if algo in CONTINUOUS_ONLY:
        return ["continuous"]
    # PPO bridges both.
    if ppo_track == "both":
        return ["continuous", "discrete"]
    return [ppo_track]


def _protocol(scenario: str, reward: str, track: str) -> str:
    base = "matched_stock" if reward == "stock" else "matched_social_risk"
    proto = f"{base}_{scenario}_respawn"
    if track == "continuous":
        proto += "_continuous"
    return proto


def _build_runs(args: argparse.Namespace) -> list[dict]:
    label = _steps_label(int(args.steps))
    runs: list[dict] = []
    seen: set[str] = set()
    for scenario in args.scenarios:
        for reward in args.reward_profiles:
            for algo in args.algos:
                for track in _track_for(algo, args.ppo_track):
                    proto = _protocol(scenario, reward, track)
                    if proto not in METADRIVE_PROTOCOLS:
                        raise SystemExit(f"Unknown protocol '{proto}' (scenario={scenario}).")
                    reward_tag = "stock" if reward == "stock" else "social"
                    # Disambiguate the two PPO tracks in the run name.
                    track_tag = ""
                    if algo == "ppo" and args.ppo_track == "both":
                        track_tag = "_cont" if track == "continuous" else "_disc"
                    run_name = f"{args.tag}_{scenario}_{reward_tag}_{algo}{track_tag}_{label}"
                    if run_name in seen:
                        continue
                    seen.add(run_name)
                    runs.append({
                        "scenario": scenario, "reward": reward, "algo": algo,
                        "track": track, "protocol": proto, "run_name": run_name,
                    })
    return runs


def _final_ckpt(run_name: str) -> Path:
    return REPO_ROOT / "rl" / "checkpoints" / "metadrive" / run_name / "final.zip"


def _command(run: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "-m", "rl.train_metadrive_sb3",
        "--protocol", run["protocol"],
        "--algo", run["algo"],
        "--steps", str(int(args.steps)),
        "--n-envs", str(int(args.n_envs)),
        "--device", str(args.device),
        "--seed", str(int(args.seed)),
        "--run-name", run["run_name"],
    ]
    if run["reward"] == "social_full":
        cmd += ["--reward-profile", "social_full"]
    if args.overwrite:
        cmd += ["--allow-overwrite"]
    if args.extra:
        cmd += list(args.extra)
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=["intersection"], choices=SCENARIOS)
    p.add_argument("--algos", nargs="+", default=list(ALL_ALGOS), choices=ALL_ALGOS)
    p.add_argument("--reward-profiles", nargs="+", default=list(REWARDS), choices=REWARDS,
                   help="Which reward arms to train (stock baseline and/or social_full).")
    p.add_argument("--ppo-track", choices=("continuous", "discrete", "both"), default="continuous",
                   help="Action space for PPO. 'both' also trains the discrete PPO that pairs with DQN.")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="socialbench",
                   help="Run-name prefix / checkpoint namespace (kept disjoint from old runs).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-train and overwrite runs in THIS namespace (old matched_* runs are never touched).")
    p.add_argument("--dry-run", action="store_true", help="Print the planned commands and exit.")
    p.add_argument("--continue-on-error", action="store_true", default=True,
                   help="Keep going if a run fails (default).")
    p.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is forwarded verbatim to the trainer.")
    args = p.parse_args()

    runs = _build_runs(args)
    print(f"Planned {len(runs)} runs "
          f"(scenarios={args.scenarios}, algos={args.algos}, "
          f"rewards={args.reward_profiles}, ppo-track={args.ppo_track}, "
          f"steps={args.steps}, device={args.device}, tag={args.tag})\n")

    pending, skipped = [], []
    for r in runs:
        if _final_ckpt(r["run_name"]).exists() and not args.overwrite:
            skipped.append(r)
        else:
            pending.append(r)

    for r in runs:
        mark = "SKIP (final.zip exists)" if r in skipped else "RUN "
        print(f"  [{mark}] {r['run_name']:<48} <- {r['protocol']} ({r['algo']}/{r['track']})")
    print(f"\n{len(pending)} to run, {len(skipped)} already done.\n")

    if args.dry_run:
        print("--- commands (dry run) ---")
        for r in pending:
            print("  " + " ".join(_command(r, args)))
        return 0

    failures: list[tuple[str, int]] = []
    for i, r in enumerate(pending, 1):
        cmd = _command(r, args)
        print("=" * 72)
        print(f"[{i}/{len(pending)}] {r['run_name']}")
        print("  " + " ".join(cmd))
        print("=" * 72)
        # cwd=REPO_ROOT + `-m` makes the repo root sys.path[0] so `import config`
        # resolves to top-level config.py (not rl/config/). Do NOT inject
        # PYTHONPATH here; the trainer manages it for SubprocVecEnv workers.
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        # Judge success by the checkpoint, not the exit code: MetaDrive/Panda3D
        # + CUDA can segfault during interpreter teardown on Windows (exit
        # 0xC0000005 / 0xC0000374) *after* training and model.save() complete.
        if _final_ckpt(r["run_name"]).exists():
            if proc.returncode != 0:
                print(f"[OK*] {r['run_name']} trained and saved final.zip; "
                      f"ignoring teardown exit code {proc.returncode}.")
            else:
                print(f"[OK ] {r['run_name']}")
        else:
            failures.append((r["run_name"], proc.returncode))
            print(f"[FAIL] {r['run_name']} exited {proc.returncode} with no final.zip")
            if not args.continue_on_error:
                break

    print("\n" + "=" * 72)
    print(f"Done. {len(pending) - len(failures)}/{len(pending)} succeeded, "
          f"{len(skipped)} skipped, {len(failures)} failed.")
    for name, rc in failures:
        print(f"  FAILED ({rc}): {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
