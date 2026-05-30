from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401

    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False


EXPECTED_RUNS = [
    {
        "key": "highway",
        "train_env": "highway-fast-v0",
        "eval_envs": ["highway-v0", "merge-v0"],
        "runs": {
            "ppo": Path("rl/logs/social_ppo_a5"),
            "dqn": Path("rl/logs/social_dqn_a5"),
        },
    },
    {
        "key": "roundabout",
        "train_env": "roundabout-v0",
        "eval_envs": ["roundabout-v0"],
        "runs": {
            "ppo": Path("rl/logs/social_ppo_a5_roundabout"),
            "dqn": Path("rl/logs/social_dqn_a5_roundabout"),
        },
    },
    {
        "key": "intersection",
        "train_env": "intersection-v0",
        "eval_envs": ["intersection-v0"],
        "runs": {
            "ppo": Path("rl/logs/social_ppo_a5_intersection"),
            "dqn": Path("rl/logs/social_dqn_a5_intersection"),
        },
    },
]

STOCK_CURVE_RUNS = [
    {
        "key": "highway",
        "env_id": "highway-v0",
        "runs": {
            "ppo": Path("rl/logs/highway_v0_curve_compare/ppo_curve.json"),
            "dqn": Path("rl/logs/highway_v0_curve_compare/dqn_curve.json"),
        },
    },
    {
        "key": "roundabout",
        "env_id": "roundabout-v0",
        "runs": {
            "ppo": Path("rl/logs/roundabout_stock_curve_compare/ppo_curve.json"),
            "dqn": Path("rl/logs/roundabout_stock_curve_compare/dqn_curve.json"),
        },
    },
    {
        "key": "intersection",
        "env_id": "intersection-v0",
        "runs": {
            "ppo": Path("rl/logs/intersection_stock_curve_compare/ppo_curve.json"),
            "dqn": Path("rl/logs/intersection_stock_curve_compare/dqn_curve.json"),
        },
    },
]


def _apply_style() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "grid", "no-latex"])
    else:
        plt.style.use("default")


def _save(fig, path_base: Path) -> None:
    fig.savefig(f"{path_base}.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def _exists_run(run_dir: Path) -> bool:
    return _run_summary_path(run_dir).exists()


def _load_run(run_dir: Path) -> dict:
    with open(_run_summary_path(run_dir), "r", encoding="utf-8") as f:
        summary = json.load(f)
    return {
        "run_dir": run_dir,
        "summary": summary,
        "records": summary.get("records", []),
        "config": summary.get("config", {}),
        "algo": str(summary.get("config", {}).get("algo", "")).lower(),
    }


def _metric_key(env_id: str, suffix: str) -> str:
    return f"{env_id.replace('-', '_')}_{suffix}"


def _scenario_title(env_id: str) -> str:
    return str(env_id).replace("-v0", "").replace("-", " ").title()


def _inventory_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in EXPECTED_RUNS:
        for algo, run_dir in spec["runs"].items():
            status = "trained" if _exists_run(run_dir) else "missing"
            checkpoint = ""
            summary = ""
            if status == "trained":
                ckpt = run_dir / "checkpoints" / "best_model.zip"
                checkpoint = str(ckpt) if ckpt.exists() else ""
                summary = str(_run_summary_path(run_dir))
            rows.append(
                {
                    "scenario": spec["key"],
                    "train_env": spec["train_env"],
                    "algo": algo.upper(),
                    "status": status,
                    "run_dir": str(run_dir),
                    "summary_json": summary,
                    "best_checkpoint": checkpoint,
                }
            )
    return rows


def _save_inventory(out_dir: Path, rows: list[dict[str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model_inventory.csv"
    md_path = out_dir / "model_inventory.md"
    fields = ["scenario", "train_env", "algo", "status", "run_dir", "summary_json", "best_checkpoint"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Social-RL Model Inventory",
        "",
        "| scenario | train_env | algo | status | best_checkpoint |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['train_env']} | {row['algo']} | {row['status']} | {row['best_checkpoint'] or '-'} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _stock_inventory_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in STOCK_CURVE_RUNS:
        for algo, curve_path in spec["runs"].items():
            rows.append(
                {
                    "scenario": spec["key"],
                    "env_id": spec["env_id"],
                    "algo": algo.upper(),
                    "status": "trained" if curve_path.exists() else "missing",
                    "curve_json": str(curve_path) if curve_path.exists() else "",
                }
            )
    return rows


def _save_stock_inventory(out_dir: Path, rows: list[dict[str, str]]) -> None:
    csv_path = out_dir / "stock_curve_inventory.csv"
    md_path = out_dir / "stock_curve_inventory.md"
    fields = ["scenario", "env_id", "algo", "status", "curve_json"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Stock RL Curve Inventory",
        "",
        "| scenario | env_id | algo | status | curve_json |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['env_id']} | {row['algo']} | {row['status']} | {row['curve_json'] or '-'} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _load_available_runs() -> tuple[dict[str, list[dict]], list[str]]:
    by_env: dict[str, list[dict]] = {}
    env_order: list[str] = []
    for spec in EXPECTED_RUNS:
        for algo, run_dir in spec["runs"].items():
            if not _exists_run(run_dir):
                continue
            run = _load_run(run_dir)
            for env_id in spec["eval_envs"]:
                by_env.setdefault(env_id, []).append(run)
                if env_id not in env_order:
                    env_order.append(env_id)
    return by_env, env_order


def _load_runs_from_root(root: Path) -> tuple[dict[str, list[dict]], list[str], list[dict[str, str]]]:
    by_env: dict[str, list[dict]] = {}
    env_order: list[str] = []
    inventory: list[dict[str, str]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        run_dir = summary_path.parent
        run = _load_run(run_dir)
        summary = run.get("summary", {})
        config = summary.get("config", {})
        final_eval = summary.get("final_eval", {})
        env_ids = list(final_eval.keys())
        if not env_ids:
            cfg_envs = config.get("eval_env_id", [])
            if isinstance(cfg_envs, list):
                env_ids = [str(x) for x in cfg_envs]
            elif cfg_envs:
                env_ids = [str(cfg_envs)]
        for env_id in env_ids:
            by_env.setdefault(env_id, []).append(run)
            if env_id not in env_order:
                env_order.append(env_id)
        inventory.append(
            {
                "scenario": str(config.get("env_id", "")),
                "train_env": str(config.get("env_id", "")),
                "algo": str(config.get("algo", "")).upper(),
                "status": "trained",
                "run_dir": str(run_dir),
                "summary_json": str(summary_path),
                "best_checkpoint": str(run_dir / "checkpoints" / "best_model.zip"),
            }
        )
    return by_env, env_order, inventory


def _load_stock_runs() -> dict[str, list[dict]]:
    by_env: dict[str, list[dict]] = {}
    for spec in STOCK_CURVE_RUNS:
        env_id = spec["env_id"]
        for algo, curve_path in spec["runs"].items():
            if not curve_path.exists():
                continue
            with open(curve_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            by_env.setdefault(env_id, []).append(
                {
                    "algo": str(algo).lower(),
                    "kind": "stock",
                    "records": records,
                    "label": f"Stock {str(algo).upper()}",
                }
            )
    return by_env


def _algo_label(run: dict) -> str:
    if run.get("kind") == "stock":
        return str(run.get("label", "Stock"))
    algo = str(run.get("algo", "")).upper() or "RUN"
    config = run.get("config", {})
    ablation = str(config.get("ablation", "full"))
    return f"{algo}-{ablation}"


def _plot_eval_reward(by_env: dict[str, list[dict]], env_order: list[str], out_dir: Path) -> None:
    if not env_order:
        return
    fig, axes = plt.subplots(1, len(env_order), figsize=(6.4 * len(env_order), 4.8), squeeze=False)
    axes = axes.ravel()
    for ax, env_id in zip(axes, env_order):
        for run in by_env.get(env_id, []):
            records = run.get("records", [])
            if not records:
                continue
            xs = np.asarray([row["timesteps"] for row in records], dtype=float)
            ys = np.asarray([row.get(_metric_key(env_id, "return_mean"), np.nan) for row in records], dtype=float)
            ax.plot(xs, ys, linewidth=2.0, label=_algo_label(run))
        ax.set_title(f"{_scenario_title(env_id)} Reward")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Avg Reward")
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Social-RL Episode Average Reward Across Scenarios")
    _save(fig, out_dir / "overview_eval_reward")


def _plot_eval_reward_with_stock(
    social_by_env: dict[str, list[dict]],
    stock_by_env: dict[str, list[dict]],
    env_order: list[str],
    out_dir: Path,
) -> None:
    available_envs = [env_id for env_id in env_order if social_by_env.get(env_id) or stock_by_env.get(env_id)]
    if not available_envs:
        return
    fig, axes = plt.subplots(1, len(available_envs), figsize=(6.8 * len(available_envs), 4.8), squeeze=False)
    axes = axes.ravel()
    style_map = {
        "Stock PPO": {"color": "#7f7f7f", "linestyle": "--"},
        "Stock DQN": {"color": "#9467bd", "linestyle": "--"},
    }
    for ax, env_id in zip(axes, available_envs):
        plotted_any = False
        for run in social_by_env.get(env_id, []):
            records = run.get("records", [])
            if not records:
                continue
            xs = np.asarray([row["timesteps"] for row in records], dtype=float)
            ys = np.asarray([row.get(_metric_key(env_id, "return_mean"), np.nan) for row in records], dtype=float)
            ax.plot(xs, ys, linewidth=2.0, label=_algo_label(run))
            plotted_any = True
        for run in stock_by_env.get(env_id, []):
            records = run.get("records", [])
            if not records:
                continue
            xs = np.asarray([row["timesteps"] for row in records], dtype=float)
            ys = np.asarray([row.get("return_mean", np.nan) for row in records], dtype=float)
            style = style_map.get(_algo_label(run), {})
            ax.plot(
                xs,
                ys,
                linewidth=2.0,
                label=_algo_label(run),
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                alpha=0.95,
            )
            plotted_any = True
        if plotted_any:
            ax.legend(frameon=True, fontsize=8)
        ax.set_title(_scenario_title(env_id))
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Avg Reward")
    fig.suptitle("Episode Reward Curves: Social-RL And Stock PPO/DQN")
    _save(fig, out_dir / "overview_eval_reward_with_stock")


def _plot_objective_metrics(by_env: dict[str, list[dict]], env_order: list[str], out_dir: Path) -> None:
    if not env_order:
        return
    metric_specs = [
        ("corridor_risk_mean", "Corridor Risk", "lower"),
        ("imposed_rear_decel_max_mean", "Rear Decel", "lower"),
        ("collision_rate", "Collision Rate", "lower"),
        ("ttc_min_mean", "TTC Min", "higher"),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(env_order), figsize=(6.2 * len(env_order), 3.2 * len(metric_specs)), squeeze=False)
    for row_idx, (suffix, title, _better) in enumerate(metric_specs):
        for col_idx, env_id in enumerate(env_order):
            ax = axes[row_idx, col_idx]
            for run in by_env.get(env_id, []):
                records = run.get("records", [])
                if not records:
                    continue
                xs = np.asarray([record["timesteps"] for record in records], dtype=float)
                ys = np.asarray([record.get(_metric_key(env_id, suffix), np.nan) for record in records], dtype=float)
                ax.plot(xs, ys, linewidth=2.0, label=_algo_label(run))
            if row_idx == 0:
                ax.set_title(_scenario_title(env_id))
            if col_idx == 0:
                ax.set_ylabel(title)
            ax.set_xlabel("Timesteps")
    axes[0, 0].legend(frameon=True, fontsize=8)
    fig.suptitle("Risk And Social Objective Metrics Across Scenarios")
    _save(fig, out_dir / "overview_objective_metrics")


def _plot_reward_components(by_env: dict[str, list[dict]], env_order: list[str], out_dir: Path) -> None:
    if not env_order:
        return
    component_specs = [
        ("train_reward_total_mean", "Train Total Reward"),
        ("train_r_prog_safe_mean", "Risk-Gated Progress"),
        ("train_c_ttc_mean", "TTC Cost"),
        ("train_c_social_mean", "Courtesy Cost"),
        ("train_c_field_mean", "Field Cost"),
    ]
    fig, axes = plt.subplots(len(component_specs), len(env_order), figsize=(6.2 * len(env_order), 3.0 * len(component_specs)), squeeze=False)
    for row_idx, (key, title) in enumerate(component_specs):
        for col_idx, env_id in enumerate(env_order):
            ax = axes[row_idx, col_idx]
            for run in by_env.get(env_id, []):
                records = run.get("records", [])
                if not records:
                    continue
                xs = np.asarray([record["timesteps"] for record in records], dtype=float)
                ys = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
                ax.plot(xs, ys, linewidth=2.0, label=_algo_label(run))
            if row_idx == 0:
                ax.set_title(_scenario_title(env_id))
            if col_idx == 0:
                ax.set_ylabel(title)
            ax.set_xlabel("Timesteps")
    axes[0, 0].legend(frameon=True, fontsize=8)
    fig.suptitle("Reward-Design Components Across Scenarios")
    _save(fig, out_dir / "overview_reward_components")


def _plot_final_bars(by_env: dict[str, list[dict]], env_order: list[str], out_dir: Path) -> None:
    if not env_order:
        return
    metric_specs = [
        ("return_mean", "Return"),
        ("corridor_risk_mean", "Corridor Risk"),
        ("collision_rate", "Collision Rate"),
        ("imposed_rear_decel_max_mean", "Rear Decel"),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(env_order), figsize=(5.6 * len(env_order), 3.0 * len(metric_specs)), squeeze=False)
    for row_idx, (suffix, title) in enumerate(metric_specs):
        for col_idx, env_id in enumerate(env_order):
            ax = axes[row_idx, col_idx]
            runs = by_env.get(env_id, [])
            labels = []
            values = []
            for run in runs:
                labels.append(_algo_label(run))
                final_eval = run.get("summary", {}).get("final_eval", {})
                values.append(float(final_eval.get(env_id, {}).get(suffix, np.nan)))
            x = np.arange(len(labels), dtype=float)
            ax.bar(x, values, width=0.65)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            if row_idx == 0:
                ax.set_title(_scenario_title(env_id))
            if col_idx == 0:
                ax.set_ylabel(title)
    fig.suptitle("Final Evaluation Summary Across Scenarios")
    _save(fig, out_dir / "overview_final_eval")


def _plot_comfort_latency(by_env: dict[str, list[dict]], env_order: list[str], out_dir: Path) -> None:
    if not env_order:
        return
    metric_specs = [
        ("mean_abs_accel", "Abs. Acceleration"),
        ("mean_jerk_abs", "Abs. Jerk"),
        ("action_change_rate", "Action Change"),
        ("mean_action_selection_ms", "Decision Time (ms)"),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(env_order), figsize=(6.2 * len(env_order), 3.0 * len(metric_specs)), squeeze=False)
    plotted = False
    for row_idx, (suffix, title) in enumerate(metric_specs):
        for col_idx, env_id in enumerate(env_order):
            ax = axes[row_idx, col_idx]
            key = _metric_key(env_id, suffix)
            for run in by_env.get(env_id, []):
                records = run.get("records", [])
                if not records:
                    continue
                xs = np.asarray([record["timesteps"] for record in records], dtype=float)
                ys = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
                if not np.any(np.isfinite(ys)):
                    continue
                ax.plot(xs, ys, linewidth=2.0, label=_algo_label(run))
                plotted = True
            if row_idx == 0:
                ax.set_title(_scenario_title(env_id))
            if col_idx == 0:
                ax.set_ylabel(title)
            ax.set_xlabel("Timesteps")
    if not plotted:
        plt.close(fig)
        return
    axes[0, 0].legend(frameon=True, fontsize=8)
    fig.suptitle("Comfort And Inference Metrics Across Scenarios")
    _save(fig, out_dir / "overview_comfort_latency")


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform overview plots for social-RL runs across envs and algorithms.")
    parser.add_argument("--out-dir", default="rl/logs/social_overview", help="Directory to write the overview plots and model inventory.")
    parser.add_argument(
        "--run-root",
        default="",
        help="Optional parent directory containing run subdirectories with summary.json, e.g. rl/logs/highwayenv.",
    )
    args = parser.parse_args()

    _apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_root:
        by_env, env_order, inventory = _load_runs_from_root(Path(args.run_root))
    else:
        inventory = _inventory_rows()
        by_env, env_order = _load_available_runs()
    _save_inventory(out_dir, inventory)
    stock_inventory = _stock_inventory_rows()
    _save_stock_inventory(out_dir, stock_inventory)
    stock_by_env = _load_stock_runs()
    _plot_eval_reward(by_env, env_order, out_dir)
    _plot_eval_reward_with_stock(by_env, stock_by_env, env_order, out_dir)
    _plot_objective_metrics(by_env, env_order, out_dir)
    _plot_reward_components(by_env, env_order, out_dir)
    _plot_final_bars(by_env, env_order, out_dir)
    _plot_comfort_latency(by_env, env_order, out_dir)

    summary = {
        "out_dir": str(out_dir),
        "scenarios_found": env_order,
        "trained_models": [row for row in inventory if row["status"] == "trained"],
        "missing_models": [row for row in inventory if row["status"] == "missing"],
        "trained_stock_curves": [row for row in stock_inventory if row["status"] == "trained"],
        "missing_stock_curves": [row for row in stock_inventory if row["status"] == "missing"],
    }
    with open(out_dir / "overview_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
