from __future__ import annotations

import argparse
import json
import os
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


def _apply_style() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "grid", "no-latex"])
    else:
        plt.style.use("default")


def _discover_run_dirs(root: Path) -> list[Path]:
    if (root / "summary.json").exists():
        return [root]
    return sorted(path.parent for path in root.glob("*/summary.json"))


def _load_run(run_dir: Path) -> dict:
    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    records = summary.get("records", [])
    config = summary.get("config", {})
    label = f"{config.get('algo', 'run').upper()}-{config.get('ablation', 'full')}-{config.get('interface', 'stock')}"
    return {"run_dir": run_dir, "summary": summary, "records": records, "label": label}


def _save(fig, path_base: Path) -> None:
    fig.savefig(f"{path_base}.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def _eval_env_ids(runs: list[dict]) -> list[str]:
    env_ids: list[str] = []
    for run in runs:
        summary = run.get("summary", {})
        final_eval = summary.get("final_eval", {})
        candidates = list(final_eval.keys())
        if not candidates:
            config = summary.get("config", {})
            cfg_envs = config.get("eval_env_id", [])
            if isinstance(cfg_envs, list):
                candidates = [str(x) for x in cfg_envs]
            elif cfg_envs:
                candidates = [str(cfg_envs)]
        for env_id in candidates:
            if env_id not in env_ids:
                env_ids.append(env_id)
    return env_ids or ["highway-v0", "merge-v0"]


def _metric_key(env_id: str, suffix: str) -> str:
    return f"{env_id.replace('-', '_')}_{suffix}"


def _env_title(env_id: str) -> str:
    return str(env_id).replace("-v0", "").replace("-", " ").title()


def _plot_learning_curves(runs: list[dict], out_dir: Path) -> None:
    env_ids = _eval_env_ids(runs)
    fig, axes = plt.subplots(1, len(env_ids), figsize=(6.0 * len(env_ids), 4.8), squeeze=False)
    axes = axes.ravel()
    for run in runs:
        records = run["records"]
        if not records:
            continue
        xs = np.asarray([row["timesteps"] for row in records], dtype=float)
        for ax, env_id in zip(axes, env_ids):
            key = _metric_key(env_id, "return_mean")
            ys = np.asarray([row.get(key, np.nan) for row in records], dtype=float)
            ax.plot(xs, ys, label=run["label"], linewidth=2.0)
            ax.set_title(f"{_env_title(env_id)} Return")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Return")
    for ax in axes:
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Social-RL Learning Curves")
    _save(fig, out_dir / "learning_curves")


def _plot_reward_components(runs: list[dict], out_dir: Path) -> None:
    component_keys = [
        "train_reward_total_mean",
        "train_r_prog_safe_mean",
        "train_c_ttc_mean",
        "train_c_social_mean",
        "train_c_field_mean",
    ]
    titles = {
        "train_reward_total_mean": "Total Reward",
        "train_r_prog_safe_mean": "Risk-Gated Progress",
        "train_c_ttc_mean": "TTC Cost",
        "train_c_social_mean": "Courtesy Cost",
        "train_c_field_mean": "Field Cost",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for idx, key in enumerate(component_keys):
        ax = axes[idx]
        for run in runs:
            records = run["records"]
            if not records:
                continue
            xs = np.asarray([row["timesteps"] for row in records], dtype=float)
            ys = np.asarray([row.get(key, np.nan) for row in records], dtype=float)
            ax.plot(xs, ys, label=run["label"], linewidth=2.0)
        ax.set_title(titles[key])
        ax.set_xlabel("Timesteps")
    axes[-1].axis("off")
    axes[0].legend(frameon=True, fontsize=8)
    fig.suptitle("Reward Components")
    _save(fig, out_dir / "reward_components")


def _plot_safety_courtesy(runs: list[dict], out_dir: Path) -> None:
    env_ids = _eval_env_ids(runs)
    fig, axes = plt.subplots(2, len(env_ids), figsize=(6.0 * len(env_ids), 8.0), squeeze=False)
    for col, env_id in enumerate(env_ids):
        collision_key = _metric_key(env_id, "collision_rate")
        ttc_key = _metric_key(env_id, "ttc_min_mean")
        for run in runs:
            records = run["records"]
            if not records:
                continue
            xs = np.asarray([row["timesteps"] for row in records], dtype=float)
            ys_collision = np.asarray([row.get(collision_key, np.nan) for row in records], dtype=float)
            ys_ttc = np.asarray([row.get(ttc_key, np.nan) for row in records], dtype=float)
            axes[0, col].plot(xs, ys_collision, label=run["label"], linewidth=2.0)
            axes[1, col].plot(xs, ys_ttc, label=run["label"], linewidth=2.0)
        axes[0, col].set_title(f"{_env_title(env_id)} Collision Rate")
        axes[0, col].set_xlabel("Timesteps")
        axes[1, col].set_title(f"{_env_title(env_id)} TTC Min")
        axes[1, col].set_xlabel("Timesteps")
    axes[0, 0].legend(frameon=True, fontsize=8)
    fig.suptitle("Safety and Courtesy Metrics")
    _save(fig, out_dir / "safety_courtesy")


def _plot_final_eval_bars(runs: list[dict], out_dir: Path) -> None:
    env_ids = _eval_env_ids(runs)
    metrics = [
        ("return_mean", "Return"),
        ("collision_rate", "Collision Rate"),
        ("ttc_min_mean", "TTC Min"),
        ("imposed_rear_decel_max_mean", "Rear Decel Max"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    width = 0.8 / max(1, len(env_ids))
    for ax, (metric_key, title) in zip(axes, metrics):
        labels = []
        env_vals = {env_id: [] for env_id in env_ids}
        for run in runs:
            labels.append(run["label"])
            final_eval = run["summary"].get("final_eval", {})
            for env_id in env_ids:
                env_vals[env_id].append(float(final_eval.get(env_id, {}).get(metric_key, np.nan)))
        x = np.arange(len(labels), dtype=float)
        offsets = np.linspace(-(len(env_ids) - 1) * width / 2, (len(env_ids) - 1) * width / 2, len(env_ids))
        for offset, env_id in zip(offsets, env_ids):
            ax.bar(x + offset, env_vals[env_id], width=width, label=env_id)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
    axes[0].legend(frameon=True, fontsize=8)
    fig.suptitle("Final Evaluation Comparison")
    _save(fig, out_dir / "final_eval_bars")


def _has_finite(records: list[dict], key: str) -> bool:
    for row in records:
        try:
            if np.isfinite(float(row.get(key, np.nan))):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _plot_comfort_latency(runs: list[dict], out_dir: Path) -> None:
    env_ids = _eval_env_ids(runs)
    metrics = [
        ("mean_abs_accel", "Abs. Acceleration"),
        ("mean_jerk_abs", "Abs. Jerk"),
        ("action_change_rate", "Action Change"),
        ("mean_action_selection_ms", "Decision Time (ms)"),
    ]
    fig, axes = plt.subplots(len(metrics), len(env_ids), figsize=(6.0 * len(env_ids), 3.0 * len(metrics)), squeeze=False)
    plotted = False
    for row_idx, (suffix, title) in enumerate(metrics):
        for col_idx, env_id in enumerate(env_ids):
            ax = axes[row_idx, col_idx]
            key = _metric_key(env_id, suffix)
            for run in runs:
                records = run["records"]
                if not records or not _has_finite(records, key):
                    continue
                xs = np.asarray([record["timesteps"] for record in records], dtype=float)
                ys = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
                ax.plot(xs, ys, label=run["label"], linewidth=2.0)
                plotted = True
            if row_idx == 0:
                ax.set_title(_env_title(env_id))
            if col_idx == 0:
                ax.set_ylabel(title)
            ax.set_xlabel("Timesteps")
    if not plotted:
        plt.close(fig)
        return
    axes[0, 0].legend(frameon=True, fontsize=8)
    fig.suptitle("Comfort and Inference Metrics")
    _save(fig, out_dir / "comfort_latency")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SciencePlots figures for social HighwayEnv SB3 runs.")
    parser.add_argument("--run-dir", required=True, help="Single run directory or a parent directory containing multiple run subdirectories.")
    args = parser.parse_args()

    _apply_style()
    root = Path(args.run_dir)
    runs = [_load_run(path) for path in _discover_run_dirs(root)]
    if not runs:
        raise FileNotFoundError(f"No summary.json found under {root}")
    out_dir = root if (root / "summary.json").exists() else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_learning_curves(runs, out_dir)
    _plot_reward_components(runs, out_dir)
    _plot_safety_courtesy(runs, out_dir)
    _plot_final_eval_bars(runs, out_dir)
    _plot_comfort_latency(runs, out_dir)
    print(json.dumps({"run_count": len(runs), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
