"""
Unified RL training reward/return plotter for HighwayEnv and MetaDrive.

The script normalizes several local log formats into:

    timesteps, avg_reward, episode_return

Supported inputs:
    - MetaDrive SB3 `progress.csv`
    - HighwayEnv social-RL `summary.json`
    - HighwayEnv baseline curve JSON files
    - Generic CSV/JSON logs with common reward/return columns

Usage
-----
python rl/plot_rl_training_reward_return.py ^
  --runs "MetaDrive Stock PPO=rl/logs/metadrive/matched_stock_ppo/progress.csv" ^
         "MetaDrive Risk DQN=rl/logs/metadrive/matched_social_risk_intersection_respawn_dqn_1m/progress.csv" ^
         "HighwayEnv Social PPO=rl/logs/social_ppo_a5/summary.json" ^
  --out rl/logs/figures/training_reward_return
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401

    _HAS_SCIENCEPLOTS = True
except ImportError:  # pragma: no cover
    _HAS_SCIENCEPLOTS = False


COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#000000",
]


def _apply_style() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "grid", "no-latex"])
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def _to_float(value, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _first_finite(row: dict, keys: Iterable[str]) -> float:
    for key in keys:
        val = _to_float(row.get(key))
        if math.isfinite(val):
            return val
    return float("nan")


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or window <= 1:
        return values
    out = np.empty_like(values, dtype=float)
    for idx in range(values.size):
        lo = max(0, idx - window + 1)
        chunk = values[lo : idx + 1]
        out[idx] = float(np.nanmean(chunk)) if np.any(np.isfinite(chunk)) else float("nan")
    return out


def _infer_family(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "/metadrive/" in text or "metadrive" in text:
        return "MetaDrive"
    if "highway" in text or "merge_v0" in text or "roundabout" in text or "intersection" in text:
        return "HighwayEnv"
    return "RL"


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec.strip())
    return path.parent.name if path.name in {"progress.csv", "summary.json"} else path.stem, path


def _load_json_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        records = data.get("records")
        if isinstance(records, list):
            return [dict(row) for row in records if isinstance(row, dict)]
        curve = data.get("curve")
        if isinstance(curve, list):
            return [dict(row) for row in curve if isinstance(row, dict)]
        final_record = data.get("final_record")
        if isinstance(final_record, dict):
            return [dict(final_record)]
    return []


def _load_csv_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _normalise_records(path: Path, label: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw_rows = _load_csv_records(path) if path.suffix.lower() == ".csv" else _load_json_records(path)
    family = _infer_family(path)
    if family == "RL" and raw_rows:
        row_keys = set().union(*(row.keys() for row in raw_rows[:5]))
        if any(
            key.startswith(("highway_v0_", "merge_v0_", "roundabout_v0_", "intersection_v0_"))
            for key in row_keys
        ):
            family = "HighwayEnv"
    rows: list[dict] = []
    for idx, row in enumerate(raw_rows):
        step = _first_finite(row, ("timesteps", "steps", "num_timesteps", "step"))
        if not math.isfinite(step):
            step = float(idx)

        episode_return = _first_finite(
            row,
            (
                "ep_reward",
                "train_return_mean",
                "return_mean",
                "ep_return_mean",
                "episode_return",
            ),
        )
        if not math.isfinite(episode_return):
            # Prefer the first evaluation return column in HighwayEnv summary rows.
            for key, value in row.items():
                if key.endswith("_return_mean"):
                    episode_return = _to_float(value)
                    if math.isfinite(episode_return):
                        break

        avg_reward = _first_finite(
            row,
            (
                "train_reward_total_mean",
                "reward_mean",
                "mean_reward",
                "avg_reward",
            ),
        )
        if "ep_reward" in row and "ep_len" in row:
            ep_len = max(1.0, _to_float(row.get("ep_len"), 1.0))
            avg_reward = _to_float(row.get("ep_reward")) / ep_len
        elif not math.isfinite(avg_reward):
            length = _first_finite(row, ("train_episode_length_mean", "episode_length_mean", "ep_len"))
            if math.isfinite(episode_return) and math.isfinite(length) and length > 0:
                avg_reward = episode_return / length

        if not math.isfinite(episode_return) and not math.isfinite(avg_reward):
            continue
        rows.append(
            {
                "family": family,
                "label": label,
                "source": str(path),
                "timesteps": float(step),
                "avg_reward": float(avg_reward),
                "episode_return": float(episode_return),
            }
        )
    rows.sort(key=lambda item: item["timesteps"])
    return rows


def _write_normalized_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["family", "label", "source", "timesteps", "avg_reward", "episode_return"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot(runs: list[list[dict]], out_base: Path, *, smooth: int) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    metrics = [
        ("avg_reward", "Average Reward", "Average reward per step"),
        ("episode_return", "Episode Return", "Episode return"),
    ]
    for idx, rows in enumerate(runs):
        if not rows:
            continue
        label = rows[0]["label"]
        family = rows[0]["family"]
        x = np.asarray([row["timesteps"] for row in rows], dtype=float)
        color = COLORS[idx % len(COLORS)]
        linestyle = "-" if family == "MetaDrive" else "--"
        for ax, (key, title, ylabel) in zip(axes, metrics):
            y = np.asarray([row[key] for row in rows], dtype=float)
            y_s = _rolling_mean(y, smooth)
            ax.plot(x, y, color=color, alpha=0.18, linewidth=0.7, linestyle=linestyle)
            ax.plot(x, y_s, color=color, linewidth=1.5, linestyle=linestyle, label=f"{family}: {label}")
            ax.set_title(title)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 6))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.14))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run specs as 'Label=path/to/log.csv' or plain paths.",
    )
    parser.add_argument("--out", default="rl/logs/figures/training_reward_return")
    parser.add_argument("--smooth", type=int, default=25)
    args = parser.parse_args()

    run_rows: list[list[dict]] = []
    all_rows: list[dict] = []
    for spec in args.runs:
        label, path = _parse_run_spec(spec)
        rows = _normalise_records(path, label)
        if not rows:
            print(f"[warn] no plottable rows in {path}")
            continue
        run_rows.append(rows)
        all_rows.extend(rows)
        print(f"[load] {label}: {len(rows)} rows from {path}")

    if not run_rows:
        raise SystemExit("No plottable runs found.")

    out_base = Path(args.out)
    _write_normalized_csv(all_rows, out_base.with_name(out_base.name + "_data.csv"))
    _plot(run_rows, out_base, smooth=max(1, int(args.smooth)))
    print(f"[plot] wrote {out_base.with_suffix('.png')}")
    print(f"[plot] wrote {out_base.with_suffix('.pdf')}")
    print(f"[data] wrote {out_base.with_name(out_base.name + '_data.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
