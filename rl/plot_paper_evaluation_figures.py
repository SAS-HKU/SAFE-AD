"""
Paper-facing evaluation plots for HighwayEnv and MetaDrive.

The script is intentionally higher-level than the diagnostic plotters. It
groups the many raw metrics into a small set of figures:

* HighwayEnv: shaded learning curves, behavioral joint distributions, and a
  compact metric heatmap for mechanistic interpretation.
* MetaDrive: four-pillar radar plots, box plots, and dense algorithm/scenario
  summary tables for high-fidelity robustness.

Examples
--------
python rl/plot_paper_evaluation_figures.py ^
  --highway-suite-dirs rl/logs/highwayenv_sb3_suite_highway/highway_v0_medium ^
                       rl/logs/highwayenv_sb3_suite_merge/merge_v0_medium ^
                       rl/logs/highwayenv_sb3_suite_intersection/intersection_v0_medium ^
                       rl/logs/highwayenv_sb3_suite_roundabout/roundabout_v0_medium ^
  --highway-train-runs "Social PPO=rl/logs/social_ppo_a5/summary.json" ^
                       "Social DQN=rl/logs/social_dqn_a5/summary.json" ^
  --metadrive-eval-dirs rl/logs/metadrive/eval_merge_3arm_full ^
                        rl/logs/metadrive/eval_intersection_3arm_full ^
                        rl/logs/metadrive/eval_roundabout_3arm_full ^
  --out-dir rl/logs/paper_eval_figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This plotting script requires pandas. Install pandas or use the existing diagnostic plotters.") from exc

try:
    import scienceplots  # noqa: F401

    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False


PALETTE = {
    "stock": "#4D4D4D",
    "risk": "#0072B2",
    "social": "#D55E00",
    "idm": "#009E73",
    "other": "#CC79A7",
}

PILLARS = ["Safety", "Efficiency", "Comfort", "Sociality"]
FAMILY_ORDER = ["stock", "risk", "social", "idm", "other"]
FAMILY_LABELS = {
    "stock": "Stock",
    "risk": "+Risk",
    "social": "+Social",
    "idm": "IDM",
    "other": "Other",
}


def _apply_style() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "grid", "no-latex"])
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 140,
        }
    )


def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _parse_label_path(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec.strip())
    return path.parent.name, path


def _method_family(name: str) -> str:
    text = str(name).lower()
    if "idm" in text:
        return "idm"
    if "social" in text or "full" in text:
        return "social"
    if "risk" in text:
        return "risk"
    if "stock" in text:
        return "stock"
    return "other"


def _family_label(name: str) -> str:
    return FAMILY_LABELS.get(str(name), str(name).title())


def _method_algo(name: str) -> str:
    text = str(name).lower()
    for algo in ("ppo", "dqn", "sac", "td3", "ddpg"):
        if algo in text:
            return algo.upper()
    if "idm" in text:
        return "IDM"
    return str(name).upper()


def _sort_families(values: Iterable[str]) -> list[str]:
    seen = list(dict.fromkeys(str(v) for v in values if pd.notna(v)))
    return sorted(seen, key=lambda item: FAMILY_ORDER.index(item) if item in FAMILY_ORDER else len(FAMILY_ORDER))


def _scenario_from_text(text: str) -> str:
    text = str(text).lower()
    for key in ("roundabout", "intersection", "merge", "highway", "curve", "straight", "mixed"):
        if key in text:
            return key
    return "scenario"


def _to_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _normalise_higher(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").astype(float)
    lo = float(vals.min()) if vals.notna().any() else 0.0
    hi = float(vals.max()) if vals.notna().any() else 1.0
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-9:
        return pd.Series(np.ones(len(vals)) * 0.5, index=vals.index)
    return ((vals - lo) / (hi - lo)).clip(0.0, 1.0)


def _normalise_lower(values: pd.Series) -> pd.Series:
    return 1.0 - _normalise_higher(values)


def _normalise_grouped(df: pd.DataFrame, values: pd.Series, *, by: str = "scenario", higher: bool = True) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    groups = df[by].astype(str) if by in df.columns else pd.Series("all", index=df.index)
    for _, idx in groups.groupby(groups).groups.items():
        sub = vals.loc[idx]
        out.loc[idx] = _normalise_higher(sub) if higher else _normalise_lower(sub)
    return out.fillna(0.5).clip(0.0, 1.0)


def _metric_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(float)
    return pd.Series(float(default), index=df.index, dtype=float)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_training_runs(specs: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in specs:
        label, path = _parse_label_path(spec)
        if not path.exists():
            continue
        data = _read_json(path)
        records = data.get("records", []) if isinstance(data, dict) else data
        for row in records:
            if not isinstance(row, dict):
                continue
            step = _to_float(row.get("timesteps"))
            if not math.isfinite(step):
                continue
            returns = []
            for key, value in row.items():
                if key.endswith("_return_mean"):
                    val = _to_float(value)
                    if math.isfinite(val):
                        returns.append(val)
            ret = _to_float(row.get("train_return_mean"))
            if not math.isfinite(ret) and returns:
                ret = float(np.nanmean(returns))
            if math.isfinite(ret):
                rows.append(
                    {
                        "label": label,
                        "scenario": _scenario_from_text(f"{label} {path}"),
                        "family": _method_family(f"{label} {path}"),
                        "algo": _method_algo(f"{label} {path}"),
                        "timesteps": step,
                        "episode_return": ret,
                    }
                )
    return pd.DataFrame(rows)


def _plot_learning_by_scenario(df: pd.DataFrame, out_dir: Path, *, out_name: str, title: str, value_col: str) -> None:
    if df.empty:
        return
    if value_col not in df.columns:
        return
    scenarios = list(dict.fromkeys(df["scenario"].dropna().astype(str).tolist()))
    if not scenarios:
        return
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 2.7), squeeze=False)
    for ax, scenario in zip(axes.ravel(), scenarios):
        sub_s = df[df["scenario"].astype(str) == scenario]
        for family in _sort_families(sub_s["family"].unique()):
            sub = sub_s[sub_s["family"] == family]
            if sub.empty:
                continue
            grouped = sub.groupby("timesteps")[value_col]
            mean = grouped.mean()
            count = grouped.count()
            std = grouped.std().fillna(0.0)
            ci = 1.96 * std / np.sqrt(np.maximum(count.to_numpy(dtype=float), 1.0))
            x = mean.index.to_numpy(dtype=float)
            y = mean.to_numpy(dtype=float)
            color = PALETTE.get(family, None)
            ax.plot(x, y, linewidth=1.6, label=_family_label(family), color=color)
            if np.any(ci > 0):
                ax.fill_between(x, y - ci, y + ci, alpha=0.16, color=color)
        ax.set_title(str(scenario).title())
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode return")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 6))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(title, y=1.10)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    _save(fig, out_dir / out_name)


def _load_highway_training_root(root: str | None) -> pd.DataFrame:
    if not root:
        return pd.DataFrame()
    base = Path(root)
    if not base.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for summary_path in sorted(base.glob("*/summary.json")):
        data = _read_json(summary_path)
        config = data.get("config", {}) if isinstance(data, dict) else {}
        records = data.get("records", []) if isinstance(data, dict) else []
        env_id = str(config.get("env_id", summary_path.parent.name))
        scenario = _scenario_from_text(env_id)
        ablation = str(config.get("ablation", ""))
        family = "stock" if ablation == "A0" else "risk" if ablation == "A2" else _method_family(f"{summary_path.parent.name} {ablation}")
        algo = str(config.get("algo", _method_algo(summary_path.parent.name))).upper()
        for record in records:
            if not isinstance(record, dict):
                continue
            step = _to_float(record.get("timesteps"))
            if not math.isfinite(step):
                continue
            ret = _to_float(record.get("train_return_mean"))
            if not math.isfinite(ret):
                vals = [_to_float(v) for k, v in record.items() if k.endswith("_return_mean")]
                vals = [v for v in vals if math.isfinite(v)]
                ret = float(np.nanmean(vals)) if vals else float("nan")
            if not math.isfinite(ret):
                continue
            rows.append(
                {
                    "label": summary_path.parent.name,
                    "scenario": scenario,
                    "family": family,
                    "algo": algo,
                    "timesteps": step,
                    "episode_return": ret,
                }
            )
    return pd.DataFrame(rows)


def _load_metadrive_training_root(root: str | None) -> pd.DataFrame:
    if not root:
        return pd.DataFrame()
    base = Path(root)
    if not base.exists():
        return pd.DataFrame()
    candidates: dict[tuple[str, str, str], tuple[int, Path]] = {}
    for progress_path in sorted(base.glob("*/progress.csv")):
        name = progress_path.parent.name.lower()
        if any(token in name for token in ("lite", "lj001", "eval", "smoke")):
            continue
        scenario = _scenario_from_text(name)
        if scenario not in {"merge", "intersection", "roundabout"}:
            continue
        family = _method_family(name)
        if family not in {"stock", "risk", "social"}:
            continue
        algo = _method_algo(name)
        priority = 0
        if "socialbench" in name:
            priority += 4
        if "decoupled" not in name:
            priority += 2
        if name.endswith("_1m"):
            priority += 1
        key = (scenario, family, algo)
        if key not in candidates or priority > candidates[key][0]:
            candidates[key] = (priority, progress_path)

    rows: list[dict] = []
    for (scenario, family, algo), (_priority, path) in sorted(candidates.items()):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "timesteps" not in df.columns or "ep_reward" not in df.columns:
            continue
        for _, row in df.iterrows():
            step = _to_float(row.get("timesteps"))
            reward = _to_float(row.get("ep_reward"))
            if not (math.isfinite(step) and math.isfinite(reward)):
                continue
            rows.append(
                {
                    "label": path.parent.name,
                    "scenario": scenario,
                    "family": family,
                    "algo": algo,
                    "timesteps": step,
                    "episode_return": reward,
                }
            )
    return pd.DataFrame(rows)


def _load_highway_suite_dirs(dirs: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict] = []
    summary_rows: list[dict] = []
    for raw in dirs:
        base = Path(raw)
        candidate_dirs = [base] if (base / "summary.json").exists() else sorted(base.glob("*/summary.json"))
        for item in candidate_dirs:
            run_dir = item.parent if item.name == "summary.json" else item
            summary_path = run_dir / "summary.json"
            episodes_path = run_dir / "episodes.json"
            if not summary_path.exists():
                continue
            summary = _read_json(summary_path)
            scenario = str(summary.get("env_id", _scenario_from_text(run_dir))).replace("-v0", "")
            traffic = str(summary.get("traffic_label", ""))
            metrics = summary.get("metrics", {})
            if isinstance(metrics, dict):
                for planner, row in metrics.items():
                    if not isinstance(row, dict):
                        continue
                    flat = dict(row)
                    flat.update(
                        {
                            "planner": planner,
                            "family": _method_family(planner),
                            "algo": _method_algo(planner),
                            "scenario": scenario,
                            "traffic": traffic,
                        }
                    )
                    summary_rows.append(flat)
            if not episodes_path.exists():
                continue
            episodes = _read_json(episodes_path)
            if not isinstance(episodes, dict):
                continue
            for planner, eps in episodes.items():
                if not isinstance(eps, list):
                    continue
                for ep in eps:
                    if not isinstance(ep, dict):
                        continue
                    row = dict(ep)
                    social = row.pop("social_summary", {})
                    if isinstance(social, dict):
                        for key, value in social.items():
                            row[f"social_{key}"] = value
                    row.update(
                        {
                            "planner": planner,
                            "family": _method_family(planner),
                            "algo": _method_algo(planner),
                            "scenario": scenario,
                            "traffic": traffic,
                        }
                    )
                    episode_rows.append(row)
    return pd.DataFrame(episode_rows), pd.DataFrame(summary_rows)


def _plot_highway_joint(episodes: pd.DataFrame, out_dir: Path) -> None:
    if episodes.empty or not {"mean_speed", "mean_abs_jerk", "planner"}.issubset(episodes.columns):
        return
    scenarios = list(episodes["scenario"].dropna().unique())[:4]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 2.65), squeeze=False, constrained_layout=True)
    for ax, scenario in zip(axes.ravel(), scenarios):
        sub = episodes[episodes["scenario"] == scenario]
        for planner, grp in sub.groupby("planner", sort=False):
            x = pd.to_numeric(grp["mean_speed"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(grp["mean_abs_jerk"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                continue
            color = PALETTE.get(_method_family(planner), None)
            ax.scatter(x[mask], y[mask], s=13, alpha=0.46, label=planner, color=color, edgecolors="none")
        ax.set_title(str(scenario).title())
        ax.set_xlabel("Mean speed (m/s)")
        ax.set_ylabel("Mean |jerk|")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.16))
    _save(fig, out_dir / "highway_behavior_joint_speed_jerk")


def _plot_highway_metric_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    metric_specs = [
        ("collision_rate", False, "Collision"),
        ("ttc_min_mean", True, "TTC"),
        ("mean_abs_jerk", False, "Jerk"),
        ("risk_mass_others_mean", False, "Imposed risk"),
        ("risk_flux_backward_mean", False, "Back flux"),
        ("safety_efficiency_index_sei_mean", True, "SEI"),
        ("social_friendliness_score_mean", True, "Social score"),
    ]
    rows = []
    for _, row in summary.iterrows():
        out = {"scenario": row.get("scenario"), "planner": row.get("planner")}
        for key, higher, label in metric_specs:
            out[label] = _to_float(row.get(key))
        rows.append(out)
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df["method"] = df["scenario"].astype(str).str.title() + "\n" + df["planner"].astype(str)
    vals = df[[label for _, _, label in metric_specs]].copy()
    for key, higher, label in metric_specs:
        vals[label] = _normalise_higher(vals[label]) if higher else _normalise_lower(vals[label])
    fig_h = max(2.6, 0.27 * len(df) + 0.8)
    fig, ax = plt.subplots(figsize=(4.5, fig_h), constrained_layout=True)
    im = ax.imshow(vals.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["method"].tolist())
    ax.set_xticks(np.arange(len(vals.columns)))
    ax.set_xticklabels(vals.columns, rotation=25, ha="right")
    ax.set_title("HighwayEnv Normalized Diagnostic Metrics")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("normalized score")
    _save(fig, out_dir / "highway_diagnostic_heatmap")


def _plot_highway_sei_surface(episodes: pd.DataFrame, out_dir: Path) -> None:
    if episodes.empty:
        return
    speed_col = "social_mean_speed_frame" if "social_mean_speed_frame" in episodes.columns else "mean_speed"
    distance_col = "social_min_distance_m" if "social_min_distance_m" in episodes.columns else "min_spacing"
    sei_col = "social_safety_efficiency_index_sei"
    if sei_col not in episodes.columns:
        return
    cols = ["scenario", "family", "planner", speed_col, distance_col, sei_col]
    df = episodes[[c for c in cols if c in episodes.columns]].copy()
    df["speed"] = pd.to_numeric(df.get(speed_col), errors="coerce")
    df["leader_distance"] = pd.to_numeric(df.get(distance_col), errors="coerce")
    df["sei"] = pd.to_numeric(df.get(sei_col), errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["speed", "leader_distance", "sei"])
    df = df[(df["sei"] > 0.0) & (df["speed"] >= 0.0) & (df["leader_distance"] >= 0.0)]
    if df.empty:
        return

    scenarios = list(dict.fromkeys(df["scenario"].dropna().astype(str).tolist()))[:4]
    if not scenarios:
        return
    ncols = 2 if len(scenarios) > 1 else 1
    nrows = int(math.ceil(len(scenarios) / ncols))
    fig = plt.figure(figsize=(3.9 * ncols, 3.25 * nrows))
    legend_handles = []
    legend_labels = []
    for i, scenario in enumerate(scenarios, start=1):
        ax = fig.add_subplot(nrows, ncols, i, projection="3d")
        sub = df[df["scenario"].astype(str) == scenario].copy()
        if len(sub) < 6:
            continue
        sub["speed"] = sub["speed"].clip(0.0, 32.0)
        sub["leader_distance"] = sub["leader_distance"].clip(0.0, 80.0)
        speed_edges = np.linspace(float(sub["speed"].min()), float(sub["speed"].max()) + 1e-6, 12)
        dist_edges = np.linspace(float(sub["leader_distance"].min()), float(sub["leader_distance"].max()) + 1e-6, 12)
        speed_centers = 0.5 * (speed_edges[:-1] + speed_edges[1:])
        dist_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])
        sub["speed_bin"] = pd.cut(sub["speed"], speed_edges, labels=speed_centers, include_lowest=True)
        sub["dist_bin"] = pd.cut(sub["leader_distance"], dist_edges, labels=dist_centers, include_lowest=True)
        pivot = sub.pivot_table(index="dist_bin", columns="speed_bin", values="sei", aggfunc="mean", observed=False)
        pivot = pivot.astype(float).interpolate(axis=0, limit_direction="both").interpolate(axis=1, limit_direction="both")
        pivot = pivot.fillna(float(sub["sei"].mean()))
        X, Y = np.meshgrid(pivot.columns.astype(float).to_numpy(), pivot.index.astype(float).to_numpy())
        Z = pivot.to_numpy(dtype=float)
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.74, linewidth=0, antialiased=True)
        for family in ("stock", "risk", "social"):
            grp = sub[sub["family"] == family]
            if grp.empty:
                continue
            ax.scatter(
                grp["speed"],
                grp["leader_distance"],
                grp["sei"],
                s=11,
                alpha=0.65,
                color=PALETTE.get(family),
                label=_family_label(family),
                depthshade=False,
            )
        if not legend_handles:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(str(scenario).title())
        ax.set_xlabel("Speed (m/s)")
        ax.set_ylabel("Leader distance (m)")
        ax.set_zlabel("SEI")
        ax.view_init(elev=24, azim=-55)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0.25
            axis._axinfo["grid"]["color"] = (0.80, 0.80, 0.80, 0.55)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(3, len(legend_labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
    fig.suptitle("HighwayEnv SEI Surface and Policy Occupancy", y=0.99)
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
    _save(fig, out_dir / "highway_sei_surface")


def _write_highway_ei_sei_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    cols = [
        "scenario",
        "planner",
        "algo",
        "family",
        "efficiency_index_ei_mean",
        "safety_efficiency_index_sei_mean",
        "social_traffic_efficiency_index_mean",
        "mean_speed_frame_mean",
        "speed_variance_frame_mean",
        "frac_critical_ttc_mean",
        "risk_flux_backward_frame_mean",
        "social_friendliness_score_mean",
    ]
    keep = [col for col in cols if col in summary.columns]
    if not keep:
        return
    out = summary[keep].copy()
    stock = out[out["family"] == "stock"][["scenario", "algo", "safety_efficiency_index_sei_mean"]].rename(
        columns={"safety_efficiency_index_sei_mean": "stock_sei"}
    )
    if "safety_efficiency_index_sei_mean" in out.columns:
        out = out.merge(stock, on=["scenario", "algo"], how="left")
        out["EI"] = out.get("efficiency_index_ei_mean", np.nan)
        out["SEI"] = out["safety_efficiency_index_sei_mean"]
        out["stock_reference_SEI"] = out["stock_sei"]
        out["delta_sei_pct_vs_stock"] = 100.0 * (
            out["safety_efficiency_index_sei_mean"] - out["stock_sei"]
        ) / out["stock_sei"].replace(0.0, np.nan)
    out.to_csv(out_dir / "highway_ei_sei_summary.csv", index=False)


def _load_metadrive_eval_dirs(dirs: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_frames: list[pd.DataFrame] = []
    step_frames: list[pd.DataFrame] = []
    for raw in dirs:
        base = Path(raw)
        candidates = [base] if (base / "eval_episodes.csv").exists() else [p.parent for p in base.glob("*/eval_episodes.csv")]
        for run_dir in candidates:
            ep_path = run_dir / "eval_episodes.csv"
            if ep_path.exists():
                ep = pd.read_csv(ep_path)
                if "scenario" not in ep.columns:
                    ep["scenario"] = ep.apply(
                        lambda row: _scenario_from_text(f"{run_dir} {row.get('protocol', '')}"), axis=1
                    )
                ep["eval_dir"] = str(run_dir)
                ep["family"] = ep["planner"].map(_method_family)
                ep["algo"] = ep["planner"].map(_method_algo)
                episode_frames.append(ep)
            step_path = run_dir / "eval_steps.csv"
            if step_path.exists():
                steps = pd.read_csv(step_path)
                if "scenario" not in steps.columns:
                    steps["scenario"] = steps.apply(
                        lambda row: _scenario_from_text(f"{run_dir} {row.get('protocol', '')}"), axis=1
                    )
                steps["eval_dir"] = str(run_dir)
                steps["family"] = steps["planner"].map(_method_family)
                steps["algo"] = steps["planner"].map(_method_algo)
                step_frames.append(steps)
    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    steps = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    return episodes, steps


def _add_metadrive_pillars(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return episodes
    df = episodes.copy()
    route = _metric_series(df, "route_completion", 0.0).clip(0, 1)
    success = _metric_series(df, "success", 0.0).clip(0, 1)
    ep_len = _metric_series(df, "ep_len", 1.0).clip(lower=1.0)
    risk_score = _normalise_grouped(df, _metric_series(df, "cumulative_risk_exposure", 0.0), higher=False)
    crash_score = (1.0 - _metric_series(df, "crash_any", 0.0).clip(0, 1)).clip(0, 1)
    ttc_rate = (_metric_series(df, "ttc_violation_steps", 0.0) / ep_len).clip(0, 1)
    ttc_score = (1.0 - 5.0 * ttc_rate).clip(0, 1)
    df["Safety"] = (0.45 * risk_score + 0.35 * crash_score + 0.20 * ttc_score).clip(0, 1)
    df["Efficiency"] = (0.65 * route + 0.35 * success).clip(0, 1)
    jerk_score = _normalise_grouped(df, _metric_series(df, "mean_jerk_abs", 0.0), higher=False)
    steer_score = _normalise_grouped(df, _metric_series(df, "steering_change_rate", 0.0), higher=False)
    throttle_score = _normalise_grouped(df, _metric_series(df, "throttle_change_rate", 0.0), higher=False)
    df["Comfort"] = (0.50 * jerk_score + 0.25 * steer_score + 0.25 * throttle_score).clip(0, 1)
    social_base = _metric_series(df, "social_friendliness_score", 0.5).clip(0, 1)
    backward_score = _normalise_grouped(df, _metric_series(df, "mean_backward_pressure", 0.0), higher=False)
    hard_brake_score = (1.0 - 8.0 * _metric_series(df, "hard_brake_imposed_rate", 0.0)).clip(0, 1)
    bad_cut_score = (1.0 - 8.0 * _metric_series(df, "bad_cut_in_rate", 0.0)).clip(0, 1)
    rear_ttc_score = _normalise_grouped(df, _metric_series(df, "mean_rear_ttc_loss", 0.0), higher=False)
    externality_score = (0.40 * backward_score + 0.25 * hard_brake_score + 0.25 * bad_cut_score + 0.10 * rear_ttc_score)
    df["Sociality"] = (0.55 * social_base + 0.45 * externality_score).clip(0, 1)
    df["four_pillar_score"] = df[PILLARS].mean(axis=1)
    alpha = 0.6
    df["semi_alpha06"] = (
        df["Efficiency"] * df["Safety"] - alpha * (1.0 - df["Sociality"]) - 0.25 * (1.0 - df["Comfort"])
    ).clip(lower=0.0)
    return df


def _plot_metadrive_radar(episodes: pd.DataFrame, out_dir: Path) -> None:
    if episodes.empty:
        return
    df = _add_metadrive_pillars(episodes)
    df = df[df["family"].isin(["stock", "risk", "social"])].copy()
    if df.empty:
        return
    agg = df.groupby(["scenario", "family"], as_index=False)[PILLARS].mean()
    scenarios = list(agg["scenario"].dropna().unique())[:3]
    if not scenarios:
        return
    angles = np.linspace(0, 2 * np.pi, len(PILLARS), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, len(scenarios), subplot_kw={"projection": "polar"}, figsize=(3.25 * len(scenarios), 3.45))
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        sub = agg[agg["scenario"] == scenario]
        for family in _sort_families(sub["family"].unique()):
            row = sub[sub["family"] == family].iloc[0]
            values = [float(row[p]) for p in PILLARS]
            values += values[:1]
            color = PALETTE.get(family, None)
            linestyle = "--" if family == "stock" else "-"
            ax.plot(angles, values, linewidth=1.8, linestyle=linestyle, label=_family_label(family), color=color)
            ax.fill(angles, values, alpha=0.10, color=color)
        ax.set_title(str(scenario).title(), y=1.16)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(PILLARS)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(0.0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("MetaDrive RL Ablation Four-Pillar Trade-Offs", y=0.98)
    fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.91])
    _save(fig, out_dir / "metadrive_four_pillar_radar")


def _plot_metadrive_boxplots(episodes: pd.DataFrame, out_dir: Path) -> None:
    if episodes.empty:
        return
    df = _add_metadrive_pillars(episodes)
    metrics = [
        ("route_completion", "Route completion"),
        ("cumulative_risk_exposure", "Risk exposure"),
        ("mean_jerk_abs", "Mean |jerk|"),
        ("mean_backward_pressure", "Backward pressure"),
        ("four_pillar_score", "Four-pillar score"),
    ]
    available = [(key, title) for key, title in metrics if key in df.columns]
    if not available:
        return
    families = _sort_families(df["family"].dropna().unique())
    fig, axes = plt.subplots(1, len(available), figsize=(2.15 * len(available), 2.9), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, available):
        data = []
        labels = []
        box_families = []
        for family in families:
            vals = pd.to_numeric(df.loc[df["family"] == family, key], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                data.append(vals)
                labels.append(_family_label(family))
                box_families.append(family)
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.62)
        for patch, family in zip(bp["boxes"], box_families):
            patch.set_facecolor(PALETTE.get(family, "#999999"))
            patch.set_alpha(0.46)
        ax.set_title(title)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    _save(fig, out_dir / "metadrive_metric_boxplots")


def _plot_metadrive_objective_deltas(episodes: pd.DataFrame, out_dir: Path) -> None:
    if episodes.empty:
        return
    df = _add_metadrive_pillars(episodes)
    df = df[df["family"].isin(["stock", "risk", "social"])].copy()
    if df.empty:
        return
    agg = (
        df.groupby(["scenario", "algo", "family"], as_index=False)
        .agg(
            risk_exposure=("cumulative_risk_exposure", "mean"),
            mean_jerk_abs=("mean_jerk_abs", "mean"),
            sociality=("Sociality", "mean"),
            semi_alpha06=("semi_alpha06", "mean"),
        )
    )
    records = []
    for (scenario, algo), sub in agg.groupby(["scenario", "algo"], sort=False):
        by_family = sub.set_index("family")
        if "stock" not in by_family.index:
            continue
        stock = by_family.loc["stock"]
        row = {"scenario": scenario, "algo": algo, "label": f"{str(scenario).title()}\n{algo}"}
        if "risk" in by_family.index:
            risk = by_family.loc["risk"]
            row["+Risk risk exposure reduction"] = float(stock["risk_exposure"]) - float(risk["risk_exposure"])
            row["+Risk SEMI gain"] = float(risk["semi_alpha06"]) - float(stock["semi_alpha06"])
        else:
            row["+Risk risk exposure reduction"] = np.nan
            row["+Risk SEMI gain"] = np.nan
        if "social" in by_family.index:
            social = by_family.loc["social"]
            denom = max(float(stock["mean_jerk_abs"]), 1e-9)
            row["+Social jerk reduction (%)"] = 100.0 * (float(stock["mean_jerk_abs"]) - float(social["mean_jerk_abs"])) / denom
            row["+Social sociality gain"] = float(social["sociality"]) - float(stock["sociality"])
            row["+Social SEMI gain"] = float(social["semi_alpha06"]) - float(stock["semi_alpha06"])
        else:
            row["+Social jerk reduction (%)"] = np.nan
            row["+Social sociality gain"] = np.nan
            row["+Social SEMI gain"] = np.nan
        records.append(row)
    out = pd.DataFrame(records)
    if out.empty:
        return
    metric_cols = [c for c in out.columns if c not in {"scenario", "algo", "label"}]
    out.to_csv(out_dir / "metadrive_objective_deltas_vs_stock.csv", index=False)

    vals = out[metric_cols].to_numpy(dtype=float)
    color_vals = np.full_like(vals, np.nan, dtype=float)
    for c in range(vals.shape[1]):
        col = vals[:, c]
        scale = float(np.nanpercentile(np.abs(col), 90)) if np.isfinite(col).any() else 1.0
        scale = max(scale, 1e-6)
        color_vals[:, c] = np.clip(col / scale, -1.0, 1.0)
    fig_h = max(3.2, 0.26 * len(out) + 1.0)
    fig, ax = plt.subplots(figsize=(6.6, fig_h), constrained_layout=True)
    im = ax.imshow(color_vals, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_yticks(np.arange(len(out)))
    ax.set_yticklabels(out["label"].tolist())
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=25, ha="right")
    ax.set_title("MetaDrive Objective Deltas Relative to Stock")
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            value = vals[r, c]
            if np.isfinite(value):
                text = f"{value:+.1f}" if "%" in metric_cols[c] else f"{value:+.2f}"
                ax.text(c, r, text, ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("positive is better")
    _save(fig, out_dir / "metadrive_objective_deltas_vs_stock")


def _write_metadrive_summary_tables(episodes: pd.DataFrame, out_dir: Path) -> None:
    if episodes.empty:
        return
    df = _add_metadrive_pillars(episodes)
    agg = (
        df.groupby(["scenario", "planner", "algo", "family"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            route_completion=("route_completion", "mean"),
            risk_exposure=("cumulative_risk_exposure", "mean"),
            mean_jerk_abs=("mean_jerk_abs", "mean"),
            comfort_cost=("mean_comfort_cost", "mean"),
            social_score=("Sociality", "mean"),
            safety_pillar=("Safety", "mean"),
            efficiency_pillar=("Efficiency", "mean"),
            comfort_pillar=("Comfort", "mean"),
            sociality_pillar=("Sociality", "mean"),
            four_pillar_score=("four_pillar_score", "mean"),
            semi_alpha06=("semi_alpha06", "mean"),
            decision_ms=("mean_action_selection_ms", "mean") if "mean_action_selection_ms" in df.columns else ("four_pillar_score", "count"),
        )
        .sort_values(["scenario", "algo", "family", "planner"])
    )
    stock = agg[agg["family"] == "stock"][["scenario", "algo", "semi_alpha06", "success_rate"]].rename(
        columns={"semi_alpha06": "stock_semi", "success_rate": "stock_success"}
    )
    merged = agg.merge(stock, on=["scenario", "algo"], how="left")
    merged["delta_semi_pct_vs_stock"] = 100.0 * (merged["semi_alpha06"] - merged["stock_semi"]) / merged["stock_semi"].replace(0.0, np.nan)
    merged["delta_success_pct_vs_stock"] = 100.0 * (merged["success_rate"] - merged["stock_success"]) / merged["stock_success"].replace(0.0, np.nan)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "metadrive_algorithm_scenario_summary.csv", index=False)

    pivot = merged.pivot_table(index=["scenario", "algo"], columns="family", values="semi_alpha06", aggfunc="mean")
    pivot.to_csv(out_dir / "metadrive_semi_alpha06_pivot.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--highway-suite-dirs", nargs="*", default=[])
    parser.add_argument("--highway-train-runs", nargs="*", default=[])
    parser.add_argument(
        "--highway-train-root",
        default="",
        help="Optional parent directory containing HighwayEnv training run folders with summary.json.",
    )
    parser.add_argument(
        "--metadrive-train-root",
        default="",
        help="Optional parent directory containing MetaDrive training run folders with progress.csv.",
    )
    parser.add_argument("--metadrive-eval-dirs", nargs="*", default=[])
    parser.add_argument("--out-dir", default="rl/logs/paper_eval_figures")
    args = parser.parse_args()

    _apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    highway_train_df = pd.concat(
        [
            _load_training_runs(args.highway_train_runs),
            _load_highway_training_root(args.highway_train_root),
        ],
        ignore_index=True,
    )
    _plot_learning_by_scenario(
        highway_train_df,
        out_dir,
        out_name="highway_training_by_scenario",
        title="HighwayEnv Training Dynamics",
        value_col="episode_return",
    )

    metadrive_train_df = _load_metadrive_training_root(args.metadrive_train_root)
    _plot_learning_by_scenario(
        metadrive_train_df,
        out_dir,
        out_name="metadrive_training_by_scenario",
        title="MetaDrive Training Dynamics",
        value_col="episode_return",
    )

    highway_episodes, highway_summary = _load_highway_suite_dirs(args.highway_suite_dirs)
    _plot_highway_joint(highway_episodes, out_dir)
    _plot_highway_sei_surface(highway_episodes, out_dir)
    _plot_highway_metric_heatmap(highway_summary, out_dir)
    _write_highway_ei_sei_summary(highway_summary, out_dir)
    if not highway_episodes.empty:
        highway_episodes.to_csv(out_dir / "highway_episode_plot_data.csv", index=False)
    if not highway_summary.empty:
        highway_summary.to_csv(out_dir / "highway_summary_plot_data.csv", index=False)

    metadrive_episodes, metadrive_steps = _load_metadrive_eval_dirs(args.metadrive_eval_dirs)
    _plot_metadrive_radar(metadrive_episodes, out_dir)
    _plot_metadrive_boxplots(metadrive_episodes, out_dir)
    _plot_metadrive_objective_deltas(metadrive_episodes, out_dir)
    _write_metadrive_summary_tables(metadrive_episodes, out_dir)
    if not metadrive_steps.empty:
        # Keep this data for optional future joint-distribution plots without
        # duplicating large step CSVs in the paper figure directory.
        metadrive_steps.groupby(["scenario", "planner"], as_index=False).agg(
            mean_speed=("speed", "mean"),
            mean_abs_steer=("control_steer", lambda x: float(np.mean(np.abs(pd.to_numeric(x, errors="coerce"))))),
            mean_abs_jerk=("jerk", lambda x: float(np.mean(np.abs(pd.to_numeric(x, errors="coerce"))))),
            mean_action_selection_ms=("action_selection_ms", "mean"),
        ).to_csv(out_dir / "metadrive_step_metric_summary.csv", index=False)

    manifest = {
        "out_dir": str(out_dir),
        "highway_training_rows": int(len(highway_train_df)),
        "metadrive_training_rows": int(len(metadrive_train_df)),
        "highway_episode_rows": int(len(highway_episodes)),
        "highway_summary_rows": int(len(highway_summary)),
        "metadrive_episode_rows": int(len(metadrive_episodes)),
        "metadrive_step_rows": int(len(metadrive_steps)),
        "figures": sorted(str(p.name) for p in out_dir.glob("*.pdf")),
    }
    (out_dir / "paper_eval_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
