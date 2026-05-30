from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCIENCEPLOTS_SRC = REPO_ROOT / "SciencePlots" / "src"
if SCIENCEPLOTS_SRC.is_dir() and str(SCIENCEPLOTS_SRC) not in sys.path:
    sys.path.insert(0, str(SCIENCEPLOTS_SRC))

try:
    import scienceplots  # noqa: F401

    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False


METHOD_LABELS = {
    "idm": "IDM",
    "random": "Random",
    "stock_ppo": "Stock PPO",
    "stock_notebook": "Notebook PPO",
    "stock_traffic": "Traffic PPO",
    "risk_ppo": "Social-Risk PPO",
    "risk01": "Risk PPO $\\lambda=0.1$",
    "risk10": "Risk PPO $\\lambda=1.0$",
    "dqn": "DQN",
    "sac": "SAC",
    "ddpg": "DDPG",
    "td3": "TD3",
    "a2c": "A2C",
}

CANONICAL_ORDER = [
    "idm",
    "stock_ppo",
    "risk_ppo",
    "dqn",
    "sac",
    "ddpg",
    "td3",
    "a2c",
    "random",
    "stock_notebook",
    "stock_traffic",
    "risk01",
    "risk10",
]

METRIC_SPECS = {
    "route_completion": ("Route completion", True),
    "success": ("Success rate", True),
    "crash_any": ("Crash rate", False),
    "cumulative_risk_exposure": ("Risk exposure", False),
    "peak_risk": ("Peak risk", False),
    "risk_adjusted_efficiency": ("Risk-adjusted efficiency", True),
    "mean_speed": ("Mean speed (m/s)", True),
    "jerk_l1": ("Jerk $L_1$", False),
    "near_miss_steps": ("Near-miss steps", False),
    "ttc_violation_steps": ("TTC violation steps", False),
}


DEFAULT_SCENARIO_LABEL = "Scenario: MetaDrive PG curved road map C; Trigger traffic; IDM-controlled surrounding vehicles"


def _apply_style() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "grid", "no-latex"])
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _default_eval_csvs() -> list[Path]:
    preferred = REPO_ROOT / "rl" / "logs" / "metadrive" / "matched_stock_vs_social_risk_eval" / "eval_episodes.csv"
    if preferred.exists():
        return [preferred]
    return sorted((REPO_ROOT / "rl" / "logs" / "metadrive").glob("*/eval_episodes.csv"))


def _load_eval_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No eval_episodes.csv files were found.")
    df = pd.concat(frames, ignore_index=True)
    for col in ["success", "crash_any", "crash_vehicle", "out_of_road", "max_step"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    numeric_cols = [
        "seed",
        "traffic_density",
        "ep_len",
        "ep_reward",
        "ep_base_reward",
        "ep_cost",
        "route_completion",
        "overtake_vehicle_num",
        "cumulative_risk_exposure",
        "peak_risk",
        "high_risk_steps",
        "near_miss_steps",
        "ttc_violation_steps",
        "mean_speed",
        "mean_accel_abs",
        "jerk_l1",
        "risk_adjusted_efficiency",
        "overtakes_per_second",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_pseudo"] = False
    df["pseudo_note"] = ""
    df["method"] = df["planner"].astype(str)
    if "scenario_type" not in df.columns:
        df["scenario_type"] = df["protocol"].map(_scenario_from_protocol).fillna("MetaDrive PG map")
    return df


def _scenario_from_protocol(protocol: object) -> str:
    name = str(protocol)
    if name in {"official_notebook_reference", "matched_stock", "matched_social_risk"}:
        return "Curved road (map C)"
    if name == "safe_metadrive_risk":
        return "SafeMetaDrive curved road"
    if name == "pseudo_metadrive_preview":
        return "Preview placeholder"
    return name


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _positive_normal(rng: np.random.Generator, mean: float, scale: float, floor: float = 0.0) -> float:
    return float(max(floor, rng.normal(mean, max(1e-9, scale))))


def _density_values(real_df: pd.DataFrame) -> list[float]:
    densities = sorted(float(x) for x in real_df["traffic_density"].dropna().unique())
    return densities or [0.1, 0.3]


def _base_stats(real_df: pd.DataFrame, density: float) -> dict[str, float]:
    density_df = real_df[np.isclose(real_df["traffic_density"], density)]
    stock = density_df[density_df["method"].isin(["stock_ppo", "stock_traffic", "stock_notebook"])]
    if stock.empty:
        stock = density_df
    if stock.empty:
        stock = real_df
    return {
        "n": max(20, int(len(stock))),
        "success": float(stock["success"].mean()) if "success" in stock else 0.2,
        "crash_any": float(stock["crash_any"].mean()) if "crash_any" in stock else 0.45,
        "route_completion": float(stock["route_completion"].mean()),
        "ep_reward": float(stock["ep_reward"].mean()),
        "risk": float(stock["cumulative_risk_exposure"].mean()),
        "peak_risk": float(stock["peak_risk"].mean()),
        "efficiency": float(stock["risk_adjusted_efficiency"].mean()),
        "mean_speed": float(stock["mean_speed"].mean()),
        "jerk": float(stock["jerk_l1"].mean()),
        "near_miss": float(stock["near_miss_steps"].mean()),
        "ttc": float(stock["ttc_violation_steps"].mean()),
    }


PSEUDO_PROFILES = {
    "dqn": {
        "success": 0.75,
        "crash": 1.15,
        "route": 0.88,
        "reward": 0.88,
        "risk": 1.10,
        "peak": 1.08,
        "eff": 0.85,
        "speed": 0.95,
        "jerk": 1.20,
    },
    "sac": {
        "success": 1.18,
        "crash": 0.88,
        "route": 1.06,
        "reward": 1.06,
        "risk": 0.92,
        "peak": 0.95,
        "eff": 1.08,
        "speed": 1.02,
        "jerk": 0.82,
    },
    "ddpg": {
        "success": 0.60,
        "crash": 1.30,
        "route": 0.74,
        "reward": 0.75,
        "risk": 1.25,
        "peak": 1.18,
        "eff": 0.72,
        "speed": 1.00,
        "jerk": 1.45,
    },
    "td3": {
        "success": 0.95,
        "crash": 1.02,
        "route": 0.96,
        "reward": 0.96,
        "risk": 1.03,
        "peak": 1.02,
        "eff": 0.96,
        "speed": 1.00,
        "jerk": 1.05,
    },
    "a2c": {
        "success": 0.68,
        "crash": 1.20,
        "route": 0.80,
        "reward": 0.80,
        "risk": 1.15,
        "peak": 1.12,
        "eff": 0.78,
        "speed": 0.92,
        "jerk": 1.18,
    },
}


def _make_pseudo_rows(real_df: pd.DataFrame, methods: list[str], seed: int, n_per_density: int | None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for density in _density_values(real_df):
        base = _base_stats(real_df, density)
        n = int(n_per_density or max(30, min(100, base["n"])))
        density_penalty = 1.0 + max(0.0, density - 0.1) * 0.75
        for method in methods:
            if method not in PSEUDO_PROFILES:
                continue
            p = PSEUDO_PROFILES[method]
            success_rate = _clip01(base["success"] * p["success"] / density_penalty)
            crash_rate = _clip01(base["crash_any"] * p["crash"] * density_penalty)
            route_mean = _clip01(base["route_completion"] * p["route"] / math.sqrt(density_penalty))
            reward_mean = base["ep_reward"] * p["reward"] / math.sqrt(density_penalty)
            risk_mean = max(0.0, base["risk"] * p["risk"] * density_penalty)
            peak_mean = max(0.0, base["peak_risk"] * p["peak"] * density_penalty)
            eff_mean = max(0.0, base["efficiency"] * p["eff"] / density_penalty)
            speed_mean = max(0.2, base["mean_speed"] * p["speed"])
            jerk_mean = max(0.01, base["jerk"] * p["jerk"] * density_penalty)
            for idx in range(n):
                success = float(rng.random() < success_rate)
                crash = float((not success) and (rng.random() < crash_rate))
                route = _clip01(rng.normal(route_mean + 0.18 * success - 0.12 * crash, 0.18))
                risk = _positive_normal(rng, risk_mean * (1.35 if crash else 0.85), max(0.05, risk_mean * 0.45))
                peak_risk = _positive_normal(rng, peak_mean * (1.25 if crash else 0.9), max(0.01, peak_mean * 0.35))
                near_miss = int(rng.poisson(max(0.05, base["near_miss"] * p["risk"] + 0.3 * crash)))
                ttc_viol = int(rng.poisson(max(0.03, base["ttc"] * p["risk"] + 0.2 * crash)))
                mean_speed = _positive_normal(rng, speed_mean * (0.85 if crash else 1.0), 0.9, floor=0.1)
                jerk = _positive_normal(rng, jerk_mean * (1.25 if crash else 1.0), max(0.03, jerk_mean * 0.35))
                rows.append(
                    {
                        "planner": method,
                        "protocol": "pseudo_metadrive_preview",
                        "seed": 900000 + idx,
                        "traffic_density": density,
                        "ep_len": int(max(20, rng.normal(110 + 120 * route, 35))),
                        "ep_reward": rng.normal(reward_mean + 35.0 * success - 25.0 * crash, max(1.0, abs(reward_mean) * 0.20)),
                        "ep_base_reward": rng.normal(reward_mean + 35.0 * success - 25.0 * crash, max(1.0, abs(reward_mean) * 0.20)),
                        "ep_cost": crash,
                        "route_completion": route,
                        "success": success,
                        "crash_any": crash,
                        "crash_vehicle": crash * float(rng.random() < 0.75),
                        "out_of_road": crash * float(rng.random() < 0.55),
                        "max_step": float((not crash) and (not success) and (rng.random() < 0.05)),
                        "overtake_vehicle_num": int(rng.poisson(0.2 + 0.8 * route)),
                        "cumulative_risk_exposure": risk,
                        "peak_risk": peak_risk,
                        "high_risk_steps": int(rng.poisson(max(0.02, 0.7 * risk))),
                        "near_miss_steps": near_miss,
                        "ttc_violation_steps": ttc_viol,
                        "mean_speed": mean_speed,
                        "mean_accel_abs": _positive_normal(rng, 2.8 + 0.6 * crash, 0.45),
                        "jerk_l1": jerk,
                        "risk_adjusted_efficiency": max(0.0, rng.normal(eff_mean + 0.08 * success - 0.08 * crash, 0.08)),
                        "overtakes_per_second": max(0.0, rng.normal(0.01 + 0.02 * route, 0.015)),
                        "source_file": "synthetic_preview",
                        "is_pseudo": True,
                        "pseudo_note": "Synthetic placeholder for visual layout only; replace with trained/evaluated algorithm results.",
                        "method": method,
                        "scenario_type": "Preview placeholder",
                    }
                )
    return pd.DataFrame(rows)


def _method_order(df: pd.DataFrame) -> list[str]:
    seen = list(dict.fromkeys(df["method"].astype(str).tolist()))
    ordered = [m for m in CANONICAL_ORDER if m in seen]
    ordered.extend([m for m in seen if m not in ordered])
    return ordered


def _display_label(method: str, df: pd.DataFrame) -> str:
    label = METHOD_LABELS.get(method, method)
    if bool(df.loc[df["method"] == method, "is_pseudo"].any()):
        return f"{label}*"
    return label


def _palette(methods: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {method: cmap(idx % 10) for idx, method in enumerate(methods)}


def _save(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_boxplots(df: pd.DataFrame, out_dir: Path) -> Path:
    methods = _method_order(df)
    colors = _palette(methods)
    metrics = [
        "route_completion",
        "risk_adjusted_efficiency",
        "cumulative_risk_exposure",
        "peak_risk",
        "mean_speed",
        "jerk_l1",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    axes = axes.ravel()
    labels = [_display_label(m, df) for m in methods]
    for ax, metric in zip(axes, metrics):
        data = [df.loc[df["method"] == method, metric].dropna().to_numpy() for method in methods]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.58 if not bool(df.loc[df["method"] == method, "is_pseudo"].any()) else 0.28)
            patch.set_edgecolor(colors[method])
            if bool(df.loc[df["method"] == method, "is_pseudo"].any()):
                patch.set_hatch("//")
        title, higher_better = METRIC_SPECS[metric]
        arrow = "$\\uparrow$" if higher_better else "$\\downarrow$"
        ax.set_title(f"{title} {arrow}")
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        if metric in {"route_completion", "risk_adjusted_efficiency"}:
            ax.set_ylim(bottom=0.0, top=max(1.0, float(df[metric].quantile(0.98)) * 1.1))
        else:
            ax.set_ylim(bottom=0.0)
    fig.suptitle("MetaDrive Policy Comparison: Episode-Level Distributions", y=1.03)
    fig.text(
        0.01,
        -0.03,
        "* synthetic placeholder. Box plots use real eval rows where available and generated rows only for missing algorithms.",
        fontsize=7,
    )
    out_base = out_dir / "metadrive_boxplot_preview"
    _save(fig, out_base)
    return out_base.with_suffix(".png")


def _plot_outcome_stack(df: pd.DataFrame, out_dir: Path, scenario_label: str) -> Path:
    methods = _method_order(df)
    labels = [_display_label(m, df) for m in methods]
    outcomes = ["success", "crash_any", "out_of_road", "timeout"]
    colors = {
        "success": "#2ca02c",
        "crash_any": "#d62728",
        "out_of_road": "#ff7f0e",
        "timeout": "#7f7f7f",
    }
    rates: dict[str, list[float]] = {key: [] for key in outcomes}
    for method in methods:
        sub = df[df["method"] == method]
        success = float(sub["success"].mean())
        crash = float(sub["crash_any"].mean())
        if "out_of_road" in sub:
            oor_mask = (
                (sub["out_of_road"].astype(float) > 0.5)
                & (sub["crash_any"].astype(float) <= 0.5)
                & (sub["success"].astype(float) <= 0.5)
            )
            oor = float(oor_mask.mean())
        else:
            oor = 0.0
        timeout = max(0.0, 1.0 - min(1.0, success + crash + oor))
        rates["success"].append(success)
        rates["crash_any"].append(crash)
        rates["out_of_road"].append(oor)
        rates["timeout"].append(timeout)

    fig, ax = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for key in outcomes:
        values = np.asarray(rates[key])
        label = {
            "success": "Success",
            "crash_any": "Crash",
            "out_of_road": "Out of road",
            "timeout": "Timeout/other",
        }[key]
        ax.bar(x, values, bottom=bottom, color=colors[key], label=label, width=0.72)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Episode fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Outcome Composition by Policy")
    ax.legend(frameon=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.text(0.01, -0.10, scenario_label, fontsize=7)
    out_base = out_dir / "metadrive_outcome_stack_preview"
    _save(fig, out_base)
    return out_base.with_suffix(".png")


def _plot_tradeoff_scatter(df: pd.DataFrame, out_dir: Path, scenario_label: str) -> Path:
    methods = _method_order(df)
    colors = _palette(methods)
    agg = df.groupby("method", dropna=False).agg(
        route_completion=("route_completion", "mean"),
        risk_exposure=("cumulative_risk_exposure", "mean"),
        crash_rate=("crash_any", "mean"),
        peak_risk=("peak_risk", "mean"),
        is_pseudo=("is_pseudo", "max"),
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    for method in methods:
        row = agg.loc[method]
        pseudo = bool(row["is_pseudo"])
        size = 90.0 + 520.0 * float(row["crash_rate"])
        marker = "s" if pseudo else "o"
        ax.scatter(
            row["risk_exposure"],
            row["route_completion"],
            s=size,
            marker=marker,
            color=colors[method],
            alpha=0.40 if pseudo else 0.78,
            edgecolor="black",
            linewidth=0.5,
            label=_display_label(method, df),
        )
        ax.annotate(
            _display_label(method, df),
            (row["risk_exposure"], row["route_completion"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Mean cumulative risk exposure $\\downarrow$")
    ax.set_ylabel("Mean route completion $\\uparrow$")
    ax.set_title("Risk-Efficiency Trade-Off")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0, top=max(1.0, float(agg["route_completion"].max()) * 1.12))
    ax.text(0.99, 0.03, "Bubble area: crash rate", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    fig.text(0.01, -0.04, scenario_label, fontsize=7)
    out_base = out_dir / "metadrive_risk_efficiency_tradeoff_preview"
    _save(fig, out_base)
    return out_base.with_suffix(".png")


def _normalise_metric(series: pd.Series, higher_better: bool) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if math.isclose(lo, hi):
        return pd.Series(np.ones(len(series)) * 0.5, index=series.index)
    score = (series - lo) / (hi - lo)
    return score if higher_better else 1.0 - score


def _plot_score_radar(df: pd.DataFrame, out_dir: Path, scenario_label: str) -> Path:
    methods = _method_order(df)
    colors = _palette(methods)
    agg = df.groupby("method", dropna=False).agg(
        success=("success", "mean"),
        crash_any=("crash_any", "mean"),
        route_completion=("route_completion", "mean"),
        cumulative_risk_exposure=("cumulative_risk_exposure", "mean"),
        jerk_l1=("jerk_l1", "mean"),
        is_pseudo=("is_pseudo", "max"),
    )
    axes_metrics = [
        ("success", "Success", True),
        ("crash_any", "Low crash", False),
        ("route_completion", "Progress", True),
        ("cumulative_risk_exposure", "Low risk", False),
        ("jerk_l1", "Comfort", False),
    ]
    score = pd.DataFrame(index=agg.index)
    for key, _label, higher_better in axes_metrics:
        score[key] = _normalise_metric(agg[key], higher_better)
    angles = np.linspace(0, 2 * np.pi, len(axes_metrics), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    fig, ax = plt.subplots(figsize=(4.9, 4.6), subplot_kw={"projection": "polar"}, constrained_layout=True)
    for method in methods:
        values = score.loc[method, [m[0] for m in axes_metrics]].to_numpy(dtype=float)
        values = np.concatenate([values, values[:1]])
        pseudo = bool(agg.loc[method, "is_pseudo"])
        if method not in {"idm", "stock_ppo", "risk_ppo", "sac", "td3"} and not pseudo:
            continue
        ax.plot(
            angles,
            values,
            color=colors[method],
            linestyle="--" if pseudo else "-",
            linewidth=1.5,
            label=_display_label(method, df),
        )
        ax.fill(angles, values, color=colors[method], alpha=0.04 if pseudo else 0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _key, label, _higher in axes_metrics])
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_ylim(0, 1)
    ax.set_title("Normalized Social-Risk Score Profile")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=True)
    fig.text(0.01, -0.04, scenario_label, fontsize=7)
    out_base = out_dir / "metadrive_score_radar_preview"
    _save(fig, out_base)
    return out_base.with_suffix(".png")


def _sem(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def _plot_density_lines(df: pd.DataFrame, out_dir: Path, scenario_label: str) -> Path:
    methods = _method_order(df)
    colors = _palette(methods)
    metrics = ["success", "crash_any", "route_completion", "cumulative_risk_exposure"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    axes = axes.ravel()
    grouped = (
        df.groupby(["method", "traffic_density"], dropna=False)
        .agg({metric: ["mean", _sem] for metric in metrics})
        .reset_index()
    )
    for ax, metric in zip(axes, metrics):
        for method in methods:
            sub = grouped[grouped["method"] == method].sort_values("traffic_density")
            if sub.empty:
                continue
            x = sub["traffic_density"].to_numpy(dtype=float)
            y = sub[(metric, "mean")].to_numpy(dtype=float)
            err = sub[(metric, "_sem")].to_numpy(dtype=float)
            is_pseudo = bool(df.loc[df["method"] == method, "is_pseudo"].any())
            line_style = "--" if is_pseudo else "-"
            marker = "s" if is_pseudo else "o"
            label = _display_label(method, df)
            ax.plot(x, y, color=colors[method], linestyle=line_style, marker=marker, markersize=3.2, label=label)
            ax.fill_between(x, y - err, y + err, color=colors[method], alpha=0.10 if not is_pseudo else 0.06)
        title, higher_better = METRIC_SPECS[metric]
        arrow = "$\\uparrow$" if higher_better else "$\\downarrow$"
        ax.set_title(f"{title} {arrow}")
        ax.set_xlabel("Traffic density")
        ax.set_ylim(bottom=0.0)
        if metric in {"success", "crash_any", "route_completion"}:
            ax.set_ylim(0.0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=True, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("MetaDrive Policy Comparison Across Traffic Density", y=1.16)
    fig.text(
        0.01,
        -0.03,
        f"{scenario_label}. Dashed lines and * labels are synthetic placeholders for plotting preview only.",
        fontsize=7,
    )
    out_base = out_dir / "metadrive_density_line_preview"
    _save(fig, out_base)
    return out_base.with_suffix(".png")


def _write_summary(df: pd.DataFrame, out_dir: Path) -> Path:
    metrics = [
        "success",
        "crash_any",
        "route_completion",
        "cumulative_risk_exposure",
        "peak_risk",
        "risk_adjusted_efficiency",
        "mean_speed",
        "jerk_l1",
        "near_miss_steps",
        "ttc_violation_steps",
    ]
    summary = (
        df.groupby(["method", "traffic_density", "is_pseudo"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    path = out_dir / "metadrive_preview_summary.csv"
    summary.to_csv(path, index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create SciencePlots-style MetaDrive benchmark preview figures. "
            "Real rows come from eval_episodes.csv; optional missing algorithms are synthetic placeholders."
        )
    )
    parser.add_argument(
        "--eval-csv",
        action="append",
        default=None,
        help="Path to eval_episodes.csv. Can be repeated. Defaults to matched_stock_vs_social_risk_eval if present.",
    )
    parser.add_argument(
        "--override-method-csv",
        action="append",
        default=[],
        metavar="METHOD=CSV",
        help=(
            "Replace rows for METHOD from the main eval CSVs with rows from CSV. "
            "Useful when a baseline such as IDM has been re-evaluated separately."
        ),
    )
    parser.add_argument("--out-dir", default="rl/logs/metadrive/benchmark_preview", help="Output directory.")
    parser.add_argument(
        "--pseudo-methods",
        default="dqn,sac,ddpg,td3,a2c",
        help="Comma-separated missing algorithms to synthesize for preview. Empty string disables pseudo rows.",
    )
    parser.add_argument("--pseudo-seed", type=int, default=7)
    parser.add_argument("--pseudo-n-per-density", type=int, default=0)
    parser.add_argument("--scenario-label", default=DEFAULT_SCENARIO_LABEL)
    args = parser.parse_args()

    _apply_style()
    paths = [Path(p) for p in args.eval_csv] if args.eval_csv else _default_eval_csvs()
    real_df = _load_eval_csvs(paths)
    override_specs: dict[str, Path] = {}
    for spec in args.override_method_csv:
        if "=" not in spec:
            raise ValueError(f"--override-method-csv must use METHOD=CSV, got: {spec}")
        method, path = spec.split("=", 1)
        override_specs[method.strip().lower()] = Path(path.strip())
    for method, path in override_specs.items():
        override_df = _load_eval_csvs([path])
        override_df = override_df[override_df["method"].astype(str).str.lower() == method]
        if override_df.empty:
            raise ValueError(f"Override CSV {path} contains no rows for method '{method}'")
        real_df = real_df[real_df["method"].astype(str).str.lower() != method]
        real_df = pd.concat([real_df, override_df], ignore_index=True, sort=False)
    pseudo_methods = [m.strip().lower() for m in args.pseudo_methods.split(",") if m.strip()]
    pseudo_methods = [m for m in pseudo_methods if m not in set(real_df["method"].astype(str))]
    pseudo_df = (
        _make_pseudo_rows(real_df, pseudo_methods, args.pseudo_seed, args.pseudo_n_per_density or None)
        if pseudo_methods
        else pd.DataFrame()
    )
    df = pd.concat([real_df, pseudo_df], ignore_index=True, sort=False)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "metadrive_preview_plot_data.csv"
    df.to_csv(data_path, index=False)
    summary_path = _write_summary(df, out_dir)
    boxplot_path = _plot_boxplots(df, out_dir)
    line_path = _plot_density_lines(df, out_dir, args.scenario_label)
    outcome_path = _plot_outcome_stack(df, out_dir, args.scenario_label)
    tradeoff_path = _plot_tradeoff_scatter(df, out_dir, args.scenario_label)
    radar_path = _plot_score_radar(df, out_dir, args.scenario_label)

    payload = {
        "real_eval_csvs": [str(p) for p in paths],
        "pseudo_methods": pseudo_methods,
        "rows": int(len(df)),
        "real_rows": int((~df["is_pseudo"]).sum()),
        "pseudo_rows": int(df["is_pseudo"].sum()),
        "outputs": {
            "plot_data": str(data_path),
            "summary": str(summary_path),
            "boxplot_png": str(boxplot_path),
            "boxplot_pdf": str(boxplot_path.with_suffix(".pdf")),
            "line_png": str(line_path),
            "line_pdf": str(line_path.with_suffix(".pdf")),
            "outcome_stack_png": str(outcome_path),
            "outcome_stack_pdf": str(outcome_path.with_suffix(".pdf")),
            "tradeoff_png": str(tradeoff_path),
            "tradeoff_pdf": str(tradeoff_path.with_suffix(".pdf")),
            "score_radar_png": str(radar_path),
            "score_radar_pdf": str(radar_path.with_suffix(".pdf")),
        },
        "scienceplots": bool(_HAS_SCIENCEPLOTS),
        "warning": "Pseudo rows are for visualization layout only and must be replaced by real trained/evaluated baselines.",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
