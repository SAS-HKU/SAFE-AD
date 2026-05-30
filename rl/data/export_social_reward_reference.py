from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rl.data.historical_extractor import summarize_dataset
from rl.reward.social_reward import DEFAULT_SOCIAL_REWARD_CONFIG, SocialRewardConfig


SOCIAL_KEYS = (
    "rear_decel_peak_3s",
    "rear_ttc_delta",
    "rear_thw_delta",
    "hard_brake_imposed_flag",
    "bad_cut_in_flag",
    "missed_opportunity_flag",
    "bad_lane_change_flag",
    "risk_mass_total",
    "risk_mass_others",
    "risk_gradient_peak",
    "risk_flux_backward",
    "risk_field_entropy",
    "safety_score",
    "progress_score",
    "courtesy_score",
    "social_friendliness_score",
    "social_class",
)


def _load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def build_reference(inputs: list[str]) -> dict:
    datasets = []
    aggregate_social: dict[str, list[np.ndarray]] = {k: [] for k in SOCIAL_KEYS}
    for path in inputs:
        data = _load_npz(path)
        summary = summarize_dataset(data)
        social_summary = {}
        for key in SOCIAL_KEYS:
            if key in data:
                aggregate_social[key].append(np.asarray(data[key]))
                social_summary[key] = _safe_stats(np.asarray(data[key]))
        datasets.append(
            {
                "path": str(path),
                "summary": summary,
                "social_summary": social_summary,
            }
        )

    combined_social = {}
    for key, chunks in aggregate_social.items():
        if chunks:
            combined_social[key] = _safe_stats(np.concatenate([np.ravel(np.asarray(x)) for x in chunks]))

    cfg = DEFAULT_SOCIAL_REWARD_CONFIG.to_dict()
    cfg["calibration"] = {
        "inputs": [str(p) for p in inputs],
        "dataset_count": len(inputs),
        "dataset_summaries": datasets,
        "combined_social_summary": combined_social,
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a frozen social-reward reference JSON from dataset NPZ files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input NPZ files produced by historical_extractor.py --include-social")
    parser.add_argument("--out", required=True, help="Output JSON path, e.g. rl/config/social_reward_v1.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref = build_reference([str(Path(p)) for p in args.inputs])
    SocialRewardConfig(**{k: v for k, v in ref.items() if k in SocialRewardConfig.__dataclass_fields__}).save(str(out_path))
    print(json.dumps({"out": str(out_path), "inputs": args.inputs}, indent=2))


if __name__ == "__main__":
    main()
