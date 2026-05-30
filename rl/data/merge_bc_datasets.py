"""
Merge two or more BC .npz datasets into a single file.

All inputs must share the same ``schema_version``, ``obs_dim``,
``n_actions`` and ``horizon_sec``. Per-sample arrays (obs, actions,
track_ids, recording_ids, t_rel, future_*, lc_success) are concatenated.
To avoid track_id / recording_id collisions across source datasets, each
source gets a large integer offset added to its recording_ids and
track_ids (configurable via ``--id-offset-step``).

Usage
-----
    python -m rl.data.merge_bc_datasets \
        --inputs rl/checkpoints/bc_dataset.npz \
                 rl/checkpoints/bc_highd_full.npz \
        --labels exiD highD \
        --out    rl/checkpoints/bc_combined.npz
"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path


META_KEYS = ("schema_version", "obs_dim", "n_actions", "horizon_sec")


def _load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Two or more BC .npz files to merge")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Optional short name per input (e.g. exiD highD). "
                         "Used only for log output.")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--id-offset-step", type=int, default=100_000,
                    help="Added to track_ids and recording_ids per source "
                         "to prevent collisions (default 100000)")
    args = ap.parse_args()

    datasets = [_load(p) for p in args.inputs]
    labels = args.labels or [Path(p).stem for p in args.inputs]
    assert len(labels) == len(datasets), "--labels must match --inputs count"

    # --- schema compatibility check -----------------------------------------
    base = datasets[0]
    for k in META_KEYS:
        for d, lbl in zip(datasets[1:], labels[1:]):
            if k in base and k in d:
                if not np.array_equal(base[k], d[k]):
                    raise SystemExit(
                        f"[merge] schema mismatch on '{k}': "
                        f"{labels[0]}={base[k]}  vs  {lbl}={d[k]}")

    # --- gather sample-axis keys -------------------------------------------
    n0 = int(base["obs"].shape[0])
    sample_keys = [k for k, v in base.items()
                   if hasattr(v, "shape") and v.ndim >= 1
                   and int(v.shape[0]) == n0 and k not in META_KEYS]
    print(f"[merge] sample-axis keys: {sample_keys}")

    # --- concat with id offsets --------------------------------------------
    acc = {k: [] for k in sample_keys}
    for i, (d, lbl) in enumerate(zip(datasets, labels)):
        n = int(d["obs"].shape[0])
        off = i * args.id_offset_step
        for k in sample_keys:
            if k not in d:
                raise SystemExit(f"[merge] '{lbl}' missing key '{k}'")
            arr = d[k]
            if k in ("track_ids", "recording_ids"):
                arr = arr.astype(np.int64) + off
            acc[k].append(arr)
        print(f"[merge]   {lbl:10s}  {n:>8d} samples  "
              f"(id_offset=+{off})")

    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    for k in META_KEYS:
        if k in base:
            out[k] = base[k]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    total = int(out["obs"].shape[0])
    print(f"[merge] wrote {total} samples to {args.out}")

    # quick action histogram so the user can see the LC share shift
    acts = out["actions"].astype(np.int64)
    hist = np.bincount(acts, minlength=int(out.get("n_actions", 9)))
    frac = hist / max(1, int(hist.sum()))
    lc_frac = float(1.0 - frac[0] - frac[1] - frac[2])  # actions 0..2 are lane_keep
    # (lane_bin = action // 3 — bin 0 is keep)
    lane_keep = float(frac[::3].sum() if acts.size else 0)
    lane_lc   = float(1.0 - lane_keep)
    print(f"[merge] combined LC fraction (lane_bin != 0): {lane_lc:.3%}")


if __name__ == "__main__":
    main()
