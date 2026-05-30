"""
Behaviour-cloning pretraining for the DREAM decision policy.
=============================================================

Given an (obs, action) dataset extracted with
:mod:`rl.data.historical_extractor`, fit a
:class:`rl.policy.decision_policy.DecisionPolicy` via supervised
cross-entropy on the demonstration actions.

The resulting checkpoint is the "warm start" for either
(a) direct deployment into uncertainty_merger_DREAM.py as the
    rl-decision baseline, or
(b) PPO fine-tuning inside the merger decision env.

Usage
-----
    python -m rl.train_bc \
        --dataset rl/checkpoints/bc_dataset.npz \
        --out     rl/checkpoints/decision_policy_bc.pt \
        --epochs  20

The train/val split is by track_id (so the same human's frames are not
leaked across sets), and the action distribution is class-weighted
because lane-change demonstrations are rare compared with lane keeps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from rl.policy.decision_policy import (
    DEC_OBS_DIM,
    DEC_N_ACTIONS,
    DecisionPolicy,
)


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------

EXPECTED_SCHEMA_VERSION = 2   # v2: outcome-aware labels (backward-compatible load)


def _load_dataset(path: str):
    d = np.load(path)
    obs     = d["obs"].astype(np.float32)
    actions = d["actions"].astype(np.int64)
    tids    = d["track_ids"].astype(np.int64)

    if obs.shape[1] != DEC_OBS_DIM:
        raise ValueError(f"dataset obs dim {obs.shape[1]} != "
                         f"DEC_OBS_DIM {DEC_OBS_DIM}")
    if actions.max() >= DEC_N_ACTIONS or actions.min() < 0:
        raise ValueError(f"action out of range: [{actions.min()}, "
                         f"{actions.max()}]")

    # Optional but recommended: reject datasets produced by an older
    # extractor schema so we fail loudly instead of training silently on
    # mismatched obs layout.
    if "schema_version" in d.files:
        sv = int(d["schema_version"])
        if sv != EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"dataset schema_version={sv} != "
                f"EXPECTED_SCHEMA_VERSION={EXPECTED_SCHEMA_VERSION}. "
                "Re-extract with the current rl.data.historical_extractor.")

    return obs, actions, tids


def _split_by_track(tids: np.ndarray, val_frac: float = 0.15,
                    seed: int = 0) -> tuple:
    """Partition indices so whole tracks stay together."""
    rng = np.random.default_rng(seed)
    unique = np.unique(tids)
    rng.shuffle(unique)
    n_val_tracks = max(1, int(round(len(unique) * val_frac)))
    val_tracks = set(unique[:n_val_tracks].tolist())

    val_mask = np.array([int(t) in val_tracks for t in tids], dtype=bool)
    train_mask = ~val_mask
    return np.where(train_mask)[0], np.where(val_mask)[0]


def _class_weights(actions: np.ndarray, n_classes: int) -> torch.Tensor:
    """Mild inverse-frequency weighting for the CE loss (not the sampler)."""
    hist = np.bincount(actions, minlength=n_classes).astype(np.float32)
    hist = np.where(hist > 0, hist, 1.0)
    # sqrt-inverse frequency — less aggressive than raw inv-freq but still
    # tilts the loss toward rare classes. The heavy lifting is done by the
    # WeightedRandomSampler below; this just stops the common class from
    # dominating once the batch is already balanced.
    inv = 1.0 / np.sqrt(hist)
    inv *= n_classes / inv.sum()
    inv = np.clip(inv, a_min=0.5, a_max=3.0)
    return torch.as_tensor(inv, dtype=torch.float32)


def _sample_weights(actions: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-sample weights for a WeightedRandomSampler: 1/freq of own class."""
    hist = np.bincount(actions, minlength=n_classes).astype(np.float64)
    hist = np.where(hist > 0, hist, 1.0)
    per_class = 1.0 / hist
    return per_class[actions]


class ClassBalancedBatchSampler(torch.utils.data.Sampler):
    """Yield class-balanced batches without hitting the 2^24 category
    limit of ``torch.multinomial`` / ``WeightedRandomSampler``.

    For each batch, ``batch_size / n_classes`` indices are drawn uniformly
    at random (with replacement) from each class's index pool. This has
    the same end effect as inverse-frequency weighted sampling but scales
    to datasets of arbitrary size (we only call ``randint`` on small pools).
    """

    def __init__(self, actions: np.ndarray, n_classes: int,
                 batch_size: int, num_samples: int):
        self.n_classes = int(n_classes)
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        # Build per-class index pools once.
        a = np.asarray(actions, dtype=np.int64)
        self.pools = [np.where(a == c)[0] for c in range(n_classes)]
        # Classes that have zero samples drop out of sampling.
        self.active = [c for c, p in enumerate(self.pools) if len(p) > 0]
        if not self.active:
            raise ValueError("ClassBalancedBatchSampler: no classes populated")

    def __iter__(self):
        per_class = max(1, self.batch_size // len(self.active))
        emitted = 0
        while emitted < self.num_samples:
            batch = []
            for c in self.active:
                pool = self.pools[c]
                idx = np.random.randint(0, len(pool), size=per_class)
                batch.extend(int(i) for i in pool[idx])
            # Pad up to batch_size if per_class*active < batch_size.
            short = self.batch_size - len(batch)
            if short > 0:
                c = self.active[np.random.randint(0, len(self.active))]
                pool = self.pools[c]
                idx = np.random.randint(0, len(pool), size=short)
                batch.extend(int(i) for i in pool[idx])
            np.random.shuffle(batch)
            for i in batch:
                if emitted >= self.num_samples:
                    return
                yield i
                emitted += 1

    def __len__(self):
        return self.num_samples


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train_bc(
    dataset_path: str,
    out_path: str,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 3e-4,
    hidden: int = 128,
    val_frac: float = 0.15,
    seed: int = 0,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[bc] loading dataset: {dataset_path}")
    obs, actions, tids = _load_dataset(dataset_path)
    print(f"[bc] N={len(obs)}  unique_tracks={len(np.unique(tids))}")

    train_idx, val_idx = _split_by_track(tids, val_frac=val_frac, seed=seed)
    print(f"[bc] train={len(train_idx)}  val={len(val_idx)}")

    weights = _class_weights(actions[train_idx], DEC_N_ACTIONS).to(device)
    print(f"[bc] class weights: {weights.cpu().numpy().round(3)}")

    x_train = torch.as_tensor(obs[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(actions[train_idx], dtype=torch.int64)
    x_val   = torch.as_tensor(obs[val_idx],   dtype=torch.float32)
    y_val   = torch.as_tensor(actions[val_idx],   dtype=torch.int64)

    # Balanced minibatching. Without this the BC policy collapses to always
    # predicting class 0 (lane-keep + speed maintain) because that class is
    # ~88% of the dataset. For datasets up to ~16M samples we use
    # WeightedRandomSampler; beyond that we fall back to a custom
    # ClassBalancedBatchSampler because torch.multinomial caps categories
    # at 2^24.
    _WRS_LIMIT = 2 ** 24 - 1  # torch.multinomial limit
    if len(train_idx) <= _WRS_LIMIT:
        sw = _sample_weights(actions[train_idx], DEC_N_ACTIONS)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sw, dtype=torch.float64),
            num_samples=len(train_idx),
            replacement=True,
        )
        print(f"[bc] sampler: WeightedRandomSampler (N={len(train_idx)})")
    else:
        sampler = ClassBalancedBatchSampler(
            actions[train_idx], DEC_N_ACTIONS,
            batch_size=batch_size, num_samples=len(train_idx),
        )
        print(f"[bc] sampler: ClassBalancedBatchSampler "
              f"(N={len(train_idx)} > 2^24 — WeightedRandomSampler unavailable)")
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, sampler=sampler, drop_last=False,
    )

    model = DecisionPolicy(obs_dim=DEC_OBS_DIM,
                           n_actions=DEC_N_ACTIONS,
                           hidden=hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_val_acc   = -1.0
    best_state     = None
    best_per_class = [0.0] * DEC_N_ACTIONS

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss_sum = 0.0
        tr_acc_sum  = 0
        tr_n        = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss_sum += float(loss.item()) * len(xb)
            tr_acc_sum  += int((logits.argmax(-1) == yb).sum().item())
            tr_n        += len(xb)

        # Validation
        model.eval()
        with torch.no_grad():
            xb = x_val.to(device)
            yb = y_val.to(device)
            logits, _ = model(xb)
            val_loss = float(F.cross_entropy(logits, yb).item())
            preds    = logits.argmax(-1)
            val_acc  = float((preds == yb).float().mean().item())
            # Per-class accuracy (captures tail-class collapse).
            per_class_acc = np.zeros(DEC_N_ACTIONS, dtype=np.float32)
            per_class_n   = np.zeros(DEC_N_ACTIONS, dtype=np.int64)
            yb_cpu    = yb.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            for c in range(DEC_N_ACTIONS):
                m = yb_cpu == c
                per_class_n[c] = int(m.sum())
                if per_class_n[c] > 0:
                    per_class_acc[c] = float((preds_cpu[m] == c).mean())

        tr_loss = tr_loss_sum / max(1, tr_n)
        tr_acc  = tr_acc_sum  / max(1, tr_n)
        dt = time.time() - t0
        print(f"[bc] epoch {epoch:>2d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"({dt:.1f}s)")
        # Compact per-class line (useful because action class-0 dominates).
        pc_str = "  ".join(f"a{c}={per_class_acc[c]:.2f}"
                           for c in range(DEC_N_ACTIONS))
        print(f"[bc]          per_class_val_acc: {pc_str}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            best_per_class = per_class_acc.tolist()

    if best_state is None:
        best_state = model.state_dict()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "state_dict":     best_state,
        "obs_dim":        DEC_OBS_DIM,
        "n_actions":      DEC_N_ACTIONS,
        "hidden":         hidden,
        "val_acc":        best_val_acc,
        "per_class_acc":  best_per_class,
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "dataset":        os.path.abspath(dataset_path),
    }, out_path)
    print(f"[bc] saved best checkpoint (val_acc={best_val_acc:.3f}) → {out_path}")

    # Also save the final-epoch weights alongside the best checkpoint.
    final_path = out_path.replace(".pt", "_final.pt")
    torch.save({
        "state_dict":     {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()},
        "obs_dim":        DEC_OBS_DIM,
        "n_actions":      DEC_N_ACTIONS,
        "hidden":         hidden,
        "val_acc":        val_acc,
        "per_class_acc":  per_class_acc.tolist(),
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "dataset":        os.path.abspath(dataset_path),
    }, final_path)
    print(f"[bc] saved final-epoch checkpoint → {final_path}")

    # Also dump training metadata as a small sidecar json.
    meta = {
        "dataset":        os.path.abspath(dataset_path),
        "epochs":         epochs,
        "batch_size":     batch_size,
        "lr":             lr,
        "hidden":         hidden,
        "val_frac":       val_frac,
        "seed":           seed,
        "best_val_acc":   best_val_acc,
        "best_per_class": best_per_class,
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "n_samples":      int(len(obs)),
        "n_train":        int(len(train_idx)),
        "n_val":          int(len(val_idx)),
    }
    with open(out_path.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_decision_policy(
    checkpoint_path: str,
    device: str = "cpu",
) -> DecisionPolicy:
    """Utility: reload a trained policy from a BC checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden = int(ckpt.get("hidden", 128))
    obs_dim   = int(ckpt.get("obs_dim",   DEC_OBS_DIM))
    n_actions = int(ckpt.get("n_actions", DEC_N_ACTIONS))
    if obs_dim != DEC_OBS_DIM or n_actions != DEC_N_ACTIONS:
        raise ValueError(
            f"checkpoint obs/action dims ({obs_dim},{n_actions}) don't match "
            f"runtime ({DEC_OBS_DIM},{DEC_N_ACTIONS}); re-train after schema "
            "changes."
        )
    sv = int(ckpt.get("schema_version", -1))
    if sv != -1 and sv != EXPECTED_SCHEMA_VERSION:
        print(f"[bc] WARNING: checkpoint schema_version={sv} != "
              f"EXPECTED={EXPECTED_SCHEMA_VERSION}")
    model = DecisionPolicy(obs_dim=obs_dim,
                           n_actions=n_actions,
                           hidden=hidden).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out",     required=True)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--val-frac",    type=float, default=0.15)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--device",      type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_bc(
        dataset_path=args.dataset,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
    )
