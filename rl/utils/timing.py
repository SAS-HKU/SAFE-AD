"""Lightweight timing instrumentation for the DREAM/RL stack.

Five compute categories are tracked:
  - pinn_train       : per-epoch training step
  - pinn_output      : per-call PINN inference (full grid or ego-local)
  - rl_train         : per-rollout / per-update RL training time
  - rl_inference     : per-decision wall time during evaluation
  - mpc_cbf          : per-solve LMPC / CBF QP wall time

All measurements use time.perf_counter for ms-resolution. When CUDA is
available and sync_cuda=True, the timer issues torch.cuda.synchronize()
around the measured block so GPU launches are not under-counted.
"""

from __future__ import annotations

import contextlib
import csv
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


CATEGORY_PINN_TRAIN = "pinn_train"
CATEGORY_PINN_OUTPUT = "pinn_output"
CATEGORY_RL_TRAIN = "rl_train"
CATEGORY_RL_INFERENCE = "rl_inference"
CATEGORY_MPC_CBF = "mpc_cbf"

KNOWN_CATEGORIES = (
    CATEGORY_PINN_TRAIN,
    CATEGORY_PINN_OUTPUT,
    CATEGORY_RL_TRAIN,
    CATEGORY_RL_INFERENCE,
    CATEGORY_MPC_CBF,
)


@dataclass
class TimingCategory:
    name: str
    n: int = 0
    total_s: float = 0.0
    samples: list = field(default_factory=list)
    cuda_peak_mb: float = 0.0
    extras: dict = field(default_factory=dict)

    def add(self, dt: float, peak_mem_mb: float = 0.0) -> None:
        self.n += 1
        self.total_s += float(dt)
        self.samples.append(float(dt))
        if peak_mem_mb > self.cuda_peak_mb:
            self.cuda_peak_mb = float(peak_mem_mb)

    def stats(self) -> dict:
        if not self.samples:
            return {
                "n": 0,
                "total_s": 0.0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "max_ms": 0.0,
                "cuda_peak_mb": 0.0,
            }
        if _HAS_NP:
            s = np.asarray(self.samples, dtype=float) * 1000.0
            return {
                "n": int(self.n),
                "total_s": float(self.total_s),
                "mean_ms": float(s.mean()),
                "p50_ms": float(np.percentile(s, 50)),
                "p95_ms": float(np.percentile(s, 95)),
                "p99_ms": float(np.percentile(s, 99)),
                "max_ms": float(s.max()),
                "cuda_peak_mb": float(self.cuda_peak_mb),
            }
        # numpy-free fallback
        sorted_ms = sorted(x * 1000.0 for x in self.samples)
        n = len(sorted_ms)
        def _p(q: float) -> float:
            idx = max(0, min(n - 1, int(round(q * (n - 1)))))
            return float(sorted_ms[idx])
        return {
            "n": n,
            "total_s": float(self.total_s),
            "mean_ms": float(sum(sorted_ms) / n),
            "p50_ms": _p(0.50),
            "p95_ms": _p(0.95),
            "p99_ms": _p(0.99),
            "max_ms": float(sorted_ms[-1]),
            "cuda_peak_mb": float(self.cuda_peak_mb),
        }


class Timer:
    """Thread-safe registry of TimingCategory objects."""

    def __init__(self, csv_path: Optional[str] = None) -> None:
        self.csv_path = csv_path
        self.cats: dict[str, TimingCategory] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> TimingCategory:
        with self._lock:
            if name not in self.cats:
                self.cats[name] = TimingCategory(name=name)
            return self.cats[name]

    @contextlib.contextmanager
    def measure(self, name: str, sync_cuda: bool = False):
        cat = self.get(name)
        if sync_cuda and _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        t0 = time.perf_counter()
        try:
            yield cat
        finally:
            if sync_cuda and _HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak_mb = 0.0
            if _HAS_TORCH and torch.cuda.is_available():
                try:
                    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
                except Exception:
                    peak_mb = 0.0
            cat.add(dt, peak_mem_mb=peak_mb)

    def add(self, name: str, dt: float, peak_mb: float = 0.0) -> None:
        self.get(name).add(dt, peak_mem_mb=peak_mb)

    def set_extra(self, name: str, key: str, value) -> None:
        self.get(name).extras[key] = value

    def write_csv(self, path: Optional[str] = None) -> Optional[str]:
        out = path or self.csv_path
        if not out:
            return None
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        rows = []
        for name in sorted(self.cats):
            stats = self.cats[name].stats()
            stats["name"] = name
            rows.append(stats)
        keys = ["name", "n", "total_s", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "cuda_peak_mb"]
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in keys})
        return out

    def summary(self) -> dict:
        return {name: cat.stats() for name, cat in self.cats.items()}


_GLOBAL_TIMER: Optional[Timer] = None
_GLOBAL_LOCK = threading.Lock()


def get_timer(csv_path: Optional[str] = None) -> Timer:
    """Return the process-wide Timer, creating it if needed."""
    global _GLOBAL_TIMER
    with _GLOBAL_LOCK:
        if _GLOBAL_TIMER is None:
            _GLOBAL_TIMER = Timer(csv_path=csv_path)
        elif csv_path is not None:
            _GLOBAL_TIMER.csv_path = csv_path
    return _GLOBAL_TIMER


def reset_timer() -> None:
    global _GLOBAL_TIMER
    with _GLOBAL_LOCK:
        _GLOBAL_TIMER = None


def measure(name: str, sync_cuda: bool = False):
    """Shorthand: ``with measure('pinn_output'): ...``"""
    return get_timer().measure(name, sync_cuda=sync_cuda)
