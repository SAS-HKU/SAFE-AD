"""
Analytic DRIFT-shape risk proxy for offline dataset extraction.
================================================================

exiD / highD recordings have no DRIFT PDE field attached. The BC
pipeline still needs risk features so that the policy can later be
deployed against the live :class:`Integration.drift_interface.DRIFTInterface`
without a feature-distribution shift.

This module provides a purely analytic, neighbour-kernel stand-in whose
**shape** and **range** match the simulator's risk queries:

* :func:`risk_at`                    ↔ ``DRIFTInterface.get_risk_cartesian``
* :func:`risk_max_along_segment`     ↔ ``DRIFTInterface.get_risk_corridor``
* :func:`risk_gradient`              ↔ ``DRIFTInterface.get_risk_gradient``

All routines operate in the *ego-rotated frame*:

    x = longitudinal (along ego heading, +ve forward)
    y = lateral      (+ve to ego's left)

The ego itself is at the origin.  Callers must rotate neighbours into
this frame before calling (see :func:`rotate_neighbours_to_ego`).

Design note
-----------
Running a DRIFT PDE per exiD frame would be prohibitive — the proxy is a
feature-extraction convenience, not a claim about risk theory.  We
preserve:

* non-negativity,
* magnitude (~[0, 2]) so that ``rl_config.NORM_RISK = 5.0`` is shared,
* sign of the lateral gradient (left-risky vs right-risky),
* saturation at a fixed closing-speed cap,

which is enough for the BC auxiliary heads to learn features that remain
meaningful when the simulator switches the field source.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tuning constants (single source of truth — reuse everywhere)
# ---------------------------------------------------------------------------
SIGMA_KERNEL: float = 6.0   # Gaussian width (m); mean(risk_ego_now) ≈ [0, 2]
V0_KERNEL:    float = 5.0   # closing-speed normaliser (m/s); matches LANE_ADV_DV_V0
W_CLOSING_CAP: float = 2.0  # cap on positive closing weight (saturates tail)


# ---------------------------------------------------------------------------
# Neighbour prep
# ---------------------------------------------------------------------------

def rotate_neighbours_to_ego(
    ego_x: float, ego_y: float, ego_psi: float,
    v_ego_long: float,
    entries: list,
    self_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate a ``frame_index`` neighbour list into the ego frame and
    compute per-neighbour longitudinal closing speed.

    Args:
        ego_x, ego_y   : ego position in world coordinates (m)
        ego_psi        : ego heading (rad)
        v_ego_long     : ego speed along ego heading (m/s)
        entries        : list of tuples ``(tid, x, y, vx, vy, heading_rad)``
                         as produced by
                         :func:`rl.data.historical_extractor._build_frame_index`
        self_id        : ego track id (excluded from the output)

    Returns:
        ``(xs_ego, ys_ego, closing)`` each shape ``[K]``, float32, where
        ``closing = v_ego_long − v_neighbour_along_ego_heading`` — positive
        means the ego is approaching that neighbour.  ``K`` is the number
        of neighbours after ``self_id`` is dropped (may be 0).
    """
    if not entries:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty.copy(), empty.copy()

    c = float(np.cos(ego_psi))
    s = float(np.sin(ego_psi))

    # Unpack into columns (drop self)
    xs = []
    ys = []
    vxs = []
    vys = []
    for (tid, x, y, vx, vy, _psi) in entries:
        if int(tid) == int(self_id):
            continue
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)

    if not xs:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty.copy(), empty.copy()

    xs  = np.asarray(xs,  dtype=np.float32)
    ys  = np.asarray(ys,  dtype=np.float32)
    vxs = np.asarray(vxs, dtype=np.float32)
    vys = np.asarray(vys, dtype=np.float32)

    dx = xs - np.float32(ego_x)
    dy = ys - np.float32(ego_y)

    xs_ego = dx * c + dy * s        # longitudinal (forward = +)
    ys_ego = -dx * s + dy * c       # lateral      (left    = +)

    v_long_other = vxs * c + vys * s
    closing = np.float32(v_ego_long) - v_long_other   # +ve = ego closing

    return xs_ego, ys_ego, closing


# ---------------------------------------------------------------------------
# Risk queries (ego frame)
# ---------------------------------------------------------------------------

def risk_at(
    px: float, py: float,
    nbr_xs_ego: np.ndarray,
    nbr_ys_ego: np.ndarray,
    nbr_closing: np.ndarray,
    sigma: float = SIGMA_KERNEL,
) -> float:
    """
    Scalar risk at ego-frame query point ``(px, py)``.

    Non-negative. Units comparable to
    :meth:`Integration.drift_interface.DRIFTInterface.get_risk_cartesian`.
    """
    if nbr_xs_ego.shape[0] == 0:
        return 0.0
    dx = nbr_xs_ego - np.float32(px)
    dy = nbr_ys_ego - np.float32(py)
    d2 = dx * dx + dy * dy
    weight = np.clip(nbr_closing / np.float32(V0_KERNEL),
                     0.0, np.float32(W_CLOSING_CAP))
    contrib = weight * np.exp(-d2 / np.float32(2.0 * sigma * sigma))
    return float(contrib.sum())


def risk_max_along_segment(
    x_start: float, y_lane: float,
    length: float, n_samples: int,
    nbr_xs_ego: np.ndarray,
    nbr_ys_ego: np.ndarray,
    nbr_closing: np.ndarray,
    sigma: float = SIGMA_KERNEL,
) -> float:
    """Maximum risk along a lane-parallel segment (matches ``get_risk_corridor``)."""
    n = max(1, int(n_samples))
    if length <= 0.0:
        return risk_at(x_start, y_lane,
                       nbr_xs_ego, nbr_ys_ego, nbr_closing, sigma=sigma)
    xs = np.linspace(float(x_start), float(x_start) + float(length), n)
    best = 0.0
    for x in xs:
        r = risk_at(float(x), float(y_lane),
                    nbr_xs_ego, nbr_ys_ego, nbr_closing, sigma=sigma)
        if r > best:
            best = r
    return best


def risk_corridor_tau(
    ego_vx: float, y_lane: float,
    tau_sec: float, n_samples: int,
    nbr_xs_ego: np.ndarray,
    nbr_ys_ego: np.ndarray,
    nbr_closing: np.ndarray,
    sigma: float = SIGMA_KERNEL,
) -> float:
    """
    Max risk along a forward corridor whose length is ``τ · v_ego``.

    Mirrors the time-parametrised lookahead done at PPO-observation time
    for the live DRIFT field.
    """
    length = max(0.0, float(ego_vx) * float(tau_sec))
    return risk_max_along_segment(
        0.0, y_lane, length, n_samples,
        nbr_xs_ego, nbr_ys_ego, nbr_closing, sigma=sigma,
    )


def risk_gradient(
    px: float, py: float,
    nbr_xs_ego: np.ndarray,
    nbr_ys_ego: np.ndarray,
    nbr_closing: np.ndarray,
    eps: float = 1.0,
    sigma: float = SIGMA_KERNEL,
) -> Tuple[float, float]:
    """Central finite-difference gradient ``(∂R/∂x, ∂R/∂y)``."""
    gx = 0.5 * (
        risk_at(px + eps, py, nbr_xs_ego, nbr_ys_ego, nbr_closing, sigma=sigma)
        - risk_at(px - eps, py, nbr_xs_ego, nbr_ys_ego, nbr_closing, sigma=sigma)
    )
    gy = 0.5 * (
        risk_at(px, py + eps, nbr_xs_ego, nbr_ys_ego, nbr_closing, sigma=sigma)
        - risk_at(px, py - eps, nbr_xs_ego, nbr_ys_ego, nbr_closing, sigma=sigma)
    )
    return float(gx), float(gy)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic scene: one approaching leader 20 m ahead in the same lane,
    # one far vehicle 50 m ahead (should not dominate the ego cell).
    xs_ego = np.array([20.0, 50.0, -15.0], dtype=np.float32)
    ys_ego = np.array([ 0.0,  0.0,   3.5], dtype=np.float32)
    closing = np.array([4.0, -1.0, 0.0], dtype=np.float32)   # only leader approaches

    r0 = risk_at(0.0, 0.0, xs_ego, ys_ego, closing)
    r_fwd_10 = risk_at(10.0, 0.0, xs_ego, ys_ego, closing)
    r_fwd_20 = risk_at(20.0, 0.0, xs_ego, ys_ego, closing)
    r_left_lane = risk_max_along_segment(0.0, 3.5, 20.0, 6,
                                         xs_ego, ys_ego, closing)
    gx, gy = risk_gradient(0.0, 0.0, xs_ego, ys_ego, closing)

    print(f"risk_ego_now     = {r0:.4f}")
    print(f"risk_fwd_10m     = {r_fwd_10:.4f}  (should exceed risk_ego_now)")
    print(f"risk_fwd_20m     = {r_fwd_20:.4f}  (peak at leader)")
    # NB: SIGMA_KERNEL=6 m is wider than a lane (3.5 m), so the leader's
    # Gaussian bleeds into the adjacent-lane corridor.  We therefore expect
    # risk_left_lane < risk_fwd_20m (leader off-axis by 3.5 m) but > 0.
    print(f"risk_left_lane   = {r_left_lane:.4f}  (< risk_fwd_20m, leader spillover)")
    print(f"risk_grad        = ({gx:+.4f}, {gy:+.4f})  (grad_x should be +)")
    assert r_fwd_20 > r0, "forward risk should grow toward the leader"
    assert r_left_lane < r_fwd_20, "leader spillover must attenuate at 3.5 m offset"
    assert gx > 0, "longitudinal gradient should point toward leader"
    print("OK")
