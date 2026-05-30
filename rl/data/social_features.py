"""
Social-friendliness feature primitives (schema v4).
====================================================

Pure-numpy helpers that turn the v3 per-frame state (already computed by
:mod:`rl.data.historical_extractor`) into the three families of social
features the project's behaviour-analysis review calls out:

A. **Courtesy / disturbance to others.**  How much braking, TTC loss
   and time-headway loss the ego *imposes* on the rear vehicle in the
   lane it is moving into.
B. **Decision quality.**  Whether the human's chosen action is
   consistent with the lane utility used by the policy reward
   (``missed_opportunity_flag``, ``bad_lc_flag``).
C. **Field-theoretic externality.**  Risk-mass / gradient / flux
   metrics over a small BEV grid sampled with the analytic
   :mod:`rl.data.risk_proxy` (DRIFT-calibrated; PINN/PDE overlay
   optional via ``RiskFieldQuery``).
D. **Composite scores + 5-class social label**, computable from A–C
   plus the v3 outcome labels.

These helpers are intentionally pure: pass arrays in, get arrays out.
This keeps them easy to unit-test and lets the caller decide the
batching strategy (per-frame in the extractor, vectorised in the
plotting script, etc.).

Schema-v4 additions
-------------------
The extractor's ``--include-social`` mode produces, for every sample::

    rear_decel_peak_3s, rear_ttc_now, rear_ttc_after, rear_ttc_delta,
    rear_thw_now, rear_thw_after, rear_thw_delta,
    hard_brake_imposed_flag, bad_cut_in_flag,
    missed_opportunity_flag, bad_lane_change_flag,
    risk_mass_total, risk_mass_others, risk_gradient_peak,
    risk_flux_backward, risk_field_entropy,
    safety_score, progress_score, courtesy_score,
    social_friendliness_score, social_class

``social_class`` is one of:

    0 = social_good        (safe, progressive, low disturbance)
    1 = social_defensive   (safe, low disturbance, poor progress)
    2 = social_aggressive  (progressive but high imposed risk/braking)
    3 = social_passive     (missed an obvious safe opportunity)
    4 = social_harmful     (raised risk or caused near-miss/braking)

Constants are exposed at module level so the plotting and reward
modules can reuse them without divergence.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from rl.data.risk_proxy import (
    SIGMA_KERNEL, V0_KERNEL,
    risk_at,
)


# ---------------------------------------------------------------------------
# Tunable constants — duplicated nowhere else.  Bump SCHEMA_VERSION on
# any change, otherwise downstream tools will silently consume stale
# semantics.
# ---------------------------------------------------------------------------

# Hard-brake threshold (m/s²): rear vehicle decelerating below this is
# treated as a "forced brake" caused by the ego maneuver.
HARD_BRAKE_DECEL_MPS2 = -3.0

# Bad-cut-in: TTC of the new rear collapses below this absolute value
# OR loses more than 30 % of its prior TTC after the ego maneuver.
BAD_CUT_TTC_ABS       = 2.5         # s
BAD_CUT_TTC_DROP_FRAC = 0.30

# Missed opportunity: best_adv (lane-utility margin) above this while
# the human chose to keep lane and was blocked by a slow leader.
MISSED_OPP_BEST_ADV   = 0.6

# Bad LC: human changed lane while best_adv was clearly negative.
BAD_LC_BEST_ADV       = -0.3

# BEV sampling for the risk-field metrics.  Cell area = DX_M × DY_M.
GRID_X_M  = (-15.0, 35.0)   # longitudinal, ego-frame
GRID_Y_M  = (-10.0,  10.0)  # lateral
DX_M = 1.0
DY_M = 1.0

# Composite-score normalisers.  Picked so that each component in
# {safety, progress, courtesy} sits roughly in [0, 1] on the existing
# datasets and so that the social score lies in [-1, 1].
SAFETY_RISK_NORM      = 0.6     # divides max(0, future_risk_change)
PROGRESS_SPEED_NORM   = 5.0     # m/s
COURTESY_DECEL_NORM   = 6.0     # m/s², 2× HARD_BRAKE
COURTESY_TTC_NORM     = 4.0     # s

# Composite weights.  Exposed so the policy reward can mirror them later.
W_SAFETY    = 0.40
W_PROGRESS  = 0.30
W_COURTESY  = 0.30
W_HESITATE  = 0.10
W_AGGRESS   = 0.10

# Class-decision thresholds operate on the three component scores.
GOOD_HIGH       = 0.55
DEFENSIVE_HIGH  = 0.50
AGGRESS_LOW     = 0.30
HARMFUL_LOW     = 0.20


# ---------------------------------------------------------------------------
# A. Courtesy / disturbance helpers
# ---------------------------------------------------------------------------

def _safe_ttc(gap_m: float, closing_mps: float) -> float:
    """TTC in seconds; +inf when not closing."""
    if closing_mps <= 1e-3 or gap_m <= 0:
        return float('inf')
    return float(gap_m) / float(closing_mps)


def _safe_thw(gap_m: float, follower_speed_mps: float) -> float:
    """Time-headway = gap / follower_speed."""
    if follower_speed_mps <= 1e-3 or gap_m <= 0:
        return float('inf')
    return float(gap_m) / float(follower_speed_mps)


def courtesy_block(
    target_rear_now: Optional[Mapping[str, float]],
    target_rear_traj: Optional[np.ndarray],
    ego_speed_now: float,
    ego_speed_after: float,
    dt: float,
) -> Dict[str, float]:
    """
    Compute the rear-vehicle disturbance features for one ego-frame.

    Args:
        target_rear_now : dict ``{ds, dvx, vx_long, x, y}`` describing the
                          target-lane rear neighbour (the vehicle that
                          becomes ego's follower after the maneuver).
                          ``ds`` is signed (negative = behind), ``dvx`` is
                          ``v_other_long − v_ego_long``.  ``None`` if no
                          such neighbour exists within the perception
                          range.
        target_rear_traj: ``np.ndarray`` shape ``[T, 4]`` of
                          ``[x, y, lon_speed, lon_accel]`` for that
                          neighbour over the outcome horizon.  ``None`` /
                          empty when the neighbour leaves the scene
                          before the horizon ends.
        ego_speed_now   : ego speed at i (m/s).
        dt              : timestep (s).

    Returns:
        Dict with keys::

            rear_decel_peak_3s, rear_ttc_now, rear_ttc_after,
            rear_ttc_delta, rear_thw_now, rear_thw_after,
            rear_thw_delta, hard_brake_imposed_flag, bad_cut_in_flag

        Missing values are filled with ``np.nan`` (continuous) or ``0``
        (flags) — never None — so the npz schema is fixed-size.
    """
    out = {
        'rear_decel_peak_3s':       float('nan'),
        'rear_ttc_now':             float('nan'),
        'rear_ttc_after':           float('nan'),
        'rear_ttc_delta':           float('nan'),
        'rear_thw_now':             float('nan'),
        'rear_thw_after':           float('nan'),
        'rear_thw_delta':           float('nan'),
        'hard_brake_imposed_flag':  0,
        'bad_cut_in_flag':          0,
    }
    if target_rear_now is None:
        return out

    ds = float(target_rear_now['ds'])
    if ds >= 0:                      # not behind us
        return out
    gap_now = abs(ds)
    dvx_now = float(target_rear_now.get('dvx', 0.0))
    follower_speed_now = float(target_rear_now.get('vx_long',
                                                   ego_speed_now + dvx_now))

    # +ve closing (ego frame): follower faster than ego  →  follower closes
    closing_now = float(follower_speed_now - ego_speed_now)
    out['rear_ttc_now'] = _safe_ttc(gap_now, closing_now)
    out['rear_thw_now'] = _safe_thw(gap_now, follower_speed_now)

    if target_rear_traj is None or len(target_rear_traj) == 0:
        return out

    # rear_decel_peak_3s = min lon_acc over horizon
    lon_acc = np.asarray(target_rear_traj[:, 3], dtype=np.float64)
    lon_acc = lon_acc[np.isfinite(lon_acc)]
    if lon_acc.size:
        out['rear_decel_peak_3s'] = float(lon_acc.min())
        if lon_acc.min() <= HARD_BRAKE_DECEL_MPS2:
            out['hard_brake_imposed_flag'] = 1

    # End-of-horizon TTC and THW.  Caller has already rotated the last
    # row of the trajectory into the ego frame at i+H so we can compute
    # gap and closing directly.
    last = target_rear_traj[-1]
    follower_x_after, follower_y_after = float(last[0]), float(last[1])
    follower_speed_after = float(last[2])

    gap_after = float(np.hypot(follower_x_after, follower_y_after))
    if follower_x_after > 0:        # follower has overshot ego
        gap_after = -gap_after

    closing_after = follower_speed_after - float(ego_speed_after)
    out['rear_ttc_after'] = _safe_ttc(gap_after if gap_after > 0 else 0.0,
                                      closing_after)
    out['rear_thw_after'] = _safe_thw(gap_after if gap_after > 0 else 0.0,
                                      follower_speed_after)

    # Deltas — use a finite cap so np.inf doesn't poison the npz.
    def _diff(a: float, b: float) -> float:
        if not (np.isfinite(a) and np.isfinite(b)):
            return float('nan')
        return float(a - b)

    out['rear_ttc_delta'] = _diff(out['rear_ttc_after'], out['rear_ttc_now'])
    out['rear_thw_delta'] = _diff(out['rear_thw_after'], out['rear_thw_now'])

    # Bad cut-in detection — only fires when the rear vehicle is
    # actually closing on ego after the maneuver (finite rear_ttc_after).
    if np.isfinite(out['rear_ttc_after']):
        if out['rear_ttc_after'] < BAD_CUT_TTC_ABS:
            out['bad_cut_in_flag'] = 1
        elif (np.isfinite(out['rear_ttc_now']) and out['rear_ttc_now'] > 0
              and (out['rear_ttc_now'] - out['rear_ttc_after']) /
                  out['rear_ttc_now'] > BAD_CUT_TTC_DROP_FRAC):
            out['bad_cut_in_flag'] = 1

    return out


# ---------------------------------------------------------------------------
# B. Decision-quality helpers
# ---------------------------------------------------------------------------

def decision_block(
    best_adv:                float,
    lane_delta_label:        int,
    blocked_by_leader_flag:  int,
) -> Dict[str, int]:
    """
    Cheap, frame-local decision-consistency labels.

    Returns ``missed_opportunity_flag`` and ``bad_lane_change_flag``.
    Track-level features (oscillation, hesitation, commitment) need a
    second pass over the per-track sequence and are intentionally not
    computed here — they belong on the per-agent case-study side.
    """
    is_keep = (lane_delta_label == 0)
    lc_taken = (lane_delta_label != 0)
    return {
        'missed_opportunity_flag': int(
            is_keep and (blocked_by_leader_flag == 1)
            and (best_adv > MISSED_OPP_BEST_ADV)
        ),
        'bad_lane_change_flag': int(
            lc_taken and (best_adv < BAD_LC_BEST_ADV)
        ),
    }


# ---------------------------------------------------------------------------
# C. Field-theoretic externality helpers
# ---------------------------------------------------------------------------

class RiskFieldQuery:
    """
    Adapter over a risk-field source.

    The default uses :func:`rl.data.risk_proxy.risk_at` so the field is
    DRIFT-calibrated and has no I/O.  The ``pinn_callable`` and
    ``drift_callable`` constructors let downstream tools plug in:

    * a trained PINN's :class:`pinn_risk_field.FieldInterpolator` (via
      ``.query(x_np, y_np, t_np)``);
    * a precomputed numerical PDE snapshot bundle (via a callable that
      takes ``(x, y)`` and returns ``R(x, y)``).

    All inputs are in the *ego-rotated* frame (forward = +x, left = +y).
    """

    def __init__(self, mode: str = 'analytic',
                 callable_=None, sigma: float = SIGMA_KERNEL):
        if mode not in ('analytic', 'pinn', 'drift'):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.callable_ = callable_
        self.sigma = sigma

    def field_grid(
        self,
        nbr_xs_ego: np.ndarray, nbr_ys_ego: np.ndarray,
        nbr_closing: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample R on the canonical BEV grid.

        Returns ``(R, X, Y)`` with shape ``[ny, nx]`` each, where R is
        non-negative.  X, Y are returned for downstream gradient and
        integration helpers.
        """
        xs = np.arange(GRID_X_M[0], GRID_X_M[1] + 1e-3, DX_M, dtype=np.float32)
        ys = np.arange(GRID_Y_M[0], GRID_Y_M[1] + 1e-3, DY_M, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        flat = np.zeros(X.size, dtype=np.float32)

        if self.mode == 'analytic':
            for k in range(X.size):
                flat[k] = risk_at(float(X.ravel()[k]), float(Y.ravel()[k]),
                                  nbr_xs_ego, nbr_ys_ego, nbr_closing,
                                  sigma=self.sigma)
        elif self.mode == 'pinn':
            # Caller-supplied .query(x_np, y_np, t_np) — t broadcast as 0.
            xs_np = X.ravel().astype(np.float32)
            ys_np = Y.ravel().astype(np.float32)
            ts_np = np.zeros_like(xs_np)
            flat[:] = np.asarray(self.callable_(xs_np, ys_np, ts_np),
                                 dtype=np.float32).ravel()
        elif self.mode == 'drift':
            for k in range(X.size):
                flat[k] = float(self.callable_(float(X.ravel()[k]),
                                               float(Y.ravel()[k])))

        R = np.maximum(flat.reshape(X.shape), 0.0)
        return R, X, Y


def field_metrics(
    R: np.ndarray, X: np.ndarray, Y: np.ndarray,
    nbr_xs_ego: np.ndarray, nbr_ys_ego: np.ndarray,
    nbr_closing: np.ndarray,
) -> Dict[str, float]:
    """
    Reduce a sampled BEV risk field to scalar externality features.

    All keys returned::

        risk_mass_total       — Σ R · dx · dy (proportional to integral)
        risk_mass_others      — Σ R · dx · dy weighted by Gaussian bumps
                                centred on every non-ego neighbour
        risk_gradient_peak    — max ‖∇R‖ over the grid
        risk_flux_backward    — backward-direction risk flux:
                                Σ R · max(0, -closing) summed cell-wise
                                over neighbours that are *behind* ego
        risk_field_entropy    — -Σ p log p (p = R / ΣR) — high = diffuse,
                                low = concentrated
    """
    cell = float(DX_M * DY_M)
    total = float(R.sum() * cell)

    # Risk mass attributed to "others' neighbourhoods": Gaussian
    # weighting with the same sigma as the kernel itself.
    mass_others = 0.0
    if nbr_xs_ego.shape[0]:
        sigma = SIGMA_KERNEL
        for nx, ny in zip(nbr_xs_ego, nbr_ys_ego):
            d2 = (X - float(nx))**2 + (Y - float(ny))**2
            w = np.exp(-d2 / (2.0 * sigma * sigma))
            mass_others += float((R * w).sum() * cell)

    # Gradient peak via finite differences
    gx, gy = np.gradient(R, DX_M, DY_M)  # gy first because Y is rows
    grad_mag = np.hypot(gx, gy)
    grad_peak = float(grad_mag.max()) if grad_mag.size else 0.0

    # Backward flux: contribution from neighbours that are receding from
    # behind (closing < 0 and behind ego).  This is a coarse surrogate
    # for the "shockwave/braking pressure" the expert calls out.
    flux_back = 0.0
    if nbr_xs_ego.shape[0]:
        is_behind = (nbr_xs_ego < 0)
        for nx, ny, c in zip(nbr_xs_ego[is_behind],
                             nbr_ys_ego[is_behind],
                             nbr_closing[is_behind]):
            if c >= 0:                # not receding
                continue
            sigma = SIGMA_KERNEL
            d2 = (X - float(nx))**2 + (Y - float(ny))**2
            w = np.exp(-d2 / (2.0 * sigma * sigma))
            flux_back += float((R * w * (-float(c))).sum() * cell)

    # Field entropy
    if total > 1e-9:
        p = R / R.sum()
        p_pos = p[p > 0]
        entropy = -float((p_pos * np.log(p_pos)).sum())
    else:
        entropy = 0.0

    return {
        'risk_mass_total':    total,
        'risk_mass_others':   mass_others,
        'risk_gradient_peak': grad_peak,
        'risk_flux_backward': flux_back,
        'risk_field_entropy': entropy,
    }


# ---------------------------------------------------------------------------
# D. Composite scores + 5-class label
# ---------------------------------------------------------------------------

def composite_scores(
    *,
    future_risk_change:        float,
    near_miss_future:          int,
    collision_future:          int,
    future_speed_gain:         float,
    escape_success_flag:       int,
    blocked_by_leader_flag:    int,
    rear_decel_peak_3s:        float,
    rear_ttc_delta:            float,
    hard_brake_imposed_flag:   int,
    bad_cut_in_flag:           int,
    missed_opportunity_flag:   int,
    bad_lane_change_flag:      int,
) -> Dict[str, float]:
    """
    Combine v3 outcome labels and v4 social features into one score
    per family + a 5-class categorical label.

    All component scores are clipped to [0, 1].  ``social_class`` is in
    {0, 1, 2, 3, 4} per the legend at the top of the module.
    """
    # Safety: bad if we raised risk OR are headed to a near-miss.
    raised_risk = max(0.0, float(future_risk_change)) / max(1e-3, SAFETY_RISK_NORM)
    safety = 1.0 - min(1.0, raised_risk
                       + 0.6 * float(near_miss_future)
                       + 1.0 * float(collision_future))
    safety = max(0.0, safety)

    # Progress: speed gain + successful escape, minus missed opportunities.
    progress = (max(0.0, float(future_speed_gain)) / PROGRESS_SPEED_NORM
                + 0.5 * float(escape_success_flag)
                - 0.5 * float(missed_opportunity_flag))
    progress = max(0.0, min(1.0, progress))

    # Courtesy: bad if we forced others to brake or shrank their TTC.
    decel_pen  = 0.0
    if np.isfinite(rear_decel_peak_3s):
        decel_pen = max(0.0, -float(rear_decel_peak_3s)) / COURTESY_DECEL_NORM
    ttc_pen = 0.0
    if np.isfinite(rear_ttc_delta):
        ttc_pen = max(0.0, -float(rear_ttc_delta)) / COURTESY_TTC_NORM
    courtesy = 1.0 - min(
        1.0,
        decel_pen
        + ttc_pen
        + 0.5 * float(hard_brake_imposed_flag)
        + 0.5 * float(bad_cut_in_flag),
    )
    courtesy = max(0.0, courtesy)

    social = (
        W_SAFETY   * safety
        + W_PROGRESS * progress
        + W_COURTESY * courtesy
        - W_HESITATE * float(missed_opportunity_flag)
        - W_AGGRESS  * float(bad_lane_change_flag)
    )
    social = float(np.clip(social, -1.0, 1.0))

    # 5-class label
    if (collision_future == 1 or near_miss_future == 1
            or hard_brake_imposed_flag == 1 or bad_cut_in_flag == 1
            or safety < HARMFUL_LOW or courtesy < HARMFUL_LOW):
        social_class = 4    # harmful
    elif (missed_opportunity_flag == 1
          or (blocked_by_leader_flag == 1 and progress < AGGRESS_LOW)):
        social_class = 3    # passive
    elif (bad_lane_change_flag == 1 or progress > GOOD_HIGH and safety < AGGRESS_LOW):
        social_class = 2    # aggressive
    elif (safety > GOOD_HIGH and courtesy > GOOD_HIGH and progress < DEFENSIVE_HIGH):
        social_class = 1    # defensive
    elif (safety > GOOD_HIGH and courtesy > GOOD_HIGH and progress > DEFENSIVE_HIGH):
        social_class = 0    # good
    else:
        # In-between: lean defensive if courtesy high, else passive
        social_class = 1 if courtesy > GOOD_HIGH else 3

    return {
        'safety_score':              safety,
        'progress_score':            progress,
        'courtesy_score':            courtesy,
        'social_friendliness_score': social,
        'social_class':              int(social_class),
    }


SOCIAL_CLASS_NAMES = (
    'social_good',          # 0
    'social_defensive',     # 1
    'social_aggressive',    # 2
    'social_passive',       # 3
    'social_harmful',       # 4
)


# ---------------------------------------------------------------------------
# Convenience: empty record (for the extractor's _empty_output path)
# ---------------------------------------------------------------------------

V4_KEYS_FLOAT = (
    'rear_decel_peak_3s', 'rear_ttc_now', 'rear_ttc_after', 'rear_ttc_delta',
    'rear_thw_now', 'rear_thw_after', 'rear_thw_delta',
    'risk_mass_total', 'risk_mass_others', 'risk_gradient_peak',
    'risk_flux_backward', 'risk_field_entropy',
    'safety_score', 'progress_score', 'courtesy_score',
    'social_friendliness_score',
)
V4_KEYS_INT = (
    'hard_brake_imposed_flag', 'bad_cut_in_flag',
    'missed_opportunity_flag', 'bad_lane_change_flag',
    'social_class',
)
V4_KEYS_ALL = V4_KEYS_FLOAT + V4_KEYS_INT


# ===========================================================================
# E. v5 PDE-propagated frame-level traffic-efficiency metrics
# ===========================================================================
# These are FRAME-level (not per-(ego, frame)) so they need to be
# pre-computed once per frame and broadcast onto every ego row that
# falls in that frame.  The extractor's per-track loop reads them out
# of a `frame_aggregates: dict[frame_id -> dict]` cache.
#
# Why frame-level and not per-ego?  The expert's brief makes the point
# that "traffic efficiency is a spatiotemporal propagation problem"
# and that the right object is the *scene* — number of agents in this
# frame, the close-pair interaction graph, the backward risk flux for
# the whole frame, etc.  Per-ego metrics like `risk_mass_others` then
# answer "what does ego contribute to this frame's externality".

# Pair-distance threshold for the interaction graph.
CLOSE_PAIR_DIST_M = 30.0
# Shockwave detection thresholds — heuristic; tunable per dataset.
SHOCKWAVE_GROWTH_THR     = 0.05      # Δrisk / dt above this counts
SHOCKWAVE_SPEED_VAR_DIFF = 0.50      # m²/s² jump in speed_variance
SHOCKWAVE_MIN_BACK_RATIO = 0.30      # backward / total

# STEI composite weights.  The defaults below produce STEI in roughly
# [-1, +2] on highD; use --stei-weights for a per-dataset override.
STEI_W_PROGRESS    = 1.00
STEI_W_RISK_PER_N  = 0.30
STEI_W_RISK_OTHERS = 0.30
STEI_W_BACK_FLUX   = 0.20
STEI_W_SPEED_VAR   = 0.10
STEI_W_HARD_BRAKE  = 0.10


def frame_interaction_block(
    entries: list,
    sigma_kernel: float = SIGMA_KERNEL,
    close_pair_dist: float = CLOSE_PAIR_DIST_M,
) -> Dict[str, float]:
    """
    Compute frame-level interaction-graph and propagated-risk metrics.

    Args:
        entries: list of ``(tid, x, y, vx, vy, heading_rad)`` for every
                 agent visible in the frame (output of
                 ``_build_frame_index``).

    Returns:
        Dict with frame-level keys::

            num_agents_frame, close_pair_count, closing_pair_count,
            interaction_density, closing_interaction_density,
            risk_mass_frame, risk_mass_per_agent,
            risk_per_close_pair, risk_per_closing_pair,
            risk_flux_backward_frame, backward_risk_flux_ratio,
            mean_speed_frame, speed_variance_frame,
            total_progress_rate_frame

    All values are scalars; the caller broadcasts to the per-(ego,
    frame) buffer.
    """
    n = len(entries)
    out = _empty_frame_block(n)
    if n == 0:
        return out

    xs  = np.asarray([e[1] for e in entries], dtype=np.float32)
    ys  = np.asarray([e[2] for e in entries], dtype=np.float32)
    vxs = np.asarray([e[3] for e in entries], dtype=np.float32)
    vys = np.asarray([e[4] for e in entries], dtype=np.float32)
    speeds = np.hypot(vxs, vys)

    out['num_agents_frame']         = float(n)
    out['mean_speed_frame']         = float(speeds.mean())
    out['speed_variance_frame']     = float(speeds.var())
    out['total_progress_rate_frame'] = float(speeds.sum())     # m/s, integrate later

    if n < 2:
        return out

    # Pairwise differences (n,n) — fine up to ~few hundred agents.
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    d  = np.sqrt(d2, where=np.isfinite(d2), out=np.full_like(d2, np.inf))

    close_mask = d < close_pair_dist
    n_close = int(close_mask.sum() // 2)         # unique unordered pairs
    out['close_pair_count']    = float(n_close)
    out['interaction_density'] = n_close / max(n, 1)

    # Closing-rate of each pair.  Positive = approaching.
    rel_dot = dx * (vxs[:, None] - vxs[None, :]) + dy * (vys[:, None] - vys[None, :])
    closing = -rel_dot / np.where(d > 0, d, 1.0)
    closing_mask = close_mask & (closing > 0.0)
    n_closing = int(closing_mask.sum() // 2)
    out['closing_pair_count']           = float(n_closing)
    out['closing_interaction_density']  = n_closing / max(n, 1)

    # Pairwise risk kernel, summed over close pairs only.
    pair_risk = (
        np.exp(-d2 / (2.0 * sigma_kernel * sigma_kernel))
        * np.clip(1.0 + np.maximum(0.0, closing) / V0_KERNEL, 1.0, 2.0)
    )
    pair_risk = pair_risk * close_mask
    risk_mass = float(pair_risk.sum() / 2.0)
    out['risk_mass_frame']         = risk_mass
    out['risk_mass_per_agent']     = risk_mass / max(n, 1)
    out['risk_per_close_pair']     = risk_mass / max(n_close, 1)
    out['risk_per_closing_pair']   = risk_mass / max(n_closing, 1)

    # Backward risk flux: pair-risk where agent j is downstream
    # (forward of i in world x) and ego-i is closing.  Captures the
    # pressure travelling against traffic flow.
    is_downstream = dx > 0          # j ahead of i (since dx[i,j] = x_i - x_j)
    backward_mask = closing_mask & is_downstream
    risk_back = float((pair_risk * backward_mask).sum() / 2.0)
    out['risk_flux_backward_frame'] = risk_back
    out['backward_risk_flux_ratio'] = risk_back / max(risk_mass, 1e-6)

    return out


def _empty_frame_block(n: int) -> Dict[str, float]:
    return {
        'num_agents_frame':            float(n),
        'close_pair_count':            0.0,
        'closing_pair_count':          0.0,
        'interaction_density':         0.0,
        'closing_interaction_density': 0.0,
        'risk_mass_frame':             0.0,
        'risk_mass_per_agent':         0.0,
        'risk_per_close_pair':         0.0,
        'risk_per_closing_pair':       0.0,
        'risk_flux_backward_frame':    0.0,
        'backward_risk_flux_ratio':    0.0,
        'mean_speed_frame':            0.0,
        'speed_variance_frame':        0.0,
        'total_progress_rate_frame':   0.0,
    }


def temporal_propagation_block(
    prev_risk_mass: Optional[float],
    curr_risk_mass: float,
    prev_speed_var: Optional[float],
    curr_speed_var: float,
    dt: float,
) -> Dict[str, float]:
    """Compute risk_mass_delta, growth rate, and the shockwave flag."""
    out = {
        'risk_mass_delta_frame':       0.0,
        'risk_mass_growth_rate_frame': 0.0,
        'shockwave_onset_flag':        0,
    }
    if prev_risk_mass is None:
        return out
    delta = float(curr_risk_mass - prev_risk_mass)
    out['risk_mass_delta_frame']       = delta
    out['risk_mass_growth_rate_frame'] = delta / max(dt, 1e-6)

    if prev_speed_var is not None:
        var_jump = curr_speed_var - prev_speed_var
        out['shockwave_onset_flag'] = int(
            out['risk_mass_growth_rate_frame'] > SHOCKWAVE_GROWTH_THR
            and var_jump > SHOCKWAVE_SPEED_VAR_DIFF
        )
    return out


def social_traffic_efficiency_index(
    *,
    progress_rate:           float,
    risk_mass_per_agent:     float,
    risk_others_per_agent:   float,
    backward_risk_flux_ratio: float,
    speed_variance:          float,
    hard_brake_rate:         float,
) -> float:
    """
    Scalar STEI (Social Traffic Efficiency Index) per frame.

    Designed so that a "good" frame — high progress, low per-agent
    risk, no backward shockwave, smooth speed — sits near +1, while a
    stop-and-go frame sits near 0 or below.

    All inputs should be normalised by the caller (typical range
    [0, 1] before weighting).  The defaults defined above produce
    STEI ≈ 0.0–1.5 on highD when the caller normalises by:
      * progress_rate / max(progress_rate over recording)
      * risk_mass_per_agent  ≈ /1.0 (already small in highD)
      * speed_variance / typical_speed²
    """
    return (
        STEI_W_PROGRESS    * float(progress_rate)
        - STEI_W_RISK_PER_N  * float(risk_mass_per_agent)
        - STEI_W_RISK_OTHERS * float(risk_others_per_agent)
        - STEI_W_BACK_FLUX   * float(backward_risk_flux_ratio)
        - STEI_W_SPEED_VAR   * float(speed_variance)
        - STEI_W_HARD_BRAKE  * float(hard_brake_rate)
    )


# Per-(ego, frame) v5 keys broadcast from the frame aggregates above.
V5_KEYS_FLOAT = (
    'num_agents_frame',
    'close_pair_count', 'closing_pair_count',
    'interaction_density', 'closing_interaction_density',
    'risk_mass_frame', 'risk_mass_per_agent',
    'risk_per_close_pair', 'risk_per_closing_pair',
    'risk_flux_backward_frame', 'backward_risk_flux_ratio',
    'mean_speed_frame', 'speed_variance_frame',
    'total_progress_rate_frame',
    'risk_mass_delta_frame', 'risk_mass_growth_rate_frame',
    'risk_adjusted_progress', 'social_traffic_efficiency_index',
)
V5_KEYS_INT = (
    'shockwave_onset_flag',
)
V5_KEYS_ALL = V5_KEYS_FLOAT + V5_KEYS_INT


# ===========================================================================
# F. Literature-backed Surrogate Safety Measures (SSMs) — schema v6
# ===========================================================================
# These are *named* metrics backed by published transportation-safety
# literature.  Adding them here (alongside our hand-crafted v4/v5
# metrics) gives reviewers a credible reference frame:
#
#   * TTC  — Time-To-Collision, the canonical surrogate-safety measure
#            (Hayward 1972; widely surveyed in Mahmud et al. 2017).
#            Critical threshold ≈ 1.5 s.
#   * THW  — Time Headway = gap / follower_speed.  Regulatory targets
#            commonly 1.8–3.0 s.
#   * DRAC — Deceleration Rate to Avoid Collision (Cooper & Ferguson
#            1976; Jiang et al. 2023).  Required follower decel; >4 m/s²
#            is treated as a critical event.
#   * MIN_DIST — minimum bumper-to-bumper distance over the horizon.
#   * SEI  — Safety–Efficiency Index (Andreotti et al. 2023): a
#            unitless composite penalising critical-TTC frames within
#            an efficiency normaliser.
#
# Reading the names: every key below carries an explicit unit suffix
# (`_s` for seconds, `_mps2` for accelerations, `_m` for distances)
# so reviewers and downstream tools cannot confuse them.

CRITICAL_TTC_S      = 1.5      # threshold for "critical" TTC events
CRITICAL_THW_S      = 1.0      # below this counts as a "tailgate"
CRITICAL_DRAC_MPS2  = 4.0      # collision threshold per Jiang 2023
TARGET_SPEED_MPS    = 27.78    # 100 km/h — highway target for EI
LANE_WIDTH_M_FOR_SAME_LANE = 3.5  # filter for same-lane pair detection


def surrogate_safety_block(
    entries: list,
    sigma_kernel: float = SIGMA_KERNEL,
    close_pair_dist: float = CLOSE_PAIR_DIST_M,
) -> Dict[str, float]:
    """
    Compute the per-frame distribution of literature SSMs (TTC, THW,
    DRAC, MIN_DIST) and the Andreotti-style EI / SEI composites.

    All inputs are in world coordinates, all outputs carry an explicit
    unit suffix.

    Returns:
        ttc_min_s, ttc_p10_s, frac_critical_ttc:
            min, 10th-percentile and rate of TTC<1.5s over closing pairs.
        mean_thw_s, frac_tailgate_thw:
            mean Time Headway and rate of THW<1.0s over close pairs.
        max_drac_mps2, frac_critical_drac:
            max DRAC and rate of DRAC>4 m/s² over closing pairs.
        min_distance_m:
            closest pair separation in the frame.
        efficiency_index_ei, safety_efficiency_index_sei:
            Andreotti et al. composites (unitless).

    Empty / single-agent frames return all NaN/0.
    """
    out = _empty_ssm_block()
    n = len(entries)
    if n < 2:
        return out

    xs  = np.asarray([e[1] for e in entries], dtype=np.float32)
    ys  = np.asarray([e[2] for e in entries], dtype=np.float32)
    vxs = np.asarray([e[3] for e in entries], dtype=np.float32)
    vys = np.asarray([e[4] for e in entries], dtype=np.float32)
    speeds = np.hypot(vxs, vys)

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    d  = np.sqrt(d2, where=np.isfinite(d2), out=np.full_like(d2, np.inf))

    out['min_distance_m'] = float(d.min())

    close_mask = d < close_pair_dist
    if not close_mask.any():
        return out

    abs_dy = np.abs(dy)
    same_lane = abs_dy < (LANE_WIDTH_M_FOR_SAME_LANE / 2.0)

    # --- TTC (same-lane closing pairs only) -----------------------------
    rel_dot = dx * (vxs[:, None] - vxs[None, :]) + dy * (vys[:, None] - vys[None, :])
    closing = -rel_dot / np.where(d > 0, d, 1.0)            # +ve = approaching
    closing_mask = close_mask & same_lane & (closing > 0.1)
    ttc = np.full_like(d, np.inf)
    np.divide(d, closing, out=ttc, where=closing_mask)
    finite_ttc = ttc[closing_mask]
    if finite_ttc.size:
        out['ttc_min_s']           = float(finite_ttc.min())
        out['ttc_p10_s']           = float(np.percentile(finite_ttc, 10))
        out['frac_critical_ttc']   = float((finite_ttc < CRITICAL_TTC_S).mean())

    # --- THW (same-lane follower→leader pairs) --------------------------
    # THW = longitudinal_gap / v_follower.  Pair (i,j) where j is ahead
    # of i (dx > 0)  →  i is the follower with absolute speed speeds[i].
    long_gap = np.abs(dx)
    follower_speed = speeds[:, None] + np.zeros_like(d)
    thw = np.full_like(d, np.inf)
    valid_thw = same_lane & (dx > 0) & (follower_speed > 0.5)
    np.divide(long_gap, follower_speed, out=thw, where=valid_thw)
    thw_finite = thw[valid_thw]
    if thw_finite.size:
        out['mean_thw_s']         = float(thw_finite.mean())
        out['frac_tailgate_thw']  = float((thw_finite < CRITICAL_THW_S).mean())

    # --- DRAC (required follower decel to avoid collision) -------------
    # DRAC_ij = (v_follower − v_lead)² / (2 · gap_long)   for *same-lane*
    # closing follower–leader pairs (Cooper & Ferguson 1976; Jiang et al.
    # 2023).  Same-lane filter and longitudinal gap match the 1-D
    # follower–leader assumption the literature definition makes.
    follower_mask = (dx > 0)            # j ahead of i → i is follower
    drac_pair_mask = closing_mask & follower_mask
    drac = np.zeros_like(d)
    np.divide(closing * closing, 2.0 * np.maximum(long_gap, 0.5),
              out=drac, where=drac_pair_mask)
    drac_in = drac[drac_pair_mask]
    if drac_in.size:
        # Clip the upper tail at 20 m/s² — beyond physical braking
        # capacity, the flag is more meaningful than the raw number.
        drac_clipped = np.minimum(drac_in, 20.0)
        out['max_drac_mps2']        = float(drac_clipped.max())
        out['frac_critical_drac']   = float((drac_in > CRITICAL_DRAC_MPS2).mean())

    # --- Efficiency Index (Andreotti et al. 2023 §4.2) -----------------
    # EI = mean_speed / V_target + (1 − coefficient_of_variation_spacing)
    # Spacing = nearest forward gap per agent.
    gap_each = d.copy()
    gap_each[~(dx > 0)] = np.inf                            # only forward neighbours
    gap_each[gap_each > 200.0] = np.inf
    nearest_fwd = gap_each.min(axis=1)
    valid_fwd = nearest_fwd[np.isfinite(nearest_fwd)]
    cv_spacing = (valid_fwd.std() / max(valid_fwd.mean(), 1e-6)
                  if valid_fwd.size > 1 else 0.0)
    ei = float(speeds.mean() / TARGET_SPEED_MPS + (1.0 - min(cv_spacing, 1.0)))
    out['efficiency_index_ei'] = ei
    out['safety_efficiency_index_sei'] = float(
        ei * (1.0 - out['frac_critical_ttc'])
    )
    return out


def _empty_ssm_block() -> Dict[str, float]:
    return {
        'ttc_min_s':                  float('nan'),
        'ttc_p10_s':                  float('nan'),
        'frac_critical_ttc':          0.0,
        'mean_thw_s':                 float('nan'),
        'frac_tailgate_thw':          0.0,
        'max_drac_mps2':              0.0,
        'frac_critical_drac':         0.0,
        'min_distance_m':             float('nan'),
        'efficiency_index_ei':        0.0,
        'safety_efficiency_index_sei': 0.0,
    }


# v6 keys exposed on the per-(ego, frame) row.  All carry an explicit
# unit suffix so the column meaning is unambiguous to a reviewer.
V6_KEYS_FLOAT = (
    'ttc_min_s', 'ttc_p10_s',
    'mean_thw_s', 'min_distance_m',
    'max_drac_mps2',
    'efficiency_index_ei', 'safety_efficiency_index_sei',
    'frac_critical_ttc', 'frac_tailgate_thw', 'frac_critical_drac',
)
V6_KEYS_INT = ()
V6_KEYS_ALL = V6_KEYS_FLOAT + V6_KEYS_INT


# ===========================================================================
# G. Display-name mapping — keep our internal column names stable but
# expose a literature-aligned title for every plot panel.
# ===========================================================================

DISPLAY_NAMES = {
    # Field-level renames (from Table 1 in metrics.md)
    'risk_mass_frame':              'Total Risk Potential',
    'risk_mass_total':              'Total Risk Potential',
    'risk_mass_others':             'Imposed Risk Potential',
    'risk_mass_per_agent':          'Mean Risk per Vehicle',
    'risk_per_close_pair':          'Risk per Close Pair',
    'risk_flux_backward_frame':     'Backward Risk Flux',
    'risk_flux_backward':           'Backward Risk Flux',
    'backward_risk_flux_ratio':     'Backward Flux Ratio',
    'risk_gradient_peak':           'Risk Gradient Peak',
    'risk_field_entropy':           'Risk Field Entropy',
    # Courtesy renames
    'rear_decel_peak_3s':           'Follower Peak Decel (DRAC, 3 s)',
    'rear_ttc_delta':               'Follower ΔTTC',
    'rear_thw_delta':               'Follower ΔTHW',
    'hard_brake_imposed_flag':      'Hard-Brake Imposed Rate',
    'bad_cut_in_flag':              'Bad Cut-In Rate',
    # Literature SSMs (no rename, just the canonical name)
    'ttc_min_s':                    'Min TTC (s)',
    'ttc_p10_s':                    'TTC P10 (s)',
    'mean_thw_s':                   'Mean THW (s)',
    'min_distance_m':               'Min Distance (m)',
    'max_drac_mps2':                'Max DRAC (m/s²)',
    'frac_critical_ttc':            'Frac TTC < 1.5 s',
    'frac_tailgate_thw':            'Frac THW < 1.0 s',
    'frac_critical_drac':           'Frac DRAC > 4 m/s²',
    'efficiency_index_ei':          'Efficiency Index (EI)',
    'safety_efficiency_index_sei':  'Safety–Efficiency Index (SEI)',
    # Frame-level
    'num_agents_frame':             'Agents in Frame',
    'mean_speed_frame':             'Mean Speed (m/s)',
    'speed_variance_frame':         'Speed Variance (m²/s²)',
    'interaction_density':          'Interaction Density',
    # v4 composites — kept but flagged "(experimental)" in plots
    'social_friendliness_score':    'Social Friendliness Score (experimental)',
    'safety_score':                 'Safety Score (experimental)',
    'progress_score':               'Progress Score (experimental)',
    'courtesy_score':               'Courtesy Score (experimental)',
}


def display_name(key: str) -> str:
    """Look up a literature-aligned display label for a key."""
    return DISPLAY_NAMES.get(key, key)
