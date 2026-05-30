"""
Historical trajectory extractor for behaviour-cloning pretraining.
==================================================================

Reads normalized traffic recordings from the integrated datasets
(``exiD``, ``highD``, ``inD``, ``rounD``, ``uniD``, ``SQM-N-4``,
``YTDJ-3``, ``XAM-N-5``, ``XAM-N-6``), picks every moving car track as an
"ego candidate", rotates the scene into that track's ego frame,
detects lane-change events from the ``laneChange`` stream, and emits
one record per timestep containing:

1. the 17-dim decision observation (`build_decision_obs`),
2. raw ego / surrounding-traffic scalars needed by auxiliary features,
3. a DRIFT-shape **analytic risk proxy** (see :mod:`rl.data.risk_proxy`)
   — ``risk_ego_now``, time-parametrised forward corridors
   (``risk_corridor_tau[τ∈{1,2,3,4}s]``), adjacent-lane corridor risks,
   and ``(∂R/∂x, ∂R/∂y)``,
4. lane-wise tactical utilities (``gap_fwd_j``, ``rel_speed_j``,
   ``lane_risk_j``, ``utility_j``) plus ``adv_{left,right}`` / ``best_adv``,
5. the primary human action label in both the original 9-way form and
   decomposed (``lane_delta_label``, ``speed_mode_label``),
6. outcome-aware future labels over a configurable horizon:
   ``future_lane_delta``, ``future_speed_delta / gain``, ``future_gap_gain``,
   ``future_risk_change``, ``lane_change_success``, ``near_miss_future``,
   ``collision_future``, ``blocked_by_leader_flag``, ``escape_success_flag``,
   ``lane_change_advantage_flag``, ``short_horizon_return_proxy``.

Schema v3 (this file) is **not** backward-compatible with v2 — train_bc
will reject older ``.npz`` files. Re-extract after pulling this commit.

Usage
-----
    python -m rl.data.historical_extractor \
        --dataset-format exiD \
        --data-dir data/exiD \
        --recordings 00,01,02 \
        --out-path rl/checkpoints/bc_dataset_v3.npz \
        --horizon-sec 1.5 --outcome-horizon-sec 3.0

Design choices
--------------
* Ego frame is obtained by rotating world vectors by the per-frame ego
  heading so the analytic risk proxy can reuse the rotated-neighbour
  arrays unchanged.
* Lane indices are assigned by cumulatively integrating ``laneChange``
  events and resolving direction via lateral displacement over a ±30
  frame window (unchanged from v2).
* Action labels are "what the human did over the next *action* horizon".
  Outcome labels are "what happened to the human over the next *outcome*
  horizon" — separately configurable (default 1.5 s / 3.0 s).
* These datasets have **no DRIFT field**; all risk features use the
  analytic proxy in :mod:`rl.data.risk_proxy` whose magnitude is
  calibrated to match ``DRIFTInterface.get_risk_cartesian``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from rl.policy.decision_policy import (
    DEC_OBS_DIM,
    DEC_N_ACTIONS,
    build_decision_obs,
    encode_action,
    PERCEPTION_RANGE,
)
from rl.data.risk_proxy import (
    SIGMA_KERNEL, V0_KERNEL,
    rotate_neighbours_to_ego,
    risk_at, risk_max_along_segment, risk_corridor_tau, risk_gradient,
)


DATASET_DIR_NAMES = {
    "exid": "exiD",
    "highd": "highD",
    "ind": "inD",
    "round": "rounD",
    "roundd": "rounD",
    "unid": "uniD",
    "sqm-n-4": "SQM-N-4",
    "ytdj-3": "YTDJ-3",
    "xam-n-5": "XAM-N-5",
    "xam-n-6": "XAM-N-6",
}
SPECIAL_DATASET_KEYS = {"sqm-n-4", "ytdj-3", "xam-n-5", "xam-n-6"}
UNSUPPORTED_DATASET_KEYS = {"pkdd-8", "rml-7"}
SUPPORTED_DATASET_HELP = ", ".join(
    ["exiD", "highD", "inD", "rounD/round", "uniD", "SQM-N-4", "YTDJ-3", "XAM-N-5", "XAM-N-6"]
)


# tracks_import pulls in pandas. Defer so the module can be imported in
# environments that only need the extraction primitives.
def _lazy_read_from_dataset(*args, **kwargs):
    from tracks_import import read_from_dataset
    return read_from_dataset(*args, **kwargs)


def _canonical_dataset_name(dataset_format: str) -> str:
    dataset_key = str(dataset_format).strip().lower()
    return DATASET_DIR_NAMES.get(dataset_key, str(dataset_format).strip())


def _dataset_key(dataset_format: str) -> str:
    return _canonical_dataset_name(dataset_format).lower()


def _is_special_dataset(dataset_format: str) -> bool:
    return _dataset_key(dataset_format) in SPECIAL_DATASET_KEYS


def _default_data_dir(dataset_format: str) -> str:
    return os.path.join(REPO_ROOT, "data", _canonical_dataset_name(dataset_format))


def _recording_id_to_int(recording_id: str, fallback: int) -> int:
    token = str(recording_id).strip()
    return int(token) if token.isdigit() else int(fallback)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

MIN_TRACK_DURATION_S = 4.0   # skip very short tracks
MIN_EGO_SPEED        = 2.0   # skip parked / very slow vehicles
HORIZON_SEC_DEFAULT  = 1.5   # action-label horizon (s)
OUTCOME_HORIZON_SEC  = 3.0   # outcome-label horizon (s)
SPEED_MODE_THRESH    = 0.8   # m/s — smaller dv counted as "maintain"
LANE_WIDTH_ASSUMED   = 3.5   # used for adjacent-lane risk sampling
MAX_NEIGHBOURS       = 6     # slots exposed via build_decision_obs

# Lane utility constants — MUST match rl.config.rl_config and
# rl.reward.reward_fn._lane_advantage so offline and online agree.
LANE_ADV_D0 = 30.0
LANE_ADV_V0 =  5.0
LANE_ADV_R0 =  2.0

# Outcome thresholds (frozen here so labels are reproducible)
NEAR_MISS_THR      = 6.0    # m — future near-miss
COLLISION_THR      = 2.0    # m — future collision surrogate
BLOCKED_GAP_THR    = 25.0   # m — "leader is close"
BLOCKED_SPEED_FRAC = 0.7    # fraction of ego speed → "leader is slow"
TAU_D              = 5.0    # m — future_gap_gain threshold
TAU_R              = 0.2    # normalised risk — future_risk_change threshold
SETTLING_FRAMES_DEFAULT = 10

# Risk corridor sampling
CORRIDOR_N_SAMPLES = 6
TAU_SECONDS        = (1.0, 2.0, 3.0, 4.0)   # τ grid for risk_corridor_tau

# Short-horizon return proxy weights (match the paper's tactical shaping;
# keep low relative magnitudes so the target stays bounded)
RET_W_PROGRESS   = 1.0
RET_W_RISK       = 0.2
RET_W_COMFORT    = 0.1
RET_W_NEAR_MISS  = 3.0

# Bump when schema or labelling rules change so old datasets are rejected
# loudly at training time.
#   v3: propagated-risk, lane-utility, and outcome-aware labels.
#   v4: rear-vehicle courtesy features, decision-quality flags, BEV
#       risk-field externality metrics, composite social scores +
#       5-class label.  See rl/data/social_features.py for definitions.
#   v5: PDE-propagated frame-level traffic-efficiency metrics —
#       agent-normalised risk, interaction-graph statistics, temporal
#       propagation, Social Traffic Efficiency Index (STEI).
#   v6: literature-backed Surrogate Safety Measures (TTC, THW, DRAC,
#       MIN_DIST) and the Andreotti EI / SEI composites — exposes a
#       reviewer-credible reference frame next to the v4/v5
#       hand-crafted metrics.  See rl/data/metrics.md.
SCHEMA_VERSION = 6


# ---------------------------------------------------------------------------
# Per-recording helpers (unchanged from v2)
# ---------------------------------------------------------------------------

def _primary_lanelet(lanelet_row) -> int | None:
    """Return the first non-NaN lanelet id in the row, or None."""
    for v in lanelet_row:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        return int(v)
    return None


def _assign_lane_index(track) -> np.ndarray:
    """
    Walk a track and assign each frame a signed lane index (0 at start).
    Uses the per-frame ``laneChange`` marker and the sign of rotated
    world-lateral displacement over a ±30-frame window.
    """
    n = len(track["xCenter"])
    lane_idx = np.zeros(n, dtype=np.int64)
    if n < 3:
        return lane_idx

    xs = np.asarray(track["xCenter"], dtype=np.float32)
    ys = np.asarray(track["yCenter"], dtype=np.float32)

    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    theta = np.arctan2(dy, dx) if abs(dx) + abs(dy) > 1e-3 else 0.0
    c, s = np.cos(theta), np.sin(theta)
    lat_world = -(xs - xs[0]) * s + (ys - ys[0]) * c

    lc = track.get("laneChange", None)
    if lc is None:
        return lane_idx
    lc_arr = np.asarray(lc).reshape(-1)

    current = 0
    WIN = 30
    for i in range(n):
        if lc_arr[i] == 0:
            lane_idx[i] = current
            continue
        i0 = max(0, i - WIN)
        i1 = min(n - 1, i + WIN)
        d_lat = float(np.mean(lat_world[i + 1:i1 + 1]) -
                      np.mean(lat_world[i0:i])) if (i - i0 >= 2 and i1 - i >= 2) else 0.0
        if d_lat > 0:
            current += 1
        elif d_lat < 0:
            current -= 1
        lane_idx[i] = current

    return lane_idx


def _build_frame_index(tracks) -> dict:
    """frame_id -> list of (track_id, x, y, vx, vy, heading_rad)."""
    frame_idx: dict = defaultdict(list)
    for tr in tracks:
        tid = int(tr["trackId"])
        for i, f in enumerate(tr["frame"]):
            frame_idx[int(f)].append((
                tid,
                float(tr["xCenter"][i]),
                float(tr["yCenter"][i]),
                float(tr["xVelocity"][i]),
                float(tr["yVelocity"][i]),
                float(tr["heading"][i]) * np.pi / 180.0,
            ))
    return dict(frame_idx)


def _find_neighbours(
    ego_id: int,
    frame_entries: list,
    ego_x: float, ego_y: float,
    ego_vx: float, ego_vy: float,
    ego_psi: float,
) -> dict:
    """
    Group neighbours into (front/rear) × (same/left/right) slots; nearest wins.
    ``slots[k]`` is either ``None`` or ``(ds, dvx)`` in the *signed* ego-
    longitudinal frame (ds > 0 ahead, ds < 0 behind).
    """
    slots = {k: None for k in (
        "front_same", "front_left", "front_right",
        "rear_same",  "rear_left",  "rear_right",
    )}
    nearest = {k: np.inf for k in slots}

    c, s = np.cos(ego_psi), np.sin(ego_psi)
    v_ego_longitudinal = ego_vx * c + ego_vy * s

    for (tid, x, y, vx, vy, _psi) in frame_entries:
        if tid == ego_id:
            continue
        dx = x - ego_x
        dy = y - ego_y
        ds = dx * c + dy * s          # +ve = ahead
        d_lat = -dx * s + dy * c      # +ve = left
        if abs(ds) > PERCEPTION_RANGE:
            continue
        if abs(d_lat) > 1.8 * LANE_WIDTH_ASSUMED:
            continue

        if d_lat >  0.5 * LANE_WIDTH_ASSUMED:
            lane_key = "left"
        elif d_lat < -0.5 * LANE_WIDTH_ASSUMED:
            lane_key = "right"
        else:
            lane_key = "same"
        fr_key = "front" if ds >= 0 else "rear"
        slot   = f"{fr_key}_{lane_key}"

        v_other_long = vx * c + vy * s
        dvx = v_other_long - v_ego_longitudinal

        if abs(ds) < nearest[slot]:
            nearest[slot] = abs(ds)
            slots[slot] = (float(ds), float(dvx))

    return slots


def _find_neighbours_with_ids(
    ego_id: int,
    frame_entries: list,
    ego_x: float, ego_y: float,
    ego_vx: float, ego_vy: float,
    ego_psi: float,
) -> dict:
    """
    Same lane × longitudinal slotting as :func:`_find_neighbours` but
    also returns the chosen neighbour's ``track_id`` and absolute
    longitudinal velocity in the ego frame.

    Each non-empty slot is::

        {'tid':      int,
         'ds':       float,    # signed (+ve = ahead)
         'dvx':      float,    # v_other_long − v_ego_long
         'vx_long':  float}    # absolute longitudinal speed of neighbour

    Exposed only when v4 social features are requested — the v3 hot
    path keeps using ``_find_neighbours``'s tuple shape.
    """
    slots = {k: None for k in (
        "front_same", "front_left", "front_right",
        "rear_same",  "rear_left",  "rear_right",
    )}
    nearest = {k: np.inf for k in slots}

    c, s = np.cos(ego_psi), np.sin(ego_psi)
    v_ego_longitudinal = ego_vx * c + ego_vy * s

    for (tid, x, y, vx, vy, _psi) in frame_entries:
        if tid == ego_id:
            continue
        dx = x - ego_x
        dy = y - ego_y
        ds = dx * c + dy * s
        d_lat = -dx * s + dy * c
        if abs(ds) > PERCEPTION_RANGE:
            continue
        if abs(d_lat) > 1.8 * LANE_WIDTH_ASSUMED:
            continue

        if d_lat > 0.5 * LANE_WIDTH_ASSUMED:
            lane_key = "left"
        elif d_lat < -0.5 * LANE_WIDTH_ASSUMED:
            lane_key = "right"
        else:
            lane_key = "same"
        fr_key = "front" if ds >= 0 else "rear"
        slot = f"{fr_key}_{lane_key}"

        v_other_long = vx * c + vy * s
        if abs(ds) < nearest[slot]:
            nearest[slot] = abs(ds)
            slots[slot] = {
                'tid': int(tid),
                'ds': float(ds),
                'dvx': float(v_other_long - v_ego_longitudinal),
                'vx_long': float(v_other_long),
            }

    return slots


def _build_track_lookup(tracks) -> dict:
    """
    Build per-track {trackId -> (frames_arr, x_arr, y_arr, vx_arr, vy_arr, lon_acc_arr)}.
    Used by social-feature mode to query a neighbour's future
    trajectory without walking the frame index.
    """
    lookup = {}
    for tr in tracks:
        tid = int(tr["trackId"])
        frames = np.asarray(tr["frame"], dtype=np.int64)
        order = np.argsort(frames)
        frames = frames[order]
        lookup[tid] = {
            'frames': frames,
            'x':      np.asarray(tr["xCenter"],   dtype=np.float32)[order],
            'y':      np.asarray(tr["yCenter"],   dtype=np.float32)[order],
            'vx':     np.asarray(tr["xVelocity"], dtype=np.float32)[order],
            'vy':     np.asarray(tr["yVelocity"], dtype=np.float32)[order],
            'lon_acc': (np.asarray(tr["lonAcceleration"], dtype=np.float32)[order]
                        if "lonAcceleration" in tr
                        else np.zeros(frames.size, dtype=np.float32)),
        }
    return lookup


def _follower_trajectory(
    track_lookup: dict,
    tid: int,
    frame_start: int,
    frame_end: int,
    ego_x_end: float, ego_y_end: float,
    ego_psi_end: float,
) -> np.ndarray | None:
    """
    Pull a neighbour's per-frame state over ``[frame_start, frame_end]``
    and rotate the *final* row into the ego-frame at ``frame_end`` so
    that callers can compute "after-maneuver" gap to ego without
    re-rotating manually.

    Returns a [T, 4] array of ``[x_ego_at_end, y_ego_at_end, lon_speed,
    lon_accel]`` or ``None`` when the neighbour is missing for any
    frame in the window.

    The first three rows use *world-frame* x/y so callers can compute
    the rear vehicle's deceleration profile directly; only the final
    row's x/y is rotated into the ego frame at i+H so the gap
    computation in :func:`courtesy_block` is meaningful.
    """
    rec = track_lookup.get(int(tid))
    if rec is None:
        return None
    frames = rec['frames']
    if frames.size == 0 or frames[0] > frame_start or frames[-1] < frame_end:
        return None
    i_start = int(np.searchsorted(frames, frame_start))
    i_end = int(np.searchsorted(frames, frame_end))
    if (i_start >= frames.size or i_end >= frames.size
            or frames[i_start] != frame_start or frames[i_end] != frame_end):
        return None

    n_rows = i_end - i_start + 1
    out = np.zeros((n_rows, 4), dtype=np.float32)
    out[:, 0] = rec['x'][i_start:i_end + 1]
    out[:, 1] = rec['y'][i_start:i_end + 1]
    out[:, 2] = np.hypot(rec['vx'][i_start:i_end + 1],
                         rec['vy'][i_start:i_end + 1])
    out[:, 3] = rec['lon_acc'][i_start:i_end + 1]

    # Rotate the LAST row into the ego frame at i+H.
    cos_psi = float(np.cos(ego_psi_end))
    sin_psi = float(np.sin(ego_psi_end))
    dx = float(out[-1, 0] - ego_x_end)
    dy = float(out[-1, 1] - ego_y_end)
    out[-1, 0] = dx * cos_psi + dy * sin_psi          # forward = +
    out[-1, 1] = -dx * sin_psi + dy * cos_psi         # left = +
    return out


# ---------------------------------------------------------------------------
# Per-frame feature block (shared by "now" and "outcome" timestamps)
# ---------------------------------------------------------------------------

def _per_frame_features(
    i: int,
    tid: int,
    frames: np.ndarray,
    xs: np.ndarray, ys: np.ndarray,
    vxs: np.ndarray, vys: np.ndarray,
    psis: np.ndarray,
    frame_index: dict,
) -> dict:
    """
    Compute all per-frame scalars/vectors needed for obs + risk + utility.

    Returned dict keys:
        ego_x, ego_y, ego_psi, ego_vx_world, ego_vy_world, ego_speed,
        ego_vx_body, ego_vy_body,
        nb                       : slot dict (see _find_neighbours),
        xs_ego, ys_ego, closing  : ego-frame neighbour arrays,
        risk_ego, risk_fwd_5m, risk_fwd_10m, risk_fwd_20m,
        risk_grad_x, risk_grad_y,
        risk_left_lane, risk_right_lane,
        risk_corridor_tau        : np.ndarray [4] for τ∈TAU_SECONDS, current lane,
        corridor_tau_curr / left / right : np.ndarray [len(TAU_SECONDS)] per lane,
        gap_fwd                  : np.ndarray [3]  (current, left, right),
        rel_speed                : np.ndarray [3],
        lane_risk                : np.ndarray [3],
        utility                  : np.ndarray [3],
        adv_left, adv_right, best_adv,
        leader_v_abs_curr        : absolute speed of current-lane leader (0 if none).
    """
    f = int(frames[i])
    ego_x = float(xs[i])
    ego_y = float(ys[i])
    ego_psi = float(psis[i])
    ego_vx_world = float(vxs[i])
    ego_vy_world = float(vys[i])
    ego_speed = float(np.hypot(ego_vx_world, ego_vy_world))

    c, s = np.cos(ego_psi), np.sin(ego_psi)
    ego_vx_body = ego_vx_world * c + ego_vy_world * s
    ego_vy_body = -ego_vx_world * s + ego_vy_world * c

    entries = frame_index.get(f, [])

    # Slotted neighbours for the obs vector
    nb = _find_neighbours(tid, entries, ego_x, ego_y,
                          ego_vx_world, ego_vy_world, ego_psi)

    # Full ego-frame neighbour arrays for risk proxy queries
    xs_ego, ys_ego, closing = rotate_neighbours_to_ego(
        ego_x, ego_y, ego_psi, ego_vx_body, entries, tid,
    )

    # --- Risk at ego + spatial lookahead ---
    risk_ego = risk_at(0.0, 0.0, xs_ego, ys_ego, closing)
    risk_fwd_5m  = risk_at( 5.0, 0.0, xs_ego, ys_ego, closing)
    risk_fwd_10m = risk_at(10.0, 0.0, xs_ego, ys_ego, closing)
    risk_fwd_20m = risk_at(20.0, 0.0, xs_ego, ys_ego, closing)
    grad_x, grad_y = risk_gradient(0.0, 0.0, xs_ego, ys_ego, closing)

    # Adjacent-lane forward corridors (scalar: max over 20 m × 6 samples)
    risk_left_lane  = risk_max_along_segment(
        0.0, +LANE_WIDTH_ASSUMED, 20.0, CORRIDOR_N_SAMPLES,
        xs_ego, ys_ego, closing,
    )
    risk_right_lane = risk_max_along_segment(
        0.0, -LANE_WIDTH_ASSUMED, 20.0, CORRIDOR_N_SAMPLES,
        xs_ego, ys_ego, closing,
    )

    # Time-parametrised forward corridors on each of the three lanes
    corridor_curr  = np.zeros(len(TAU_SECONDS), dtype=np.float32)
    corridor_left  = np.zeros(len(TAU_SECONDS), dtype=np.float32)
    corridor_right = np.zeros(len(TAU_SECONDS), dtype=np.float32)
    for k, tau in enumerate(TAU_SECONDS):
        corridor_curr[k]  = risk_corridor_tau(
            ego_vx_body, 0.0, tau, CORRIDOR_N_SAMPLES,
            xs_ego, ys_ego, closing,
        )
        corridor_left[k]  = risk_corridor_tau(
            ego_vx_body, +LANE_WIDTH_ASSUMED, tau, CORRIDOR_N_SAMPLES,
            xs_ego, ys_ego, closing,
        )
        corridor_right[k] = risk_corridor_tau(
            ego_vx_body, -LANE_WIDTH_ASSUMED, tau, CORRIDOR_N_SAMPLES,
            xs_ego, ys_ego, closing,
        )

    # Lane-wise tactical utility
    def _lane_utility(gap: float, dv: float, lrisk: float) -> float:
        # Matches rl.reward.reward_fn._lane_advantage exactly.
        gap_score  = min(gap, 80.0) / LANE_ADV_D0
        dv_score   = -dv / LANE_ADV_V0      # dv > 0 means leader pulling away
        risk_score = -lrisk / LANE_ADV_R0
        return float(gap_score + dv_score + risk_score)

    def _slot_gap_dv(slot_key: str, default_gap: float = 80.0) -> tuple:
        slot = nb.get(slot_key)
        if slot is None:
            return default_gap, 0.0
        ds, dvx = slot
        return float(abs(ds)), float(dvx)

    g_curr, dv_curr   = _slot_gap_dv("front_same")
    g_left, dv_left   = _slot_gap_dv("front_left")
    g_right, dv_right = _slot_gap_dv("front_right")

    gap_fwd   = np.asarray([g_curr,  g_left,  g_right],  dtype=np.float32)
    rel_speed = np.asarray([dv_curr, dv_left, dv_right], dtype=np.float32)

    # Lane-aggregate risk: mean over τ∈{1, 2, 3}s of the per-lane corridor
    lane_risk = np.asarray([
        float(np.mean(corridor_curr [:3])),
        float(np.mean(corridor_left [:3])),
        float(np.mean(corridor_right[:3])),
    ], dtype=np.float32)

    utility = np.asarray([
        _lane_utility(gap_fwd[j], rel_speed[j], lane_risk[j])
        for j in range(3)
    ], dtype=np.float32)

    adv_left  = float(utility[1] - utility[0])
    adv_right = float(utility[2] - utility[0])
    best_adv  = float(max(adv_left, adv_right))

    # Leader absolute speed (0 if no leader): v_leader_long = v_ego_body - (-dvx)
    # because dvx = v_other_long - v_ego_long → v_other_long = v_ego_long + dvx
    leader_v_abs_curr = ego_vx_body + dv_curr if nb.get("front_same") is not None else 0.0

    return {
        "ego_x": ego_x, "ego_y": ego_y, "ego_psi": ego_psi,
        "ego_vx_world": ego_vx_world, "ego_vy_world": ego_vy_world,
        "ego_vx_body":  ego_vx_body,  "ego_vy_body":  ego_vy_body,
        "ego_speed":    ego_speed,
        "nb":           nb,
        "xs_ego":       xs_ego, "ys_ego": ys_ego, "closing": closing,
        "risk_ego":     float(risk_ego),
        "risk_fwd_5m":  float(risk_fwd_5m),
        "risk_fwd_10m": float(risk_fwd_10m),
        "risk_fwd_20m": float(risk_fwd_20m),
        "risk_grad_x":  float(grad_x),
        "risk_grad_y":  float(grad_y),
        "risk_left_lane":  float(risk_left_lane),
        "risk_right_lane": float(risk_right_lane),
        "risk_corridor_tau": corridor_curr.copy(),
        "corridor_curr":  corridor_curr,
        "corridor_left":  corridor_left,
        "corridor_right": corridor_right,
        "gap_fwd":   gap_fwd,
        "rel_speed": rel_speed,
        "lane_risk": lane_risk,
        "utility":   utility,
        "adv_left":  adv_left,
        "adv_right": adv_right,
        "best_adv":  best_adv,
        "leader_v_abs_curr": float(leader_v_abs_curr),
    }


def _min_neighbour_gap_over_window(
    i: int, i_end: int,
    tid: int,
    frames: np.ndarray,
    xs: np.ndarray, ys: np.ndarray,
    frame_index: dict,
) -> float:
    """Min Euclidean distance to any non-self neighbour over [i, i_end]."""
    min_d = np.inf
    for k in range(i, i_end + 1):
        ego_x_k = float(xs[k])
        ego_y_k = float(ys[k])
        for (tid_k, nx, ny, _vx, _vy, _psi) in frame_index.get(int(frames[k]), []):
            if int(tid_k) == int(tid):
                continue
            d = float(np.hypot(nx - ego_x_k, ny - ego_y_k))
            if d < min_d:
                min_d = d
    return float(min_d) if np.isfinite(min_d) else 1e6


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_from_recording(
    data_dir: str,
    recording_id: str,
    horizon_sec: float = HORIZON_SEC_DEFAULT,
    outcome_horizon_sec: float = OUTCOME_HORIZON_SEC,
    limit_tracks: int | None = None,
    dataset_format: str = "exiD",
    include_social: bool = False,
) -> dict:
    """
    Extract one recording into the schema-v3 dict.

    Returns a dict of aligned numpy arrays (axis 0 = sample index) —
    see :func:`extract_many` for the full list of keys.
    """
    if recording_id.isdigit() and len(recording_id) < 2:
        recording_id = recording_id.zfill(2)
    dataset_name = _canonical_dataset_name(dataset_format)
    if not _is_special_dataset(dataset_name):
        tf = os.path.join(data_dir, f"{recording_id}_tracks.csv")
        tmf = os.path.join(data_dir, f"{recording_id}_tracksMeta.csv")
        rmf = os.path.join(data_dir, f"{recording_id}_recordingMeta.csv")
        if not (os.path.exists(tf) and os.path.exists(tmf) and os.path.exists(rmf)):
            print(f"[extractor] skipping {recording_id}: CSV files missing in {data_dir}")
            return _empty_output()

    tracks, tracks_meta, rec_meta = _lazy_read_from_dataset(
        data_dir,
        dataset_name=dataset_name,
        recording=recording_id,
    )

    frame_rate     = float(rec_meta["frameRate"])
    horizon_frames = max(1, int(round(horizon_sec         * frame_rate)))
    outcome_frames = max(1, int(round(outcome_horizon_sec * frame_rate)))
    settling_frames = max(1, int(round(1.0 * frame_rate)))  # 1 s settling

    meta_by_id  = {m["trackId"]: m for m in tracks_meta}
    frame_index = _build_frame_index(tracks)

    track_lookup = None
    field_query = None
    frame_aggregates: Dict[int, dict] = {}
    if include_social:
        # Lazy import: only needed when social features are requested.
        from rl.data.social_features import (
            courtesy_block, decision_block, RiskFieldQuery,
            field_metrics, composite_scores,
            frame_interaction_block, temporal_propagation_block,
            social_traffic_efficiency_index,
            surrogate_safety_block,
        )
        track_lookup = _build_track_lookup(tracks)
        field_query = RiskFieldQuery(mode='analytic')

        # ---------- v5 frame-level pre-pass ----------
        # Iterate frames in chronological order so the temporal-
        # propagation block can read the previous frame's risk mass
        # and speed variance.  Cost: O(N_frames * N_agents²) but
        # vectorised inside frame_interaction_block.
        prev_total = None
        prev_speed_var = None
        dt = 1.0 / float(frame_rate)
        for f in sorted(frame_index.keys()):
            agg = frame_interaction_block(frame_index[f])
            tmp = temporal_propagation_block(
                prev_risk_mass=prev_total,
                curr_risk_mass=agg['risk_mass_frame'],
                prev_speed_var=prev_speed_var,
                curr_speed_var=agg['speed_variance_frame'],
                dt=dt,
            )
            agg.update(tmp)
            # v6 literature-backed SSMs (TTC, THW, DRAC, MIN_DIST, EI/SEI)
            agg.update(surrogate_safety_block(frame_index[f]))

            # Composite per-frame STEI.  ``risk_others_per_agent`` is
            # left zero here because it is intrinsically per-ego; the
            # downstream aggregator can replace it with
            # mean_per_agent(risk_mass_others) when comparing
            # recordings.
            risk_adjusted_progress = (
                agg['total_progress_rate_frame']
                / (1.0 + agg['risk_mass_per_agent'])
            )
            agg['risk_adjusted_progress'] = float(risk_adjusted_progress)
            agg['social_traffic_efficiency_index'] = social_traffic_efficiency_index(
                progress_rate=agg['mean_speed_frame'],
                risk_mass_per_agent=agg['risk_mass_per_agent'],
                risk_others_per_agent=0.0,           # filled per-ego below
                backward_risk_flux_ratio=agg['backward_risk_flux_ratio'],
                speed_variance=agg['speed_variance_frame'],
                hard_brake_rate=0.0,                 # filled per-ego below
            )

            frame_aggregates[int(f)] = agg
            prev_total = agg['risk_mass_frame']
            prev_speed_var = agg['speed_variance_frame']

    # Collect per-sample fields into lists; stack once at the end.
    bufs: Dict[str, list] = defaultdict(list)

    n_ego_used = 0
    for tr in tracks:
        if limit_tracks is not None and n_ego_used >= limit_tracks:
            break
        tid = int(tr["trackId"])
        meta = meta_by_id.get(tid, {})
        vclass = str(meta.get("class", "car")).lower()
        if vclass not in ("car", "van", "motorcycle"):
            continue

        n = len(tr["xCenter"])
        duration = n / frame_rate
        if duration < MIN_TRACK_DURATION_S:
            continue

        speeds_all = np.hypot(tr["xVelocity"], tr["yVelocity"])
        if float(np.mean(speeds_all)) < MIN_EGO_SPEED:
            continue

        lane_idx = _assign_lane_index(tr)
        frames   = np.asarray(tr["frame"],     dtype=np.int64)
        xs       = np.asarray(tr["xCenter"],   dtype=np.float32)
        ys       = np.asarray(tr["yCenter"],   dtype=np.float32)
        vxs      = np.asarray(tr["xVelocity"], dtype=np.float32)
        vys      = np.asarray(tr["yVelocity"], dtype=np.float32)
        psis     = np.asarray(tr["heading"],   dtype=np.float32) * np.pi / 180.0

        for i in range(0, n - horizon_frames):
            if speeds_all[i] < MIN_EGO_SPEED:
                continue

            state_now = _per_frame_features(
                i, tid, frames, xs, ys, vxs, vys, psis, frame_index,
            )

            # ey — lateral offset from current lane centre (fallback 0)
            try:
                off = tr["latLaneCenterOffset"][i]
                ey = float(off[0]) if isinstance(off, list) else float(off)
                if np.isnan(ey):
                    ey = 0.0
            except Exception:
                ey = 0.0
            # Heading error relative to track start heading
            psi_ref = float(psis[0])
            epsi = float(psis[i]) - psi_ref
            epsi = (epsi + np.pi) % (2 * np.pi) - np.pi

            # 17-dim decision observation
            lane_rel = int(lane_idx[i] - lane_idx[0])
            obs = build_decision_obs(
                ego_vx=state_now["ego_vx_body"],
                ego_vy=state_now["ego_vy_body"],
                lane_rel=lane_rel,
                ey=ey, epsi=epsi,
                neighbours=state_now["nb"],
            )

            # ---------- Action label (next action horizon) ----------
            lane_future = int(lane_idx[i + horizon_frames] - lane_idx[i])
            if lane_future > 0:
                lane_delta = +1
            elif lane_future < 0:
                lane_delta = -1
            else:
                lane_delta = 0

            v_future_action = float(np.hypot(
                vxs[i + horizon_frames], vys[i + horizon_frames],
            ))
            dv_action = v_future_action - state_now["ego_speed"]
            if dv_action > SPEED_MODE_THRESH:
                speed_mode = 2
            elif dv_action < -SPEED_MODE_THRESH:
                speed_mode = 1
            else:
                speed_mode = 0

            action_9way = encode_action(lane_delta, speed_mode)
            # Decomposed labels for the multi-head BC loss.
            #   lane_delta_3way: 0=keep(0), 1=down(-1), 2=up(+1)   — matches
            #     rl.policy.decision_policy.LANE_DELTAS layout and can be
            #     trained with CrossEntropyLoss directly.
            lane_delta_3way = {0: 0, -1: 1, +1: 2}[lane_delta]

            # ---------- Outcome features (outcome horizon) ----------
            i_out = min(i + outcome_frames, n - 1)
            state_fut = _per_frame_features(
                i_out, tid, frames, xs, ys, vxs, vys, psis, frame_index,
            )

            future_lane_delta_val = int(np.sign(lane_idx[i_out] - lane_idx[i]))
            future_speed_delta    = float(state_fut["ego_speed"] - state_now["ego_speed"])
            future_speed_gain     = float(max(0.0, future_speed_delta))
            future_gap_gain       = float(state_fut["gap_fwd"][0] - state_now["gap_fwd"][0])

            # future_risk_change: use τ=2s corridor of the lane ego will
            # actually occupy at i_out.  If the future lane slot would be
            # absent (e.g., ego shifted to a lane that has no left/right
            # relative at i_out) fall back to the current-lane corridor.
            fut_corridor_tau2 = float(state_fut["corridor_curr"][1])
            now_corridor_tau2 = float(state_now["corridor_curr"][1])
            future_risk_change = fut_corridor_tau2 - now_corridor_tau2

            # min neighbour gap over [i, i_out]
            min_gap_future = _min_neighbour_gap_over_window(
                i, i_out, tid, frames, xs, ys, frame_index,
            )
            near_miss_future_val = int(min_gap_future < NEAR_MISS_THR)
            collision_future_val = int(min_gap_future < COLLISION_THR)

            # blocked_by_leader_flag at t
            blocked = 0
            if (state_now["gap_fwd"][0] < BLOCKED_GAP_THR
                    and state_now["ego_speed"] > 2.0
                    and state_now["leader_v_abs_curr"]
                        < BLOCKED_SPEED_FRAC * state_now["ego_speed"]):
                blocked = 1

            # lane_change_success: lane changed AND stable settling
            lc_success_val = 0
            if future_lane_delta_val != 0 and i_out < n - 1:
                # Settling: lane_idx stays at its i_out value for the next S frames
                s_end = min(n - 1, i_out + settling_frames)
                stable = bool(np.all(lane_idx[i_out:s_end + 1] == lane_idx[i_out]))
                no_collision = (min_gap_future >= COLLISION_THR)
                lc_success_val = int(stable and no_collision)

            # escape_success_flag: depends on blocked + outcome
            escape_success_val = int(
                blocked == 1
                and future_lane_delta_val != 0
                and future_risk_change < 0.0
                and (future_gap_gain > 0.0 or future_speed_delta > 0.0)
            )

            # lane_change_advantage_flag
            lc_advantage_val = int(
                future_gap_gain > TAU_D
                and future_risk_change < -TAU_R
            )

            # ---------- v4 social features ----------
            social = None
            if include_social:
                # Identify the target-lane rear neighbour.  Convention:
                # lane_delta == +1 (left), follower in left rear; -1
                # (right) → right rear; 0 → same-lane rear.
                slots_id = _find_neighbours_with_ids(
                    tid, frame_index.get(int(frames[i]), []),
                    state_now['ego_x'], state_now['ego_y'],
                    state_now['ego_vx_world'], state_now['ego_vy_world'],
                    state_now['ego_psi'],
                )
                target_slot_key = (
                    'rear_same'  if lane_delta == 0
                    else 'rear_left' if lane_delta == +1
                    else 'rear_right'
                )
                target_rear_now = slots_id.get(target_slot_key)
                target_rear_traj = None
                if target_rear_now is not None:
                    target_rear_traj = _follower_trajectory(
                        track_lookup,
                        target_rear_now['tid'],
                        int(frames[i]),
                        int(frames[i_out]),
                        state_fut['ego_x'], state_fut['ego_y'],
                        state_fut['ego_psi'],
                    )

                courtesy = courtesy_block(
                    target_rear_now=target_rear_now,
                    target_rear_traj=target_rear_traj,
                    ego_speed_now=state_now['ego_speed'],
                    ego_speed_after=state_fut['ego_speed'],
                    dt=1.0 / frame_rate,
                )
                decision = decision_block(
                    best_adv=state_now['best_adv'],
                    lane_delta_label=lane_delta_3way,
                    blocked_by_leader_flag=blocked,
                )
                R_grid, X_grid, Y_grid = field_query.field_grid(
                    state_now['xs_ego'], state_now['ys_ego'],
                    state_now['closing'],
                )
                fields = field_metrics(
                    R_grid, X_grid, Y_grid,
                    state_now['xs_ego'], state_now['ys_ego'],
                    state_now['closing'],
                )
                comp = composite_scores(
                    future_risk_change=float(future_risk_change),
                    near_miss_future=int(near_miss_future_val),
                    collision_future=int(collision_future_val),
                    future_speed_gain=float(future_speed_gain),
                    escape_success_flag=int(escape_success_val),
                    blocked_by_leader_flag=int(blocked),
                    rear_decel_peak_3s=courtesy['rear_decel_peak_3s'],
                    rear_ttc_delta=courtesy['rear_ttc_delta'],
                    hard_brake_imposed_flag=int(courtesy['hard_brake_imposed_flag']),
                    bad_cut_in_flag=int(courtesy['bad_cut_in_flag']),
                    missed_opportunity_flag=int(decision['missed_opportunity_flag']),
                    bad_lane_change_flag=int(decision['bad_lane_change_flag']),
                )
                social = {**courtesy, **decision, **fields, **comp}

            # Short-horizon return proxy
            progress_H = float(max(
                0.0,
                (state_fut["ego_x"] - state_now["ego_x"]) * np.cos(state_now["ego_psi"])
                + (state_fut["ego_y"] - state_now["ego_y"]) * np.sin(state_now["ego_psi"]),
            ))
            # Cumulative risk ∝ mean(risk_ego) × duration — we approximate
            # as mean of endpoints (cheap and stable).
            cumulative_risk_H = 0.5 * (state_now["risk_ego"] + state_fut["risk_ego"]) \
                * (i_out - i) / frame_rate
            # Comfort proxy: mean |Δv_lat / dt| over the window
            if i_out > i + 1:
                lat_vs = (-vxs[i:i_out] * np.sin(state_now["ego_psi"])
                          + vys[i:i_out] * np.cos(state_now["ego_psi"]))
                comfort_cost_H = float(np.mean(np.abs(np.diff(lat_vs)) * frame_rate))
            else:
                comfort_cost_H = 0.0
            short_horizon_return = float(
                RET_W_PROGRESS * progress_H / max(1e-3,
                    state_now["ego_speed"] * (i_out - i) / frame_rate)
                - RET_W_RISK      * cumulative_risk_H
                - RET_W_COMFORT   * comfort_cost_H
                - RET_W_NEAR_MISS * near_miss_future_val
            )

            # ---------- Append to buffers ----------
            # Metadata (A)
            bufs["scene_id"].append(int(recording_id))
            bufs["ego_id"].append(int(tid))
            bufs["frame_id"].append(int(frames[i]))
            bufs["t_sec"].append(float(i / frame_rate))
            bufs["current_lane_id"].append(int(lane_rel))

            # Ego state (B)
            bufs["ego_x"].append(state_now["ego_x"])
            bufs["ego_y"].append(state_now["ego_y"])
            bufs["ego_vx"].append(state_now["ego_vx_world"])
            bufs["ego_vy"].append(state_now["ego_vy_world"])
            bufs["ego_speed"].append(state_now["ego_speed"])
            bufs["ego_yaw"].append(state_now["ego_psi"])

            # Surround (C) — raw scalars in addition to the obs vector
            bufs["obs"].append(obs)
            ds_front_curr = float(state_now["gap_fwd"][0])
            dv_front_curr = float(state_now["rel_speed"][0])

            rear_left  = state_now["nb"].get("rear_left")
            rear_right = state_now["nb"].get("rear_right")
            ds_rear_left  = float(abs(rear_left[0]))  if rear_left  is not None else 80.0
            ds_rear_right = float(abs(rear_right[0])) if rear_right is not None else 80.0
            bufs["ds_front_curr"].append(ds_front_curr)
            bufs["dv_front_curr"].append(dv_front_curr)
            bufs["ds_rear_left"].append(ds_rear_left)
            bufs["ds_rear_right"].append(ds_rear_right)

            # Risk proxy (D)
            bufs["risk_ego_now"].append(state_now["risk_ego"])
            bufs["risk_fwd_5m"].append(state_now["risk_fwd_5m"])
            bufs["risk_fwd_10m"].append(state_now["risk_fwd_10m"])
            bufs["risk_fwd_20m"].append(state_now["risk_fwd_20m"])
            bufs["risk_grad_x"].append(state_now["risk_grad_x"])
            bufs["risk_grad_y"].append(state_now["risk_grad_y"])
            bufs["risk_left_lane"].append(state_now["risk_left_lane"])
            bufs["risk_right_lane"].append(state_now["risk_right_lane"])
            bufs["risk_corridor_tau"].append(state_now["risk_corridor_tau"])

            # Lane-wise tactical utility (E)
            bufs["gap_fwd"].append(state_now["gap_fwd"])
            bufs["rel_speed"].append(state_now["rel_speed"])
            bufs["lane_risk"].append(state_now["lane_risk"])
            bufs["utility"].append(state_now["utility"])
            bufs["adv_left"].append(state_now["adv_left"])
            bufs["adv_right"].append(state_now["adv_right"])
            bufs["best_adv"].append(state_now["best_adv"])

            # Action labels (F)
            bufs["action_9way"].append(int(action_9way))
            bufs["lane_delta_label"].append(int(lane_delta_3way))
            bufs["speed_mode_label"].append(int(speed_mode))

            # Outcome labels (G)
            bufs["future_lane_delta"].append(int(future_lane_delta_val))
            bufs["future_speed_delta"].append(float(future_speed_delta))
            bufs["future_speed_gain"].append(float(future_speed_gain))
            bufs["future_gap_gain"].append(float(future_gap_gain))
            bufs["future_risk_change"].append(float(future_risk_change))
            bufs["lane_change_success"].append(int(lc_success_val))
            bufs["near_miss_future"].append(int(near_miss_future_val))
            bufs["collision_future"].append(int(collision_future_val))
            bufs["blocked_by_leader_flag"].append(int(blocked))
            bufs["escape_success_flag"].append(int(escape_success_val))
            bufs["lane_change_advantage_flag"].append(int(lc_advantage_val))
            bufs["short_horizon_return_proxy"].append(float(short_horizon_return))

            # v4 social features
            if social is not None:
                from rl.data.social_features import (
                    V4_KEYS_FLOAT, V4_KEYS_INT,
                    V5_KEYS_FLOAT, V5_KEYS_INT,
                    V6_KEYS_FLOAT, V6_KEYS_INT,
                )
                for k in V4_KEYS_FLOAT:
                    v = social.get(k, float('nan'))
                    # ±inf would poison aggregate stats — collapse to NaN
                    # and let downstream filters use np.isfinite().
                    if not np.isfinite(v):
                        v = float('nan')
                    bufs[k].append(float(v))
                for k in V4_KEYS_INT:
                    bufs[k].append(int(social.get(k, 0)))

                # v5/v6 frame-level metrics, broadcast onto this row
                fa = frame_aggregates.get(int(frames[i]), {})
                for k in V5_KEYS_FLOAT + V6_KEYS_FLOAT:
                    v = fa.get(k, float('nan'))
                    if not np.isfinite(v):
                        v = float('nan')
                    bufs[k].append(float(v))
                for k in V5_KEYS_INT + V6_KEYS_INT:
                    bufs[k].append(int(fa.get(k, 0)))

            # Backward-compat aliases (keep v2 names so downstream tools
            # that already look for ``lc_success`` / ``future_return``
            # continue to work without an extra migration step).
            bufs["lc_success"].append(int(lc_success_val))
            bufs["future_return"].append(float(short_horizon_return))

            # Legacy keys preserved
            bufs["actions"].append(int(action_9way))
            bufs["track_ids"].append(int(tid))
            bufs["t_rel"].append(float(i / frame_rate))

        n_ego_used += 1

    if not bufs["obs"]:
        return _empty_output()

    out: Dict[str, np.ndarray] = {}
    # Vector arrays (axis 1 exists)
    out["obs"]               = np.stack(bufs["obs"],               axis=0).astype(np.float32)
    out["risk_corridor_tau"] = np.stack(bufs["risk_corridor_tau"], axis=0).astype(np.float32)
    out["gap_fwd"]           = np.stack(bufs["gap_fwd"],           axis=0).astype(np.float32)
    out["rel_speed"]         = np.stack(bufs["rel_speed"],         axis=0).astype(np.float32)
    out["lane_risk"]         = np.stack(bufs["lane_risk"],         axis=0).astype(np.float32)
    out["utility"]           = np.stack(bufs["utility"],           axis=0).astype(np.float32)

    # Scalar arrays
    from rl.data.social_features import (
        V4_KEYS_FLOAT as _SOC_F, V4_KEYS_INT as _SOC_I,
        V5_KEYS_FLOAT as _TRA_F, V5_KEYS_INT as _TRA_I,
        V6_KEYS_FLOAT as _SSM_F, V6_KEYS_INT as _SSM_I,
    )
    _f32 = {
        "t_sec", "ego_x", "ego_y", "ego_vx", "ego_vy", "ego_speed", "ego_yaw",
        "ds_front_curr", "dv_front_curr", "ds_rear_left", "ds_rear_right",
        "risk_ego_now", "risk_fwd_5m", "risk_fwd_10m", "risk_fwd_20m",
        "risk_grad_x", "risk_grad_y", "risk_left_lane", "risk_right_lane",
        "adv_left", "adv_right", "best_adv",
        "future_speed_delta", "future_speed_gain", "future_gap_gain",
        "future_risk_change", "short_horizon_return_proxy", "future_return",
        "t_rel",
    } | set(_SOC_F) | set(_TRA_F) | set(_SSM_F)
    _i64 = {"scene_id", "ego_id", "frame_id", "action_9way", "actions", "track_ids"}
    _i8  = {
        "current_lane_id", "lane_delta_label", "speed_mode_label",
        "future_lane_delta", "lane_change_success", "near_miss_future",
        "collision_future", "blocked_by_leader_flag", "escape_success_flag",
        "lane_change_advantage_flag", "lc_success",
    } | set(_SOC_I) | set(_TRA_I) | set(_SSM_I)
    for k, lst in bufs.items():
        if k in out:
            continue
        if k in _f32:
            out[k] = np.asarray(lst, dtype=np.float32)
        elif k in _i64:
            out[k] = np.asarray(lst, dtype=np.int64)
        elif k in _i8:
            out[k] = np.asarray(lst, dtype=np.int8)
        else:
            # Unknown key — fall back to float32
            out[k] = np.asarray(lst, dtype=np.float32)

    return out


def _empty_output() -> dict:
    """Zero-sample return shape — keeps merge code branch-free."""
    empty_f32 = np.zeros((0,), dtype=np.float32)
    empty_i64 = np.zeros((0,), dtype=np.int64)
    empty_i8  = np.zeros((0,), dtype=np.int8)
    out = {
        "obs":               np.zeros((0, DEC_OBS_DIM), dtype=np.float32),
        "risk_corridor_tau": np.zeros((0, len(TAU_SECONDS)), dtype=np.float32),
        "gap_fwd":           np.zeros((0, 3), dtype=np.float32),
        "rel_speed":         np.zeros((0, 3), dtype=np.float32),
        "lane_risk":         np.zeros((0, 3), dtype=np.float32),
        "utility":           np.zeros((0, 3), dtype=np.float32),
    }
    for k in ("t_sec", "ego_x", "ego_y", "ego_vx", "ego_vy", "ego_speed", "ego_yaw",
              "ds_front_curr", "dv_front_curr", "ds_rear_left", "ds_rear_right",
              "risk_ego_now", "risk_fwd_5m", "risk_fwd_10m", "risk_fwd_20m",
              "risk_grad_x", "risk_grad_y", "risk_left_lane", "risk_right_lane",
              "adv_left", "adv_right", "best_adv",
              "future_speed_delta", "future_speed_gain", "future_gap_gain",
              "future_risk_change", "short_horizon_return_proxy", "future_return",
              "t_rel"):
        out[k] = empty_f32.copy()
    for k in ("scene_id", "ego_id", "frame_id", "action_9way", "actions", "track_ids"):
        out[k] = empty_i64.copy()
    for k in ("current_lane_id", "lane_delta_label", "speed_mode_label",
              "future_lane_delta", "lane_change_success", "near_miss_future",
              "collision_future", "blocked_by_leader_flag", "escape_success_flag",
              "lane_change_advantage_flag", "lc_success"):
        out[k] = empty_i8.copy()
    return out


# ---------------------------------------------------------------------------
# Multi-recording driver
# ---------------------------------------------------------------------------

_ARRAY_KEYS: Tuple[str, ...] = (
    # Metadata (A)
    "scene_id", "ego_id", "frame_id", "t_sec", "current_lane_id",
    # Ego state (B)
    "ego_x", "ego_y", "ego_vx", "ego_vy", "ego_speed", "ego_yaw",
    # Surround (C)
    "obs", "ds_front_curr", "dv_front_curr", "ds_rear_left", "ds_rear_right",
    # Risk proxy (D)
    "risk_ego_now", "risk_fwd_5m", "risk_fwd_10m", "risk_fwd_20m",
    "risk_grad_x", "risk_grad_y", "risk_left_lane", "risk_right_lane",
    "risk_corridor_tau",
    # Lane-wise tactical utility (E)
    "gap_fwd", "rel_speed", "lane_risk", "utility",
    "adv_left", "adv_right", "best_adv",
    # Action labels (F)
    "action_9way", "lane_delta_label", "speed_mode_label",
    # Outcome labels (G)
    "future_lane_delta", "future_speed_delta", "future_speed_gain",
    "future_gap_gain", "future_risk_change",
    "lane_change_success", "near_miss_future", "collision_future",
    "blocked_by_leader_flag", "escape_success_flag",
    "lane_change_advantage_flag", "short_horizon_return_proxy",
    # Backward-compat aliases
    "actions", "track_ids", "t_rel", "lc_success", "future_return",
)


def _v4_array_keys() -> Tuple[str, ...]:
    from rl.data.social_features import (
        V4_KEYS_FLOAT, V4_KEYS_INT,
        V5_KEYS_FLOAT, V5_KEYS_INT,
        V6_KEYS_FLOAT, V6_KEYS_INT,
    )
    return (tuple(V4_KEYS_FLOAT) + tuple(V4_KEYS_INT)
            + tuple(V5_KEYS_FLOAT) + tuple(V5_KEYS_INT)
            + tuple(V6_KEYS_FLOAT) + tuple(V6_KEYS_INT))


def _placeholder_array(name: str, n: int) -> np.ndarray:
    """Fill missing v4/v5/v6 keys with a fixed-shape placeholder array."""
    from rl.data.social_features import (
        V4_KEYS_FLOAT, V4_KEYS_INT,
        V5_KEYS_FLOAT, V5_KEYS_INT,
        V6_KEYS_FLOAT, V6_KEYS_INT,
    )
    if name in V4_KEYS_FLOAT or name in V5_KEYS_FLOAT or name in V6_KEYS_FLOAT:
        return np.full((n,), np.nan, dtype=np.float32)
    if name in V4_KEYS_INT or name in V5_KEYS_INT or name in V6_KEYS_INT:
        return np.zeros((n,), dtype=np.int8)
    return np.zeros((n,), dtype=np.float32)


def extract_many(
    data_dir: str,
    recording_ids: List[str],
    horizon_sec: float = HORIZON_SEC_DEFAULT,
    outcome_horizon_sec: float = OUTCOME_HORIZON_SEC,
    limit_tracks: int | None = None,
    dataset_format: str = "exiD",
    include_social: bool = False,
) -> dict:
    array_keys = _ARRAY_KEYS + (_v4_array_keys() if include_social else ())
    merged: Dict[str, List[np.ndarray]] = {k: [] for k in array_keys}
    merged["recording_ids"] = []

    for rid_index, rid in enumerate(recording_ids):
        t0 = time.time()
        d = extract_from_recording(
            data_dir, rid,
            horizon_sec=horizon_sec,
            outcome_horizon_sec=outcome_horizon_sec,
            limit_tracks=limit_tracks,
            dataset_format=dataset_format,
            include_social=include_social,
        )
        n = d["obs"].shape[0]
        dt = time.time() - t0
        print(f"[extractor] rec={rid}: {n:>6d} samples  ({dt:.1f}s)")
        if n == 0:
            continue
        for k in array_keys:
            if k not in d:                # v3 dataset; fill placeholder
                d[k] = _placeholder_array(k, n)
            merged[k].append(d[k])
        merged["recording_ids"].append(
            np.full((n,), _recording_id_to_int(rid, rid_index), dtype=np.int64)
        )

    if not merged["obs"]:
        raise RuntimeError("No samples extracted — check data path and recording ids.")

    out: Dict[str, np.ndarray] = {}
    for k in list(array_keys) + ["recording_ids"]:
        out[k] = np.concatenate(merged[k], axis=0)

    # Sidecar scalars — documented in the schema spec.
    out["schema_version"]      = np.asarray(SCHEMA_VERSION,      dtype=np.int64)
    out["horizon_sec"]         = np.asarray(horizon_sec,         dtype=np.float32)
    out["outcome_horizon_sec"] = np.asarray(outcome_horizon_sec, dtype=np.float32)
    out["obs_dim"]             = np.asarray(DEC_OBS_DIM,         dtype=np.int64)
    out["n_actions"]           = np.asarray(DEC_N_ACTIONS,       dtype=np.int64)
    # Tuning constants (useful at training time for consistency checks)
    out["sigma_kernel"]        = np.asarray(SIGMA_KERNEL,        dtype=np.float32)
    out["v0_kernel"]           = np.asarray(V0_KERNEL,           dtype=np.float32)
    out["lane_adv_D0"]         = np.asarray(LANE_ADV_D0,         dtype=np.float32)
    out["lane_adv_V0"]         = np.asarray(LANE_ADV_V0,         dtype=np.float32)
    out["lane_adv_R0"]         = np.asarray(LANE_ADV_R0,         dtype=np.float32)
    out["near_miss_thr"]       = np.asarray(NEAR_MISS_THR,       dtype=np.float32)
    out["collision_thr"]       = np.asarray(COLLISION_THR,       dtype=np.float32)
    out["blocked_gap_thr"]     = np.asarray(BLOCKED_GAP_THR,     dtype=np.float32)
    out["blocked_speed_frac"]  = np.asarray(BLOCKED_SPEED_FRAC,  dtype=np.float32)
    out["tau_d"]               = np.asarray(TAU_D,               dtype=np.float32)
    out["tau_r"]               = np.asarray(TAU_R,               dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# Sanity report (schema-v3 aware)
# ---------------------------------------------------------------------------

def summarize_dataset(d: dict) -> dict:
    """Compact sanity report — prints and returns a dict."""
    obs      = np.asarray(d["obs"])
    actions  = np.asarray(d.get("action_9way", d.get("actions"))).astype(np.int64)
    tids     = np.asarray(d.get("ego_id", d.get("track_ids"))).astype(np.int64)
    scene    = np.asarray(d.get("scene_id", d.get("recording_ids", np.zeros_like(tids)))).astype(np.int64)

    n = int(obs.shape[0])
    if n == 0:
        print("[summary] empty dataset")
        return {"n": 0}

    if obs.shape[1] != DEC_OBS_DIM:
        print(f"[summary] WARNING obs_dim={obs.shape[1]} != DEC_OBS_DIM={DEC_OBS_DIM}")

    hist = np.bincount(actions, minlength=DEC_N_ACTIONS)
    frac = hist.astype(np.float64) / max(1, int(hist.sum()))
    lane_bin  = actions // 3
    speed_bin = actions %  3

    lane_keep  = float(np.mean(lane_bin == 0))
    lane_left  = float(np.mean(lane_bin == 1))
    lane_right = float(np.mean(lane_bin == 2))
    sp_maintain = float(np.mean(speed_bin == 0))
    sp_slower   = float(np.mean(speed_bin == 1))
    sp_faster   = float(np.mean(speed_bin == 2))

    n_tracks  = int(np.unique(tids).size)
    n_scenes  = int(np.unique(scene).size)
    per_scene = {int(r): int(np.sum(scene == r)) for r in np.unique(scene).tolist()}

    schema_v = int(d.get("schema_version", np.asarray(-1)))
    horizon  = float(d.get("horizon_sec",  np.asarray(-1.0)))
    ohorizon = float(d.get("outcome_horizon_sec", np.asarray(-1.0)))

    print("[summary] ================ BC dataset sanity (schema v3) ================")
    print(f"[summary] samples:       {n}")
    print(f"[summary] unique tracks: {n_tracks}")
    print(f"[summary] scenes:        {n_scenes}  coverage={per_scene}")
    print(f"[summary] schema_version={schema_v}  horizon_sec={horizon:.3f}  "
          f"outcome_horizon_sec={ohorizon:.3f}")
    print(f"[summary] obs shape {obs.shape}  dtype {obs.dtype}")
    print(f"[summary] obs stats  mean={obs.mean():+.3f}  std={obs.std():.3f}  "
          f"min={obs.min():+.3f}  max={obs.max():+.3f}")
    print("[summary] action histogram (9-way):")
    for a in range(DEC_N_ACTIONS):
        print(f"           a={a}: {hist[a]:>7d}  ({100*frac[a]:5.2f}%)")
    print(f"[summary] lane    : keep={lane_keep:.3f}  "
          f"left={lane_left:.3f}  right={lane_right:.3f}  "
          f"LC_total={(lane_left+lane_right):.3f}")
    print(f"[summary] speed   : maintain={sp_maintain:.3f}  "
          f"slower={sp_slower:.3f}  faster={sp_faster:.3f}")

    # Risk feature sanity
    risk_ego = np.asarray(d.get("risk_ego_now", []), dtype=np.float32)
    if risk_ego.size == n:
        r_fwd_20 = np.asarray(d["risk_fwd_20m"], dtype=np.float32)
        r_left   = np.asarray(d["risk_left_lane"],  dtype=np.float32)
        r_right  = np.asarray(d["risk_right_lane"], dtype=np.float32)
        r_tau    = np.asarray(d["risk_corridor_tau"], dtype=np.float32)  # [N, 4]
        print("[summary] risk    :")
        print(f"           risk_ego_now    mean={float(risk_ego.mean()):.3f}  "
              f"std={float(risk_ego.std()):.3f}  max={float(risk_ego.max()):.3f}")
        print(f"           risk_fwd_20m    mean={float(r_fwd_20.mean()):.3f}  "
              f"std={float(r_fwd_20.std()):.3f}")
        print(f"           risk_left_lane  mean={float(r_left.mean()):.3f}  "
              f"risk_right_lane mean={float(r_right.mean()):.3f}")
        print(f"           risk_corridor_tau mean per-τ: "
              f"{np.round(r_tau.mean(axis=0), 3).tolist()}")

    # Outcome label sanity (the part that matters for this schema bump)
    lc_mask = (lane_bin != 0)
    n_lc = int(lc_mask.sum())
    lc_adv = np.asarray(d.get("lane_change_advantage_flag", []), dtype=np.int64)
    lc_succ = np.asarray(d.get("lane_change_success",       []), dtype=np.int64)
    escape  = np.asarray(d.get("escape_success_flag",       []), dtype=np.int64)
    blocked = np.asarray(d.get("blocked_by_leader_flag",    []), dtype=np.int64)
    near_m  = np.asarray(d.get("near_miss_future",          []), dtype=np.int64)
    fsg     = np.asarray(d.get("future_speed_gain",         []), dtype=np.float32)
    fgg     = np.asarray(d.get("future_gap_gain",           []), dtype=np.float32)
    frc     = np.asarray(d.get("future_risk_change",        []), dtype=np.float32)

    lc_adv_pos_all = float(lc_adv.mean()) if lc_adv.size == n else float("nan")
    lc_adv_pos_lc  = float(lc_adv[lc_mask].mean()) if (lc_adv.size == n and n_lc > 0) else float("nan")
    lc_succ_rate_lc = float(lc_succ[lc_mask].mean()) if (lc_succ.size == n and n_lc > 0) else float("nan")
    blocked_rate    = float(blocked.mean()) if blocked.size == n else float("nan")
    escape_given_blocked = (
        float(escape[blocked == 1].mean())
        if (escape.size == n and blocked.size == n and (blocked == 1).any())
        else float("nan")
    )
    near_miss_rate  = float(near_m.mean()) if near_m.size == n else float("nan")

    print("[summary] outcome :")
    print(f"           LC samples           : {n_lc}  ({100*n_lc/max(1,n):.2f}%)")
    print(f"           lc_advantage (all)   : {lc_adv_pos_all:.3f}")
    print(f"           lc_advantage (LC)    : {lc_adv_pos_lc:.3f}    "
          f"<— primary new label; want 0.05–0.30")
    print(f"           lc_success (LC)      : {lc_succ_rate_lc:.3f}")
    print(f"           blocked_by_leader    : {blocked_rate:.3f}")
    print(f"           escape | blocked     : {escape_given_blocked:.3f}")
    print(f"           near_miss_future     : {near_miss_rate:.3f}")
    if fsg.size == n:
        print(f"           future_speed_gain    mean={float(fsg.mean()):+.2f}  "
              f"std={float(fsg.std()):.2f}")
    if fgg.size == n:
        print(f"           future_gap_gain      mean={float(fgg.mean()):+.2f}  "
              f"std={float(fgg.std()):.2f}")
    if frc.size == n:
        print(f"           future_risk_change   mean={float(frc.mean()):+.3f}  "
              f"std={float(frc.std()):.3f}")
        # Separation check: risk-drop should correlate with advantage label
        if lc_adv.size == n and n_lc > 0:
            m_pos = (lc_adv == 1) & lc_mask
            m_neg = (lc_adv == 0) & lc_mask
            if m_pos.any() and m_neg.any():
                mu_pos = float(frc[m_pos].mean())
                mu_neg = float(frc[m_neg].mean())
                print(f"           Δrisk  (adv=1 vs 0) : {mu_pos:+.3f}  vs  "
                      f"{mu_neg:+.3f}  <— advantage should have lower risk change")

    return {
        "n": n,
        "n_tracks": n_tracks,
        "n_scenes": n_scenes,
        "per_scene": per_scene,
        "schema_version": schema_v,
        "horizon_sec": horizon,
        "outcome_horizon_sec": ohorizon,
        "action_hist": hist.tolist(),
        "lane_keep_frac": lane_keep,
        "lane_change_frac": float(lane_left + lane_right),
        "lc_advantage_frac_all": lc_adv_pos_all,
        "lc_advantage_frac_lc":  lc_adv_pos_lc,
        "lc_success_frac_lc":    lc_succ_rate_lc,
        "blocked_frac":          blocked_rate,
        "escape_given_blocked":  escape_given_blocked,
        "near_miss_frac":        near_miss_rate,
    }


def summarize_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    return summarize_dataset(d)


# ---------------------------------------------------------------------------
# Scene-level split manifest
# ---------------------------------------------------------------------------

def build_split_manifest(
    out: dict,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 0,
) -> dict:
    """
    Produce a {scene_id: "train"|"val"|"test"} manifest.

    Fallback: if fewer than ~6 unique scenes exist, split by ``ego_id``
    within each scene so the same human never crosses splits.
    """
    scenes = np.unique(np.asarray(out["scene_id"], dtype=np.int64)).tolist()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(scenes)

    manifest = {"split_by": "scene_id", "seed": int(seed)}

    if len(scenes) >= 6:
        n_train = max(1, int(round(len(scenes) * train_frac)))
        n_val   = max(1, int(round(len(scenes) * val_frac)))
        train = scenes[:n_train]
        val   = scenes[n_train:n_train + n_val]
        test  = scenes[n_train + n_val:]
    else:
        manifest["split_by"] = "ego_id_within_scene"
        egos = np.unique(np.asarray(out["ego_id"], dtype=np.int64)).tolist()
        rng.shuffle(egos)
        n_train = max(1, int(round(len(egos) * train_frac)))
        n_val   = max(1, int(round(len(egos) * val_frac)))
        train = egos[:n_train]
        val   = egos[n_train:n_train + n_val]
        test  = egos[n_train + n_val:]

    manifest["train"] = [int(x) for x in train]
    manifest["val"]   = [int(x) for x in val]
    manifest["test"]  = [int(x) for x in test]
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_recordings(data_dir: str, dataset_format: str = "exiD") -> List[str]:
    """Find recording ids for one dataset directory."""
    if _is_special_dataset(dataset_format):
        return ["00"]

    import glob
    pattern = os.path.join(data_dir, "*_tracks.csv")
    rids = sorted(set(
        os.path.basename(f).replace("_tracks.csv", "")
        for f in glob.glob(pattern)
    ))
    return rids


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None,
                   help="Dataset directory. Defaults to data/<dataset-format>.")
    p.add_argument("--recordings", default=None,
                   help="Comma-separated list, e.g. 00,01,02,03  "
                        "Use 'all' to auto-discover every recording in --data-dir")
    p.add_argument("--out-path", default=None,
                   help="Target .npz file for extraction output")
    p.add_argument("--horizon-sec", type=float, default=HORIZON_SEC_DEFAULT,
                   help="Action-label horizon (s)")
    p.add_argument("--outcome-horizon-sec", type=float, default=OUTCOME_HORIZON_SEC,
                   help="Outcome-label horizon (s)")
    p.add_argument("--limit-tracks", type=int, default=None,
                   help="Cap ego candidates per recording (smoke tests)")
    p.add_argument("--dataset-format", default="exiD",
                   help=f"Dataset schema family. Supported: {SUPPORTED_DATASET_HELP}")
    p.add_argument("--summary", default=None,
                   help="Instead of extracting, load an existing .npz and "
                        "print a sanity report")
    p.add_argument("--split-seed", type=int, default=0,
                   help="Seed for scene-level train/val/test split manifest")
    p.add_argument("--no-manifest", action="store_true",
                   help="Skip writing the split manifest JSON sidecar")
    p.add_argument("--include-social", action="store_true",
                   help="Compute schema-v4 social-friendliness features "
                        "(courtesy, decision-quality, BEV risk-field "
                        "externality, composite scores).  Slower (~3-5x) "
                        "but required for plot_social_externality.py.")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.summary is not None:
        summarize_npz(args.summary)
        return

    dataset_name = _canonical_dataset_name(args.dataset_format)
    dataset_key = dataset_name.lower()
    if dataset_key in UNSUPPORTED_DATASET_KEYS:
        raise SystemExit(
            f"dataset-format '{args.dataset_format}' is present in data/ but not yet integrated here."
        )

    args.data_dir = os.path.abspath(args.data_dir or _default_data_dir(dataset_name))
    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"--data-dir not found: {args.data_dir}")

    if not (args.recordings and args.out_path):
        raise SystemExit("--recordings and --out-path are required unless --summary is given")

    if args.recordings.strip().lower() == "all":
        rids = _discover_recordings(args.data_dir, dataset_name)
        print(f"[extractor] auto-discovered {len(rids)} recordings: "
              f"{rids[0] if rids else '—'}..{rids[-1] if rids else '—'}")
    else:
        rids = [r.strip() for r in args.recordings.split(",") if r.strip()]

    out = extract_many(
        args.data_dir, rids,
        horizon_sec=args.horizon_sec,
        outcome_horizon_sec=args.outcome_horizon_sec,
        limit_tracks=args.limit_tracks,
        dataset_format=dataset_name,
        include_social=args.include_social,
    )

    summarize_dataset(out)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    np.savez_compressed(args.out_path, **out)
    print(f"[extractor] wrote {out['obs'].shape[0]} samples to {args.out_path}")

    if not args.no_manifest:
        manifest = build_split_manifest(out, seed=args.split_seed)
        manifest_path = args.out_path.replace(".npz", "_split.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[extractor] wrote split manifest → {manifest_path}  "
              f"(split_by={manifest['split_by']}, "
              f"train={len(manifest['train'])}, val={len(manifest['val'])}, "
              f"test={len(manifest['test'])})")


if __name__ == "__main__":
    main()
