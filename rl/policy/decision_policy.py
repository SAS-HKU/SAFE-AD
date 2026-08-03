"""
Decision-Level RL Policy
========================
A high-level policy that outputs a discrete *decision* (lane change ±
target speed bucket) rather than low-level control. The actual steering
and throttle are produced by the downstream MPC+CBF stack, which reads
the commanded lane and speed from this policy.

Design rationale
----------------
End-to-end RL on (accel, steer) competes with the MPC controller and
yields unstable, un-benchmarkable behaviour (see
docs/rl_debug_failure_analysis.md). Keeping RL at the decision layer:

  * matches how the historical exiD data is structured (we can label
    discrete lane-change decisions from human trajectories),
  * is compatible with behaviour cloning from recorded humans,
  * lets the learned policy be dropped into uncertainty_merger_DREAM.py
    alongside DREAM / IDEAM / ADA / APF / OA-CMPC at the same
    abstraction level.

Observation schema (see OBS_NAMES below) is *hand-engineered* and can be
filled in two ways:

  1. from exiD tracks during offline dataset extraction
     (rl/data/historical_extractor.py), and
  2. from the live simulator during online fine-tuning / rollout
     (rl/env/merger_decision_env.py).

Both paths produce the same-shaped vector. This is the *only* place the
schema is defined — importers must reuse DEC_OBS_DIM / DEC_N_ACTIONS /
encode_action / decode_action to stay in sync.
"""

from __future__ import annotations

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Observation schema
# ---------------------------------------------------------------------------
#
# Ego lane-relative kinematics + neighbour gaps per lane. 17 dims total.
#   0  : ego longitudinal speed      / NORM_V
#   1  : ego lateral speed           / NORM_V
#   2  : lane index (−1 / 0 / +1 relative to ego start lane)
#   3  : ey (lateral error from current lane centre) / NORM_EY
#   4  : heading error (epsi)        / NORM_EPSI
#   5,6: front-same   (ds, dvx) normalised
#   7,8: front-left   (ds, dvx) normalised
#   9,10: front-right (ds, dvx) normalised
#   11,12: rear-same  (ds, dvx) normalised
#   13,14: rear-left  (ds, dvx) normalised
#   15,16: rear-right (ds, dvx) normalised
#
# Gaps are clipped to ±PERCEPTION_RANGE and set to (+1, 0) if no vehicle
# present (i.e. "clear lane").

DEC_OBS_DIM = 17

OBS_NAMES = [
    "ego_vx", "ego_vy", "lane_rel", "ey", "epsi",
    "ds_fs", "dvx_fs",
    "ds_fl", "dvx_fl",
    "ds_fr", "dvx_fr",
    "ds_rs", "dvx_rs",
    "ds_rl", "dvx_rl",
    "ds_rr", "dvx_rr",
]

# Normalisation scales (match rl.config.rl_config defaults so obs from
# live env and from offline extractor are numerically comparable).
NORM = {
    "V":   15.0,
    "EY":  2.0,
    "EPSI": 0.4,
    "DS":  60.0,   # perception range used for gap normalisation
    "DV":  10.0,
}

PERCEPTION_RANGE = 60.0   # metres — gaps beyond this are treated as "clear"


# ---------------------------------------------------------------------------
# Action schema
# ---------------------------------------------------------------------------
#
# 9 discrete actions = {lane_delta ∈ [-1,0,+1]} × {speed_mode ∈ {slower, keep, faster}}
# This matches rl/config/rl_config.py (N_ACTIONS=9) so the existing smoke
# tests and reward function continue to work unchanged.
#
# Convention (MUST match rl_config.LANE_DELTAS / SPEED_OFFSETS):
#   idx 0,1,2 : lane_delta  0   (KEEP)    — speed: maintain / slower / faster
#   idx 3,4,5 : lane_delta −1   (LC_DOWN) — speed: maintain / slower / faster
#   idx 6,7,8 : lane_delta +1   (LC_UP)   — speed: maintain / slower / faster
#
# LC_DOWN / LC_UP here mean *lane-index delta* in the env's lane numbering
# (0 = bottom/right, 2 = top/left), which is opposite of "go left".

DEC_N_ACTIONS = 9
LANE_DELTAS   = np.array([ 0,  0,  0, -1, -1, -1, +1, +1, +1], dtype=np.int64)
SPEED_OFFSETS = np.array([ 0.0, -2.0, +2.0,
                           0.0, -2.0, +2.0,
                           0.0, -2.0, +2.0], dtype=np.float32)


def encode_action(lane_delta: int, speed_mode: int) -> int:
    """Encode a (lane_delta, speed_mode) pair into a discrete action index.

    Args:
        lane_delta : -1, 0, or +1
        speed_mode :  0 = maintain, 1 = slower, 2 = faster

    Returns:
        int in [0, DEC_N_ACTIONS)
    """
    assert lane_delta in (-1, 0, 1), f"invalid lane_delta {lane_delta}"
    assert speed_mode in (0, 1, 2), f"invalid speed_mode {speed_mode}"
    lane_block = {0: 0, -1: 1, +1: 2}[lane_delta]
    return lane_block * 3 + speed_mode


def decode_action(a: int) -> tuple:
    """Decode a discrete action into (lane_delta, speed_mode)."""
    a = int(a)
    assert 0 <= a < DEC_N_ACTIONS
    return int(LANE_DELTAS[a]), int(a % 3)


def target_speed_from_action(a: int, v_ref: float) -> float:
    """Compute the MPC reference speed [m/s] for the chosen action."""
    return float(v_ref + SPEED_OFFSETS[int(a)])


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def _normalise_gap(ds: float | None, dvx: float | None) -> tuple:
    if ds is None:
        return 1.0, 0.0
    ds_n  = float(np.clip(ds, -PERCEPTION_RANGE, PERCEPTION_RANGE) / NORM["DS"])
    dvx_n = float(np.clip((dvx if dvx is not None else 0.0), -20.0, 20.0) / NORM["DV"])
    return ds_n, dvx_n


def build_decision_obs(
    ego_vx: float,
    ego_vy: float,
    lane_rel: int,
    ey: float,
    epsi: float,
    neighbours: dict,
) -> np.ndarray:
    """Build the 17-dim decision observation.

    Args:
        ego_vx, ego_vy : ego velocity components [m/s]
        lane_rel       : current lane index relative to episode start lane
                         (ego starts at 0; -1 if shifted one lane "down",
                         +1 "up"). This lets the policy know *how far it
                         has already moved laterally*.
        ey             : lateral error from current lane centre [m]
        epsi           : heading error [rad]
        neighbours     : dict with keys
            {'front_same','front_left','front_right',
             'rear_same', 'rear_left', 'rear_right'}
            each either None (no vehicle) or (ds, dvx) where
            ds  = longitudinal gap [m], +ve forward (rear_* values are −ve)
            dvx = relative longitudinal speed  (v_other − v_ego) [m/s]

    Returns:
        np.ndarray shape (DEC_OBS_DIM,), dtype float32
    """
    obs = np.zeros(DEC_OBS_DIM, dtype=np.float32)
    obs[0] = float(ego_vx) / NORM["V"]
    obs[1] = float(ego_vy) / NORM["V"]
    obs[2] = float(lane_rel)
    obs[3] = float(ey)   / NORM["EY"]
    obs[4] = float(epsi) / NORM["EPSI"]

    order = ["front_same", "front_left", "front_right",
             "rear_same",  "rear_left",  "rear_right"]
    idx = 5
    for key in order:
        nb = neighbours.get(key)
        if nb is None:
            ds_n, dvx_n = 1.0, 0.0
        else:
            ds, dvx = nb
            ds_n, dvx_n = _normalise_gap(ds, dvx)
        obs[idx]     = ds_n
        obs[idx + 1] = dvx_n
        idx += 2

    return obs


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class DecisionPolicy(nn.Module):
        """
        MLP categorical policy + value head.

        Head layout kept small because the underlying observation is
        low-dimensional and the training signal (BC + light PPO fine-tune)
        is small. 2-layer 128-unit torso is enough.
        """

        def __init__(self, obs_dim: int = DEC_OBS_DIM,
                     n_actions: int = DEC_N_ACTIONS,
                     hidden: int = 128):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            )
            self.pi_head = nn.Linear(hidden, n_actions)
            self.v_head  = nn.Linear(hidden, 1)

        def forward(self, obs: torch.Tensor):
            h = self.trunk(obs)
            return self.pi_head(h), self.v_head(h).squeeze(-1)

        def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
            self.eval()
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _ = self.forward(x)
                if deterministic:
                    return int(logits.argmax(dim=-1).item())
                probs = F.softmax(logits, dim=-1)
                return int(torch.multinomial(probs, 1).item())

        def log_prob(self, obs: torch.Tensor, actions: torch.Tensor):
            logits, v = self.forward(obs)
            logp = F.log_softmax(logits, dim=-1)
            return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1), v, logits


else:
    # Torch-less fallback: allow obs builder imports to work in environments
    # where PyTorch is not installed (e.g. exiD extraction on a CI box).
    class DecisionPolicy:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch not available — install torch to use DecisionPolicy")


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test: obs and action round-trips.
    nb = {
        "front_same":  (25.0, -3.0),
        "front_left":  None,
        "front_right": (12.0,  1.5),
        "rear_same":   (-8.0,  2.0),
        "rear_left":   None,
        "rear_right":  None,
    }
    obs = build_decision_obs(10.0, 0.1, 0, 0.05, 0.01, nb)
    print("obs shape:", obs.shape, "dtype:", obs.dtype)
    print("obs:", obs)
    assert obs.shape == (DEC_OBS_DIM,)

    for a in range(DEC_N_ACTIONS):
        ld, sm = decode_action(a)
        a_back = encode_action(ld, sm)
        assert a == a_back, (a, ld, sm, a_back)
        print(f"action {a}: lane_delta={ld:+d} speed_mode={sm} "
              f"-> target_speed(10.0)={target_speed_from_action(a, 10.0):.1f}")

    if _HAS_TORCH:
        pol = DecisionPolicy()
        a = pol.act(obs, deterministic=False)
        print("sampled action:", a)
    print("OK")
