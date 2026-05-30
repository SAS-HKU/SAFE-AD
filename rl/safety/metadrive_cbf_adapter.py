"""
MetaDrive adapter for the analytic CBF safety filter.
========================================================
Projects a MetaDrive normalised action ``[steer, throttle_brake] in [-1, 1]^2``
onto the CBF-safe-and-comfortable set defined by :class:`CBFSafetyFilter`,
operating in the **ego body frame** (ego at the origin, heading = +x).

Enabled barriers in the MetaDrive setting:
  - longitudinal gap / time-headway CBF  (body-frame leader in the lane band)
  - adjacent-vehicle lateral CBF         (body-frame neighbours)
  - comfort-rate limits                  (jerk, steer-rate, comfort-decel)

The road-edge *lane-boundary* CBF is intentionally neutralised here: MetaDrive
maps are procedural/curved, so the filter is fed the ego at the lane centre
(``ego_y = (LANE_LEFT_LIMIT + LANE_RIGHT_LIMIT)/2``) which makes the boundary
barrier inactive. MetaDrive's own out-of-road termination covers lane departure.
This keeps the projection frame-robust; the part that backs the paper claim --
that the CBF *enforces steering and jerk directly* -- is the comfort-rate layer,
which is frame-agnostic.

Unit mapping (normalised action <-> physical):
  steer (rad)  : delta = steer_norm * STEER_ABS_MAX               (and inverse)
  accel (m/s^2): a = tb * ACCEL_MAX (tb>=0) | tb * |ACCEL_MIN| (tb<0)  (and inverse)
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rl.safety.cbf_filter import CBFConfig, CBFSafetyFilter, VehicleState
from rl.risk.metadrive_scene_adapter import (
    enumerate_traffic_vehicles,
    velocity_to_ego_body,
    world_to_ego_body,
)


class MetaDriveCBFProjector:
    """Stateful CBF projection layer for MetaDrive single-agent envs."""

    def __init__(self, cfg: Optional[CBFConfig] = None, neighbor_radius_m: float = 60.0) -> None:
        self.cfg = cfg if cfg is not None else CBFConfig()
        self.cbf = CBFSafetyFilter(self.cfg)
        self.neighbor_radius_m = float(neighbor_radius_m)
        self._lane_centre_y = 0.5 * (self.cfg.LANE_LEFT_LIMIT + self.cfg.LANE_RIGHT_LIMIT)
        self.reset()

    def reset(self) -> None:
        self._last_steer_rad: Optional[float] = None
        self._last_accel: Optional[float] = None
        self.n_steps = 0
        self.n_interventions = 0

    # --- unit mapping -----------------------------------------------------
    def _steer_to_rad(self, steer_norm: float) -> float:
        return float(np.clip(steer_norm, -1.0, 1.0)) * self.cfg.STEER_ABS_MAX

    def _rad_to_steer(self, delta_rad: float) -> float:
        return float(np.clip(delta_rad / max(1e-6, self.cfg.STEER_ABS_MAX), -1.0, 1.0))

    def _throttle_to_accel(self, tb: float) -> float:
        tb = float(np.clip(tb, -1.0, 1.0))
        return tb * self.cfg.ACCEL_MAX if tb >= 0.0 else tb * abs(self.cfg.ACCEL_MIN)

    def _accel_to_throttle(self, a: float) -> float:
        if a >= 0.0:
            return float(np.clip(a / max(1e-6, self.cfg.ACCEL_MAX), 0.0, 1.0))
        return float(np.clip(a / max(1e-6, abs(self.cfg.ACCEL_MIN)), -1.0, 0.0))

    # --- scene -> body-frame VehicleState ---------------------------------
    def _body_frame_neighbors(self, env) -> List[VehicleState]:
        ego = env.unwrapped.agent
        ego_pos = np.asarray(ego.position, dtype=float)
        ego_heading = float(getattr(ego, "heading_theta", 0.0))
        out: List[VehicleState] = []
        for other in enumerate_traffic_vehicles(env):
            if other is None or other is ego:
                continue
            try:
                pos = np.asarray(other.position, dtype=float)
                vel = np.asarray(other.velocity, dtype=float)
            except Exception:
                continue
            xb, yb = world_to_ego_body(pos[0], pos[1], ego_pos[0], ego_pos[1], ego_heading)
            if xb * xb + yb * yb > self.neighbor_radius_m ** 2:
                continue
            vxb, vyb = velocity_to_ego_body(vel[0], vel[1], ego_heading)
            out.append(VehicleState(
                x=xb, y=self._lane_centre_y + yb, vx=vxb, vy=vyb,
                length=float(getattr(other, "LENGTH", 4.5)),
            ))
        return out

    def project(self, env, action) -> Tuple[np.ndarray, dict]:
        """Return ``(safe_action, info)``; ``safe_action`` is normalised [-1,1]^2."""
        action = np.asarray(action, dtype=np.float32).flatten()
        steer_norm = float(action[0]) if action.size >= 1 else 0.0
        tb_norm = float(action[1]) if action.size >= 2 else 0.0

        ego = env.unwrapped.agent
        ego_speed = float(getattr(ego, "speed_km_h", 0.0)) / 3.6
        surrounding = self._body_frame_neighbors(env)

        a_raw = self._throttle_to_accel(tb_norm)
        d_raw = self._steer_to_rad(steer_norm)

        a_safe, d_safe, info = self.cbf.project(
            a_raw=a_raw, delta_raw=d_raw,
            ego_x=0.0, ego_y=self._lane_centre_y, ego_vx=ego_speed, ego_vy=0.0,
            surrounding=surrounding, ego_yaw=0.0,
            last_accel=self._last_accel, last_steer=self._last_steer_rad,
        )

        self._last_accel = a_safe
        self._last_steer_rad = d_safe
        self.n_steps += 1
        intervened = bool(info.get("a_clipped") or info.get("delta_clipped"))
        if intervened:
            self.n_interventions += 1
        info["cbf_intervened"] = intervened

        safe_action = np.array([self._rad_to_steer(d_safe),
                                self._accel_to_throttle(a_safe)], dtype=np.float32)
        return safe_action, info

    @property
    def intervention_rate(self) -> float:
        return float(self.n_interventions) / max(1, self.n_steps)


if __name__ == "__main__":
    # Lightweight offline check of the unit mapping + a synthetic close-leader
    # projection, without spinning up MetaDrive.
    proj = MetaDriveCBFProjector()
    # round-trip mapping
    for s in (-1.0, -0.3, 0.0, 0.5, 1.0):
        assert abs(proj._rad_to_steer(proj._steer_to_rad(s)) - s) < 1e-6
    for tb in (-1.0, -0.4, 0.0, 0.6, 1.0):
        assert abs(proj._accel_to_throttle(proj._throttle_to_accel(tb)) - tb) < 1e-6
    print("unit-mapping round trip OK")

    # Directly exercise the underlying filter through a fake leader to confirm
    # the projector's CBF wiring brakes for a close slow leader.
    a_safe, d_safe, info = proj.cbf.project(
        a_raw=0.5, delta_raw=0.0, ego_x=0.0, ego_y=proj._lane_centre_y,
        ego_vx=12.0, ego_vy=0.0,
        surrounding=[VehicleState(x=4.0, y=proj._lane_centre_y, vx=2.0)],
        ego_yaw=0.0, last_accel=0.0, last_steer=0.0,
    )
    assert a_safe < 0.0, f"expected braking, got {a_safe}"
    print(f"close-leader projection OK: a_safe={a_safe:.3f}, throttle={proj._accel_to_throttle(a_safe):.3f}")
    print("metadrive_cbf_adapter self-test passed.")
