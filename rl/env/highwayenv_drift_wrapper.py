from __future__ import annotations

import math
import os
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config as cfg
from pde_solver import (
    compute_Q_occlusion,
    compute_Q_vehicle,
    create_vehicle as drift_create_vehicle,
)
from Integration.drift_interface import DRIFTInterface


_HIGHWAY_DRIFT_GRID_READY = False
_DRIFT_NEIGHBOR_AHEAD_M = 85.0
_DRIFT_NEIGHBOR_BEHIND_M = 35.0
_DRIFT_NEIGHBOR_LATERAL_M = 14.0
_DRIFT_NEIGHBOR_RADIUS_M = 95.0
_GRID_DX_REF = (255.2 - (-150.0)) / (250 - 1)
_GRID_DY_REF = ((-45.3) - (-225.2)) / (80 - 1)

# compute_Q_vehicle was calibrated for the IDEAM scenario where surrounding
# vehicles travel at 5-10 m/s.  HighwayEnv vehicles travel at 20-30 m/s which
# inflates the relative-speed weight (omega_rel) roughly 2x, and the default
# omega_brake=3.0 amplification in pde_solver always fires (not just when
# vehicles actually brake).  Scale the Q source down aggressively so
# (a) the in-view p90 risk matches the [0, 3] band seen in uncertainty_test
# and (b) reward-gate semantics stay physically meaningful.
_HIGHWAYENV_Q_SCALE = 0.22
# Cap per-cell Q to avoid long-tail PDE explosions when several fast vehicles
# overlap.  The analytical steady-state (no diffusion) is R ≈ Q / λ_decay;
# with λ_decay = 0.15 and a cap of 0.45 this gives R ≈ 3 at hotspots, and
# diffusion brings typical peaks into the [0, 2] band that matches the
# uncertainty_test_DREAM reference rendering.
_HIGHWAYENV_Q_CAP = 0.45
# CFL stability target for the explicit PDE integrator.  With D_occ = 6 m²/s
# and the dx≈1.6, dy≈1.8 grid, the diffusion stability bound is
#     sub_dt < 0.5 / (D_occ * (1/dx² + 1/dy²)) ≈ 0.124 s.
# HighwayEnv merge-v0 / highway-v0 run at policy_frequency=1 Hz (dt=1.0 s),
# so the legacy substeps=3 yielded sub_dt=0.33 s — unstable — and the field
# saturated at the solver clip (R=10) regardless of Q scaling.  Target
# sub_dt ≤ _PDE_SUB_DT_TARGET and derive `substeps = ceil(dt / target)`.
_PDE_SUB_DT_TARGET = 0.04
# Maximum time a chunk of PDE integration may see *fixed* vehicle positions.
# HighwayEnv vehicles travel ~20 m/s, so holding sources fixed for a full
# env.step() (dt=1.0 s on merge-v0 / highway-v0) effectively teleports every
# source by ~20 m = ~12 grid cells between frames — the risk field appears to
# "pulse" instead of flow smoothly even though the PDE itself is continuous.
# Splitting the env step into chunks of ≤ _PDE_CHUNK_DT_TARGET seconds and
# back-projecting each vehicle by v·Δt within each chunk makes the source
# deposit glide with the vehicle.  At 0.1 s chunks the per-chunk displacement
# is ~2 m (≈1 grid cell), which is below visual resolution.
_PDE_CHUNK_DT_TARGET = 0.1


def _configure_highwayenv_drift_grid() -> None:
    global _HIGHWAY_DRIFT_GRID_READY
    if _HIGHWAY_DRIFT_GRID_READY:
        return
    # dx ~1.6 m, dy ~2.2 m — mirrors the IDEAM reference (config.py). With
    # dt/substeps ≈ 0.033 s and D_occ = 6 m²/s, these resolutions keep the
    # explicit PDE diffusion term well within CFL stability
    # (D*dt*(1/dx² + 1/dy²) ≈ 0.12 < 0.5). Finer dy blows up the field.
    cfg.x_min, cfg.x_max = -80.0, 1080.0
    cfg.y_min, cfg.y_max = -22.0, 26.0
    cfg.nx, cfg.ny = 340, 22
    cfg.dx = (cfg.x_max - cfg.x_min) / (cfg.nx - 1)
    cfg.dy = (cfg.y_max - cfg.y_min) / (cfg.ny - 1)
    cfg.x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    cfg.y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    cfg.X, cfg.Y = np.meshgrid(cfg.x, cfg.y)
    _HIGHWAY_DRIFT_GRID_READY = True


class DriftOverlayWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        use_drift: bool = True,
        drift_warmup_s: float = 2.0,
        reward_gate_scale_r0: float = 1.5,
        risk_clip: float = 5.0,
        record_risk_metrics: bool = False,
        gate_reward: bool = True,
    ) -> None:
        super().__init__(env)
        _configure_highwayenv_drift_grid()
        self.use_drift = bool(use_drift)
        self.drift_warmup_s = float(max(0.0, drift_warmup_s))
        self.reward_gate_scale_r0 = float(max(1e-6, reward_gate_scale_r0))
        self.risk_clip = float(max(1e-6, risk_clip))
        self.record_risk_metrics = bool(record_risk_metrics)
        self.gate_reward = bool(gate_reward)
        self.drift = None
        self._x_origin = 0.0
        self._overlay_cfg: dict[str, Any] = {}
        self._road_mask: np.ndarray | None = None
        self._episode_metrics: list[dict[str, float]] = []
        self._grid_X: np.ndarray | None = None
        self._grid_Y: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        self._episode_metrics = []
        if not self.use_drift:
            info["use_drift"] = False
            return obs, info

        raw = self._raw_env()
        self._x_origin = float(raw.vehicle.position[0]) - 20.0
        self._configure_grid_for_env(raw)
        self._capture_grid()
        self._overlay_cfg = self._resolve_overlay_config()
        self.drift = DRIFTInterface()
        self.drift.reset()
        self._road_mask = self._build_road_mask()
        if self._road_mask is not None:
            self.drift.set_road_mask(self._road_mask)
        if self.drift_warmup_s > 0.0:
            ego, vehicles = self._collect_drift_state()
            dt_env = 1.0 / float(raw.config["policy_frequency"])
            self.drift.warmup(
                vehicles,
                ego,
                dt=dt_env,
                duration=self.drift_warmup_s,
                substeps=self._pde_substeps(dt_env),
                source_fn=self._source_fn,
            )
        metrics = self._current_metrics()
        info.update(self._format_info(metrics, base_reward=None))
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        base_reward = float(np.clip(base_reward, 0.0, 1.0))
        if not self.use_drift:
            info["use_drift"] = False
            info["base_reward"] = base_reward
            info["reward_gated"] = base_reward
            return obs, base_reward, terminated, truncated, info

        ego, vehicles = self._collect_drift_state()
        raw = self._raw_env()
        dt_env = 1.0 / float(raw.config["policy_frequency"])
        self._drift_step_interpolated(vehicles, ego, dt_env)
        metrics = self._current_metrics()
        info.update(self._format_info(metrics, base_reward=base_reward))
        gated_reward = self._gate_reward(base_reward, metrics)
        info["reward_gated"] = gated_reward
        if self.record_risk_metrics:
            self._episode_metrics.append(metrics)
            if terminated or truncated:
                info["drift_episode_summary"] = self._episode_summary()
        reward_out = gated_reward if self.gate_reward else base_reward
        return obs, reward_out, terminated, truncated, info

    def _raw_env(self):
        base = self.env.unwrapped
        return getattr(base, "_inner", base)

    @staticmethod
    def _pde_substeps(dt: float) -> int:
        # CFL: sub_dt * D_occ * (1/dx² + 1/dy²) < 0.5. On the dx≈1.6/dy≈1.8
        # grid the bound is sub_dt < 0.124 s; we pick _PDE_SUB_DT_TARGET=0.04 s
        # for a comfortable safety margin. At dt=1.0 s this yields substeps=25;
        # at dt=0.2 s (highway-fast-v0) it yields substeps=5.
        return max(3, int(math.ceil(float(dt) / _PDE_SUB_DT_TARGET)))

    @staticmethod
    def _pde_chunks(dt: float) -> int:
        # How many temporal chunks to split one env step into so that source
        # positions get re-sampled along the vehicle trajectory rather than
        # jumping from frame N to frame N+1.  Target: chunk_dt ≤ 0.1 s so
        # that vehicles travelling ~20 m/s displace ≤ ~2 m (~1 grid cell)
        # between chunk boundaries.
        return max(1, int(math.ceil(float(dt) / _PDE_CHUNK_DT_TARGET)))

    @staticmethod
    def _back_project_vehicle(vehicle: dict, back_dt: float) -> dict:
        """Return a copy of `vehicle` with position advected backward by
        `back_dt * velocity`.

        The PDE source term is computed from vehicle dicts.  When one
        env.step() covers 1.0 s but we want the source to appear to move
        *continuously* from the start-of-step to the end-of-step position,
        we split the step into chunks and, in each chunk, run the PDE with
        vehicles that have been back-projected by the remaining time — so
        sources are deposited along the actual trajectory rather than all
        at the end point.
        """
        if not float(back_dt):
            return vehicle
        veh = dict(vehicle)
        vx = float(veh.get("vx", 0.0))
        vy = float(veh.get("vy", 0.0))
        veh["x"] = float(veh["x"]) - vx * float(back_dt)
        veh["y"] = float(veh["y"]) - vy * float(back_dt)
        return veh

    def _drift_step_interpolated(self, vehicles: list[dict], ego: dict, dt_env: float) -> None:
        """Advance the PDE through one env step while sliding sources along the
        vehicle trajectory.  Each of `n_chunks` inner calls passes vehicles
        back-projected to the mid-point of its sub-interval, giving a smooth
        source trace across the frame instead of a single 20-m teleport.
        """
        n_chunks = self._pde_chunks(dt_env)
        chunk_dt = float(dt_env) / n_chunks
        substeps_per_chunk = self._pde_substeps(chunk_dt)
        for k in range(n_chunks):
            # Back-project to the midpoint of chunk k (τ = (k + 0.5) * chunk_dt
            # measured from start-of-env-step).  End state is vehicles / ego
            # as captured after env.step(), so back_dt = dt_env - τ.
            back_dt = float(dt_env) - (k + 0.5) * chunk_dt
            vehicles_t = [self._back_project_vehicle(v, back_dt) for v in vehicles]
            ego_t = self._back_project_vehicle(ego, back_dt)
            self.drift.step(
                vehicles_t,
                ego_t,
                dt=chunk_dt,
                substeps=substeps_per_chunk,
                source_fn=self._source_fn,
            )

    @property
    def raw_env(self):
        return self._raw_env()

    def world_to_drift(self, x: float, y: float) -> tuple[float, float]:
        return self._transform_xy(x, y)

    def current_drift_metrics(self) -> dict[str, float]:
        if not self.use_drift:
            return {
                "r_ego": 0.0,
                "r_fwd": 0.0,
                "r_left": 0.0,
                "r_right": 0.0,
                "grad_x": 0.0,
                "grad_y": 0.0,
            }
        return dict(self._current_metrics())

    def get_risk_field(self) -> np.ndarray | None:
        if not self.use_drift or self.drift is None:
            return None
        return self.drift.risk_field

    def get_masked_risk_field(self, threshold: float = 0.05) -> np.ndarray | None:
        risk_field = self.get_risk_field()
        if risk_field is None:
            return None
        if self._road_mask is None:
            return risk_field
        return np.where(self._road_mask > float(threshold), risk_field, np.nan)

    def get_road_mask(self) -> np.ndarray | None:
        return self._road_mask

    def get_drift_grid(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._grid_X, self._grid_Y

    def lane_polygon(self, lane, *, margin: float = 0.0, n_samples: int = 96) -> np.ndarray:
        return self._lane_polygon(lane, margin=margin, n_samples=n_samples)

    def _resolve_overlay_config(self) -> dict[str, Any]:
        raw = self._raw_env()
        if hasattr(raw, "get_drift_overlay_config"):
            cfg_dict = dict(raw.get_drift_overlay_config())
        else:
            cfg_dict = self._infer_overlay_config(raw)
        if "merge_x_start" in cfg_dict:
            cfg_dict["merge_x_start"] = float(cfg_dict["merge_x_start"]) - self._x_origin
        if "merge_x_end" in cfg_dict:
            cfg_dict["merge_x_end"] = float(cfg_dict["merge_x_end"]) - self._x_origin
        return cfg_dict

    def _infer_overlay_config(self, raw) -> dict[str, Any]:
        if getattr(self.env, "env_id", "") == "merge-v0":
            road = raw.road
            if ("b" in road.network.graph) and ("c" in road.network.graph["b"]) and len(road.network.graph["b"]["c"]) >= 3:
                merge_lane = ("b", "c", 2)
                target_lane = ("b", "c", 1)
                lane = road.network.get_lane(merge_lane)
                return {
                    "use_merge_source": True,
                    "merge_x_start": float(lane.start[0]),
                    "merge_x_end": float(lane.end[0]),
                    "merge_from_lane_index": merge_lane,
                    "merge_to_lane_index": target_lane,
                }
        return {"use_merge_source": False}

    def _transform_xy(self, x: float, y: float) -> tuple[float, float]:
        return float(x - self._x_origin), float(y)

    def _configure_grid_for_env(self, raw) -> None:
        xs: list[float] = []
        ys: list[float] = []
        road = getattr(raw, "road", None)
        if road is None or not getattr(road, "network", None):
            _configure_highwayenv_drift_grid()
            return

        for _from, to_dict in road.network.graph.items():
            for _to, lanes in to_dict.items():
                for lane in lanes:
                    samples = np.linspace(0.0, float(lane.length), 80)
                    for s in samples:
                        width = float(lane.width_at(float(s)))
                        for lateral in (-0.5 * width, 0.5 * width):
                            x_t, y_t = self._transform_xy(*lane.position(float(s), lateral))
                            xs.append(float(x_t))
                            ys.append(float(y_t))

        if not xs or not ys:
            _configure_highwayenv_drift_grid()
            return

        margin_x_m = 80.0   # large margin along x so ego doesn't fall near the sponge layer
        margin_y_m = 10.0
        x_min = min(xs) - margin_x_m
        x_max = max(xs) + margin_x_m
        y_min = min(ys) - margin_y_m
        y_max = max(ys) + margin_y_m

        cfg.x_min, cfg.x_max = float(x_min), float(x_max)
        cfg.y_min, cfg.y_max = float(y_min), float(y_max)
        # Use the IDEAM reference resolution (dx≈1.6 m, dy≈2.2 m) which keeps
        # the PDE stable under the default D_occ / dt combination.
        cfg.nx = max(250, int((cfg.x_max - cfg.x_min) / _GRID_DX_REF) + 2)
        cfg.ny = max(22, int((cfg.y_max - cfg.y_min) / _GRID_DY_REF) + 2)
        cfg.dx = (cfg.x_max - cfg.x_min) / (cfg.nx - 1)
        cfg.dy = (cfg.y_max - cfg.y_min) / (cfg.ny - 1)
        cfg.x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
        cfg.y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
        cfg.X, cfg.Y = np.meshgrid(cfg.x, cfg.y)

    def _capture_grid(self) -> None:
        self._grid_X = np.array(cfg.X, copy=True)
        self._grid_Y = np.array(cfg.Y, copy=True)

    def _lane_polygon(self, lane, *, margin: float = 0.0, n_samples: int = 96) -> np.ndarray:
        samples = np.linspace(0.0, float(lane.length), max(2, int(n_samples)))
        upper = []
        lower = []
        for s in samples:
            width = float(lane.width_at(float(s)))
            upper.append(self._transform_xy(*lane.position(float(s), +0.5 * width + float(margin))))
            lower.append(self._transform_xy(*lane.position(float(s), -0.5 * width - float(margin))))
        upper_arr = np.asarray(upper, dtype=np.float32)
        lower_arr = np.asarray(lower, dtype=np.float32)
        return np.vstack([upper_arr, lower_arr[::-1], upper_arr[:1]])

    def _build_road_mask(self) -> np.ndarray | None:
        raw = self._raw_env()
        road = getattr(raw, "road", None)
        if road is None or not getattr(road, "network", None):
            return None

        query_points = np.column_stack([cfg.X.ravel(), cfg.Y.ravel()])
        mask = np.zeros(query_points.shape[0], dtype=bool)

        for _from, to_dict in road.network.graph.items():
            for _to, lanes in to_dict.items():
                for lane in lanes:
                    polygon = self._lane_polygon(lane, margin=0.45, n_samples=120)
                    if polygon.shape[0] < 4:
                        continue
                    lane_path = Path(polygon, closed=True)
                    mask |= lane_path.contains_points(query_points, radius=0.10)

        if not np.any(mask):
            return np.ones_like(cfg.X, dtype=np.float32)

        road_mask = mask.reshape(cfg.X.shape).astype(np.float32)
        road_mask = gaussian_filter(road_mask, sigma=0.8)
        max_value = float(np.max(road_mask))
        if max_value > 1e-6:
            road_mask /= max_value
        return np.clip(road_mask, 0.0, 1.0)

    def _vehicle_to_drift(self, vehicle, vid: int) -> dict[str, Any]:
        vx, vy = vehicle.velocity
        x, y = self._transform_xy(float(vehicle.position[0]), float(vehicle.position[1]))
        vclass = "truck" if bool(getattr(vehicle, "is_truck", False)) or float(getattr(vehicle, "LENGTH", 5.0)) >= 8.0 else "car"
        drift_vehicle = drift_create_vehicle(vid=vid, x=x, y=y, vx=float(vx), vy=float(vy), vclass=vclass)
        drift_vehicle["heading"] = float(vehicle.heading)
        action = getattr(vehicle, "action", None)
        accel = 0.0
        if isinstance(action, dict):
            accel = float(action.get("acceleration", 0.0))
        drift_vehicle["a"] = accel
        return drift_vehicle

    def _collect_drift_state(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        raw = self._raw_env()
        ego = self._vehicle_to_drift(raw.vehicle, 0)
        vehicles: list[dict[str, Any]] = []
        vid = 1
        ego_x_world = float(raw.vehicle.position[0])
        ego_y_world = float(raw.vehicle.position[1])
        ego_heading = float(getattr(raw.vehicle, "heading", 0.0))
        cos_h = math.cos(ego_heading)
        sin_h = math.sin(ego_heading)
        selected: dict[int, Any] = {}

        def _within_window(other) -> bool:
            dx = float(other.position[0]) - ego_x_world
            dy = float(other.position[1]) - ego_y_world
            rel_long = cos_h * dx + sin_h * dy
            rel_lat = -sin_h * dx + cos_h * dy
            if rel_long > _DRIFT_NEIGHBOR_AHEAD_M or rel_long < -_DRIFT_NEIGHBOR_BEHIND_M:
                return False
            if abs(rel_lat) > _DRIFT_NEIGHBOR_LATERAL_M and math.hypot(rel_long, rel_lat) > _DRIFT_NEIGHBOR_RADIUS_M:
                return False
            return True

        def _maybe_add(other) -> None:
            if other is None or other is raw.vehicle:
                return
            if not _within_window(other):
                return
            selected[id(other)] = other

        current_lane = getattr(raw.vehicle, "lane_index", None)
        lane_candidates = [current_lane] if current_lane is not None else []
        if current_lane is not None:
            lane_candidates.extend(raw.road.network.side_lanes(current_lane))
        for lane_index in lane_candidates:
            front, rear = raw.road.neighbour_vehicles(raw.vehicle, lane_index)
            _maybe_add(front)
            _maybe_add(rear)

        for other in raw.road.vehicles:
            if other is raw.vehicle:
                continue
            if bool(getattr(other, "is_truck", False)) or str(getattr(other, "_scenario_tag", "")) in {"truck", "blocker"}:
                _maybe_add(other)

        # Fallback for curved / intersection-like topologies where lane-neighbour
        # queries often return empty even though several nearby actors are inside
        # the ego conflict region (e.g. roundabout-v0).  Keep the straight-highway
        # prioritization above, but if it selected nothing, admit all nearby
        # vehicles inside the ego-heading window.
        if not selected:
            for other in raw.road.vehicles:
                if other is raw.vehicle:
                    continue
                _maybe_add(other)

        for other in selected.values():
            vehicles.append(self._vehicle_to_drift(other, vid))
            vid += 1
        hidden_fn = getattr(raw, "get_hidden_drift_vehicles", None)
        if callable(hidden_fn):
            for hidden in hidden_fn():
                vehicles.append(self._vehicle_to_drift(hidden, vid))
                vid += 1
        return ego, vehicles

    def _compute_merge_source(self, vehicles, ego, X, Y) -> np.ndarray:
        if not self._overlay_cfg.get("use_merge_source", False):
            return np.zeros_like(X)
        x_start = float(self._overlay_cfg.get("merge_x_start", 30.0))
        x_end = float(self._overlay_cfg.get("merge_x_end", 70.0))
        if x_end <= x_start:
            return np.zeros_like(X)
        from_lane_idx = self._overlay_cfg.get("merge_from_lane_index")
        to_lane_idx = self._overlay_cfg.get("merge_to_lane_index")
        if from_lane_idx is None or to_lane_idx is None:
            return np.zeros_like(X)
        raw = self._raw_env()
        from_lane = raw.road.network.get_lane(from_lane_idx)
        to_lane = raw.road.network.get_lane(to_lane_idx)
        from_y = self._transform_xy(*from_lane.position(0.0, 0.0))[1]
        to_y = self._transform_xy(*to_lane.position(0.0, 0.0))[1]

        # Gate merge source by presence of a merger: only fire when a surrounding
        # vehicle is close to the from-lane (lateral error < 3 m) AND roughly
        # aligned with the merge corridor along x. Without this check the merge
        # source floods the ramp with phantom risk even when no vehicle is using
        # the on-ramp.
        has_merger = False
        for vehicle in vehicles:
            if abs(float(vehicle["y"]) - from_y) > 3.0:
                continue
            if float(vehicle["x"]) < x_start - 30.0 or float(vehicle["x"]) > x_end + 10.0:
                continue
            has_merger = True
            break
        if not has_merger:
            return np.zeros_like(X)

        midpoint = 0.5 * (from_y + to_y)
        lateral_scale = max(4.0, abs(from_y - to_y))
        s = np.clip((X - x_start) / max(1.0, x_end - x_start), 0.0, 1.0)
        ramp = 3 * s ** 2 - 2 * s ** 3
        lateral = np.exp(-0.5 * ((Y - midpoint) ** 2) / max(1.0, lateral_scale ** 2))
        gore = np.exp(-((X - x_end) ** 2 + (Y - to_y) ** 2) / 100.0)
        # Density restricted to merger-lane candidates only (avoid smearing from
        # every mainline vehicle).
        density = np.zeros_like(X)
        for vehicle in vehicles:
            if abs(float(vehicle["y"]) - from_y) > 3.0:
                continue
            dist_sq = ((X - vehicle["x"]) ** 2 / 400.0) + ((Y - vehicle["y"]) ** 2 / 9.0)
            density += np.exp(-0.5 * dist_sq)
        density = np.clip(density, 0.0, 1.0)
        return (0.6 * ramp * lateral + 1.0 * gore) * density

    def _source_fn(self, vehicles, ego, X, Y):
        q_veh = compute_Q_vehicle(vehicles, ego, X, Y)
        q_occ, occ_mask = compute_Q_occlusion(vehicles, ego, X, Y)
        q_merge = self._compute_merge_source(vehicles, ego, X, Y)
        # Calibrate + clip so the steady-state PDE solution stays in the
        # uncertainty_test_DREAM range (peak R ~ 1-3 on-road).
        q_veh = _HIGHWAYENV_Q_SCALE * q_veh
        q_occ = _HIGHWAYENV_Q_SCALE * q_occ
        q_merge = _HIGHWAYENV_Q_SCALE * q_merge
        q_total = np.clip(q_veh + q_occ + q_merge, 0.0, _HIGHWAYENV_Q_CAP)
        return q_total, q_veh, q_occ, occ_mask

    def _lane_risk_ahead(self, lane_index, length: float, n_samples: int = 8) -> float:
        if lane_index is None:
            return 0.0
        raw = self._raw_env()
        ego = raw.vehicle
        lane = raw.road.network.get_lane(lane_index)
        s0, _ = lane.local_coordinates(ego.position)
        s1 = min(float(lane.length), float(s0) + float(length))
        if s1 <= s0:
            return 0.0
        samples = np.linspace(s0, s1, max(2, int(n_samples)))
        points = np.array([self._transform_xy(*lane.position(float(s), 0.0)) for s in samples], dtype=np.float32)
        risks = self.drift.get_risk_cartesian(points[:, 0], points[:, 1])
        return float(np.max(risks)) if len(risks) else 0.0

    def _current_metrics(self) -> dict[str, float]:
        raw = self._raw_env()
        ego = raw.vehicle
        x, y = self._transform_xy(float(ego.position[0]), float(ego.position[1]))
        r_ego = float(self.drift.get_risk_cartesian(x, y))
        grad_x, grad_y = self.drift.get_risk_gradient_cartesian(x, y)
        current_lane = ego.lane_index
        raw_lane = raw.road.network.get_lane(current_lane)
        s0, _ = raw_lane.local_coordinates(ego.position)
        current_y = raw_lane.position(s0, 0.0)[1]
        r_fwd = self._lane_risk_ahead(current_lane, length=25.0)
        r_left = 0.0
        r_right = 0.0
        for candidate in raw.road.network.side_lanes(current_lane):
            candidate_lane = raw.road.network.get_lane(candidate)
            candidate_y = candidate_lane.position(min(float(candidate_lane.length), max(0.0, s0)), 0.0)[1]
            risk = self._lane_risk_ahead(candidate, length=20.0)
            # HighwayEnv convention: higher y = rightward (lane 0 at y=0, lane 1 at y=4, ...).
            if candidate_y > current_y:
                r_right = max(r_right, risk)
            else:
                r_left = max(r_left, risk)
        return {
            "r_ego": r_ego,
            "r_fwd": r_fwd,
            "r_left": r_left,
            "r_right": r_right,
            "grad_x": float(grad_x),
            "grad_y": float(grad_y),
        }

    def _gate_reward(self, base_reward: float, metrics: dict[str, float]) -> float:
        r_gate = float(np.clip(max(metrics["r_ego"], metrics["r_fwd"]), 0.0, self.risk_clip))
        return float(np.clip(base_reward, 0.0, 1.0) * math.exp(-r_gate / self.reward_gate_scale_r0))

    def _format_info(self, metrics: dict[str, float], base_reward: float | None) -> dict[str, Any]:
        r_gate = float(np.clip(max(metrics["r_ego"], metrics["r_fwd"]), 0.0, self.risk_clip))
        info = {
            "use_drift": True,
            "base_reward": None if base_reward is None else float(base_reward),
            "reward_gate": r_gate,
            "reward_gate_scale_r0": self.reward_gate_scale_r0,
            "risk_clip": self.risk_clip,
            "r_ego": metrics["r_ego"],
            "r_fwd": metrics["r_fwd"],
            "r_left": metrics["r_left"],
            "r_right": metrics["r_right"],
            "grad_x": metrics["grad_x"],
            "grad_y": metrics["grad_y"],
            "drift_metrics": dict(metrics),
        }
        raw = self._raw_env()
        scenario_info_fn = getattr(raw, "get_scenario_info", None)
        if callable(scenario_info_fn):
            info["scenario_info"] = scenario_info_fn()
        return info

    def _episode_summary(self) -> dict[str, float]:
        if not self._episode_metrics:
            return {}
        keys = sorted(self._episode_metrics[0].keys())
        return {
            f"{key}_mean": float(np.mean([row[key] for row in self._episode_metrics]))
            for key in keys
        } | {
            f"{key}_max": float(np.max([row[key] for row in self._episode_metrics]))
            for key in keys
        }
