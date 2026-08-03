"""Interchangeable numerical and PINN risk-field backends.

Both backends expose the query surface used by the HighwayEnv wrapper.  The
numerical backend advances the advection-diffusion PDE, whereas the PINN
backend evaluates a trained surrogate from the same instantaneous coefficient
fields.  Keeping this contract explicit makes teacher-to-surrogate policy
swaps measurable instead of changing unrelated observation or reward logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from Integration.drift_interface import DRIFTInterface
from pde_solver import compute_diffusion_field, compute_total_Q, compute_velocity_field
from rl.risk.pinn_adapter import PINNRiskAdapter
from rl.risk.prospective_solver import ProspectiveRiskSolver, ProspectiveSolverConfig
from rl.risk.recurrent_pinn_operator import (
    INPUT_CHANNELS,
    OperatorScales,
    build_operator_input,
    checkpoint_domain_scales,
    load_recurrent_pinn_checkpoint,
    select_checkpoint_inputs,
    warp_ego_local_field,
)
from rl.risk.scene_conditioning import (
    compute_dist_nearest_field,
    summarize_selected_agents,
)


@dataclass
class FieldBackendTiming:
    """Wall-clock latency decomposition for one field update.

    ``inference_ms`` is retained for compatibility and denotes the numerical
    solve or the complete tensor inference segment.  The more specific fields
    distinguish context construction, device transfers, the neural kernel,
    and post-processing so that kernel-only and online latency are not mixed.
    """

    coefficient_ms: float = 0.0
    context_ms: float = 0.0
    transfer_in_ms: float = 0.0
    kernel_ms: float = 0.0
    transfer_out_ms: float = 0.0
    postprocess_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class NumericalFieldBackend:
    """Thin timing-aware wrapper around :class:`DRIFTInterface`."""

    name = "numerical"
    requires_temporal_substeps = True

    def __init__(self, path_funcs=None, sim_cfg=None):
        self._delegate = DRIFTInterface(path_funcs=path_funcs, sim_cfg=sim_cfg)
        self.last_timing = FieldBackendTiming()

    @property
    def risk_field(self) -> np.ndarray:
        return self._delegate.risk_field

    @property
    def last_Q(self):
        return self._delegate.last_Q

    @property
    def last_D(self):
        return self._delegate.last_D

    @property
    def last_vx(self):
        return self._delegate.last_vx

    @property
    def last_vy(self):
        return self._delegate.last_vy

    def set_road_mask(self, mask):
        self._delegate.set_road_mask(mask)

    def reset(self):
        self._delegate.reset()
        self.last_timing = FieldBackendTiming()

    def warmup(self, *args, **kwargs):
        started = time.perf_counter()
        result = self._delegate.warmup(*args, **kwargs)
        elapsed = 1000.0 * (time.perf_counter() - started)
        self.last_timing = FieldBackendTiming(total_ms=elapsed, inference_ms=elapsed)
        return result

    def step(self, *args, **kwargs):
        started = time.perf_counter()
        result = self._delegate.step(*args, **kwargs)
        elapsed = 1000.0 * (time.perf_counter() - started)
        self.last_timing = FieldBackendTiming(total_ms=elapsed, inference_ms=elapsed)
        return result

    def get_risk_cartesian(self, x, y):
        return self._delegate.get_risk_cartesian(x, y)

    def get_risk_gradient_cartesian(self, x, y):
        return self._delegate.get_risk_gradient_cartesian(x, y)


class PINNFieldBackend:
    """PINN surrogate with sparse policy and full-grid visualization paths."""

    name = "pinn"
    requires_temporal_substeps = False

    def __init__(
        self,
        *,
        checkpoint_path: str,
        sim_cfg,
        device: str = "cpu",
        time_mode: str = "error",
        selection_mode: str = "soft_topk",
        top_k: int = 5,
        threshold_ratio: float = 0.15,
    ) -> None:
        if not checkpoint_path:
            raise ValueError("A PINN checkpoint is required for field_backend='pinn'")

        self.sim_cfg = sim_cfg
        self.X = np.asarray(sim_cfg.X, dtype=np.float32)
        self.Y = np.asarray(sim_cfg.Y, dtype=np.float32)
        self.x_grid = np.asarray(sim_cfg.x, dtype=np.float32)
        self.y_grid = np.asarray(sim_cfg.y, dtype=np.float32)
        self.dx = float(sim_cfg.dx)
        self.dy = float(sim_cfg.dy)
        self.selection_mode = str(selection_mode)
        self.top_k = int(top_k)
        self.threshold_ratio = float(threshold_ratio)

        self.adapter = PINNRiskAdapter(
            checkpoint_path=checkpoint_path,
            device=device,
            inference_x_range=(float(self.x_grid.min()), float(self.x_grid.max())),
            inference_y_range=(float(self.y_grid.min()), float(self.y_grid.max())),
            time_mode=time_mode,
        )
        if not self.adapter.available:
            raise RuntimeError(f"Unable to load PINN checkpoint: {checkpoint_path}")
        self.adapter.warmup(compute_gradient=True)

        self._R = np.zeros_like(self.X, dtype=np.float32)
        self._road_mask: np.ndarray | None = None
        self._interpolator = None
        self._grad_x_interp = None
        self._grad_y_interp = None
        self._time = 0.0
        self.last_Q = None
        self.last_D = None
        self.last_vx = None
        self.last_vy = None
        self._vehicles: list = []
        self._ego: dict | None = None
        self._has_full_field = False
        self.last_timing = FieldBackendTiming()

    @property
    def risk_field(self) -> np.ndarray:
        if self.last_Q is not None and not self._has_full_field:
            self.ensure_full_field()
        return self._R

    @property
    def time(self) -> float:
        return float(self._time)

    def set_road_mask(self, mask):
        self._road_mask = np.asarray(mask, dtype=np.float32)

    def reset(self):
        self._R.fill(0.0)
        self._time = 0.0
        self._interpolator = None
        self._grad_x_interp = None
        self._grad_y_interp = None
        self.last_Q = None
        self.last_D = None
        self.last_vx = None
        self.last_vy = None
        self._vehicles = []
        self._ego = None
        self._has_full_field = False
        self.last_timing = FieldBackendTiming()

    def warmup(
        self,
        vehicles,
        ego,
        dt=0.1,
        duration=5.0,
        substeps=3,
        source_fn=None,
        velocity_fn=None,
        diffusion_fn=None,
        full_field=False,
    ):
        del dt, substeps
        self._time = float(max(0.0, duration))
        return self._evaluate(
            vehicles,
            ego,
            source_fn=source_fn,
            velocity_fn=velocity_fn,
            diffusion_fn=diffusion_fn,
            full_field=full_field,
        )

    def step(
        self,
        vehicles,
        ego,
        dt=0.1,
        substeps=3,
        source_fn=None,
        velocity_fn=None,
        diffusion_fn=None,
        full_field=False,
    ):
        del substeps
        self._time += float(dt)
        return self._evaluate(
            vehicles,
            ego,
            source_fn=source_fn,
            velocity_fn=velocity_fn,
            diffusion_fn=diffusion_fn,
            full_field=full_field,
        )

    def _evaluate(
        self,
        vehicles,
        ego,
        *,
        source_fn: Callable | None,
        velocity_fn: Callable | None,
        diffusion_fn: Callable | None,
        full_field: bool,
    ) -> np.ndarray:
        total_started = time.perf_counter()
        coeff_started = total_started

        if source_fn is None:
            Q_total, _Q_veh, _Q_occ, occ_mask = compute_total_Q(
                vehicles, ego, self.X, self.Y, config=self.sim_cfg
            )
        else:
            Q_total, _Q_veh, _Q_occ, occ_mask = source_fn(
                vehicles, ego, self.X, self.Y
            )

        velocity_output = (
            compute_velocity_field(vehicles, ego, self.X, self.Y, config=self.sim_cfg)
            if velocity_fn is None
            else velocity_fn(vehicles, ego, self.X, self.Y)
        )
        if len(velocity_output) == 2:
            vx, vy = velocity_output
        elif len(velocity_output) == 6:
            vx, vy = velocity_output[:2]
        else:
            raise ValueError(
                "velocity_fn must return (vx, vy) or the standard six-value output"
            )

        D = (
            compute_diffusion_field(
                occ_mask, self.X, self.Y, vehicles, ego, config=self.sim_cfg
            )
            if diffusion_fn is None
            else diffusion_fn(occ_mask, self.X, self.Y, vehicles, ego)
        )
        coefficient_ms = 1000.0 * (time.perf_counter() - coeff_started)
        self.last_Q = np.asarray(Q_total, dtype=np.float32)
        self.last_vx = np.asarray(vx, dtype=np.float32)
        self.last_vy = np.asarray(vy, dtype=np.float32)
        self.last_D = np.asarray(D, dtype=np.float32)
        self._vehicles = list(vehicles)
        self._ego = dict(ego)
        self._has_full_field = False

        inference_ms = 0.0
        if full_field:
            infer_started = time.perf_counter()
            self._evaluate_full_field()
            inference_ms = 1000.0 * (time.perf_counter() - infer_started)
        else:
            self._R.fill(0.0)
            self._interpolator = None
            self._grad_x_interp = None
            self._grad_y_interp = None

        self.last_timing = FieldBackendTiming(
            coefficient_ms=coefficient_ms,
            inference_ms=inference_ms,
            total_ms=1000.0 * (time.perf_counter() - total_started),
        )
        return self._R

    def _evaluate_full_field(self) -> np.ndarray:
        """Evaluate and cache the complete field for rendering or diagnostics."""
        if self.last_Q is None or self._ego is None:
            raise RuntimeError("PINN coefficients are unavailable; call step() first")
        R = self.adapter.query_grid(
            X=self.X,
            Y=self.Y,
            t=self._time,
            Q=self.last_Q,
            vx=self.last_vx,
            vy=self.last_vy,
            D=self.last_D,
            vehicles=self._vehicles,
            ego_vehicle=self._ego,
            selection_mode=self.selection_mode,
            top_k=self.top_k,
            threshold_ratio=self.threshold_ratio,
        )
        if self._road_mask is not None:
            R = np.asarray(R, dtype=np.float32) * self._road_mask
        self._R = np.asarray(R, dtype=np.float32)
        self._update_interpolators()
        self._has_full_field = True
        return self._R

    def ensure_full_field(self) -> np.ndarray:
        """Lazily materialize the current complete field."""
        if self._has_full_field:
            return self._R
        started = time.perf_counter()
        result = self._evaluate_full_field()
        inference_ms = 1000.0 * (time.perf_counter() - started)
        self.last_timing = FieldBackendTiming(
            coefficient_ms=self.last_timing.coefficient_ms,
            inference_ms=inference_ms,
            total_ms=self.last_timing.coefficient_ms + inference_ms,
        )
        return result

    def query_risk_features(
        self,
        *,
        ego_x: float,
        ego_y: float,
        lane_centers: list[float],
        current_lane: int,
    ) -> dict[str, float]:
        """Query the eight policy descriptors without evaluating the full grid."""
        if self.last_Q is None or self._ego is None:
            return {
                key: 0.0
                for key in (
                    "r_ego", "r_5m", "r_10m", "r_20m", "grad_x", "grad_y",
                    "r_left", "r_right",
                )
            }
        started = time.perf_counter()
        features = self.adapter.query_risk_features(
            ego_x=ego_x,
            ego_y=ego_y,
            t=self._time,
            Q_grid=self.last_Q,
            vx_grid=self.last_vx,
            vy_grid=self.last_vy,
            D_grid=self.last_D,
            sim_cfg=self.sim_cfg,
            lane_centers=lane_centers,
            current_lane=current_lane,
            vehicles=self._vehicles,
            ego_vehicle=self._ego,
            selection_mode=self.selection_mode,
            top_k=self.top_k,
            threshold_ratio=self.threshold_ratio,
        )
        inference_ms = 1000.0 * (time.perf_counter() - started)
        self.last_timing = FieldBackendTiming(
            coefficient_ms=self.last_timing.coefficient_ms,
            inference_ms=inference_ms,
            total_ms=self.last_timing.coefficient_ms + inference_ms,
        )
        return {key: float(value) for key, value in features.items()}

    def query_cartesian_points(self, xs, ys, *, compute_gradient=False) -> dict[str, np.ndarray]:
        """Query policy points without materializing the complete render grid."""
        xs = np.asarray(xs, dtype=np.float32).reshape(-1)
        ys = np.asarray(ys, dtype=np.float32).reshape(-1)
        if xs.shape != ys.shape:
            raise ValueError("xs and ys must have identical shapes")
        if self.last_Q is None or self._ego is None:
            zeros = np.zeros(xs.size, dtype=np.float32)
            return {"R": zeros, "grad_x": zeros.copy(), "grad_y": zeros.copy()}
        started = time.perf_counter()
        result = self.adapter.query_points(
            xs=xs,
            ys=ys,
            t=self._time,
            Q_grid=self.last_Q,
            vx_grid=self.last_vx,
            vy_grid=self.last_vy,
            D_grid=self.last_D,
            sim_cfg=self.sim_cfg,
            vehicles=self._vehicles,
            ego_vehicle=self._ego,
            selection_mode=self.selection_mode,
            top_k=self.top_k,
            threshold_ratio=self.threshold_ratio,
            compute_gradient=bool(compute_gradient),
        )
        inference_ms = 1000.0 * (time.perf_counter() - started)
        self.last_timing = FieldBackendTiming(
            coefficient_ms=self.last_timing.coefficient_ms,
            inference_ms=inference_ms,
            total_ms=self.last_timing.coefficient_ms + inference_ms,
        )
        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in result.items()
        }

    def _update_interpolators(self) -> None:
        R_smooth = gaussian_filter(self._R, sigma=0.3)
        grad_y, grad_x = np.gradient(R_smooth, self.dy, self.dx)
        points = (self.y_grid, self.x_grid)
        kwargs: dict[str, Any] = {
            "method": "linear",
            "bounds_error": False,
            "fill_value": 0.0,
        }
        self._interpolator = RegularGridInterpolator(points, R_smooth, **kwargs)
        self._grad_x_interp = RegularGridInterpolator(points, grad_x, **kwargs)
        self._grad_y_interp = RegularGridInterpolator(points, grad_y, **kwargs)

    @staticmethod
    def _query(interpolator, x, y):
        if interpolator is None:
            values = np.zeros_like(np.atleast_1d(x), dtype=float)
        else:
            values = interpolator(np.column_stack([np.atleast_1d(y), np.atleast_1d(x)]))
        if np.ndim(x) == 0:
            return float(values[0])
        return values

    def get_risk_cartesian(self, x, y):
        if self._interpolator is None and self.last_Q is not None:
            self.ensure_full_field()
        return self._query(self._interpolator, x, y)

    def get_risk_gradient_cartesian(self, x, y):
        if self._grad_x_interp is None and self.last_Q is not None:
            self.ensure_full_field()
        gx = self._query(self._grad_x_interp, x, y)
        gy = self._query(self._grad_y_interp, x, y)
        return gx, gy


class RecurrentPINNFieldBackend:
    """Stateful full-field PINN operator used for an actual online swap."""

    name = "pinn_recurrent"
    requires_temporal_substeps = False

    def __init__(
        self,
        *,
        checkpoint_path: str,
        sim_cfg,
        device: str = "cpu",
        time_mode: str = "error",
        selection_mode: str = "soft_topk",
        top_k: int = 5,
        threshold_ratio: float = 0.15,
    ) -> None:
        del time_mode
        self.sim_cfg = sim_cfg
        self.device = str(device)
        self.selection_mode = str(selection_mode)
        self.top_k = int(top_k)
        self.threshold_ratio = float(threshold_ratio)
        self.model, self.checkpoint = load_recurrent_pinn_checkpoint(
            checkpoint_path, device=self.device
        )
        self.name = (
            "pinn_prospective"
            if self.checkpoint.get("model_family") == "prospective_context_pinn"
            else "pinn_recurrent"
        )
        self._uses_history = bool(
            {"R_prev", "R_t_prev", "R_trend"}.intersection(
                self.checkpoint.get("input_channel_names", INPUT_CHANNELS)
            )
        )
        if self.checkpoint.get("coordinate_mode") != "ego_local":
            raise ValueError(
                "Closed-loop HighwayEnv deployment requires an ego-local recurrent PINN checkpoint"
            )
        self.scales = checkpoint_domain_scales(self.checkpoint, "highwayenv")
        self.local_x = np.asarray(self.checkpoint["x_grid"], dtype=np.float32)
        self.local_y = np.asarray(self.checkpoint["y_grid"], dtype=np.float32)
        self.local_X, self.local_Y = np.meshgrid(self.local_x, self.local_y)
        self.global_X = np.asarray(sim_cfg.X, dtype=np.float32)
        self.global_Y = np.asarray(sim_cfg.Y, dtype=np.float32)
        self.global_x = np.asarray(sim_cfg.x, dtype=np.float32)
        self.global_y = np.asarray(sim_cfg.y, dtype=np.float32)
        self._local_R = np.zeros_like(self.local_X, dtype=np.float32)
        self._local_R_t = np.zeros_like(self.local_X, dtype=np.float32)
        self._global_R = np.zeros_like(self.global_X, dtype=np.float32)
        self._road_mask: np.ndarray | None = None
        self._previous_pose: tuple[float, float, float] | None = None
        self._current_pose: tuple[float, float, float] | None = None
        self._local_interp = None
        self._local_grad_x_interp = None
        self._local_grad_y_interp = None
        self._has_global_field = False
        self._time = 0.0
        self.last_Q = None
        self.last_D = None
        self.last_vx = None
        self.last_vy = None
        self.last_timing = FieldBackendTiming()
        self._warm_model()

    def _warm_model(self) -> None:
        import torch

        tensor = torch.zeros(
            (1, int(self.checkpoint["input_channels"]), len(self.local_y), len(self.local_x)),
            dtype=torch.float32,
            device=self.device,
        )
        if (
            self.device == "cpu"
            and self.checkpoint.get("model_family") == "prospective_context_pinn"
        ):
            traced = torch.jit.trace(self.model, tensor, check_trace=False)
            self.model = torch.jit.freeze(traced.eval())
        with torch.inference_mode():
            self.model(tensor)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

    @property
    def risk_field(self) -> np.ndarray:
        if not self._has_global_field and self._current_pose is not None:
            self.ensure_full_field()
        return self._global_R

    @property
    def time(self) -> float:
        return float(self._time)

    def set_road_mask(self, mask) -> None:
        self._road_mask = np.asarray(mask, dtype=np.float32)

    def reset(self) -> None:
        self._local_R.fill(0.0)
        self._local_R_t.fill(0.0)
        self._global_R.fill(0.0)
        self._previous_pose = None
        self._current_pose = None
        self._local_interp = None
        self._local_grad_x_interp = None
        self._local_grad_y_interp = None
        self._has_global_field = False
        self._time = 0.0
        self.last_Q = self.last_D = self.last_vx = self.last_vy = None
        self.last_timing = FieldBackendTiming()

    @staticmethod
    def _pose(ego: dict) -> tuple[float, float, float]:
        return (
            float(ego["x"]),
            float(ego["y"]),
            float(ego.get("heading", 0.0)),
        )

    @staticmethod
    def _angle_delta(current: float, previous: float) -> float:
        return float(np.arctan2(np.sin(current - previous), np.cos(current - previous)))

    def _local_to_world(self, ego: dict) -> tuple[np.ndarray, np.ndarray]:
        heading = float(ego.get("heading", 0.0))
        c, s = float(np.cos(heading)), float(np.sin(heading))
        return (
            float(ego["x"]) + c * self.local_X - s * self.local_Y,
            float(ego["y"]) + s * self.local_X + c * self.local_Y,
        )

    def _sample_global(self, field: np.ndarray, Xw, Yw, *, fill=0.0) -> np.ndarray:
        interpolator = RegularGridInterpolator(
            (self.global_y, self.global_x),
            np.asarray(field, dtype=np.float32),
            method="linear",
            bounds_error=False,
            fill_value=float(fill),
        )
        points = np.column_stack([np.asarray(Yw).reshape(-1), np.asarray(Xw).reshape(-1)])
        return interpolator(points).reshape(self.local_X.shape).astype(np.float32)

    def _world_to_local_vehicle(self, vehicle: dict, ego: dict) -> dict:
        heading = float(ego.get("heading", 0.0))
        c, s = float(np.cos(heading)), float(np.sin(heading))
        dx = float(vehicle["x"]) - float(ego["x"])
        dy = float(vehicle["y"]) - float(ego["y"])
        result = dict(vehicle)
        result.update(
            x=c * dx + s * dy,
            y=-s * dx + c * dy,
            vx=c * float(vehicle.get("vx", 0.0)) + s * float(vehicle.get("vy", 0.0)),
            vy=-s * float(vehicle.get("vx", 0.0)) + c * float(vehicle.get("vy", 0.0)),
            heading=float(vehicle.get("heading", 0.0)) - heading,
        )
        return result

    def _coefficients(
        self,
        vehicles,
        ego,
        *,
        source_fn: Callable | None,
        velocity_fn: Callable | None,
        diffusion_fn: Callable | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if source_fn is None:
            Q, _qv, _qo, occ = compute_total_Q(
                vehicles, ego, self.global_X, self.global_Y, config=self.sim_cfg
            )
        else:
            Q, _qv, _qo, occ = source_fn(vehicles, ego, self.global_X, self.global_Y)
        velocity = (
            compute_velocity_field(
                vehicles, ego, self.global_X, self.global_Y, config=self.sim_cfg
            )
            if velocity_fn is None
            else velocity_fn(vehicles, ego, self.global_X, self.global_Y)
        )
        vx, vy = velocity[:2]
        D = (
            compute_diffusion_field(
                occ, self.global_X, self.global_Y, vehicles, ego, config=self.sim_cfg
            )
            if diffusion_fn is None
            else diffusion_fn(occ, self.global_X, self.global_Y, vehicles, ego)
        )
        self.last_Q = np.asarray(Q, dtype=np.float32)
        self.last_vx = np.asarray(vx, dtype=np.float32)
        self.last_vy = np.asarray(vy, dtype=np.float32)
        self.last_D = np.asarray(D, dtype=np.float32)
        return self.last_Q, self.last_vx, self.last_vy, self.last_D, np.asarray(occ)

    def _advance(
        self,
        vehicles,
        ego,
        *,
        dt: float,
        source_fn: Callable | None,
        velocity_fn: Callable | None,
        diffusion_fn: Callable | None,
    ) -> np.ndarray:
        import torch

        total_started = time.perf_counter()
        coefficient_started = total_started
        Q, vx, vy, D, occ = self._coefficients(
            vehicles,
            ego,
            source_fn=source_fn,
            velocity_fn=velocity_fn,
            diffusion_fn=diffusion_fn,
        )
        coefficient_ms = 1000.0 * (time.perf_counter() - coefficient_started)
        context_started = time.perf_counter()
        current_pose = self._pose(ego)
        yaw_rate = (
            0.0
            if self._previous_pose is None
            else self._angle_delta(current_pose[2], self._previous_pose[2])
            / max(float(dt), 1e-6)
        )
        if self._previous_pose is None or not self._uses_history:
            aligned_R = np.zeros_like(self._local_R)
            aligned_R_t = np.zeros_like(self._local_R_t)
        else:
            aligned_R = warp_ego_local_field(
                self._local_R,
                x_grid=self.local_x,
                y_grid=self.local_y,
                previous_pose=self._previous_pose,
                current_pose=current_pose,
            )
            aligned_R_t = warp_ego_local_field(
                self._local_R_t,
                x_grid=self.local_x,
                y_grid=self.local_y,
                previous_pose=self._previous_pose,
                current_pose=current_pose,
            )
        Xw, Yw = self._local_to_world(ego)
        q_local = self._sample_global(Q, Xw, Yw)
        vx_world = self._sample_global(vx, Xw, Yw)
        vy_world = self._sample_global(vy, Xw, Yw)
        d_local = self._sample_global(D, Xw, Yw)
        occ_local = self._sample_global(occ, Xw, Yw)
        heading = current_pose[2]
        c, s = float(np.cos(heading)), float(np.sin(heading))
        relative_vx = vx_world - float(ego.get("vx", 0.0))
        relative_vy = vy_world - float(ego.get("vy", 0.0))
        vx_local = c * relative_vx + s * relative_vy + yaw_rate * self.local_Y
        vy_local = -s * relative_vx + c * relative_vy - yaw_rate * self.local_X
        selected = summarize_selected_agents(
            ego=ego,
            vehicles=vehicles,
            perception_range=80.0,
            selection_mode=self.selection_mode,
            top_k=self.top_k,
            threshold_ratio=self.threshold_ratio,
        )
        local_agents = [
            self._world_to_local_vehicle(vehicle, ego)
            for vehicle in selected["selected_agents"]
        ]
        distance = compute_dist_nearest_field(
            self.local_X, self.local_Y, local_agents, fill_value=80.0
        )
        local_mask = (
            np.ones_like(q_local, dtype=np.float32)
            if self._road_mask is None
            else self._sample_global(self._road_mask, Xw, Yw)
        )
        snapshot = {
            "R_prev": aligned_R,
            "R_t_prev": aligned_R_t,
            "Q": q_local,
            "vx": vx_local,
            "vy": vy_local,
            "D": d_local,
            "occ_mask": occ_local,
            "road_mask": local_mask,
            "dist_nearest": distance,
            "ego_yaw_rate": yaw_rate,
            "N_agents": float(selected["N_agents_selected"]),
            "dt": float(dt),
        }
        inputs = build_operator_input(
            snapshot,
            x_grid=self.local_x,
            y_grid=self.local_y,
            scales=self.scales,
        )
        inputs = select_checkpoint_inputs(
            inputs,
            self.checkpoint,
            domain="highwayenv",
        )
        context_ms = 1000.0 * (time.perf_counter() - context_started)

        transfer_in_started = time.perf_counter()
        tensor = torch.from_numpy(inputs[None]).to(self.device)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        transfer_in_ms = 1000.0 * (time.perf_counter() - transfer_in_started)

        kernel_started = time.perf_counter()
        with torch.inference_mode():
            risk_norm, rate_norm = self.model(tensor)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
        kernel_ms = 1000.0 * (time.perf_counter() - kernel_started)

        transfer_out_started = time.perf_counter()
        self._local_R = (
            risk_norm[0, 0].detach().cpu().numpy().astype(np.float32) * self.scales.risk
        )
        self._local_R_t = (
            rate_norm[0, 0].detach().cpu().numpy().astype(np.float32)
            * self.scales.risk_rate
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        transfer_out_ms = 1000.0 * (time.perf_counter() - transfer_out_started)

        postprocess_started = time.perf_counter()
        if self._road_mask is not None:
            self._local_R *= local_mask
            self._local_R_t *= local_mask
        self._previous_pose = current_pose
        self._current_pose = current_pose
        self._time += float(dt)
        self._has_global_field = False
        self._update_local_interpolators()
        postprocess_ms = 1000.0 * (time.perf_counter() - postprocess_started)
        inference_ms = transfer_in_ms + kernel_ms + transfer_out_ms
        self.last_timing = FieldBackendTiming(
            coefficient_ms=coefficient_ms,
            context_ms=context_ms,
            transfer_in_ms=transfer_in_ms,
            kernel_ms=kernel_ms,
            transfer_out_ms=transfer_out_ms,
            postprocess_ms=postprocess_ms,
            inference_ms=inference_ms,
            total_ms=1000.0 * (time.perf_counter() - total_started),
        )
        return self._local_R

    def warmup(
        self,
        vehicles,
        ego,
        dt=0.1,
        duration=5.0,
        substeps=3,
        source_fn=None,
        velocity_fn=None,
        diffusion_fn=None,
        full_field=False,
    ):
        del substeps
        result = self._local_R
        repetitions = (
            1
            if not bool(self.checkpoint.get("stateful", True))
            else max(1, int(round(float(duration) / max(float(dt), 1e-6))))
        )
        for _ in range(repetitions):
            result = self._advance(
                vehicles,
                ego,
                dt=float(dt),
                source_fn=source_fn,
                velocity_fn=velocity_fn,
                diffusion_fn=diffusion_fn,
            )
        if full_field:
            self.ensure_full_field()
        return result

    def step(
        self,
        vehicles,
        ego,
        dt=0.1,
        substeps=3,
        source_fn=None,
        velocity_fn=None,
        diffusion_fn=None,
        full_field=False,
    ):
        del substeps
        result = self._advance(
            vehicles,
            ego,
            dt=float(dt),
            source_fn=source_fn,
            velocity_fn=velocity_fn,
            diffusion_fn=diffusion_fn,
        )
        if full_field:
            self.ensure_full_field()
        return result

    def _update_local_interpolators(self) -> None:
        smooth = gaussian_filter(self._local_R, sigma=0.3)
        gy, gx = np.gradient(
            smooth,
            float(self.local_y[1] - self.local_y[0]),
            float(self.local_x[1] - self.local_x[0]),
        )
        kwargs = {"bounds_error": False, "fill_value": 0.0, "method": "linear"}
        points = (self.local_y, self.local_x)
        self._local_interp = RegularGridInterpolator(points, smooth, **kwargs)
        self._local_grad_x_interp = RegularGridInterpolator(points, gx, **kwargs)
        self._local_grad_y_interp = RegularGridInterpolator(points, gy, **kwargs)

    def _world_points_to_local(self, xs, ys) -> tuple[np.ndarray, np.ndarray, float]:
        if self._current_pose is None:
            return np.asarray(xs), np.asarray(ys), 0.0
        cx, cy, heading = self._current_pose
        c, s = float(np.cos(heading)), float(np.sin(heading))
        dx = np.asarray(xs, dtype=np.float32) - cx
        dy = np.asarray(ys, dtype=np.float32) - cy
        return c * dx + s * dy, -s * dx + c * dy, heading

    def query_cartesian_points(self, xs, ys, *, compute_gradient=False):
        xs = np.asarray(xs, dtype=np.float32).reshape(-1)
        ys = np.asarray(ys, dtype=np.float32).reshape(-1)
        lx, ly, heading = self._world_points_to_local(xs, ys)
        points = np.column_stack([ly, lx])
        risk = (
            self._local_interp(points).astype(np.float32)
            if self._local_interp is not None
            else np.zeros_like(xs)
        )
        gx = gy = np.zeros_like(risk)
        if compute_gradient and self._local_grad_x_interp is not None:
            gx_local = self._local_grad_x_interp(points)
            gy_local = self._local_grad_y_interp(points)
            c, s = float(np.cos(heading)), float(np.sin(heading))
            gx = (c * gx_local - s * gy_local).astype(np.float32)
            gy = (s * gx_local + c * gy_local).astype(np.float32)
        return {"R": risk, "grad_x": gx, "grad_y": gy}

    def runtime_domain_mask(self, X, Y, *, ego_vehicle=None) -> np.ndarray:
        if ego_vehicle is not None:
            pose_before = self._current_pose
            self._current_pose = self._pose(ego_vehicle)
            lx, ly, _heading = self._world_points_to_local(X, Y)
            self._current_pose = pose_before
        else:
            lx, ly, _heading = self._world_points_to_local(X, Y)
        return (
            (lx >= float(self.local_x[0]))
            & (lx <= float(self.local_x[-1]))
            & (ly >= float(self.local_y[0]))
            & (ly <= float(self.local_y[-1]))
        ).astype(np.float32)

    def get_risk_cartesian(self, x, y):
        result = self.query_cartesian_points(x, y, compute_gradient=False)["R"]
        return float(result[0]) if np.ndim(x) == 0 else result

    def get_risk_gradient_cartesian(self, x, y):
        result = self.query_cartesian_points(x, y, compute_gradient=True)
        if np.ndim(x) == 0:
            return float(result["grad_x"][0]), float(result["grad_y"][0])
        return result["grad_x"], result["grad_y"]

    def ensure_full_field(self) -> np.ndarray:
        if self._current_pose is None:
            return self._global_R
        lx, ly, _heading = self._world_points_to_local(self.global_X, self.global_Y)
        points = np.column_stack([ly.reshape(-1), lx.reshape(-1)])
        self._global_R = self._local_interp(points).reshape(self.global_X.shape).astype(np.float32)
        if self._road_mask is not None:
            self._global_R *= self._road_mask
        self._has_global_field = True
        return self._global_R


class ProspectiveNumericalFieldBackend(RecurrentPINNFieldBackend):
    """Causal ego-local numerical teacher matched to the prospective PINN."""

    name = "prospective"
    requires_temporal_substeps = False

    def __init__(
        self,
        *,
        sim_cfg,
        reference_checkpoint: str | None = None,
        selection_mode: str = "soft_topk",
        top_k: int = 5,
        threshold_ratio: float = 0.15,
    ) -> None:
        self.sim_cfg = sim_cfg
        self.device = "cpu"
        self.selection_mode = str(selection_mode)
        self.top_k = int(top_k)
        self.threshold_ratio = float(threshold_ratio)
        checkpoint = None
        if reference_checkpoint:
            import torch

            checkpoint = torch.load(
                str(reference_checkpoint), map_location="cpu", weights_only=False
            )
            if checkpoint.get("model_family") != "prospective_context_pinn":
                raise ValueError(
                    "A prospective numerical reference must use a prospective_context_pinn checkpoint"
                )
        self.checkpoint = checkpoint or {}
        self.local_x = np.asarray(
            self.checkpoint.get(
                "x_grid", np.linspace(-40.0, 100.0, 141, dtype=np.float32)
            ),
            dtype=np.float32,
        )
        self.local_y = np.asarray(
            self.checkpoint.get(
                "y_grid", np.linspace(-20.0, 20.0, 41, dtype=np.float32)
            ),
            dtype=np.float32,
        )
        solver_values = self.checkpoint.get("prospective_solver", {})
        self.solver = ProspectiveRiskSolver(
            x_grid=self.local_x,
            y_grid=self.local_y,
            config=ProspectiveSolverConfig(**solver_values),
        )
        self.local_X, self.local_Y = np.meshgrid(self.local_x, self.local_y)
        self.global_X = np.asarray(sim_cfg.X, dtype=np.float32)
        self.global_Y = np.asarray(sim_cfg.Y, dtype=np.float32)
        self.global_x = np.asarray(sim_cfg.x, dtype=np.float32)
        self.global_y = np.asarray(sim_cfg.y, dtype=np.float32)
        self._local_R = np.zeros_like(self.local_X, dtype=np.float32)
        self._local_R_t = np.zeros_like(self.local_X, dtype=np.float32)
        self._global_R = np.zeros_like(self.global_X, dtype=np.float32)
        self._road_mask: np.ndarray | None = None
        self._previous_pose: tuple[float, float, float] | None = None
        self._current_pose: tuple[float, float, float] | None = None
        self._local_interp = None
        self._local_grad_x_interp = None
        self._local_grad_y_interp = None
        self._has_global_field = False
        self._time = 0.0
        self.last_Q = None
        self.last_D = None
        self.last_vx = None
        self.last_vy = None
        self.last_timing = FieldBackendTiming()

    def reset(self) -> None:
        self._local_R.fill(0.0)
        self._local_R_t.fill(0.0)
        self._global_R.fill(0.0)
        self._previous_pose = None
        self._current_pose = None
        self._local_interp = None
        self._local_grad_x_interp = None
        self._local_grad_y_interp = None
        self._has_global_field = False
        self._time = 0.0
        self.last_Q = self.last_D = self.last_vx = self.last_vy = None
        self.last_timing = FieldBackendTiming()

    def _advance(
        self,
        vehicles,
        ego,
        *,
        dt: float,
        source_fn: Callable | None,
        velocity_fn: Callable | None,
        diffusion_fn: Callable | None,
    ) -> np.ndarray:
        total_started = time.perf_counter()
        coefficient_started = total_started
        Q, vx, vy, D, occ = self._coefficients(
            vehicles,
            ego,
            source_fn=source_fn,
            velocity_fn=velocity_fn,
            diffusion_fn=diffusion_fn,
        )
        coefficient_ms = 1000.0 * (time.perf_counter() - coefficient_started)
        context_started = time.perf_counter()
        current_pose = self._pose(ego)
        yaw_rate = (
            0.0
            if self._previous_pose is None
            else self._angle_delta(current_pose[2], self._previous_pose[2])
            / max(float(dt), 1e-6)
        )
        Xw, Yw = self._local_to_world(ego)
        q_local = self._sample_global(Q, Xw, Yw)
        vx_world = self._sample_global(vx, Xw, Yw)
        vy_world = self._sample_global(vy, Xw, Yw)
        d_local = self._sample_global(D, Xw, Yw)
        _occ_local = self._sample_global(occ, Xw, Yw)
        local_mask = (
            np.ones_like(q_local, dtype=np.float32)
            if self._road_mask is None
            else self._sample_global(self._road_mask, Xw, Yw)
        )
        heading = current_pose[2]
        c, s = float(np.cos(heading)), float(np.sin(heading))
        relative_vx = vx_world - float(ego.get("vx", 0.0))
        relative_vy = vy_world - float(ego.get("vy", 0.0))
        vx_local = c * relative_vx + s * relative_vy + yaw_rate * self.local_Y
        vy_local = -s * relative_vx + c * relative_vy - yaw_rate * self.local_X
        context_ms = 1000.0 * (time.perf_counter() - context_started)

        previous = self._local_R.copy()
        solve_started = time.perf_counter()
        self._local_R = self.solver.solve(
            q_local,
            vx_local,
            vy_local,
            d_local,
            road_mask=local_mask,
        )
        inference_ms = 1000.0 * (time.perf_counter() - solve_started)
        postprocess_started = time.perf_counter()
        self._local_R_t = (self._local_R - previous) / max(float(dt), 1e-6)
        self._previous_pose = current_pose
        self._current_pose = current_pose
        self._time += float(dt)
        self._has_global_field = False
        self._update_local_interpolators()
        postprocess_ms = 1000.0 * (time.perf_counter() - postprocess_started)
        self.last_timing = FieldBackendTiming(
            coefficient_ms=coefficient_ms,
            context_ms=context_ms,
            kernel_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            inference_ms=inference_ms,
            total_ms=1000.0 * (time.perf_counter() - total_started),
        )
        return self._local_R

    def warmup(
        self,
        vehicles,
        ego,
        dt=0.1,
        duration=5.0,
        substeps=3,
        source_fn=None,
        velocity_fn=None,
        diffusion_fn=None,
        full_field=False,
    ):
        del duration, substeps
        result = self._advance(
            vehicles,
            ego,
            dt=float(dt),
            source_fn=source_fn,
            velocity_fn=velocity_fn,
            diffusion_fn=diffusion_fn,
        )
        if full_field:
            self.ensure_full_field()
        return result


def make_field_backend(
    backend: str,
    *,
    sim_cfg,
    pinn_checkpoint: str | None = None,
    pinn_device: str = "cpu",
    pinn_time_mode: str = "error",
    selection_mode: str = "soft_topk",
    top_k: int = 5,
    threshold_ratio: float = 0.15,
):
    backend = str(backend).strip().lower()
    if backend == "numerical":
        return NumericalFieldBackend(sim_cfg=sim_cfg)
    if backend == "prospective":
        return ProspectiveNumericalFieldBackend(
            sim_cfg=sim_cfg,
            reference_checkpoint=pinn_checkpoint,
            selection_mode=selection_mode,
            top_k=top_k,
            threshold_ratio=threshold_ratio,
        )
    if backend == "pinn":
        import torch

        checkpoint = torch.load(
            str(pinn_checkpoint or ""), map_location="cpu", weights_only=False
        )
        backend_cls = (
            RecurrentPINNFieldBackend
            if checkpoint.get("model_family")
            in {"recurrent_context_pinn", "prospective_context_pinn"}
            else PINNFieldBackend
        )
        return backend_cls(
            checkpoint_path=str(pinn_checkpoint or ""),
            sim_cfg=sim_cfg,
            device=pinn_device,
            time_mode=pinn_time_mode,
            selection_mode=selection_mode,
            top_k=top_k,
            threshold_ratio=threshold_ratio,
        )
    raise ValueError("field backend must be 'numerical', 'prospective', or 'pinn'")
