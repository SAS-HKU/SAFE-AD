"""Causal finite-horizon transport solver for prospective traffic risk.

The legacy DRIFT solver evolves risk accumulated from past source fields.  That
state is useful for reproducing earlier experiments, but it is not the right
teacher for a descriptor intended to anticipate where interaction pressure
will move next.  This module instead evaluates the discounted finite-horizon
transport integral

    R_H(x, t) = Z_H^{-1} integral_0^H exp(-lambda s)
                [G_{2 D s} * Q_t](x - v_t(x) s) ds,

using only information available at time ``t``.  Semi-Lagrangian backtracing
keeps the update stable for the coarse sensing intervals used by HighwayEnv,
and Gaussian convolution represents diffusion/process uncertainty.  The
implementation is deterministic and does not use future trajectory labels;
those labels are reserved for construct-validity evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass(frozen=True)
class ProspectiveSolverConfig:
    horizon_s: float = 3.0
    integration_step_s: float = 0.25
    decay_rate: float = 0.25
    transport_scale: float = 1.0
    max_diffusion_sigma_cells: float = 4.0

    def validated(self) -> "ProspectiveSolverConfig":
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if self.integration_step_s <= 0:
            raise ValueError("integration_step_s must be positive")
        if self.decay_rate < 0:
            raise ValueError("decay_rate cannot be negative")
        if self.transport_scale <= 0:
            raise ValueError("transport_scale must be positive")
        if self.max_diffusion_sigma_cells < 0:
            raise ValueError("max_diffusion_sigma_cells cannot be negative")
        return self

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


class ProspectiveRiskSolver:
    """Numerically integrate a causal risk envelope on a regular 2-D grid."""

    version = "prospective_v2"

    def __init__(
        self,
        *,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        config: ProspectiveSolverConfig | None = None,
    ) -> None:
        self.x_grid = np.asarray(x_grid, dtype=np.float32)
        self.y_grid = np.asarray(y_grid, dtype=np.float32)
        if self.x_grid.ndim != 1 or self.y_grid.ndim != 1:
            raise ValueError("x_grid and y_grid must be one-dimensional")
        if len(self.x_grid) < 2 or len(self.y_grid) < 2:
            raise ValueError("Each grid axis requires at least two points")
        self.dx = float(self.x_grid[1] - self.x_grid[0])
        self.dy = float(self.y_grid[1] - self.y_grid[0])
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("Grid axes must be strictly increasing")
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.config = (config or ProspectiveSolverConfig()).validated()
        self.last_timing_ms = 0.0
        self.last_terminal_source = np.zeros(self.shape, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.y_grid), len(self.x_grid)

    def _check_field(self, name: str, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != self.shape:
            raise ValueError(f"{name} has shape {array.shape}; expected {self.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        return array

    def solve(
        self,
        source: np.ndarray,
        velocity_x: np.ndarray,
        velocity_y: np.ndarray,
        diffusion: np.ndarray,
        *,
        road_mask: np.ndarray | None = None,
        horizon_s: float | None = None,
    ) -> np.ndarray:
        """Return the normalized discounted occupancy-pressure envelope.

        ``velocity_x`` and ``velocity_y`` must be expressed in the same frame
        as the grid.  In closed-loop use this is the ego-relative rotating
        frame, so the resulting field directly supports look-ahead queries.
        """
        started = time.perf_counter()
        q = np.clip(self._check_field("source", source), 0.0, None)
        vx = self._check_field("velocity_x", velocity_x)
        vy = self._check_field("velocity_y", velocity_y)
        D = np.clip(self._check_field("diffusion", diffusion), 0.0, None)
        mask = (
            np.ones(self.shape, dtype=np.float32)
            if road_mask is None
            else np.clip(self._check_field("road_mask", road_mask), 0.0, 1.0)
        )
        horizon = float(self.config.horizon_s if horizon_s is None else horizon_s)
        if horizon <= 0:
            raise ValueError("horizon_s must be positive")
        dt = min(float(self.config.integration_step_s), horizon)
        times = np.arange(0.0, horizon + 0.5 * dt, dt, dtype=np.float32)
        if times[-1] < horizon - 1e-6:
            times = np.append(times, np.float32(horizon))
        else:
            times[-1] = np.float32(horizon)

        # A robust scalar diffusion level prevents isolated coefficient spikes
        # from blurring the complete field while retaining uncertainty growth.
        positive_D = D[D > 0]
        diffusion_level = float(np.median(positive_D)) if positive_D.size else 0.0
        weighted_fields: list[np.ndarray] = []
        weights = np.exp(-float(self.config.decay_rate) * times).astype(np.float32)
        for tau in times:
            back_x = self.X - float(self.config.transport_scale) * vx * float(tau)
            back_y = self.Y - float(self.config.transport_scale) * vy * float(tau)
            coordinates = np.stack(
                [
                    (back_y - float(self.y_grid[0])) / self.dy,
                    (back_x - float(self.x_grid[0])) / self.dx,
                ],
                axis=0,
            )
            transported = map_coordinates(
                q,
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            if tau > 0 and diffusion_level > 0:
                spread_m = np.sqrt(2.0 * diffusion_level * float(tau))
                sigma_y = min(
                    float(self.config.max_diffusion_sigma_cells), spread_m / self.dy
                )
                sigma_x = min(
                    float(self.config.max_diffusion_sigma_cells), spread_m / self.dx
                )
                transported = gaussian_filter(
                    transported,
                    sigma=(sigma_y, sigma_x),
                    mode="nearest",
                )
            weighted_fields.append(np.asarray(transported, dtype=np.float32))

        stack = np.stack(weighted_fields, axis=0)
        if len(times) == 1:
            result = stack[0]
        else:
            numerator = np.trapz(stack * weights[:, None, None], x=times, axis=0)
            denominator = max(float(np.trapz(weights, x=times)), 1e-8)
            result = numerator / denominator
        result = np.clip(np.asarray(result, dtype=np.float32), 0.0, None) * mask
        self.last_terminal_source = (
            np.asarray(weighted_fields[-1], dtype=np.float32) * mask
        )
        self.last_timing_ms = 1000.0 * (time.perf_counter() - started)
        return result
