"""Stateful context-conditioned PINN operator for DRIFT field evolution.

The pointwise PINN cannot identify a nonlocal, history-dependent PDE solution
from instantaneous coefficients alone.  This module therefore predicts the
next complete ego-local field from the aligned previous field, telegrapher
state, and multiscale source context.  Training retains an explicit residual
of the advection-diffusion-telegrapher equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
import torch.nn.functional as F


INPUT_CHANNELS = (
    "R_prev",
    "R_t_prev",
    "R_trend",
    "Q",
    "Q_blur_2m",
    "Q_blur_6m",
    "Q_blur_15m",
    "vx",
    "vy",
    "D",
    "occ_mask",
    "road_mask",
    "dist_rbf_2m",
    "dist_rbf_6m",
    "dist_rbf_15m",
    "x",
    "y",
    "yaw_rate",
    "N_agents",
    "dt",
)


@dataclass(frozen=True)
class OperatorScales:
    risk: float
    risk_rate: float
    source: float
    velocity: float
    diffusion: float
    yaw_rate: float
    agents: float
    dt: float

    def to_dict(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, float]) -> "OperatorScales":
        return cls(**{key: float(values[key]) for key in cls.__annotations__})


class DilatedResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=width,
        )
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        groups = 4 if width % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(self.depthwise(x))
        return x + F.silu(self.norm(residual))


class RecurrentContextPINN(nn.Module):
    """A compact neural operator with an identity-initialized field update."""

    def __init__(
        self,
        input_channels: int = len(INPUT_CHANNELS),
        width: int = 24,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 1),
        channel_names: tuple[str, ...] | list[str] | None = None,
        output_mode: str = "residual",
        max_normalized_risk: float = 1.25,
        max_normalized_rate: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.channel_names = tuple(channel_names or INPUT_CHANNELS)
        if len(self.channel_names) != self.input_channels:
            raise ValueError("channel_names must match input_channels")
        self.road_mask_index = (
            self.channel_names.index("road_mask")
            if "road_mask" in self.channel_names
            else None
        )
        self.width = int(width)
        self.dilations = tuple(int(value) for value in dilations)
        self.output_mode = str(output_mode)
        if self.output_mode not in {"residual", "absolute_bounded"}:
            raise ValueError("output_mode must be 'residual' or 'absolute_bounded'")
        self.max_normalized_risk = float(max_normalized_risk)
        self.max_normalized_rate = float(max_normalized_rate)
        self.stem = nn.Conv2d(self.input_channels, self.width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [DilatedResidualBlock(self.width, value) for value in self.dilations]
        )
        self.head = nn.Conv2d(self.width, 2, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if self.output_mode == "absolute_bounded":
            initial_fraction = min(0.99, max(1e-4, 0.01 / self.max_normalized_risk))
            self.head.bias.data[0] = float(
                np.log(initial_fraction / (1.0 - initial_fraction))
            )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected (N,{self.input_channels},H,W), received {tuple(inputs.shape)}"
            )
        hidden = F.silu(self.stem(inputs))
        for block in self.blocks:
            hidden = block(hidden)
        update = self.head(hidden)
        road_mask = (
            torch.ones_like(inputs[:, 0:1])
            if self.road_mask_index is None
            else torch.clamp(
                inputs[:, self.road_mask_index : self.road_mask_index + 1], 0.0, 1.0
            )
        )
        if self.output_mode == "absolute_bounded":
            risk = (
                self.max_normalized_risk * torch.sigmoid(update[:, 0:1])
            ) * road_mask
            risk_rate = (
                self.max_normalized_rate * torch.tanh(update[:, 1:2])
            ) * road_mask
        else:
            risk = F.relu(inputs[:, 0:1] + update[:, 0:1]) * road_mask
            risk_rate = (inputs[:, 1:2] + update[:, 1:2]) * road_mask
        return risk, risk_rate


def infer_scales(recordings) -> OperatorScales:
    maxima = {
        "risk": 1e-3,
        "risk_rate": 1e-3,
        "source": 1e-3,
        "velocity": 1e-3,
        "diffusion": 1e-3,
        "yaw_rate": 1e-3,
        "agents": 1.0,
        "dt": 1e-3,
    }
    for recording in recordings:
        fields = recording._fields
        scalars = recording._scalars
        maxima["risk"] = max(maxima["risk"], float(np.max(np.abs(fields["R"]))))
        if "R_t" in fields:
            maxima["risk_rate"] = max(
                maxima["risk_rate"], float(np.max(np.abs(fields["R_t"])))
            )
        maxima["source"] = max(maxima["source"], float(np.max(np.abs(fields["Q"]))))
        maxima["velocity"] = max(
            maxima["velocity"],
            float(np.max(np.abs(fields["vx"]))),
            float(np.max(np.abs(fields["vy"]))),
        )
        maxima["diffusion"] = max(
            maxima["diffusion"], float(np.max(np.abs(fields["D"])))
        )
        maxima["yaw_rate"] = max(
            maxima["yaw_rate"],
            float(np.max(np.abs(scalars.get("ego_yaw_rate", np.zeros(1))))),
        )
        maxima["agents"] = max(
            maxima["agents"], float(np.max(scalars.get("N_agents", np.ones(1))))
        )
        maxima["dt"] = max(maxima["dt"], float(np.max(scalars.get("dt", np.ones(1)))))
    return OperatorScales(**maxima)


def build_operator_input(
    snapshot: Mapping,
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    scales: OperatorScales,
    previous_risk: np.ndarray | None = None,
    previous_rate: np.ndarray | None = None,
) -> np.ndarray:
    """Build the normalized operator tensor in the declared channel order."""
    x_grid = np.asarray(x_grid, dtype=np.float32)
    y_grid = np.asarray(y_grid, dtype=np.float32)
    X, Y = np.meshgrid(x_grid, y_grid)
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    q = np.asarray(snapshot["Q"], dtype=np.float32)
    distance = np.asarray(snapshot["dist_nearest"], dtype=np.float32)
    r_prev = np.asarray(
        snapshot.get("R_prev") if previous_risk is None else previous_risk,
        dtype=np.float32,
    )
    rt_prev = np.asarray(
        snapshot.get("R_t_prev", np.zeros_like(q))
        if previous_rate is None
        else previous_rate,
        dtype=np.float32,
    )
    q_blurs = [
        gaussian_filter(q, sigma=(scale / dy, scale / dx), mode="nearest")
        for scale in (2.0, 6.0, 15.0)
    ]
    x_norm = 2.0 * (X - float(x_grid.min())) / max(float(np.ptp(x_grid)), 1e-6) - 1.0
    y_norm = 2.0 * (Y - float(y_grid.min())) / max(float(np.ptp(y_grid)), 1e-6) - 1.0
    shape = q.shape

    def full_scalar(name: str, scale: float) -> np.ndarray:
        value = float(snapshot.get(name, 0.0)) / max(float(scale), 1e-6)
        return np.full(shape, value, dtype=np.float32)

    channels = [
        r_prev / scales.risk,
        rt_prev / scales.risk_rate,
        np.clip(
            r_prev + float(snapshot.get("dt", 0.0)) * rt_prev,
            0.0,
            1.5 * scales.risk,
        ) / scales.risk,
        q / scales.source,
        *(blur / scales.source for blur in q_blurs),
        np.asarray(snapshot["vx"], dtype=np.float32) / scales.velocity,
        np.asarray(snapshot["vy"], dtype=np.float32) / scales.velocity,
        np.asarray(snapshot["D"], dtype=np.float32) / scales.diffusion,
        np.asarray(snapshot.get("occ_mask", np.zeros_like(q)), dtype=np.float32),
        np.asarray(snapshot.get("road_mask", np.ones_like(q)), dtype=np.float32),
        *(np.exp(-distance / scale) for scale in (2.0, 6.0, 15.0)),
        x_norm,
        y_norm,
        full_scalar("ego_yaw_rate", scales.yaw_rate),
        full_scalar("N_agents", scales.agents),
        full_scalar("dt", scales.dt),
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def checkpoint_domain_scales(
    checkpoint: Mapping,
    domain: str,
) -> OperatorScales:
    """Resolve calibration-only normalization for a checkpoint domain."""
    domain_scales = checkpoint.get("domain_scales") or {}
    values = domain_scales.get(str(domain), checkpoint.get("scales"))
    if values is None:
        raise KeyError("Checkpoint does not define operator scales")
    return OperatorScales.from_dict(values)


def select_checkpoint_inputs(
    operator_input: np.ndarray,
    checkpoint: Mapping,
    *,
    domain: str,
) -> np.ndarray:
    """Select checkpoint channels and append explicit domain context."""
    index = {name: position for position, name in enumerate(INPUT_CHANNELS)}
    selected = []
    shape = tuple(np.asarray(operator_input).shape[1:])
    for name in checkpoint.get("input_channel_names", INPUT_CHANNELS):
        if name == "domain_highwayenv":
            selected.append(
                np.full(
                    shape,
                    1.0 if str(domain) == "highwayenv" else 0.0,
                    dtype=np.float32,
                )
            )
        else:
            selected.append(np.asarray(operator_input[index[name]], dtype=np.float32))
    result = np.stack(selected, axis=0).astype(np.float32)
    input_clip = checkpoint.get("input_clip")
    if input_clip is not None:
        result = np.clip(result, -float(input_clip), float(input_clip))
    return result


def warp_ego_local_field(
    field: np.ndarray,
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    previous_pose: tuple[float, float, float],
    current_pose: tuple[float, float, float],
) -> np.ndarray:
    """Reproject a previous ego-local field into the current ego frame."""
    x_grid = np.asarray(x_grid, dtype=np.float32)
    y_grid = np.asarray(y_grid, dtype=np.float32)
    Xc, Yc = np.meshgrid(x_grid, y_grid)
    px, py, ph = (float(value) for value in previous_pose)
    cx, cy, ch = (float(value) for value in current_pose)
    cc, cs = np.cos(ch), np.sin(ch)
    world_x = cx + cc * Xc - cs * Yc
    world_y = cy + cs * Xc + cc * Yc
    pc, ps = np.cos(ph), np.sin(ph)
    dx = world_x - px
    dy = world_y - py
    previous_x = pc * dx + ps * dy
    previous_y = -ps * dx + pc * dy
    # Floating-point rotation can move an exact boundary a few ulps outside
    # the interpolation domain. Snap only those near-boundary values; genuine
    # out-of-view regions must retain the zero fill used during deployment.
    tolerance = 1e-4
    previous_x = np.where(
        np.abs(previous_x - x_grid[0]) <= tolerance, x_grid[0], previous_x
    )
    previous_x = np.where(
        np.abs(previous_x - x_grid[-1]) <= tolerance, x_grid[-1], previous_x
    )
    previous_y = np.where(
        np.abs(previous_y - y_grid[0]) <= tolerance, y_grid[0], previous_y
    )
    previous_y = np.where(
        np.abs(previous_y - y_grid[-1]) <= tolerance, y_grid[-1], previous_y
    )
    interp = RegularGridInterpolator(
        (y_grid, x_grid),
        np.asarray(field, dtype=np.float32),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    points = np.column_stack([previous_y.reshape(-1), previous_x.reshape(-1)])
    return interp(points).reshape(Xc.shape).astype(np.float32)


def finite_difference_physics_losses(
    *,
    risk: torch.Tensor,
    risk_rate: torch.Tensor,
    previous_risk: torch.Tensor,
    previous_rate: torch.Tensor,
    source: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    diffusion: torch.Tensor,
    dt: torch.Tensor,
    dx: float,
    dy: float,
    tau: float,
    lambda_decay: float,
    length_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return kinematic and telegrapher residual losses in physical units."""

    def d_dx(value: torch.Tensor) -> torch.Tensor:
        core = (value[..., 2:] - value[..., :-2]) / (2.0 * dx)
        return F.pad(core, (1, 1, 0, 0), mode="replicate")

    def d_dy(value: torch.Tensor) -> torch.Tensor:
        core = (value[..., 2:, :] - value[..., :-2, :]) / (2.0 * dy)
        return F.pad(core, (0, 0, 1, 1), mode="replicate")

    dt = torch.clamp(dt, min=1e-4)
    kinematic = (risk - previous_risk) / dt - risk_rate
    grad_x = d_dx(risk)
    grad_y = d_dy(risk)
    diffusion_flux_x = diffusion * grad_x
    diffusion_flux_y = diffusion * grad_y
    div_diffusion = d_dx(diffusion_flux_x) + d_dy(diffusion_flux_y)
    div_advection = d_dx(vx * risk) + d_dy(vy * risk)
    speed = torch.sqrt(vx.square() + vy.square() + 1e-8)
    decay = float(lambda_decay) + speed / max(float(length_decay), 1e-6)
    dynamic = (
        float(tau) * (risk_rate - previous_rate) / dt
        + risk_rate
        + div_advection
        - div_diffusion
        - source
        + decay * risk
    )
    return torch.mean(kinematic.square()), torch.mean(dynamic.square())


def load_recurrent_pinn_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[RecurrentContextPINN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") not in {
        "recurrent_context_pinn",
        "prospective_context_pinn",
    }:
        raise ValueError(f"Not a context PINN checkpoint: {checkpoint_path}")
    model = RecurrentContextPINN(
        input_channels=int(checkpoint["input_channels"]),
        width=int(checkpoint["width"]),
        dilations=tuple(checkpoint["dilations"]),
        channel_names=checkpoint.get("input_channel_names"),
        output_mode=checkpoint.get("output_mode", "residual"),
        max_normalized_risk=float(checkpoint.get("max_normalized_risk", 1.25)),
        max_normalized_rate=float(checkpoint.get("max_normalized_rate", 2.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
