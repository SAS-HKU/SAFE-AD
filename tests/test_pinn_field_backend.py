from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pinn_risk_field import RiskFieldNet
import rl.risk.field_backend as field_backend_module
from rl.risk.field_backend import (
    NumericalFieldBackend,
    PINNFieldBackend,
    ProspectiveNumericalFieldBackend,
)


def _checkpoint(path):
    model = RiskFieldNet(hidden=8, depth=2, use_context=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden": 8,
            "depth": 2,
            "use_rff": False,
            "use_context": True,
            "norm_ranges": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "t": (0.0, 4.0),
                "Q": (0.0, 1.0),
                "vx": (-2.0, 2.0),
                "vy": (-2.0, 2.0),
                "D": (0.0, 1.0),
                "R": (0.0, 3.0),
                "N_agents": (0.0, 5.0),
                "dist_nearest": (0.0, 80.0),
            },
            "metadata": {"deployment_horizon_s": 4.0},
        },
        path,
    )


def test_sparse_policy_query_and_lazy_full_grid(tmp_path, monkeypatch):
    checkpoint = tmp_path / "context.pt"
    _checkpoint(checkpoint)
    x = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, 3, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    sim_cfg = SimpleNamespace(X=X, Y=Y, x=x, y=y, dx=0.5, dy=1.0)

    zeros = np.zeros_like(X, dtype=np.float32)
    ones = np.ones_like(X, dtype=np.float32)
    monkeypatch.setattr(
        field_backend_module,
        "compute_total_Q",
        lambda *_args, **_kwargs: (ones * 0.2, zeros, zeros, zeros),
    )
    monkeypatch.setattr(
        field_backend_module,
        "compute_velocity_field",
        lambda *_args, **_kwargs: (zeros, zeros),
    )
    monkeypatch.setattr(
        field_backend_module,
        "compute_diffusion_field",
        lambda *_args, **_kwargs: ones * 0.3,
    )

    backend = PINNFieldBackend(
        checkpoint_path=str(checkpoint),
        sim_cfg=sim_cfg,
        time_mode="error",
    )
    sparse_field = backend.step(
        [],
        {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        dt=0.1,
        full_field=False,
    )
    assert np.count_nonzero(sparse_field) == 0
    features = backend.query_risk_features(
        ego_x=0.0,
        ego_y=0.0,
        lane_centers=[-0.5, 0.0, 0.5],
        current_lane=1,
    )
    assert set(features) == {
        "r_ego",
        "r_5m",
        "r_10m",
        "r_20m",
        "grad_x",
        "grad_y",
        "r_left",
        "r_right",
    }
    assert np.all(np.isfinite(list(features.values())))
    assert backend.last_timing.inference_ms > 0.0

    full_field = backend.ensure_full_field()
    assert full_field.shape == X.shape
    assert np.all(np.isfinite(full_field))


def test_numerical_backend_uses_explicit_grid_configuration():
    x = np.linspace(-2.0, 2.0, 7, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    sim_cfg = SimpleNamespace(
        X=X,
        Y=Y,
        x=x,
        y=y,
        dx=float(x[1] - x[0]),
        dy=float(y[1] - y[0]),
        tau=0.0,
        lambda_decay=0.1,
        L_decay=25.0,
        sponge_length=0.0,
        lambda_sponge=0.0,
        post_smooth_sigma=0.0,
    )
    backend = NumericalFieldBackend(sim_cfg=sim_cfg)
    zeros = np.zeros_like(X, dtype=np.float32)
    ones = np.ones_like(X, dtype=np.float32)
    field = backend.step(
        [],
        {"id": 0, "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        dt=0.1,
        substeps=1,
        source_fn=lambda *_args: (ones, zeros, zeros, zeros.astype(bool)),
        velocity_fn=lambda *_args: (zeros, zeros),
        diffusion_fn=lambda *_args: ones * 0.1,
    )
    assert field.shape == X.shape
    assert np.all(np.isfinite(field))


def test_prospective_backend_queries_an_ego_local_field():
    x = np.linspace(-50.0, 120.0, 86, dtype=np.float32)
    y = np.linspace(-25.0, 25.0, 26, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    sim_cfg = SimpleNamespace(
        X=X,
        Y=Y,
        x=x,
        y=y,
        dx=float(x[1] - x[0]),
        dy=float(y[1] - y[0]),
    )
    zeros = np.zeros_like(X, dtype=np.float32)
    source = np.exp(-0.5 * ((X - 15.0) / 4.0) ** 2 - 0.5 * (Y / 2.0) ** 2)
    backend = ProspectiveNumericalFieldBackend(sim_cfg=sim_cfg)
    backend.step(
        [],
        {"id": 0, "x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "heading": 0.0},
        dt=0.1,
        source_fn=lambda *_args: (source, zeros, zeros, zeros.astype(bool)),
        velocity_fn=lambda *_args: (np.full_like(X, 10.0), zeros),
        diffusion_fn=lambda *_args: np.full_like(X, 0.5),
    )
    queried = backend.query_cartesian_points(
        np.asarray([0.0, 15.0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        compute_gradient=True,
    )
    assert np.all(np.isfinite(queried["R"]))
    assert np.all(np.isfinite(queried["grad_x"]))
    assert queried["R"][1] > queried["R"][0]
    assert backend.last_timing.inference_ms > 0.0
