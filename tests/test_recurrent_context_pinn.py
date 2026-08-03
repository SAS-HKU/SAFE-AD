import numpy as np
import torch

from rl.risk.recurrent_pinn_operator import (
    INPUT_CHANNELS,
    OperatorScales,
    RecurrentContextPINN,
    build_operator_input,
    checkpoint_domain_scales,
    finite_difference_physics_losses,
    select_checkpoint_inputs,
    warp_ego_local_field,
)
from rl.train_recurrent_context_pinn import _sobolev_gradient_loss


def _snapshot(shape):
    zeros = np.zeros(shape, dtype=np.float32)
    return {
        "R_prev": zeros,
        "R_t_prev": zeros,
        "Q": zeros,
        "vx": zeros,
        "vy": zeros,
        "D": np.ones(shape, dtype=np.float32),
        "occ_mask": zeros,
        "road_mask": np.ones(shape, dtype=np.float32),
        "dist_nearest": np.full(shape, 20.0, dtype=np.float32),
        "ego_yaw_rate": 0.0,
        "N_agents": 3.0,
        "dt": 0.2,
    }


def test_operator_input_and_model_shapes():
    x = np.linspace(-10.0, 20.0, 31, dtype=np.float32)
    y = np.linspace(-5.0, 5.0, 11, dtype=np.float32)
    scales = OperatorScales(1.0, 1.0, 1.0, 30.0, 1.0, 1.0, 10.0, 1.0)
    inputs = build_operator_input(
        _snapshot((len(y), len(x))), x_grid=x, y_grid=y, scales=scales
    )
    assert inputs.shape == (len(INPUT_CHANNELS), len(y), len(x))
    model = RecurrentContextPINN(width=8, dilations=(1, 2))
    risk, rate = model(torch.from_numpy(inputs[None]))
    assert risk.shape == (1, 1, len(y), len(x))
    assert rate.shape == risk.shape
    assert torch.all(risk >= 0.0)


def test_domain_conditioned_checkpoint_inputs_use_calibration_scales():
    x = np.linspace(-10.0, 20.0, 31, dtype=np.float32)
    y = np.linspace(-5.0, 5.0, 11, dtype=np.float32)
    naturalistic = OperatorScales(10.0, 2.0, 20.0, 30.0, 4.0, 1.0, 10.0, 1.0)
    highwayenv = OperatorScales(1.0, 1.0, 2.0, 20.0, 3.0, 1.0, 8.0, 1.0)
    checkpoint = {
        "scales": highwayenv.to_dict(),
        "domain_scales": {
            "naturalistic": naturalistic.to_dict(),
            "highwayenv": highwayenv.to_dict(),
        },
        "input_channel_names": ["Q", "road_mask", "domain_highwayenv"],
        "input_clip": 4.0,
    }
    inputs = build_operator_input(
        _snapshot((len(y), len(x))),
        x_grid=x,
        y_grid=y,
        scales=checkpoint_domain_scales(checkpoint, "highwayenv"),
    )
    selected = select_checkpoint_inputs(inputs, checkpoint, domain="highwayenv")
    assert selected.shape == (3, len(y), len(x))
    assert np.all(selected[2] == 1.0)
    assert checkpoint_domain_scales(checkpoint, "naturalistic").risk == 10.0


def test_absolute_operator_is_bounded_under_recurrence():
    model = RecurrentContextPINN(
        width=8,
        dilations=(1, 2),
        output_mode="absolute_bounded",
        max_normalized_risk=1.25,
        max_normalized_rate=2.0,
    )
    inputs = torch.randn(2, len(INPUT_CHANNELS), 7, 9) * 100.0
    inputs[:, INPUT_CHANNELS.index("road_mask")] = 1.0
    risk, rate = model(inputs)
    assert torch.all((risk >= 0.0) & (risk <= 1.25))
    assert torch.all(torch.abs(rate) <= 2.0)


def test_ego_local_warp_is_identity_for_equal_poses():
    x = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    field = np.arange(len(x) * len(y), dtype=np.float32).reshape(len(y), len(x))
    warped = warp_ego_local_field(
        field,
        x_grid=x,
        y_grid=y,
        previous_pose=(4.0, 2.0, 0.3),
        current_pose=(4.0, 2.0, 0.3),
    )
    np.testing.assert_allclose(warped, field, atol=2e-5)


def test_zero_state_satisfies_zero_source_physics():
    shape = (2, 1, 7, 9)
    zeros = torch.zeros(shape)
    dt = torch.full((2, 1, 1, 1), 0.2)
    kinematic, dynamic = finite_difference_physics_losses(
        risk=zeros,
        risk_rate=zeros,
        previous_risk=zeros,
        previous_rate=zeros,
        source=zeros,
        vx=zeros,
        vy=zeros,
        diffusion=zeros,
        dt=dt,
        dx=1.0,
        dy=1.0,
        tau=0.2,
        lambda_decay=0.15,
        length_decay=20.0,
    )
    assert float(kinematic) == 0.0
    assert float(dynamic) == 0.0


def test_sobolev_gradient_loss_distinguishes_direction():
    x = torch.linspace(-1.0, 1.0, 9).view(1, 1, 1, 9).expand(1, 1, 7, 9)
    same, _, same_direction = _sobolev_gradient_loss(
        x,
        x,
        dx=0.25,
        dy=0.25,
        hotspot_boost=4.0,
        active_quantile=0.75,
        direction_weight=0.25,
    )
    opposite, _, opposite_direction = _sobolev_gradient_loss(
        -x,
        x,
        dx=0.25,
        dy=0.25,
        hotspot_boost=4.0,
        active_quantile=0.75,
        direction_weight=0.25,
    )
    assert float(same) < 1e-6
    assert float(same_direction) < 1e-6
    assert float(opposite) > float(same)
    assert float(opposite_direction) > 1.5
