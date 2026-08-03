import numpy as np

from rl.risk.prospective_solver import ProspectiveRiskSolver, ProspectiveSolverConfig


def _solver():
    return ProspectiveRiskSolver(
        x_grid=np.linspace(-10.0, 20.0, 61),
        y_grid=np.linspace(-5.0, 5.0, 21),
        config=ProspectiveSolverConfig(
            horizon_s=2.0,
            integration_step_s=0.25,
            decay_rate=0.1,
        ),
    )


def test_zero_source_stays_zero():
    solver = _solver()
    zeros = np.zeros(solver.shape, dtype=np.float32)
    result = solver.solve(zeros, zeros, zeros, np.ones_like(zeros))
    assert result.shape == solver.shape
    assert np.all(result == 0.0)


def test_positive_flow_moves_envelope_forward():
    solver = _solver()
    source = np.zeros(solver.shape, dtype=np.float32)
    source[len(solver.y_grid) // 2, np.argmin(np.abs(solver.x_grid))] = 1.0
    vx = np.full(solver.shape, 5.0, dtype=np.float32)
    zeros = np.zeros(solver.shape, dtype=np.float32)
    field = solver.solve(source, vx, zeros, zeros)
    current = field[len(solver.y_grid) // 2, np.argmin(np.abs(solver.x_grid))]
    forward = field[len(solver.y_grid) // 2, np.argmin(np.abs(solver.x_grid - 5.0))]
    assert current > 0.0
    assert forward > 0.0
    assert np.sum(field[:, solver.x_grid > 0.0]) > np.sum(field[:, solver.x_grid < 0.0])


def test_road_mask_is_enforced():
    solver = _solver()
    source = np.ones(solver.shape, dtype=np.float32)
    zeros = np.zeros(solver.shape, dtype=np.float32)
    mask = np.ones(solver.shape, dtype=np.float32)
    mask[:, : solver.shape[1] // 2] = 0.0
    field = solver.solve(source, zeros, zeros, zeros, road_mask=mask)
    assert np.all(field[:, : solver.shape[1] // 2] == 0.0)
    assert np.all(field[:, solver.shape[1] // 2 :] > 0.0)
