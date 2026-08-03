import numpy as np

from rl.env.highwayenv_social_env import (
    FIELD_OBSERVATION_KEYS,
    make_social_highwayenv_env,
    resolve_traffic_config,
)


def test_field_observation_appends_eight_finite_descriptors():
    env = make_social_highwayenv_env(
        env_id="merge-v0",
        traffic=resolve_traffic_config(preset="sparse"),
        use_drift=True,
        append_risk_obs=True,
        drift_warmup_s=0.0,
        field_backend="numerical",
    )
    observation, info = env.reset(seed=3)
    assert observation.shape == (33,)
    assert len(FIELD_OBSERVATION_KEYS) == 8
    assert len(info["field_observation"]) == 8
    assert np.isfinite(observation).all()
    observation, _reward, _terminated, _truncated, info = env.step(1)
    assert observation.shape == (33,)
    assert np.isfinite(observation[-8:]).all()
    assert np.allclose(observation[-8:], info["field_observation"])
    env.close()


def test_stock_observation_shape_is_unchanged_by_default():
    env = make_social_highwayenv_env(
        env_id="merge-v0",
        traffic=resolve_traffic_config(preset="sparse"),
        use_drift=True,
        drift_warmup_s=0.0,
        field_backend="numerical",
    )
    observation, _info = env.reset(seed=3)
    assert observation.shape == (5, 5)
    env.close()
