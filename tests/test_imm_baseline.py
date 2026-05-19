import numpy as np

from raft_uav.baselines.imm import (
    fixed_turn_rate_matrix,
    run_async_imm_baseline,
    uniform_ctmc_transition_matrix,
)
from raft_uav.baselines.kalman import TrackingMeasurement, constant_velocity_matrix


def test_fixed_turn_rate_matrix_matches_cv_for_zero_turn_rate():
    np.testing.assert_allclose(
        fixed_turn_rate_matrix(1.25, 0.0),
        constant_velocity_matrix(1.25),
    )


def test_uniform_ctmc_transition_matrix_is_row_stochastic():
    matrix = uniform_ctmc_transition_matrix(
        n_modes=5,
        dt_s=2.0,
        mode_switch_time_constant_s=20.0,
    )

    assert matrix.shape == (5, 5)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(5))
    assert np.all(matrix >= 0.0)
    assert np.all(np.diag(matrix) > matrix[0, 1])


def test_async_imm_baseline_returns_mode_probabilities():
    covariance = np.diag([10.0, 10.0, 10.0])
    measurements = [
        TrackingMeasurement(0.0, np.array([0.0, 0.0, 0.0]), covariance, "radar"),
        TrackingMeasurement(1.0, np.array([1.0, 0.0, 0.0]), covariance, "radar"),
    ]

    records = run_async_imm_baseline(measurements)

    assert len(records) == 2
    assert records[0]["update_action"] == "bootstrap"
    assert np.asarray(records[-1]["state"]).shape == (6,)
    probabilities = np.asarray(records[-1]["mode_probabilities"], dtype=float)
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    assert len(records[-1]["mode_names"]) == probabilities.size


def test_async_imm_baseline_does_not_update_bootstrap_measurement():
    covariance = np.diag([10.0, 10.0, 10.0])
    measurement = TrackingMeasurement(0.0, np.array([0.0, 0.0, 0.0]), covariance, "radar")

    records = run_async_imm_baseline([measurement])

    assert len(records) == 1
    assert records[0]["update_action"] == "bootstrap"
    np.testing.assert_allclose(
        np.diag(np.asarray(records[0]["covariance"], dtype=float))[:3],
        np.full(3, 50.0**2),
    )
