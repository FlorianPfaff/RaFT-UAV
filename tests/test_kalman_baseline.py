import numpy as np

from raft_uav.baselines.kalman import (
    TrackingMeasurement,
    constant_velocity_matrix,
    run_async_cv_baseline,
    white_acceleration_process_noise,
)


def test_constant_velocity_matrix_places_dt_in_position_velocity_blocks():
    matrix = constant_velocity_matrix(2.5)

    assert matrix.shape == (6, 6)
    assert matrix[0, 3] == 2.5
    assert matrix[1, 4] == 2.5
    assert matrix[2, 5] == 2.5


def test_process_noise_is_symmetric():
    covariance = white_acceleration_process_noise(0.5, 4.0)

    assert covariance.shape == (6, 6)
    np.testing.assert_allclose(covariance, covariance.T)


def test_process_noise_matches_continuous_white_acceleration_model():
    covariance = white_acceleration_process_noise(2.0, 3.0)
    expected_block = np.array(
        [
            [24.0, 18.0],
            [18.0, 18.0],
        ]
    )

    for position_index, velocity_index in ((0, 3), (1, 4), (2, 5)):
        np.testing.assert_allclose(
            covariance[np.ix_([position_index, velocity_index], [position_index, velocity_index])],
            expected_block,
        )

    assert covariance[0, 1] == 0.0
    assert covariance[3, 4] == 0.0


def test_async_cv_baseline_returns_one_record_per_measurement():
    covariance = np.diag([10.0, 10.0, 10.0])
    measurements = [
        TrackingMeasurement(0.0, np.array([0.0, 0.0, 0.0]), covariance, "radar"),
        TrackingMeasurement(1.0, np.array([1.0, 0.0, 0.0]), covariance, "radar"),
    ]

    records = run_async_cv_baseline(measurements)

    assert len(records) == 2
    assert records[0]["update_action"] == "bootstrap"
    assert records[-1]["source"] == "radar"
    assert np.asarray(records[-1]["state"]).shape == (6,)


def test_async_cv_baseline_does_not_update_bootstrap_measurement():
    covariance = np.diag([10.0, 10.0, 10.0])
    measurement = TrackingMeasurement(0.0, np.array([0.0, 0.0, 0.0]), covariance, "radar")

    records = run_async_cv_baseline([measurement])

    assert len(records) == 1
    assert records[0]["update_action"] == "bootstrap"
    np.testing.assert_allclose(
        np.diag(np.asarray(records[0]["covariance"], dtype=float))[:3],
        np.full(3, 50.0**2),
    )
