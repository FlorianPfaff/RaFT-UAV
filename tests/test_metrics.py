import numpy as np

from raft_uav.evaluation.metrics import position_errors_m, summarize_errors


def test_metrics_return_finite_summaries_on_synthetic_trajectories():
    truth_times = np.array([0.0, 1.0, 2.0])
    truth_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    estimate_times = np.array([0.0, 1.0, 2.0])
    estimate_positions = truth_positions + np.array([0.0, 0.5, 0.25])

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=0.1,
        dimensions=3,
    )
    summary = summarize_errors(errors)

    assert summary["count"] == 3.0
    assert np.isfinite(summary["rmse_m"])
    assert np.isfinite(summary["mae_m"])
    assert np.isfinite(summary["p95_m"])


def test_position_errors_interpolate_estimates_to_truth_time_grid():
    truth_times = np.array([0.0, 1.0, 2.0, 3.0])
    truth_positions = np.column_stack(
        [truth_times, np.zeros_like(truth_times), np.zeros_like(truth_times)]
    )
    estimate_times = np.array([0.0, 3.0])
    estimate_positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=2.0,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.ones(4))


def test_position_errors_skip_truth_samples_without_local_estimate_support():
    truth_times = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    truth_positions = np.column_stack(
        [truth_times, np.zeros_like(truth_times), np.zeros_like(truth_times)]
    )
    estimate_times = np.array([0.0, 10.0])
    estimate_positions = truth_positions[[0, -1], :]

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=2.0,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.zeros(2))


def test_position_errors_are_order_and_duplicate_timestamp_stable():
    truth_times = np.array([0.0, 1.0])
    truth_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    estimate_times = np.array([1.0, 0.0, 0.0])
    estimate_positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [99.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.zeros(2))
