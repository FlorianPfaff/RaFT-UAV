import numpy as np

from raft_uav.baselines.kalman import TrackingMeasurement, run_async_cv_baseline


def test_async_cv_baseline_does_not_reprocess_bootstrap_measurement() -> None:
    measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([10.0, 20.0, 30.0]),
            covariance=np.eye(3),
            source="rf",
        ),
        TrackingMeasurement(
            time_s=1.0,
            vector=np.array([11.0, 20.0, 30.0]),
            covariance=np.eye(3),
            source="rf",
        ),
    ]

    records = run_async_cv_baseline(measurements, acceleration_std_mps2=0.0)

    assert len(records) == 2
    assert records[0]["update_action"] == "initialized"
    assert records[0]["accepted"] is True
    np.testing.assert_allclose(records[0]["state"][:3], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(np.diag(records[0]["covariance"]), [2500.0, 2500.0, 2500.0, 225.0, 225.0, 225.0])


def test_bootstrap_detection_does_not_skip_distinct_same_time_measurement() -> None:
    measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([0.0, 0.0, 0.0]),
            covariance=np.eye(3),
            source="rf",
        ),
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([5.0, 0.0, 0.0]),
            covariance=np.eye(3),
            source="radar",
        ),
    ]

    records = run_async_cv_baseline(measurements, acceleration_std_mps2=0.0)

    assert records[0]["update_action"] == "initialized"
    assert records[1]["update_action"] != "initialized"
    assert records[1]["accepted"] is True
    assert records[1]["covariance"][0, 0] < records[0]["covariance"][0, 0]
