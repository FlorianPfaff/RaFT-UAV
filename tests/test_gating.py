import numpy as np

from raft_uav.baselines.kalman import (
    TrackingMeasurement,
    gate_threshold_from_probability,
    huber_covariance_scale,
    run_async_cv_baseline,
    student_t_covariance_scale,
)


def _measurement(time_s: float, x: float, y: float, source: str = "rf") -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([x, y]),
        covariance=np.diag([1.0, 1.0]),
        source=source,
    )


def test_gate_threshold_from_probability_matches_chi_square_ordering():
    threshold_2d_95 = gate_threshold_from_probability(0.95, 2)
    threshold_2d_99 = gate_threshold_from_probability(0.99, 2)
    threshold_3d_99 = gate_threshold_from_probability(0.99, 3)

    assert 5.0 < threshold_2d_95 < threshold_2d_99
    assert threshold_3d_99 > threshold_2d_99


def test_heavy_tailed_covariance_scale_helpers_are_monotone():
    assert student_t_covariance_scale(2.0, 2, degrees_of_freedom=4.0) == 1.0
    assert student_t_covariance_scale(100.0, 2, degrees_of_freedom=4.0) > 1.0
    low_dof_scale = student_t_covariance_scale(100.0, 2, degrees_of_freedom=3.0)
    high_dof_scale = student_t_covariance_scale(100.0, 2, degrees_of_freedom=10.0)
    assert low_dof_scale > high_dof_scale

    assert huber_covariance_scale(4.0, threshold=2.0) == 1.0
    assert huber_covariance_scale(16.0, threshold=2.0) == 2.0
    assert huber_covariance_scale(16.0, threshold=1.0) > huber_covariance_scale(
        16.0, threshold=4.0
    )


def test_large_outlier_is_rejected_when_source_gate_is_tight():
    records = run_async_cv_baseline(
        [
            _measurement(0.0, 0.0, 0.0),
            _measurement(1.0, 1.0, 0.0),
            _measurement(2.0, 10_000.0, 10_000.0),
        ],
        gate_thresholds_by_source={"rf": 5.0},
    )

    assert len(records) == 3
    assert records[-1]["accepted"] is False
    assert records[-1]["update_action"] == "rejected"
    assert records[-1]["nis"] > 5.0
    assert np.linalg.norm(records[-1]["state"][:2]) < 100.0


def test_missing_source_gate_keeps_updates_accepted():
    records = run_async_cv_baseline(
        [
            _measurement(0.0, 0.0, 0.0),
            _measurement(1.0, 10_000.0, 10_000.0),
        ],
        gate_thresholds_by_source={"radar": 1.0},
    )

    assert records[-1]["accepted"] is True
    assert records[-1]["source"] == "rf"


def test_nis_inflation_keeps_large_outlier_but_downweights_it():
    measurements = [
        _measurement(0.0, 0.0, 0.0),
        _measurement(1.0, 1.0, 0.0),
        _measurement(2.0, 10_000.0, 10_000.0),
    ]
    rejected = run_async_cv_baseline(measurements, gate_thresholds_by_source={"rf": 5.0})
    inflated = run_async_cv_baseline(
        measurements,
        gate_thresholds_by_source={"rf": 5.0},
        robust_update_by_source={"rf": "nis-inflate"},
    )

    assert inflated[-1]["accepted"] is True
    assert inflated[-1]["update_action"] == "inflated"
    assert inflated[-1]["covariance_scale"] > 1.0
    assert np.linalg.norm(inflated[-1]["state"][:2]) > np.linalg.norm(
        rejected[-1]["state"][:2]
    )
    assert np.linalg.norm(inflated[-1]["state"][:2]) < 10_000.0


def test_nis_inflation_alpha_controls_outlier_pull():
    measurements = [
        _measurement(0.0, 0.0, 0.0),
        _measurement(1.0, 1.0, 0.0),
        _measurement(2.0, 10_000.0, 10_000.0),
    ]
    mild = run_async_cv_baseline(
        measurements,
        gate_thresholds_by_source={"rf": 5.0},
        robust_update_by_source={"rf": "nis-inflate"},
        inflation_alpha_by_source={"rf": 0.5},
    )
    strong = run_async_cv_baseline(
        measurements,
        gate_thresholds_by_source={"rf": 5.0},
        robust_update_by_source={"rf": "nis-inflate"},
        inflation_alpha_by_source={"rf": 2.0},
    )

    assert mild[-1]["update_action"] == "inflated"
    assert strong[-1]["update_action"] == "inflated"
    assert strong[-1]["covariance_scale"] > mild[-1]["covariance_scale"]
    assert np.linalg.norm(strong[-1]["state"][:2]) < np.linalg.norm(mild[-1]["state"][:2])


def test_student_t_keeps_large_outlier_but_downweights_it():
    measurements = [
        _measurement(0.0, 0.0, 0.0),
        _measurement(1.0, 1.0, 0.0),
        _measurement(2.0, 10_000.0, 10_000.0),
    ]
    plain = run_async_cv_baseline(measurements)
    robust = run_async_cv_baseline(
        measurements,
        robust_update_by_source={"rf": "student-t"},
        student_t_dof_by_source={"rf": 3.0},
    )

    assert robust[-1]["accepted"] is True
    assert robust[-1]["update_action"] == "student_t"
    assert robust[-1]["covariance_scale"] > 1.0
    assert np.linalg.norm(robust[-1]["state"][:2]) < np.linalg.norm(plain[-1]["state"][:2])


def test_huber_keeps_large_outlier_but_downweights_it():
    measurements = [
        _measurement(0.0, 0.0, 0.0),
        _measurement(1.0, 1.0, 0.0),
        _measurement(2.0, 10_000.0, 10_000.0),
    ]
    plain = run_async_cv_baseline(measurements)
    robust = run_async_cv_baseline(
        measurements,
        robust_update_by_source={"rf": "huber"},
        huber_threshold_by_source={"rf": 2.0},
    )

    assert robust[-1]["accepted"] is True
    assert robust[-1]["update_action"] == "huber"
    assert robust[-1]["covariance_scale"] > 1.0
    assert np.linalg.norm(robust[-1]["state"][:2]) < np.linalg.norm(plain[-1]["state"][:2])


def test_huber_threshold_controls_outlier_pull():
    measurements = [
        _measurement(0.0, 0.0, 0.0),
        _measurement(1.0, 1.0, 0.0),
        _measurement(2.0, 10_000.0, 10_000.0),
    ]
    tight = run_async_cv_baseline(
        measurements,
        robust_update_by_source={"rf": "huber"},
        huber_threshold_by_source={"rf": 1.0},
    )
    loose = run_async_cv_baseline(
        measurements,
        robust_update_by_source={"rf": "huber"},
        huber_threshold_by_source={"rf": 10.0},
    )

    assert tight[-1]["covariance_scale"] > loose[-1]["covariance_scale"]
    assert np.linalg.norm(tight[-1]["state"][:2]) < np.linalg.norm(loose[-1]["state"][:2])
