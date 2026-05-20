import numpy as np
import pandas as pd

from raft_uav.baselines import radar_covariance_runtime
from raft_uav.baselines.kalman import TrackingMeasurement, measurement_matrix, run_async_cv_baseline
from raft_uav.io import aerpaw
from raft_uav.io.aerpaw import radar_measurements_to_enu


def test_measurement_matrix_accepts_radar_position_velocity_observation():
    matrix = measurement_matrix(6)

    np.testing.assert_allclose(matrix, np.eye(6))


def test_radar_measurements_to_enu_defaults_to_position_only_with_velocity_columns():
    radar = pd.DataFrame(
        [
            {
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 20.0,
                "up_m": 30.0,
                "velocity_east_mps": 3.0,
                "velocity_north_mps": 4.0,
                "velocity_down_mps": -5.0,
            }
        ]
    )

    measurements = radar_measurements_to_enu(
        radar,
        default_xy_std_m=2.0,
        default_z_std_m=3.0,
        default_velocity_std_mps=4.0,
    )

    assert len(measurements) == 1
    measurement = measurements[0]
    np.testing.assert_allclose(measurement.vector, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(np.diag(measurement.covariance), [4.0, 4.0, 9.0])


def test_radar_measurements_to_enu_can_include_fortem_velocity_ned_as_enu():
    radar = pd.DataFrame(
        [
            {
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 20.0,
                "up_m": 30.0,
                "velocity_east_mps": 3.0,
                "velocity_north_mps": 4.0,
                "velocity_down_mps": -5.0,
            }
        ]
    )

    measurements = radar_measurements_to_enu(
        radar,
        default_xy_std_m=2.0,
        default_z_std_m=3.0,
        default_velocity_std_mps=4.0,
        include_velocity=True,
    )

    assert len(measurements) == 1
    measurement = measurements[0]
    np.testing.assert_allclose(measurement.vector, [10.0, 20.0, 30.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        np.diag(measurement.covariance),
        [4.0, 4.0, 9.0, 16.0, 16.0, 16.0],
    )


def test_radar_measurements_to_enu_falls_back_to_position_only_without_velocity():
    radar = pd.DataFrame(
        [
            {
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 20.0,
                "up_m": 30.0,
            }
        ]
    )

    measurements = radar_measurements_to_enu(radar)

    assert len(measurements) == 1
    assert measurements[0].vector.shape == (3,)
    assert measurements[0].covariance.shape == (3, 3)


def test_installed_radar_covariance_runtime_forwards_clock_offset(monkeypatch):
    radar_covariance_runtime.install()
    calls = {}
    projector = object()
    truth_origin_time = pd.Timestamp("2024-01-01")
    raw_radar = pd.DataFrame([{"global_time_raw_s": 0.0}])

    def normalize_spy(frame, projector_arg, truth_origin_time_arg, *, clock_offset_s):
        calls["frame"] = frame
        calls["projector"] = projector_arg
        calls["truth_origin_time"] = truth_origin_time_arg
        calls["clock_offset_s"] = clock_offset_s
        return pd.DataFrame(
            [
                {
                    "time_s": 1.0,
                    "east_m": 10.0,
                    "north_m": 20.0,
                    "up_m": 30.0,
                }
            ]
        )

    monkeypatch.setattr(aerpaw, "normalize_radar", normalize_spy)

    measurements = aerpaw.radar_measurements_to_enu(
        raw_radar,
        projector=projector,
        truth_origin_time=truth_origin_time,
        default_xy_std_m=2.0,
        default_z_std_m=3.0,
        clock_offset_s=12.5,
    )

    assert calls["frame"] is raw_radar
    assert calls["projector"] is projector
    assert calls["truth_origin_time"] is truth_origin_time
    assert calls["clock_offset_s"] == 12.5
    assert len(measurements) == 1
    np.testing.assert_allclose(measurements[0].vector, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(np.diag(measurements[0].covariance), [4.0, 4.0, 9.0])


def test_async_cv_baseline_uses_radar_velocity_measurements():
    covariance = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])
    measurements = [
        TrackingMeasurement(
            0.0,
            np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0]),
            covariance,
            "radar",
        ),
        TrackingMeasurement(
            1.0,
            np.array([5.0, 0.0, 0.0, 5.0, 0.0, 0.0]),
            covariance,
            "radar",
        ),
    ]

    records = run_async_cv_baseline(measurements)

    assert records[-1]["measurement_dim"] == 6
    assert float(np.asarray(records[-1]["state"])[3]) > 4.0
