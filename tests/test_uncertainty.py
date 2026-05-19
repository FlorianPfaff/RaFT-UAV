import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker
from raft_uav.baselines.radar_association import (
    _nis_scored_candidates,
    _radar_row_to_measurement,
)
from raft_uav.heteroscedastic_measurements import (
    radar_measurements_to_enu_with_uncertainty,
    rf_measurements_to_enu_with_uncertainty,
)
from raft_uav.uncertainty import covariance_from_row, fit_heteroscedastic_uncertainty_model


def test_fit_model_adds_positive_rf_and_radar_covariance_columns():
    truth = pd.DataFrame(
        {
            "time_s": np.arange(6, dtype=float),
            "east_m": np.arange(6, dtype=float) * 10.0,
            "north_m": np.zeros(6),
            "up_m": np.ones(6) * 20.0,
        }
    )
    rf = pd.DataFrame(
        {
            "time_s": truth["time_s"],
            "east_m": truth["east_m"] + np.array([2.0, 4.0, 10.0, 20.0, 30.0, 40.0]),
            "north_m": truth["north_m"] + np.array([1.0, 3.0, 8.0, 18.0, 25.0, 36.0]),
            "CEP": [5.0, 6.0, 15.0, 25.0, 35.0, 45.0],
            "RHO": [0.1, 0.1, 0.4, 0.7, 0.8, 1.0],
        }
    )
    radar = pd.DataFrame(
        {
            "time_s": truth["time_s"],
            "east_m": truth["east_m"] + np.array([1.0, 2.0, 4.0, 8.0, 16.0, 30.0]),
            "north_m": truth["north_m"] + np.array([1.0, 1.5, 3.0, 7.0, 15.0, 25.0]),
            "up_m": truth["up_m"] + np.array([2.0, 2.0, 5.0, 10.0, 20.0, 35.0]),
            "range_m": [50.0, 100.0, 150.0, 300.0, 500.0, 700.0],
            "cat_prob_uav": [0.9, 0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )

    model = fit_heteroscedastic_uncertainty_model(rf=rf, radar=radar, truth=truth)
    rf_out = model.apply_rf(rf)
    radar_out = model.apply_radar(radar)

    assert (rf_out[["cov_ee", "cov_nn"]] > 0.0).all().all()
    assert (radar_out[["cov_ee", "cov_nn", "cov_uu"]] > 0.0).all().all()
    assert "heteroscedastic" in rf_out["uncertainty_model"].iloc[0]


def test_covariance_from_row_prefers_association_covariance_then_model_covariance():
    fallback = np.diag([1.0, 2.0, 3.0])
    row = pd.Series(
        {
            "cov_ee": 4.0,
            "cov_nn": 5.0,
            "cov_uu": 6.0,
            "association_cov_ee": 7.0,
            "association_cov_nn": 8.0,
            "association_cov_uu": 9.0,
        }
    )

    cov = covariance_from_row(row, 3, fallback)

    assert np.allclose(np.diag(cov), [7.0, 8.0, 9.0])


def test_covariance_from_row_falls_back_for_missing_or_invalid_values():
    fallback = np.diag([10.0, 20.0])
    row = pd.Series({"cov_ee": -1.0, "cov_nn": np.nan})

    cov = covariance_from_row(row, 2, fallback)

    assert np.allclose(cov, fallback)


def test_heteroscedastic_measurement_converters_consume_covariance_columns():
    rf = pd.DataFrame(
        {
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "std_m": [75.0],
            "cov_ee": [4.0],
            "cov_nn": [9.0],
            "cov_en": [1.5],
        }
    )
    radar = pd.DataFrame(
        {
            "time_s": [2.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "cov_ee": [16.0],
            "cov_nn": [25.0],
            "cov_uu": [36.0],
            "cov_en": [2.0],
            "cov_eu": [3.0],
            "cov_nu": [4.0],
        }
    )

    [rf_measurement] = rf_measurements_to_enu_with_uncertainty(rf)
    [radar_measurement] = radar_measurements_to_enu_with_uncertainty(radar)

    assert np.allclose(rf_measurement.covariance, [[4.0, 1.5], [1.5, 9.0]])
    assert np.allclose(
        radar_measurement.covariance,
        [[16.0, 2.0, 3.0], [2.0, 25.0, 4.0], [3.0, 4.0, 36.0]],
    )


def test_radar_association_uses_rowwise_covariance_columns():
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=np.array([0.0, 0.0, 0.0]),
        initial_time_s=0.0,
        initial_position_std_m=1.0,
        initial_velocity_std_mps=1.0,
    )
    candidates = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "east_m": [100.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cov_ee": [10000.0, 1.0],
            "cov_nn": [1.0, 1.0],
            "cov_uu": [1.0, 1.0],
        }
    )

    scored = _nis_scored_candidates(candidates, tracker, np.eye(3))

    assert scored["association_nis"].iloc[0] < scored["association_nis"].iloc[1]


def test_radar_converter_keeps_velocity_block_when_velocity_is_available():
    radar = pd.DataFrame(
        {
            "time_s": [2.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "velocity_east_mps": [1.0],
            "velocity_north_mps": [2.0],
            "velocity_down_mps": [-3.0],
            "cov_ee": [16.0],
            "cov_nn": [25.0],
            "cov_uu": [36.0],
        }
    )

    [measurement] = radar_measurements_to_enu_with_uncertainty(
        radar,
        default_velocity_std_mps=7.0,
    )

    assert measurement.vector.shape == (6,)
    assert np.allclose(measurement.vector[3:], [1.0, 2.0, 3.0])
    assert np.allclose(np.diag(measurement.covariance)[:3], [16.0, 25.0, 36.0])
    assert np.allclose(np.diag(measurement.covariance)[3:], [49.0, 49.0, 49.0])


def test_radar_row_to_measurement_prefers_model_covariance():
    row = pd.Series(
        {
            "time_s": 1.0,
            "east_m": 10.0,
            "north_m": 20.0,
            "up_m": 30.0,
            "cov_ee": 16.0,
            "cov_nn": 25.0,
            "cov_uu": 36.0,
        }
    )

    measurement = _radar_row_to_measurement(row, np.diag([1.0, 2.0, 3.0]))

    assert np.allclose(measurement.vector, [10.0, 20.0, 30.0])
    assert np.allclose(np.diag(measurement.covariance), [16.0, 25.0, 36.0])
