import numpy as np
import pandas as pd

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
