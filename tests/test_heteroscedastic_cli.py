from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.heteroscedastic_cli import (
    _extract_uncertainty_model_arg,
    nis_scored_candidates_with_row_covariance,
    promote_covariance_columns_for_association,
    radar_measurements_to_enu_with_row_covariance,
    rf_measurements_to_enu_with_row_covariance,
)


class _TrackerView:
    state = np.zeros(6)
    covariance_matrix = np.zeros((6, 6))


def test_extract_uncertainty_model_arg_removes_wrapper_only_argument() -> None:
    path, delegated = _extract_uncertainty_model_arg(
        ["run-baseline", "DATA", "--flight", "F", "--uncertainty-model", "model.json"]
    )

    assert str(path) == "model.json"
    assert delegated == ["run-baseline", "DATA", "--flight", "F"]


def test_rf_measurements_prefer_row_covariance_columns() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "std_m": [75.0],
            "cov_ee": [16.0],
            "cov_nn": [25.0],
            "cov_en": [3.0],
        }
    )

    [measurement] = rf_measurements_to_enu_with_row_covariance(frame)

    np.testing.assert_allclose(measurement.vector, [10.0, 20.0])
    np.testing.assert_allclose(measurement.covariance, [[16.0, 3.0], [3.0, 25.0]])


def test_radar_measurements_prefer_row_covariance_columns() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [2.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "cov_ee": [9.0],
            "cov_nn": [16.0],
            "cov_uu": [25.0],
            "cov_en": [1.0],
            "cov_eu": [2.0],
            "cov_nu": [3.0],
        }
    )

    [measurement] = radar_measurements_to_enu_with_row_covariance(frame)

    np.testing.assert_allclose(measurement.vector, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        measurement.covariance,
        [[9.0, 1.0, 2.0], [1.0, 16.0, 3.0], [2.0, 3.0, 25.0]],
    )


def test_promote_covariance_columns_for_association() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "cov_ee": [11.0],
            "cov_nn": [12.0],
            "cov_uu": [13.0],
            "cov_en": [0.1],
            "cov_eu": [0.2],
            "cov_nu": [0.3],
            "uncertainty_model": ["heteroscedastic-loglinear"],
        }
    )

    promoted = promote_covariance_columns_for_association(frame)

    assert float(promoted.loc[0, "association_cov_ee"]) == 11.0
    assert float(promoted.loc[0, "association_cov_nn"]) == 12.0
    assert float(promoted.loc[0, "association_cov_uu"]) == 13.0
    assert promoted.loc[0, "association_covariance_mode"] == "heteroscedastic-loglinear"


def test_nis_scoring_uses_candidate_specific_covariance() -> None:
    candidates = pd.DataFrame(
        {
            "east_m": [10.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cov_ee": [1.0, 100.0],
            "cov_nn": [1.0, 100.0],
            "cov_uu": [1.0, 100.0],
            "cov_en": [0.0, 0.0],
            "cov_eu": [0.0, 0.0],
            "cov_nu": [0.0, 0.0],
        }
    )

    scored = nis_scored_candidates_with_row_covariance(candidates, _TrackerView(), np.eye(3))

    assert float(scored.loc[0, "association_nis"]) == 100.0
    assert float(scored.loc[1, "association_nis"]) == 1.0
