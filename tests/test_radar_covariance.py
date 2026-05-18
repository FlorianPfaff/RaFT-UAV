import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker
from raft_uav.baselines.radar_covariance import (
    BIAS_RESIDUAL_INCLUDED_COLUMN,
    RadarCovarianceConfig,
    append_radar_covariance_columns,
    row_radar_covariance,
)
from raft_uav.baselines.radar_association import _nis_scored_candidates
from raft_uav.calibration.bias import BIAS_RESIDUAL_STD_COLUMN_PREFIX


def test_range_angle_radar_covariance_grows_with_range():
    radar = pd.DataFrame(
        [
            {"east_m": 100.0, "north_m": 0.0, "up_m": 0.0, "range_m": 100.0},
            {"east_m": 1000.0, "north_m": 0.0, "up_m": 0.0, "range_m": 1000.0},
        ]
    )
    annotated = append_radar_covariance_columns(
        radar,
        RadarCovarianceConfig(
            mode="range-angle",
            range_std_m=1.0,
            azimuth_std_deg=2.0,
            elevation_std_deg=2.0,
            min_std_m=1.0,
        ),
    )

    near_covariance = row_radar_covariance(annotated.iloc[0])
    far_covariance = row_radar_covariance(annotated.iloc[1])

    assert near_covariance is not None
    assert far_covariance is not None
    assert far_covariance[1, 1] > near_covariance[1, 1] * 50.0
    assert annotated["association_covariance_mode"].tolist() == ["range-angle", "range-angle"]


def test_nis_scoring_uses_candidate_specific_covariance_columns():
    tracker = AsyncConstantVelocityKalmanTracker(initial_position=np.zeros(3), initial_time_s=0.0)
    tracker.covariance = np.zeros((6, 6))
    candidates = pd.DataFrame(
        [
            {
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_cov_ee": 1.0,
                "association_cov_nn": 1.0,
                "association_cov_uu": 1.0,
                "association_cov_en": 0.0,
                "association_cov_eu": 0.0,
                "association_cov_nu": 0.0,
            },
            {
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_cov_ee": 100.0,
                "association_cov_nn": 1.0,
                "association_cov_uu": 1.0,
                "association_cov_en": 0.0,
                "association_cov_eu": 0.0,
                "association_cov_nu": 0.0,
            },
        ]
    )

    scored = _nis_scored_candidates(candidates, tracker, np.eye(3))

    assert scored.loc[scored["track_id"] == 1, "association_nis"].iloc[0] == 100.0
    assert scored.loc[scored["track_id"] == 2, "association_nis"].iloc[0] == 1.0


def test_bias_residual_std_inflates_fallback_radar_covariance():
    fallback = np.diag([25.0**2, 25.0**2, 35.0**2])
    row = pd.Series(
        {
            f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}east_m": 3.0,
            f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}north_m": 4.0,
            f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}up_m": 5.0,
        }
    )

    covariance = row_radar_covariance(row, fallback)

    assert covariance is not None
    np.testing.assert_allclose(np.diag(covariance), np.diag(fallback) + np.array([9.0, 16.0, 25.0]))


def test_bias_residual_std_is_added_once_to_range_angle_covariance():
    radar = pd.DataFrame(
        [
            {
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0,
                f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}east_m": 3.0,
                f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}north_m": 4.0,
                f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}up_m": 5.0,
            }
        ]
    )
    baseline = append_radar_covariance_columns(
        radar.drop(columns=[column for column in radar.columns if column.startswith(BIAS_RESIDUAL_STD_COLUMN_PREFIX)]),
        RadarCovarianceConfig(mode="range-angle", min_std_m=1.0),
    )
    annotated = append_radar_covariance_columns(
        radar,
        RadarCovarianceConfig(mode="range-angle", min_std_m=1.0),
    )

    baseline_covariance = row_radar_covariance(baseline.iloc[0])
    annotated_covariance = row_radar_covariance(annotated.iloc[0])

    assert baseline_covariance is not None
    assert annotated_covariance is not None
    assert bool(annotated.iloc[0][BIAS_RESIDUAL_INCLUDED_COLUMN])
    np.testing.assert_allclose(
        np.diag(annotated_covariance),
        np.diag(baseline_covariance) + np.array([9.0, 16.0, 25.0]),
        rtol=1e-6,
        atol=1e-6,
    )
