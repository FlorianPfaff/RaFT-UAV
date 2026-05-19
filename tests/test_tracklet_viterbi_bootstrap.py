import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import (
    run_async_cv_baseline_with_tracklet_viterbi_association,
)
from raft_uav.baselines.tracklet_viterbi_range_covariance import (
    run_async_cv_baseline_with_tracklet_viterbi_association as run_range_covariance_tracklet_viterbi,
)
from raft_uav.baselines.tracklet_viterbi_retention import (
    run_async_cv_baseline_with_tracklet_viterbi_association as run_retention_tracklet_viterbi,
)
from raft_uav.baselines.tracklet_viterbi_fixed_lag import (
    run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay,
)
from raft_uav.baselines.tracklet_viterbi_result import (
    run_async_cv_baseline_with_tracklet_viterbi_result,
)


def _rf_measurement(time_s: float) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([1.0, 2.0, 3.0]),
        covariance=np.eye(3),
        source="rf",
    )


def _radar_with_pre_rf_distractor() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "frame_index": 0,
                "track_id": 99,
                "east_m": 1000.0,
                "north_m": 1000.0,
                "up_m": 1000.0,
                "cat_prob_uav": 0.99,
            },
            {
                "time_s": 2.0,
                "frame_index": 1,
                "track_id": 1,
                "east_m": 1.2,
                "north_m": 2.0,
                "up_m": 3.0,
                "cat_prob_uav": 0.99,
            },
        ]
    )


@pytest.mark.parametrize(
    "runner",
    [
        run_async_cv_baseline_with_tracklet_viterbi_association,
        run_retention_tracklet_viterbi,
        run_range_covariance_tracklet_viterbi,
    ],
)
def test_tracklet_viterbi_skips_pre_rf_radar_bootstrap(runner):
    radar = _radar_with_pre_rf_distractor()

    records, selected = runner(
        rf_measurements=[_rf_measurement(1.0)],
        radar=radar,
        candidate_catprob_threshold=None,
    )

    assert records
    assert records[0]["source"] == "rf"
    assert records[0]["time_s"] == 1.0
    assert all(record["time_s"] >= 1.0 for record in records)
    assert not selected.empty
    assert selected["time_s"].min() >= 1.0


def test_tracklet_viterbi_result_skips_pre_rf_radar_bootstrap():
    result = run_async_cv_baseline_with_tracklet_viterbi_result(
        rf_measurements=[_rf_measurement(1.0)],
        radar=_radar_with_pre_rf_distractor(),
        candidate_catprob_threshold=None,
    )

    assert result.records
    assert result.records[0]["source"] == "rf"
    assert result.records[0]["time_s"] == 1.0
    assert all(record["time_s"] >= 1.0 for record in result.records)
    assert not result.accepted_radar.empty
    assert result.accepted_radar["time_s"].min() >= 1.0
    assert not result.viterbi_selected_radar.empty
    assert result.viterbi_selected_radar["time_s"].min() >= 1.0
    assert not result.radar_candidate_ledger.empty
    assert result.radar_candidate_ledger["time_s"].min() >= 1.0


def test_fixed_lag_tracklet_viterbi_skips_pre_rf_radar_bootstrap():
    records, accepted, replayed = (
        run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
            rf_measurements=[_rf_measurement(1.0)],
            radar=_radar_with_pre_rf_distractor(),
            lag_s=5.0,
            candidate_catprob_threshold=None,
        )
    )

    assert records
    assert records[0]["source"] == "rf"
    assert records[0]["time_s"] == 1.0
    assert all(record["time_s"] >= 1.0 for record in records)
    assert not accepted.empty
    assert accepted["time_s"].min() >= 1.0
    assert not replayed.empty
    assert replayed["time_s"].min() >= 1.0
