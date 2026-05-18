import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import (
    run_async_cv_baseline_with_tracklet_viterbi_association,
)


def _rf_measurement(time_s: float) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([1.0, 2.0, 3.0]),
        covariance=np.eye(3),
        source="rf",
    )


def test_tracklet_viterbi_skips_pre_rf_radar_bootstrap():
    radar = pd.DataFrame(
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

    records, selected = run_async_cv_baseline_with_tracklet_viterbi_association(
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
