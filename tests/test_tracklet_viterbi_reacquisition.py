import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    run_async_cv_baseline_with_tracklet_viterbi_association,
)


def _rf_measurement(time_s: float, east_m: float, north_m: float = 0.0) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([east_m, north_m]),
        covariance=np.diag([4.0, 4.0]),
        source="rf",
    )


def test_tracklet_viterbi_reacquires_rf_anchor_after_miss_streak():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 1000.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 2,
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 500.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 2,
                "track_id": 2,
                "time_s": 2.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.3,
            },
        ]
    )

    _, selected = run_async_cv_baseline_with_tracklet_viterbi_association(
        rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(2.0, 20.0)],
        radar=radar,
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            anchor_nis_weight=0.0,
            missed_detection_cost=2.0,
            consecutive_miss_cost=0.0,
            max_speed_penalty=10_000.0,
            range_gate_m=None,
            reacquisition_miss_streak_threshold=1,
            reacquisition_gate_nis=16.0,
            reacquisition_reward=5.0,
            reacquisition_outside_gate_penalty=50.0,
        ),
    )

    assert selected["frame_index"].tolist() == [0, 2]
    assert selected["track_id"].tolist() == [1, 2]
    reacquired = selected.iloc[-1]
    assert bool(reacquired["association_reacquisition_active"])
    assert int(reacquired["association_preceding_miss_streak"]) == 1
    assert float(reacquired["association_reacquisition_cost"]) < 0.0
