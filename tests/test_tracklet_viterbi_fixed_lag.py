import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import _events
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_fixed_lag import (
    run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay,
    select_fixed_lag_tracklet_viterbi_path,
)


def _rf_measurement(time_s: float, east_m: float, north_m: float = 0.0) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([east_m, north_m]),
        covariance=np.diag([4.0, 4.0]),
        source="rf",
    )


def _row(frame_index: int, track_id: int, time_s: float, east_m: float, cat_prob_uav: float) -> dict[str, float | int]:
    return {
        "frame_index": frame_index,
        "track_id": track_id,
        "time_s": time_s,
        "east_m": east_m,
        "north_m": 0.0,
        "up_m": 0.0,
        "cat_prob_uav": cat_prob_uav,
    }


def test_fixed_lag_tracklet_viterbi_uses_bounded_future_window():
    radar = pd.DataFrame(
        [
            _row(0, 1, 0.0, 0.0, 0.8),
            _row(0, 2, 0.0, 100.0, 0.99),
            _row(1, 1, 1.0, 10.0, 0.8),
            _row(1, 2, 1.0, 1000.0, 0.99),
        ]
    )

    _, _, viterbi_selected = (
        run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
            rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(1.0, 10.0)],
            radar=radar,
            lag_s=1.0,
            candidate_catprob_threshold=None,
            config=TrackletViterbiAssociationConfig(
                anchor_nis_weight=2.0,
                track_switch_cost=20.0,
                max_speed_penalty=10_000.0,
            ),
        )
    )

    assert viterbi_selected["track_id"].tolist() == [1, 1]
    assert viterbi_selected["association_lag_s"].tolist() == [1.0, 1.0]


def test_fixed_lag_tracklet_viterbi_conditions_on_previous_committed_choice():
    radar = pd.DataFrame(
        [
            _row(0, 1, 0.0, 0.0, 0.9),
            _row(0, 2, 0.0, 100.0, 0.01),
            _row(1, 1, 1.0, 10.0, 0.3),
            _row(1, 2, 1.0, 110.0, 0.99),
        ]
    )

    selected = select_fixed_lag_tracklet_viterbi_path(
        events=_events([], radar),
        anchors={},
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            catprob_weight=2.0,
            track_switch_cost=1_000.0,
            max_speed_mps=200.0,
            max_speed_penalty=0.0,
            transition_nis_weight=0.0,
            velocity_nis_weight=0.0,
            anchor_nis_weight=0.0,
        ),
        lag_s=1.0,
    )

    assert [int(row["track_id"]) for row in selected] == [1, 1]
    assert "association_prefix_constrained" not in selected[0]
    assert bool(selected[1]["association_prefix_constrained"])
    assert int(selected[1]["association_prefix_track_id"]) == 1
    assert float(selected[1]["association_prefix_time_s"]) == 0.0


def test_fixed_lag_prefix_is_forced_even_when_missed_detections_are_free():
    radar = pd.DataFrame(
        [
            _row(0, 1, 0.0, 0.0, 1.0),
            _row(0, 2, 0.0, 100.0, 0.01),
            _row(1, 1, 1.0, 10.0, 0.01),
            _row(1, 2, 1.0, 110.0, 0.99),
        ]
    )

    selected = select_fixed_lag_tracklet_viterbi_path(
        events=_events([], radar),
        anchors={},
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            missed_detection_cost=0.0,
            consecutive_miss_cost=0.0,
            catprob_weight=10.0,
            track_switch_cost=1_000.0,
            max_speed_mps=200.0,
            max_speed_penalty=0.0,
            transition_nis_weight=0.0,
            velocity_nis_weight=0.0,
            anchor_nis_weight=0.0,
        ),
        lag_s=1.0,
    )

    assert [int(row["track_id"]) for row in selected] == [1, 1]
    assert bool(selected[1]["association_prefix_constrained"])
    assert int(selected[1]["association_prefix_track_id"]) == 1


def test_fixed_lag_tracklet_viterbi_bootstraps_from_first_rf_measurement():
    radar = pd.DataFrame(
        [
            _row(0, 9, 0.0, 900.0, 1.0),
            _row(1, 1, 2.0, 20.0, 1.0),
        ]
    )

    _, _, viterbi_selected = (
        run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
            rf_measurements=[_rf_measurement(1.0, 10.0), _rf_measurement(2.0, 20.0)],
            radar=radar,
            lag_s=1.0,
            candidate_catprob_threshold=None,
            config=TrackletViterbiAssociationConfig(
                catprob_weight=0.0,
                track_switch_cost=0.0,
                max_speed_mps=1_000.0,
                max_speed_penalty=0.0,
                transition_nis_weight=0.0,
                velocity_nis_weight=0.0,
                anchor_nis_weight=0.0,
            ),
        )
    )

    assert viterbi_selected["time_s"].tolist() == [2.0]
    assert viterbi_selected["track_id"].tolist() == [1]
    assert viterbi_selected["association_lag_commit_time_s"].tolist() == [3.0]
    assert viterbi_selected["association_lag_commit_delay_s"].tolist() == [1.0]
