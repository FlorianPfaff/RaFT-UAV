import pandas as pd

from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiConfig,
    select_tracklet_viterbi_path,
)


def test_tracklet_viterbi_prefers_dynamic_path_over_isolated_high_catprob():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.7,
            },
            {
                "frame_index": 0,
                "track_id": 99,
                "time_s": 0.0,
                "east_m": 500.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 1,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.7,
            },
            {
                "frame_index": 1,
                "track_id": 99,
                "time_s": 1.0,
                "east_m": -500.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 2,
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.7,
            },
            {
                "frame_index": 2,
                "track_id": 99,
                "time_s": 2.0,
                "east_m": 500.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
        ]
    )

    selected = select_tracklet_viterbi_path(
        radar,
        candidate_catprob_threshold=None,
        config=TrackletViterbiConfig(
            transition_std_m=20.0,
            switch_penalty=5.0,
            catprob_weight=1.0,
            max_speed_mps=80.0,
        ),
    )

    assert selected["track_id"].tolist() == [1, 1, 1]
    assert selected["association_mode"].tolist() == ["tracklet-viterbi"] * 3
    assert selected["association_viterbi_transition_cost"].iloc[1] < 1.0


def test_tracklet_viterbi_empty_input_preserves_schema():
    radar = pd.DataFrame(columns=["time_s", "east_m", "north_m", "up_m"])

    selected = select_tracklet_viterbi_path(radar)

    assert selected.empty
    assert "association_viterbi_total_cost" in selected.columns
