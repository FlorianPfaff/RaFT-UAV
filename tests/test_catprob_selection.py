import pandas as pd

from raft_uav.io.aerpaw import select_radar_measurement_rows


def test_catprob_selects_best_candidate_once_per_radar_frame():
    radar = pd.DataFrame(
        {
            "frame_index": [10, 10, 11, 11, 12],
            "time_s": [1.0, 1.0, 2.0, 2.0, 3.0],
            "track_index": [0, 1, 0, 1, 0],
            "track_id": [100, 101, 200, 201, 300],
            "cat_prob_uav": [0.60, 0.75, 0.55, 0.90, 0.20],
        }
    )

    selected = select_radar_measurement_rows(
        radar, selection="catprob", catprob_threshold=0.5
    )

    assert selected["track_id"].tolist() == [101, 201]
    assert selected["frame_index"].tolist() == [10, 11]


def test_catprob_all_preserves_legacy_multi_candidate_selection():
    radar = pd.DataFrame(
        {
            "frame_index": [10, 10, 11, 11, 12],
            "track_id": [100, 101, 200, 201, 300],
            "cat_prob_uav": [0.60, 0.75, 0.55, 0.90, 0.20],
        }
    )

    selected = select_radar_measurement_rows(
        radar, selection="catprob-all", catprob_threshold=0.5
    )

    assert selected["track_id"].tolist() == [100, 101, 200, 201]
