import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import (
    best_time_offset,
    interpolate_truth_positions,
    nearest_candidate_oracle,
    time_offset_sweep,
)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "east_m": [0.0, 10.0, 20.0, 30.0, 40.0],
            "north_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "up_m": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )


def test_interpolate_truth_positions_linearly():
    positions, valid = interpolate_truth_positions(_truth(), [0.5, 1.5], max_time_delta_s=1.0)

    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(positions[:, 0], [5.0, 15.0])
    np.testing.assert_allclose(positions[:, 2], [5.0, 5.0])


def test_nearest_candidate_oracle_selects_closest_candidate_per_frame():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 5.0,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 11.0,
                "north_m": 0.0,
                "up_m": 5.0,
            },
        ]
    )

    selected = nearest_candidate_oracle(radar, _truth())

    assert selected["track_id"].tolist() == [2]
    assert selected["oracle_error_3d_m"].tolist() == [1.0]
    assert selected["oracle_candidate_rows"].tolist() == [2]


def test_time_offset_sweep_recovers_known_positive_offset():
    radar = pd.DataFrame(
        [
            {"frame_index": 0, "time_s": 0.0, "east_m": 10.0, "north_m": 0.0, "up_m": 5.0},
            {"frame_index": 1, "time_s": 1.0, "east_m": 20.0, "north_m": 0.0, "up_m": 5.0},
            {"frame_index": 2, "time_s": 2.0, "east_m": 30.0, "north_m": 0.0, "up_m": 5.0},
        ]
    )

    sweep = time_offset_sweep(radar, _truth(), offsets_s=[-1.0, 0.0, 1.0])

    assert best_time_offset(sweep) == 1.0
    best = sweep.loc[sweep["time_offset_s"] == 1.0].iloc[0]
    assert best["mean_3d_error_m"] == 0.0
    assert best["coverage"] == 1.0
