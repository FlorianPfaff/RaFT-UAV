import numpy as np
import pandas as pd

from raft_uav.calibration.time_offset import (
    aggregate_measurement_time_offset_sweep,
    aggregate_radar_time_offset_sweep,
    apply_time_offset,
    make_offset_grid,
)
from raft_uav.evaluation.radar_oracle_diagnostics import best_time_offset


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "east_m": [0.0, 10.0, 20.0, 30.0, 40.0],
            "north_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "up_m": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )


def test_make_offset_grid_is_inclusive():
    np.testing.assert_allclose(make_offset_grid(-1.0, 1.0, 0.5), [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_apply_time_offset_preserves_uncorrected_time():
    frame = pd.DataFrame({"time_s": [1.0, 2.0], "east_m": [0.0, 1.0]})

    shifted = apply_time_offset(frame, 0.25)

    np.testing.assert_allclose(shifted["time_s"], [1.25, 2.25])
    np.testing.assert_allclose(shifted["time_s_uncorrected"], [1.0, 2.0])
    np.testing.assert_allclose(shifted["time_offset_correction_s"], [0.25, 0.25])


def test_radar_aggregate_offset_sweep_recovers_positive_offset():
    radar = pd.DataFrame(
        [
            {"frame_index": 0, "time_s": 0.0, "east_m": 10.0, "north_m": 0.0, "up_m": 5.0},
            {"frame_index": 1, "time_s": 1.0, "east_m": 20.0, "north_m": 0.0, "up_m": 5.0},
            {"frame_index": 2, "time_s": 2.0, "east_m": 30.0, "north_m": 0.0, "up_m": 5.0},
        ]
    )

    sweep = aggregate_radar_time_offset_sweep([(radar, _truth())], offsets_s=[-1.0, 0.0, 1.0])

    assert best_time_offset(sweep, metric="mean_3d_error_m") == 1.0
    best = sweep.loc[sweep["time_offset_s"] == 1.0].iloc[0]
    assert best["mean_3d_error_m"] == 0.0
    assert best["coverage"] == 1.0


def test_measurement_aggregate_offset_sweep_uses_2d_metric():
    rf = pd.DataFrame(
        [
            {"time_s": 0.0, "east_m": 10.0, "north_m": 0.0},
            {"time_s": 1.0, "east_m": 20.0, "north_m": 0.0},
        ]
    )

    sweep = aggregate_measurement_time_offset_sweep(
        [(rf, _truth())], offsets_s=[0.0, 1.0], dimensions=2
    )

    assert best_time_offset(sweep, metric="mean_2d_error_m") == 1.0
    best = sweep.loc[sweep["time_offset_s"] == 1.0].iloc[0]
    assert best["mean_2d_error_m"] == 0.0
    assert np.isnan(best["mean_3d_error_m"])
