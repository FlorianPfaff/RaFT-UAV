import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import (
    run_async_cv_baseline_with_radar_association,
)


def _rf_measurement(time_s: float) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([1.0, 2.0, 3.0]),
        covariance=np.eye(3),
        source="rf",
    )


@pytest.mark.parametrize(
    ("association", "kwargs"),
    [
        ("prediction-nis", {}),
        ("track-continuity", {}),
        ("geometry-score", {}),
        ("pda-mixture", {}),
        ("track-bank", {}),
        (
            "stable-segments",
            {"stable_segment_min_frames": 1, "stable_segment_range_gate_m": None},
        ),
    ],
)
def test_online_radar_association_skips_pre_rf_radar_bootstrap(association, kwargs):
    radar = pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "frame_index": 0,
                "track_id": 99,
                "track_index": 0,
                "east_m": 1000.0,
                "north_m": 1000.0,
                "up_m": 1000.0,
                "cat_prob_uav": 0.99,
            },
            {
                "time_s": 2.0,
                "frame_index": 1,
                "track_id": 1,
                "track_index": 0,
                "east_m": 1.2,
                "north_m": 2.0,
                "up_m": 3.0,
                "cat_prob_uav": 0.99,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(1.0)],
        radar=radar,
        association=association,
        candidate_catprob_threshold=None,
        **kwargs,
    )

    assert records
    assert records[0]["source"] == "rf"
    assert records[0]["time_s"] == pytest.approx(1.0)
    assert all(float(record["time_s"]) >= 1.0 for record in records)
    if not selected.empty:
        assert selected["time_s"].min() >= 1.0
        if "track_id" in selected.columns:
            assert 99 not in set(selected["track_id"].astype(int))


def test_online_radar_association_still_bootstraps_radar_only_runs():
    radar = pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "frame_index": 0,
                "track_id": 1,
                "track_index": 0,
                "east_m": 1.0,
                "north_m": 2.0,
                "up_m": 3.0,
                "cat_prob_uav": 0.99,
            }
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[],
        radar=radar,
        association="prediction-nis",
        candidate_catprob_threshold=None,
    )

    assert records
    assert records[0]["source"] == "radar"
    assert not selected.empty
