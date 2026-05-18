import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.diagnostics.paper_table import (
    metric_row,
    run_paper_compatible_cv_fusion,
    run_paper_longest_track_cv_fusion,
    select_stable_radar_segments,
    select_radar_for_table,
)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 10.0, 20.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )


def test_metric_row_reports_paper_style_error_columns():
    row = metric_row(
        method="synthetic",
        modality="radar",
        times_s=np.array([0.0, 1.0]),
        positions_m=np.array([[0.0, 0.0, 0.0], [13.0, 4.0, 0.0]]),
        truth=_truth(),
        candidate_count=2,
        selected_count=2,
        max_time_delta_s=0.5,
        track_ids=[1, 2],
    )

    assert row["matched_count"] == 2
    assert row["coverage"] == 1.0
    assert row["track_switches"] == 1
    assert row["error_2d_mean_m"] == 2.5
    assert row["error_3d_max_m"] == 5.0


def test_select_radar_for_table_oracle_picks_nearest_truth_per_frame():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.5,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": -100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 11.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.4,
            },
        ]
    )

    selected = select_radar_for_table(
        radar=radar,
        truth=_truth(),
        selection="radar-oracle-nearest-truth",
        catprob_threshold=0.4,
        max_time_delta_s=0.5,
    )

    assert selected["track_id"].tolist() == [1, 2]


def test_range_gated_longest_track_selection_skips_out_of_range_frames():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 10.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 900.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 900.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 11.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 11.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    selected = select_radar_for_table(
        radar=radar,
        truth=_truth(),
        selection="radar-longest-continuous-track-range-gated",
        catprob_threshold=0.4,
        range_gate_m=800.0,
        max_time_delta_s=0.5,
    )

    assert selected["frame_index"].tolist() == [0]
    assert selected["track_id"].tolist() == [1]


def test_interpolated_range_gated_longest_track_fills_radar_frame_times():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 2,
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 20.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    selected = select_radar_for_table(
        radar=radar,
        truth=_truth(),
        selection="radar-longest-track-range-gated-interpolated",
        catprob_threshold=0.4,
        range_gate_m=800.0,
        max_time_delta_s=0.5,
    )

    assert selected["time_s"].tolist() == [0.0, 1.0, 2.0]
    np.testing.assert_allclose(selected["east_m"].to_numpy(dtype=float), [0.0, 10.0, 20.0])
    assert selected["association_interpolated"].tolist() == [True, True, True]


def test_range_gated_longest_track_prefers_longest_continuous_segment():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 2,
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 20.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 10.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 2,
                "track_id": 2,
                "time_s": 2.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 20.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    selected = select_radar_for_table(
        radar=radar,
        truth=_truth(),
        selection="radar-longest-continuous-track-range-gated",
        catprob_threshold=0.4,
        range_gate_m=800.0,
        max_time_delta_s=0.5,
    )

    assert selected["track_id"].tolist() == [2, 2]


def test_stable_segments_stitch_high_confidence_tracks():
    radar = pd.DataFrame(
        [
            *[
                {
                    "frame_index": frame,
                    "track_id": 1,
                    "time_s": float(frame),
                    "east_m": float(frame),
                    "north_m": 0.0,
                    "up_m": 0.0,
                    "range_m": float(frame),
                    "cat_prob_uav": 0.9,
                }
                for frame in range(4)
            ],
            *[
                {
                    "frame_index": frame,
                    "track_id": 2,
                    "time_s": float(frame),
                    "east_m": 100.0 + float(frame),
                    "north_m": 0.0,
                    "up_m": 0.0,
                    "range_m": 100.0 + float(frame),
                    "cat_prob_uav": 0.1,
                }
                for frame in range(4, 8)
            ],
            *[
                {
                    "frame_index": frame,
                    "track_id": 3,
                    "time_s": float(frame),
                    "east_m": float(frame),
                    "north_m": 0.0,
                    "up_m": 0.0,
                    "range_m": float(frame),
                    "cat_prob_uav": 0.8,
                }
                for frame in range(8, 12)
            ],
        ]
    )

    selected = select_stable_radar_segments(
        radar=radar,
        catprob_threshold=0.4,
        range_gate_m=800.0,
        min_segment_frames=3,
    )

    assert selected["track_id"].tolist() == [1, 1, 1, 1, 3, 3, 3, 3]
    assert selected["association_segment_count"].iloc[0] == 2


def test_paper_compatible_fusion_coasts_when_all_radar_candidates_fail_hard_gate():
    rf_measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([0.0, 0.0, 0.0]),
            covariance=np.diag([1.0, 1.0, 1.0]),
            source="rf",
        )
    ]
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 900.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 900.0,
                "cat_prob_uav": 0.99,
            }
        ]
    )

    records, selected = run_paper_compatible_cv_fusion(
        rf_measurements=rf_measurements,
        radar=radar,
        radar_range_gate_m=800.0,
        radar_catprob_threshold=0.4,
    )

    assert len(records) == 2
    assert records[-1]["source"] == "radar"
    assert records[-1]["update_action"] == "missed_detection"
    assert selected.empty


def test_paper_compatible_fusion_coasts_when_catprob_threshold_has_no_fallback():
    rf_measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([0.0, 0.0, 0.0]),
            covariance=np.diag([1.0, 1.0, 1.0]),
            source="rf",
        )
    ]
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 1.0,
                "cat_prob_uav": 0.1,
            }
        ]
    )

    records, selected = run_paper_compatible_cv_fusion(
        rf_measurements=rf_measurements,
        radar=radar,
        radar_range_gate_m=800.0,
        radar_catprob_threshold=0.4,
    )

    assert records[-1]["update_action"] == "missed_detection"
    assert selected.empty


def test_paper_compatible_fusion_updates_when_candidate_passes_hard_gates():
    rf_measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([0.0, 0.0, 0.0]),
            covariance=np.diag([1.0, 1.0, 1.0]),
            source="rf",
        )
    ]
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 1.0,
                "cat_prob_uav": 0.99,
            }
        ]
    )

    records, selected = run_paper_compatible_cv_fusion(
        rf_measurements=rf_measurements,
        radar=radar,
        radar_range_gate_m=800.0,
        radar_catprob_threshold=0.4,
    )

    assert records[-1]["update_action"] == "updated"
    assert records[-1]["association_mode"] == "paper-compatible"
    assert selected["track_id"].tolist() == [1]


def test_paper_longest_track_fusion_coasts_between_stable_anchors():
    rf_measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([0.0, 0.0, 0.0]),
            covariance=np.diag([1.0, 1.0, 1.0]),
            source="rf",
        )
    ]
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 1.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 2,
                "time_s": 2.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 2,
                "track_id": 1,
                "time_s": 3.0,
                "east_m": 3.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 3.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    records, selected = run_paper_longest_track_cv_fusion(
        rf_measurements=rf_measurements,
        radar=radar,
        radar_range_gate_m=800.0,
    )

    assert [record["update_action"] for record in records if record["source"] == "radar"] == [
        "updated",
        "missed_detection",
        "updated",
    ]
    assert selected["track_id"].tolist() == [1, 1]
