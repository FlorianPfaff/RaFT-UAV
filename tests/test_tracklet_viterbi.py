import json

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    run_async_cv_baseline_with_tracklet_viterbi_association,
)
from raft_uav.baselines.tracklet_viterbi_result import (
    run_async_cv_baseline_with_tracklet_viterbi_result,
)


def _rf_measurement(time_s: float, east_m: float, north_m: float = 0.0) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([east_m, north_m]),
        covariance=np.diag([4.0, 4.0]),
        source="rf",
    )


def test_tracklet_viterbi_prefers_rf_supported_coherent_track():
    radar = pd.DataFrame(
        [
            {"frame_index": 0, "track_id": 1, "time_s": 0.0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.8},
            {"frame_index": 0, "track_id": 2, "time_s": 0.0, "east_m": 100.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.99},
            {"frame_index": 1, "track_id": 1, "time_s": 1.0, "east_m": 10.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.8},
            {"frame_index": 1, "track_id": 2, "time_s": 1.0, "east_m": 110.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.99},
            {"frame_index": 2, "track_id": 1, "time_s": 2.0, "east_m": 20.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.8},
            {"frame_index": 2, "track_id": 2, "time_s": 2.0, "east_m": 200.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.99},
        ]
    )

    _, selected = run_async_cv_baseline_with_tracklet_viterbi_association(
        rf_measurements=[
            _rf_measurement(0.0, 0.0),
            _rf_measurement(1.0, 10.0),
            _rf_measurement(2.0, 20.0),
        ],
        radar=radar,
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            anchor_nis_weight=2.0,
            track_switch_cost=20.0,
            max_speed_penalty=100.0,
        ),
    )

    assert selected["track_id"].tolist() == [1, 1, 1]
    assert selected["association_mode"].unique().tolist() == ["tracklet-viterbi"]


def test_tracklet_viterbi_can_skip_implausible_radar_frame():
    radar = pd.DataFrame(
        [
            {"frame_index": 0, "track_id": 1, "time_s": 0.0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.9},
            {"frame_index": 1, "track_id": 99, "time_s": 1.0, "east_m": 1000.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.99},
            {"frame_index": 2, "track_id": 1, "time_s": 2.0, "east_m": 20.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.9},
        ]
    )

    _, selected = run_async_cv_baseline_with_tracklet_viterbi_association(
        rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(2.0, 20.0)],
        radar=radar,
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            missed_detection_cost=5.0,
            anchor_nis_weight=2.0,
            max_speed_penalty=10_000.0,
        ),
    )

    assert selected["frame_index"].tolist() == [0, 2]
    assert selected["track_id"].tolist() == [1, 1]


def test_tracklet_viterbi_result_preserves_rejected_viterbi_choices():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 99,
                "time_s": 2.0,
                "east_m": 1000.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
        ]
    )

    result = run_async_cv_baseline_with_tracklet_viterbi_result(
        rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(1.0, 0.0)],
        radar=radar,
        candidate_catprob_threshold=None,
        safety_gate_thresholds_by_source={"radar": 1.0},
        config=TrackletViterbiAssociationConfig(
            missed_detection_cost=1_000.0,
            anchor_nis_weight=0.0,
            range_gate_m=None,
        ),
    )

    assert result.accepted_radar.empty
    assert result.viterbi_selected_radar["track_id"].tolist() == [99]
    assert result.viterbi_selected_radar["association_replay_accepted"].tolist() == [False]
    assert result.viterbi_selected_radar["association_replay_update_action"].tolist() == ["missed_detection"]


def test_tracklet_viterbi_beam_can_rerank_by_replay_rejection_cost():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 1000.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 5.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
        ]
    )

    result = run_async_cv_baseline_with_tracklet_viterbi_result(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        candidate_catprob_threshold=None,
        safety_gate_thresholds_by_source={"radar": 1.0},
        config=TrackletViterbiAssociationConfig(
            anchor_nis_weight=0.0,
            missed_detection_cost=1_000.0,
            range_gate_m=None,
            path_beam_width=2,
            replay_rejection_cost=1_000.0,
        ),
    )

    assert result.viterbi_selected_radar["track_id"].tolist() == [1]
    assert result.viterbi_selected_radar["association_viterbi_path_rank"].tolist() == [1]
    assert result.viterbi_selected_radar["association_replay_accepted"].tolist() == [True]
    assert "association_viterbi_replay_objective" in result.viterbi_selected_radar.columns
    assert result.radar_candidate_ledger["association_viterbi_selected"].tolist() == [False, True]


def test_tracklet_viterbi_smoothed_rf_anchor_uses_future_rf_context():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 5.0,
                "east_m": 50.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.5,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 5.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.5,
            },
        ]
    )
    rf_measurements = [_rf_measurement(0.0, 0.0), _rf_measurement(10.0, 100.0)]

    _, causal_selected = run_async_cv_baseline_with_tracklet_viterbi_association(
        rf_measurements=rf_measurements,
        radar=radar,
        acceleration_std_mps2=0.05,
        radar_xy_std_m=2.0,
        radar_z_std_m=2.0,
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            anchor_nis_weight=10.0,
            catprob_weight=0.0,
            rf_anchor_mode="causal",
        ),
    )
    _, smoothed_selected = run_async_cv_baseline_with_tracklet_viterbi_association(
        rf_measurements=rf_measurements,
        radar=radar,
        acceleration_std_mps2=0.05,
        radar_xy_std_m=2.0,
        radar_z_std_m=2.0,
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            anchor_nis_weight=10.0,
            catprob_weight=0.0,
            rf_anchor_mode="smoothed",
        ),
    )

    assert causal_selected["track_id"].tolist() == [2]
    assert smoothed_selected["track_id"].tolist() == [1]
    assert smoothed_selected["association_rf_anchor_mode"].tolist() == ["smoothed"]


def test_tracklet_viterbi_reranker_json_changes_unary_candidate_ranking(tmp_path):
    scorer = tmp_path / "association_reranker.json"
    scorer.write_text(
        json.dumps(
            {
                "intercept": 30.0,
                "coefficients": {"track_id": -20.0},
                "probability_floor": 1.0e-9,
                "probability_ceiling": 0.999999999,
            }
        ),
        encoding="utf-8",
    )
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.1,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 2.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
        ]
    )

    _, selected = run_async_cv_baseline_with_tracklet_viterbi_association(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            use_rf_anchor=False,
            range_gate_m=None,
            association_reranker_path=scorer,
            association_reranker_weight=2.0,
        ),
    )

    assert selected["track_id"].tolist() == [1]
    assert selected["association_reranker_probability"].iloc[0] > 0.99
    assert selected["association_reranker_cost"].iloc[0] < 1.0e-3
