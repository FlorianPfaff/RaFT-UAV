import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker, TrackingMeasurement
from raft_uav.baselines.radar_association import (
    _catprob_candidate_pool,
    _initial_measurement,
    _select_radar_candidate,
    run_async_cv_baseline_with_radar_association,
)


def _rf_measurement(time_s: float, east_m: float, north_m: float = 0.0) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([east_m, north_m]),
        covariance=np.diag([1.0, 1.0]),
        source="rf",
    )


def test_oracle_nearest_truth_selects_closest_candidate_per_frame():
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
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
        ]
    )
    truth = pd.DataFrame({"time_s": [0.0], "east_m": [101.0], "north_m": [0.0], "up_m": [0.0]})

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[],
        radar=radar,
        association="oracle-nearest-truth",
        truth=truth,
    )

    assert len(records) == 1
    assert selected["track_id"].tolist() == [2]


def test_prediction_nis_selects_candidate_near_prediction():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 2.0,
                "east_m": -100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(1.0, 10.0)],
        radar=radar,
        association="prediction-nis",
    )

    assert len(records) == 3
    assert selected["track_id"].tolist() == [1]


def test_safety_gate_makes_impossible_radar_association_a_miss():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 99,
                "time_s": 2.0,
                "east_m": 10_000.0,
                "north_m": 10_000.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(1.0, 1.0)],
        radar=radar,
        association="prediction-nis",
        gate_thresholds_by_source={"radar": 5.0},
        safety_gate_thresholds_by_source={"radar": 50.0},
        robust_update_by_source={"radar": "nis-inflate"},
    )

    assert records[-1]["source"] == "radar"
    assert records[-1]["accepted"] is False
    assert records[-1]["update_action"] == "missed_detection"
    assert records[-1]["covariance_scale"] == 1.0
    assert selected.empty


def test_catprob_candidate_pool_filters_when_possible():
    candidates = pd.DataFrame(
        {
            "track_id": [1, 2],
            "cat_prob_uav": [0.2, 0.8],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    pool = _catprob_candidate_pool(candidates, 0.4)

    assert pool["track_id"].tolist() == [2]
    assert pool["association_catprob_threshold"].tolist() == [0.4]
    assert pool["association_catprob_fallback"].tolist() == [False]


def test_catprob_candidate_pool_falls_back_to_unfiltered_candidates_when_threshold_empty():
    candidates = pd.DataFrame(
        {
            "track_id": [1, 2],
            "cat_prob_uav": [0.1, 0.2],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    pool = _catprob_candidate_pool(candidates, 0.4)

    assert pool["track_id"].tolist() == [1, 2]
    assert pool["association_catprob_threshold"].tolist() == [0.4, 0.4]
    assert pool["association_catprob_fallback"].tolist() == [True, True]
    assert pool["association_catprob_candidate_rows"].tolist() == [2, 2]
    assert pool["association_catprob_fallback_reason"].tolist() == [
        "all_candidates_below_threshold",
        "all_candidates_below_threshold",
    ]


def test_initial_radar_measurement_respects_catprob_threshold():
    candidates = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.2,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.3,
            },
        ]
    )

    measurement = _initial_measurement(
        {"kind": "radar", "time_s": 0.0, "candidates": candidates},
        association="prediction-nis",
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        candidate_catprob_threshold=0.4,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )

    assert measurement is None


def test_prediction_nis_uses_geometry_when_all_catprob_candidates_are_below_threshold():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 2.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.2,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 2.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.3,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(1.0, 1.0)],
        radar=radar,
        association="prediction-nis",
        candidate_catprob_threshold=0.4,
    )

    assert [record["source"] for record in records] == ["rf", "rf", "radar"]
    assert selected["track_id"].tolist() == [1]
    assert selected["association_catprob_fallback"].tolist() == [True]
    assert selected["association_catprob_fallback_reason"].tolist() == [
        "all_candidates_below_threshold"
    ]


def test_initial_radar_measurement_respects_catprob_threshold():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.2,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.3,
            },
        ]
    )

    measurement = _initial_measurement(
        {
            "kind": "radar",
            "time_s": 0.0,
            "candidates": radar,
        },
        association="prediction-nis",
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
        candidate_catprob_threshold=0.4,
    )

    assert measurement is None


def test_track_continuity_keeps_current_track_for_small_nis_gain():
    tracker = AsyncConstantVelocityKalmanTracker(initial_position=np.zeros(3), initial_time_s=0.0)
    candidates = pd.DataFrame(
        [
            {
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 1.1,
                "north_m": 0.0,
                "up_m": 0.0,
            },
            {
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
            },
        ]
    )

    selected = _select_radar_candidate(
        candidates,
        association="track-continuity",
        tracker=tracker,
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        truth=None,
        current_track_id=1,
        track_switch_nis_ratio=0.5,
        candidate_catprob_threshold=None,
        geometry_velocity_std_mps=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )

    assert selected is not None
    assert int(selected["track_id"]) == 1


def test_track_continuity_without_track_id_falls_back_to_prediction_nis():
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=np.zeros(3),
        initial_time_s=0.0,
    )
    candidates = pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
            },
            {
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
            },
        ]
    )

    selected = _select_radar_candidate(
        candidates,
        association="track-continuity",
        tracker=tracker,
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        truth=None,
        current_track_id=1,
        track_switch_nis_ratio=0.5,
        candidate_catprob_threshold=None,
        geometry_velocity_std_mps=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )

    assert selected is not None
    assert selected["east_m"] == 1.0


def test_track_continuity_ignores_malformed_track_ids_when_matching_current_track():
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=np.zeros(3),
        initial_time_s=0.0,
    )
    candidates = pd.DataFrame(
        [
            {
                "track_id": "not-a-number",
                "time_s": 0.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
            },
            {
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 1.1,
                "north_m": 0.0,
                "up_m": 0.0,
            },
        ]
    )

    selected = _select_radar_candidate(
        candidates,
        association="track-continuity",
        tracker=tracker,
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        truth=None,
        current_track_id=1,
        track_switch_nis_ratio=0.5,
        candidate_catprob_threshold=None,
        geometry_velocity_std_mps=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )

    assert selected is not None
    assert int(selected["track_id"]) == 1


def test_geometry_score_prefers_velocity_consistent_candidate():
    tracker = AsyncConstantVelocityKalmanTracker(initial_position=np.zeros(3), initial_time_s=0.0)
    tracker.mean[3] = 20.0
    candidates = pd.DataFrame(
        [
            {
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "velocity_east_mps": 20.0,
                "velocity_north_mps": 0.0,
                "velocity_down_mps": 0.0,
                "cat_prob_uav": 0.8,
            },
            {
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "velocity_east_mps": 0.0,
                "velocity_north_mps": 0.0,
                "velocity_down_mps": 0.0,
                "cat_prob_uav": 0.8,
            },
        ]
    )

    selected = _select_radar_candidate(
        candidates,
        association="geometry-score",
        tracker=tracker,
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        truth=None,
        current_track_id=None,
        track_switch_nis_ratio=0.5,
        candidate_catprob_threshold=None,
        geometry_velocity_std_mps=12.0,
        geometry_velocity_weight=1.0,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )

    assert selected is not None
    assert int(selected["track_id"]) == 1
    assert float(selected["association_score"]) < float("inf")


def test_rf_anchored_nis_recovers_from_prediction_drift():
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=np.array([20.0, 0.0, 0.0]),
        initial_time_s=0.0,
    )
    tracker.predict_to(2.0)
    candidates = pd.DataFrame(
        [
            {
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
            {
                "track_id": 2,
                "time_s": 2.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
        ]
    )
    covariance = np.diag([25.0**2, 25.0**2, 35.0**2])

    prediction_selected = _select_radar_candidate(
        candidates,
        association="prediction-nis",
        tracker=tracker,
        covariance=covariance,
        truth=None,
        current_track_id=None,
        track_switch_nis_ratio=0.5,
        candidate_catprob_threshold=None,
        geometry_velocity_std_mps=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    anchored_selected = _select_radar_candidate(
        candidates,
        association="rf-anchored-nis",
        tracker=tracker,
        covariance=covariance,
        truth=None,
        current_track_id=None,
        track_switch_nis_ratio=0.5,
        candidate_catprob_threshold=None,
        geometry_velocity_std_mps=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        rf_measurements=[_rf_measurement(1.9, 0.0)],
        rf_anchor_weight=1.0,
        rf_anchor_time_gate_s=0.2,
        rf_anchor_nis_cap=1_000.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )

    assert prediction_selected is not None
    assert anchored_selected is not None
    assert int(prediction_selected["track_id"]) == 2
    assert int(anchored_selected["track_id"]) == 1
    assert anchored_selected["association_action"] == "rf_anchored_nis"
    assert anchored_selected["association_anchor_nis"] == 0.0
    assert np.isclose(anchored_selected["association_anchor_time_delta_s"], 0.1)


def test_rf_gated_nis_coasts_when_all_candidates_conflict_with_recent_rf():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 120.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        association="rf-gated-nis",
        rf_anchor_time_gate_s=2.0,
        rf_anchor_gate_nis=25.0,
    )

    assert [record["source"] for record in records] == ["rf"]
    assert selected.empty


def test_rf_gated_nis_keeps_rf_consistent_candidate():
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=np.array([20.0, 0.0, 0.0]),
        initial_time_s=0.0,
    )
    tracker.predict_to(2.0)
    candidates = pd.DataFrame(
        [
            {
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
            {
                "track_id": 2,
                "time_s": 2.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
        ]
    )

    selected = _select_radar_candidate(
        candidates,
        association="rf-gated-nis",
        tracker=tracker,
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        truth=None,
        current_track_id=None,
        track_switch_nis_ratio=0.5,
        candidate_catprob_threshold=None,
        geometry_velocity_std_mps=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        rf_measurements=[_rf_measurement(1.9, 0.0)],
        rf_anchor_weight=1.0,
        rf_anchor_time_gate_s=0.2,
        rf_anchor_nis_cap=1_000.0,
        rf_anchor_gate_nis=25.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )

    assert selected is not None
    assert int(selected["track_id"]) == 1
    assert selected["association_action"] == "rf_gated_nis"
    assert selected["association_anchor_gate_rejected_count"] == 1
    assert selected["association_anchor_gate_candidate_count"] == 2


def test_pda_mixture_returns_weighted_position_and_spread_covariance():
    tracker = AsyncConstantVelocityKalmanTracker(initial_position=np.array([5.0, 0.0, 0.0]), initial_time_s=0.0)
    candidates = pd.DataFrame(
        [
            {
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 1.0,
            },
            {
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 1.0,
            },
        ]
    )

    selected = _select_radar_candidate(
        candidates,
        association="pda-mixture",
        tracker=tracker,
        covariance=np.diag([1.0, 1.0, 1.0]),
        truth=None,
        current_track_id=None,
        track_switch_nis_ratio=0.5,
        candidate_catprob_threshold=None,
        geometry_velocity_std_mps=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )

    assert selected is not None
    assert selected["association_mode"] == "pda-mixture"
    assert selected["east_m"] == 5.0
    assert selected["association_effective_candidates"] == 2.0
    assert selected["association_cov_ee"] > 1.0


def test_track_bank_uses_pyrecest_mht_and_records_hypotheses():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        association="track-bank",
        track_bank_max_hypotheses=4,
        track_bank_gate_probability=0.999999,
    )

    assert records[-1]["association_mode"] == "track-bank"
    assert int(records[-1]["hypothesis_count"]) >= 1
    assert records[-1]["hypotheses"]
    assert selected["track_id"].tolist() == [1]


def test_track_bank_bootstrap_respects_catprob_threshold():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.2,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.3,
            },
        ]
    )

    measurement = _initial_measurement(
        {"kind": "radar", "time_s": 0.0, "candidates": radar},
        association="track-bank",
        covariance=np.diag([25.0**2, 25.0**2, 35.0**2]),
        candidate_catprob_threshold=0.4,
        stable_anchor_by_key=None,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )

    assert measurement is None


def test_track_bank_does_not_reassimilate_bootstrap_radar_frame():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 7,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[],
        radar=radar,
        association="track-bank",
        track_bank_max_hypotheses=4,
        track_bank_gate_probability=0.999999,
    )

    assert len(records) == 1
    assert records[0]["source"] == "radar"
    assert records[0]["association_mode"] == "track-bank"
    assert records[0]["update_action"] == "initialized"
    assert records[0]["nis"] == 0.0
    assert selected["track_id"].tolist() == [7]


def test_stable_segments_updates_only_on_stitched_high_confidence_segments():
    radar = pd.DataFrame(
        [
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
            for frame in (1, 2, 3)
        ]
        + [
            {
                "frame_index": 4,
                "track_id": 2,
                "time_s": 4.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0,
                "cat_prob_uav": 0.9,
            }
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        association="stable-segments",
        candidate_catprob_threshold=0.4,
        stable_segment_min_frames=3,
        stable_segment_max_transition_speed_mps=65.0,
    )

    assert [record["source"] for record in records] == ["rf", "radar", "radar", "radar"]
    assert {record["association_mode"] for record in records if record["source"] == "radar"} == {
        "stable-segments"
    }
    assert selected["track_id"].tolist() == [1, 1, 1]


def test_stable_segments_respects_range_gate_and_min_frames():
    radar = pd.DataFrame(
        [
            {
                "frame_index": frame,
                "track_id": 1,
                "time_s": float(frame),
                "east_m": float(frame),
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 900.0,
                "cat_prob_uav": 0.9,
            }
            for frame in (1, 2, 3)
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        association="stable-segments",
        candidate_catprob_threshold=0.4,
        stable_segment_min_frames=3,
        stable_segment_range_gate_m=800.0,
    )

    assert [record["source"] for record in records] == ["rf"]
    assert selected.empty


def test_stable_segments_hybrid_uses_stable_anchor_then_prediction_fallback():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 1,
                "track_id": 7,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 1.0,
                "cat_prob_uav": 0.8,
            },
            {
                "frame_index": 1,
                "track_id": 99,
                "time_s": 1.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 2,
                "track_id": 7,
                "time_s": 2.0,
                "east_m": 2.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 2.0,
                "cat_prob_uav": 0.8,
            },
            {
                "frame_index": 3,
                "track_id": 88,
                "time_s": 3.0,
                "east_m": 3.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 3.0,
                "cat_prob_uav": 0.8,
            },
            {
                "frame_index": 3,
                "track_id": 89,
                "time_s": 3.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0,
                "cat_prob_uav": 0.95,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        association="stable-segments-hybrid",
        candidate_catprob_threshold=0.4,
        stable_segment_min_frames=2,
        stable_segment_max_transition_speed_mps=65.0,
    )

    assert [record["source"] for record in records] == ["rf", "radar", "radar", "radar"]
    assert selected["track_id"].tolist() == [7, 7, 88]
    assert selected["association_action"].tolist() == [
        "stable_segment_hybrid_update",
        "stable_segment_hybrid_update",
        "stable_segment_hybrid_prediction_nis",
    ]


def test_stable_segments_prefers_rf_consistent_segment_chain():
    radar = pd.DataFrame(
        [
            {
                "frame_index": frame,
                "track_id": 1,
                "time_s": float(frame),
                "east_m": float(frame),
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": float(frame),
                "cat_prob_uav": 0.8,
            }
            for frame in (1, 2, 3)
        ]
        + [
            {
                "frame_index": frame,
                "track_id": 2,
                "time_s": float(frame),
                "east_m": 100.0 + float(frame),
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0 + float(frame),
                "cat_prob_uav": 0.95,
            }
            for frame in (1, 2, 3)
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(2.0, 2.0)],
        radar=radar,
        association="stable-segments",
        candidate_catprob_threshold=0.4,
        stable_segment_min_frames=3,
        stable_segment_rf_score_weight=1.0,
        stable_segment_rf_time_gate_s=0.1,
        stable_segment_rf_nis_cap=25.0,
    )

    assert [record["source"] for record in records] == ["radar", "rf", "radar", "radar"]
    assert selected["track_id"].tolist() == [1, 1, 1]
    assert selected["association_segment_rf_support_count"].tolist() == [1, 1, 1]
    assert selected["association_segment_rf_mean_nis"].tolist() == [0.0, 0.0, 0.0]


def test_stable_segments_interpolated_fills_bracketed_radar_frames():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 1,
                "track_id": 7,
                "time_s": 1.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 2,
                "track_id": 99,
                "time_s": 2.0,
                "east_m": 1000.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 1000.0,
                "cat_prob_uav": 0.1,
            },
            {
                "frame_index": 3,
                "track_id": 7,
                "time_s": 3.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 20.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        association="stable-segments-interpolated",
        candidate_catprob_threshold=0.4,
        stable_segment_min_frames=1,
        stable_segment_max_transition_speed_mps=20.0,
        stable_segment_interpolation_max_gap_s=5.0,
        stable_segment_interpolation_max_speed_mps=20.0,
        stable_segment_interpolation_std_scale=3.0,
        stable_segment_interpolation_gap_std_mps=10.0,
    )

    assert [record["source"] for record in records] == ["rf", "radar", "radar", "radar"]
    assert selected["frame_index"].tolist() == [1, 2, 3]
    assert selected["east_m"].tolist() == [0.0, 10.0, 20.0]
    assert selected["track_id"].tolist() == [7, 7, 7]
    assert selected["association_interpolated"].tolist() == [False, True, False]
    assert selected["association_mode"].unique().tolist() == ["stable-segments-interpolated"]
    assert selected.loc[1, "association_interpolation_std_scale"] == 3.0
    assert selected.loc[1, "association_interpolation_gap_std_mps"] == 10.0
    assert selected.loc[1, "association_interpolation_gap_s"] == 2.0
    assert selected.loc[1, "association_interpolation_nearest_anchor_dt_s"] == 1.0
    assert selected.loc[1, "association_interpolation_gap_fraction"] == 1.0
    assert selected.loc[1, "association_cov_ee"] == 25.0**2 * 3.0**2 + 10.0**2
    assert selected.loc[1, "association_cov_uu"] == 35.0**2 * 3.0**2 + 10.0**2


def test_stable_segments_interpolated_gap_cap_keeps_anchors_only():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 1,
                "track_id": 7,
                "time_s": 1.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 2,
                "track_id": 99,
                "time_s": 2.0,
                "east_m": 1000.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 1000.0,
                "cat_prob_uav": 0.1,
            },
            {
                "frame_index": 3,
                "track_id": 7,
                "time_s": 3.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 20.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        association="stable-segments-interpolated",
        candidate_catprob_threshold=0.4,
        stable_segment_min_frames=1,
        stable_segment_max_transition_speed_mps=20.0,
        stable_segment_interpolation_max_gap_s=1.0,
        stable_segment_interpolation_max_speed_mps=20.0,
    )

    assert [record["source"] for record in records] == ["rf", "radar", "radar"]
    assert selected["frame_index"].tolist() == [1, 3]
    assert selected["association_interpolation_long_gap_dropped_count"].tolist() == [1, 1]
