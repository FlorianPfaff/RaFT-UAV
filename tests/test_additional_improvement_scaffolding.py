from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISAdaptiveAcceleration,
)
from raft_uav.baselines.hypothesis_mixture import GaussianHypothesis, moment_match_hypotheses
from raft_uav.baselines.radar_track_features import add_track_level_features
from raft_uav.calibration.time_offset_state import OnlineTimeOffsetEstimator, apply_time_offset
from raft_uav.evaluation.oracle_gap_decomposition import (
    OracleGapConfig,
    decompose_radar_oracle_gap,
    selected_track_stability_metrics,
)
from raft_uav.stress.perturbations import PerturbationConfig, perturb_radar


def test_oracle_gap_decomposition_flags_wrong_selection() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_id": [1, 2],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame({"time_s": [0.0], "east_m": [0.0], "north_m": [0.0], "up_m": [0.0]})
    selected = radar.iloc[[1]].copy()
    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        selected_radar=selected,
        config=OracleGapConfig(plausible_candidate_gate_m=10.0),
    )
    assert rows.loc[0, "category"] == "wrong_candidate_selected"


def test_track_stability_counts_switches() -> None:
    selected = pd.DataFrame({"time_s": [0, 1, 2, 3], "track_id": [1, 1, 2, 1]})
    metrics = selected_track_stability_metrics(selected)
    assert metrics["track_switch_count"] == 2
    assert metrics["unique_selected_track_ids"] == 2


def test_track_features_are_causal() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "frame_index": [0, 1, 2],
            "track_id": [7, 7, 7],
            "east_m": [0.0, 1.0, 3.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [0.2, 0.4, 0.8],
        }
    )
    out = add_track_level_features(radar, window_frames=2)
    assert list(out["track_age_frames"]) == [0.0, 1.0, 2.0]
    assert np.isclose(out.loc[2, "track_catprob_mean_window"], 0.6)


def test_adaptive_process_noise_scales_high_nis() -> None:
    adapter = RollingNISAdaptiveAcceleration(
        AdaptiveProcessNoiseConfig(base_acceleration_std_mps2=4.0, ewma_alpha=1.0)
    )
    adapter.observe(source="radar", measurement_dim=3, nis=12.0)
    assert adapter.acceleration_std_mps2() > 4.0


def test_time_offset_state_updates_and_applies() -> None:
    estimator = OnlineTimeOffsetEstimator(offset_s=0.0, variance_s2=1.0)
    estimator.update_from_position_residual(
        residual_m=np.array([10.0, 0.0]),
        velocity_mps=np.array([10.0, 0.0]),
        measurement_variance_m2=25.0,
    )
    assert estimator.offset_s > 0.0
    shifted = apply_time_offset(pd.DataFrame({"time_s": [1.0]}), offset_s=estimator.offset_s)
    assert shifted.loc[0, "time_s"] > 1.0


def test_hypothesis_mixture_moment_matches() -> None:
    mean, covariance, weights = moment_match_hypotheses(
        [
            GaussianHypothesis(np.array([0.0]), np.array([[1.0]]), log_weight=0.0),
            GaussianHypothesis(np.array([2.0]), np.array([[1.0]]), log_weight=0.0),
        ]
    )
    assert np.isclose(mean[0], 1.0)
    assert covariance[0, 0] > 1.0
    assert np.allclose(weights, [0.5, 0.5])


def test_stress_perturbation_injects_false_tracks() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0],
            "frame_index": [0],
            "track_id": [1],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
            "cat_prob_uav": [0.9],
        }
    )
    out = perturb_radar(radar, PerturbationConfig(name="false", false_tracks_per_frame=2, seed=1))
    assert len(out) == 3
    assert out["stress_false_track"].sum() == 2
