from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.fifth_wave_diagnostics import (
    adaptive_smoothing_lag_s,
    block_bootstrap_interval,
    candidate_ambiguity_index,
    conservative_leaderboard_rank,
    deterministic_artifact_summary,
    do_no_harm_radar_decision,
    estimate_error_frame,
    method_family_ensemble_decision,
    paired_delta_summary,
    paired_error_delta_frame,
    pseudo_label_candidates,
    recovery_events,
    residual_whiteness_summary,
    track_purity_summary,
    vertical_horizontal_error_summary,
)


def _trajectory(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "east_m": [0.0 + offset, 1.0 + offset, 2.0 + offset, 3.0 + offset],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
            "source": ["rf", "radar", "radar", "rf"],
        }
    )


def test_block_bootstrap_interval_and_paired_delta() -> None:
    ci = block_bootstrap_interval(np.arange(10.0), metric="mean", block_size=2, resamples=50, seed=1)
    assert ci.samples == 10
    assert ci.lower <= ci.estimate <= ci.upper

    truth = _trajectory(0.0)
    a = _trajectory(1.0)
    b = _trajectory(2.0)
    deltas = paired_error_delta_frame(a, b, truth, max_time_delta_s=0.1)
    assert len(deltas) == 4
    summary = paired_delta_summary(deltas, block_size=2, resamples=50)
    assert summary["fraction_a_better"] == 1.0


def test_do_no_harm_and_ambiguity() -> None:
    decision = do_no_harm_radar_decision(
        association_nis=20.0,
        gate_threshold=10.0,
        association_confidence=0.2,
        candidate_entropy=1.5,
        rf_anchor_nis=30.0,
        rf_anchor_gate_nis=10.0,
    )
    assert decision.action in {"skip", "defer"}
    assert not decision.should_apply or decision.action == "soften"

    candidates = pd.DataFrame({"association_score": [0.1, 0.2, 2.0]})
    ambiguity = candidate_ambiguity_index(candidates)
    assert ambiguity["candidate_count"] == 3
    assert ambiguity["effective_candidate_count"] > 1.0


def test_recovery_vertical_track_purity_and_error_alignment() -> None:
    truth = _trajectory(0.0)
    estimates = _trajectory(0.0)
    estimates.loc[2, "up_m"] = 10.0
    errors = estimate_error_frame(estimates, truth, max_time_delta_s=0.1)
    assert errors["error_3d_m"].max() == 10.0
    vertical = vertical_horizontal_error_summary(estimates, truth, max_time_delta_s=0.1)
    assert vertical["up_rmse_m"] > 0.0

    events = recovery_events([0, 1, 2, 3], [1, 20, 30, 2], threshold_m=10.0)
    assert len(events) == 1
    assert events.iloc[0]["max_error_m"] == 30.0

    purity = track_purity_summary(pd.DataFrame({"track_id": [4, 4, 5]}))
    assert purity["dominant_track_id"] == 4
    assert np.isclose(purity["dominant_track_fraction"], 2 / 3)


def test_conservative_rank_determinism_and_pseudo_labels() -> None:
    rows = pd.DataFrame(
        {
            "method": ["a", "b", "c"],
            "p95_3d_error_m": [5.0, 4.0, 2.0],
            "truth_coverage_rate": [0.96, 0.8, 0.97],
        }
    )
    ranked = conservative_leaderboard_rank(
        rows,
        objective="p95_3d_error_m",
        constraints={"truth_coverage_rate": ("ge", 0.95)},
    )
    assert ranked.loc[ranked["method"] == "c", "robust_rank"].iloc[0] == 1
    assert not bool(ranked.loc[ranked["method"] == "b", "eligible"].iloc[0])

    det = deterministic_artifact_summary(_trajectory(), _trajectory())
    assert det["estimates_nearly_equal"]

    labeled = pseudo_label_candidates(
        pd.DataFrame({"cat_prob_uav": [0.95, 0.05, 0.5], "association_anchor_nis": [1.0, 30.0, 5.0]})
    )
    assert labeled["pseudo_label"].notna().sum() == 2


def test_whiteness_and_ensemble_helpers() -> None:
    diagnostics = pd.DataFrame({"source": ["rf"] * 6, "nis": [1, 2, 1, 2, 1, 2]})
    whiteness = residual_whiteness_summary(diagnostics, max_lag=3)
    assert len(whiteness) == 1
    assert "ljung_box_q" in whiteness.columns

    assert adaptive_smoothing_lag_s(candidate_entropy=2.0, miss_streak=4) > 2.0
    assert method_family_ensemble_decision(stable_segment_available=True) == "stable_segments"
    assert method_family_ensemble_decision(recovery_mode=True) == "rf_fallback_or_recovery"
