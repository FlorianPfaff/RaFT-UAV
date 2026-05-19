"""Deprecated compatibility wrapper for replay-preserving tracklet-Viterbi results.

Use :func:`raft_uav.baselines.tracklet_viterbi_result.run_async_cv_baseline_with_tracklet_viterbi_result`
for new code.  This module remains as a tuple-returning compatibility shim for
older scripts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import warnings

import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_result import (
    run_async_cv_baseline_with_tracklet_viterbi_result,
)


def run_async_cv_baseline_with_tracklet_viterbi_association_and_replay(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    candidate_catprob_threshold: float | None = 0.4,
    config: TrackletViterbiAssociationConfig | None = None,
    replay_tracker_kind: str = "cv",
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    """Return legacy tuple outputs from the canonical result-object API."""

    warnings.warn(
        "run_async_cv_baseline_with_tracklet_viterbi_association_and_replay "
        "is deprecated; use run_async_cv_baseline_with_tracklet_viterbi_result "
        "and its TrackletViterbiResult return object instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = run_async_cv_baseline_with_tracklet_viterbi_result(
        rf_measurements=rf_measurements,
        radar=radar,
        acceleration_std_mps2=acceleration_std_mps2,
        radar_xy_std_m=radar_xy_std_m,
        radar_z_std_m=radar_z_std_m,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=config,
        replay_tracker_kind=replay_tracker_kind,
    )
    return result.records, result.accepted_radar, result.viterbi_selected_radar
