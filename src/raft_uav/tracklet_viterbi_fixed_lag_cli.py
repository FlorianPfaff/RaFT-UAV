"""Command-line wrapper for fixed-lag tracklet-Viterbi radar association.

The standard ``raft-uav`` CLI does not yet expose an online/fixed-lag variant
of tracklet Viterbi. This wrapper registers ``tracklet-viterbi-fixed-lag`` in
process and dispatches it to the fixed-lag runner while preserving the existing
CLI surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import os

import pandas as pd

from raft_uav import cli as _base_cli
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import (
    RADAR_ASSOCIATION_MODES as _BASE_RADAR_ASSOCIATION_MODES,
    run_async_cv_baseline_with_radar_association as _base_radar_association_runner,
)
from raft_uav.baselines.tracklet_viterbi_fixed_lag import (
    run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay,
)
from raft_uav.baselines.tracklet_viterbi_retention import (
    _track_aware_node_builder,
    _track_support_by_id,
)

_TRACKLET_FIXED_LAG_MODE = "tracklet-viterbi-fixed-lag"
_FIXED_LAG_ENV = "RAFT_UAV_TRACKLET_VITERBI_LAG_S"
_DEFAULT_FIXED_LAG_S = 20.0


def run_async_cv_baseline_with_radar_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    association: str,
    truth: pd.DataFrame | None = None,
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
    track_switch_nis_ratio: float = 0.5,
    candidate_catprob_threshold: float | None = 0.5,
    geometry_velocity_std_mps: float = 12.0,
    geometry_velocity_weight: float = 0.25,
    geometry_switch_penalty: float = 4.0,
    geometry_catprob_weight: float = 2.0,
    pda_nis_temperature: float = 1.0,
    pda_catprob_exponent: float = 1.0,
    track_bank_max_hypotheses: int = 16,
    track_bank_max_assignments: int = 16,
    track_bank_max_candidates: int = 16,
    track_bank_gate_probability: float = 0.9999999,
    track_bank_detection_probability: float = 0.999,
    track_bank_clutter_intensity: float = 1.0e-12,
    track_bank_prune_log_weight_delta: float = 80.0,
    stable_segment_min_frames: int = 100,
    stable_segment_max_transition_speed_mps: float = 65.0,
    stable_segment_range_gate_m: float | None = 800.0,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Dispatch to fixed-lag tracklet Viterbi when requested."""

    if association == _TRACKLET_FIXED_LAG_MODE:
        del truth, track_switch_nis_ratio, geometry_velocity_std_mps
        del geometry_velocity_weight, geometry_switch_penalty, geometry_catprob_weight
        del pda_nis_temperature, pda_catprob_exponent, track_bank_max_hypotheses
        del track_bank_max_assignments, track_bank_max_candidates, track_bank_gate_probability
        del track_bank_detection_probability, track_bank_clutter_intensity
        del track_bank_prune_log_weight_delta, stable_segment_min_frames
        del stable_segment_max_transition_speed_mps, stable_segment_range_gate_m
        del truth_gate_m, truth_time_gate_s
        with _track_aware_node_builder(_track_support_by_id(radar)):
            records, accepted, _replayed = (
                run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
                    rf_measurements=list(rf_measurements),
                    radar=radar,
                    lag_s=_fixed_lag_s_from_env(),
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
                )
            )
        return records, accepted

    return _base_radar_association_runner(
        rf_measurements=rf_measurements,
        radar=radar,
        association=association,
        truth=truth,
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
        track_switch_nis_ratio=track_switch_nis_ratio,
        candidate_catprob_threshold=candidate_catprob_threshold,
        geometry_velocity_std_mps=geometry_velocity_std_mps,
        geometry_velocity_weight=geometry_velocity_weight,
        geometry_switch_penalty=geometry_switch_penalty,
        geometry_catprob_weight=geometry_catprob_weight,
        pda_nis_temperature=pda_nis_temperature,
        pda_catprob_exponent=pda_catprob_exponent,
        track_bank_max_hypotheses=track_bank_max_hypotheses,
        track_bank_max_assignments=track_bank_max_assignments,
        track_bank_max_candidates=track_bank_max_candidates,
        track_bank_gate_probability=track_bank_gate_probability,
        track_bank_detection_probability=track_bank_detection_probability,
        track_bank_clutter_intensity=track_bank_clutter_intensity,
        track_bank_prune_log_weight_delta=track_bank_prune_log_weight_delta,
        stable_segment_min_frames=stable_segment_min_frames,
        stable_segment_max_transition_speed_mps=stable_segment_max_transition_speed_mps,
        stable_segment_range_gate_m=stable_segment_range_gate_m,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the standard CLI with fixed-lag tracklet Viterbi enabled."""

    modes = tuple(dict.fromkeys((*_BASE_RADAR_ASSOCIATION_MODES, _TRACKLET_FIXED_LAG_MODE)))
    _base_cli.RADAR_ASSOCIATION_MODES = modes
    _base_cli.run_async_cv_baseline_with_radar_association = (
        run_async_cv_baseline_with_radar_association
    )
    return _base_cli.main(argv)


def _fixed_lag_s_from_env() -> float:
    value = os.environ.get(_FIXED_LAG_ENV)
    if value is None or value.strip() == "":
        return _DEFAULT_FIXED_LAG_S
    lag_s = float(value)
    if lag_s <= 0.0:
        raise ValueError(f"{_FIXED_LAG_ENV} must be positive")
    return lag_s


if __name__ == "__main__":
    raise SystemExit(main())
