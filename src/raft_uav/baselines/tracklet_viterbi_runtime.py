"""Runtime integration for first-class tracklet-Viterbi association.

This keeps the standard ``raft-uav run-baseline`` entry point usable with
``--radar-association tracklet-viterbi`` without requiring the compatibility
``raft-uav-tracklet-viterbi`` wrapper.
"""

from __future__ import annotations

from typing import Any
import os

import pandas as pd

from raft_uav.baselines.tracklet_viterbi_retention import (
    TrackletViterbiAssociationConfig,
    run_async_cv_baseline_with_tracklet_viterbi_association,
)

_TRACKLET_MODE = "tracklet-viterbi"
_INSTALLED = False
_ORIGINAL_RUNNER: Any = None


def install() -> None:
    """Register tracklet-Viterbi as a normal radar association mode."""

    global _INSTALLED, _ORIGINAL_RUNNER
    if _INSTALLED:
        return

    from raft_uav.baselines import radar_association

    _ORIGINAL_RUNNER = radar_association.run_async_cv_baseline_with_radar_association
    if _TRACKLET_MODE not in radar_association.RADAR_ASSOCIATION_MODES:
        radar_association.RADAR_ASSOCIATION_MODES = (
            *radar_association.RADAR_ASSOCIATION_MODES,
            _TRACKLET_MODE,
        )
    radar_association.run_async_cv_baseline_with_radar_association = (
        _run_async_cv_baseline_with_tracklet_dispatch
    )
    _INSTALLED = True


def _run_async_cv_baseline_with_tracklet_dispatch(
    *,
    rf_measurements: Any,
    radar: pd.DataFrame,
    association: str,
    truth: pd.DataFrame | None = None,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    gate_probabilities_by_source: Any = None,
    gate_thresholds_by_source: Any = None,
    safety_gate_probabilities_by_source: Any = None,
    safety_gate_thresholds_by_source: Any = None,
    robust_update_by_source: Any = None,
    inflation_alpha_by_source: Any = None,
    max_residual_norms_by_source: Any = None,
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
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Dispatch standard association modes or the tracklet-Viterbi runner."""

    if association == _TRACKLET_MODE:
        del truth, track_switch_nis_ratio, geometry_velocity_std_mps
        del geometry_velocity_weight, geometry_switch_penalty, geometry_catprob_weight
        del pda_nis_temperature, pda_catprob_exponent, track_bank_max_hypotheses
        del track_bank_max_assignments, track_bank_max_candidates, track_bank_gate_probability
        del track_bank_detection_probability, track_bank_clutter_intensity
        del track_bank_prune_log_weight_delta, truth_gate_m, truth_time_gate_s
        return run_async_cv_baseline_with_tracklet_viterbi_association(
            rf_measurements=list(rf_measurements),
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
            config=_config_from_environment(),
        )

    if _ORIGINAL_RUNNER is None:
        raise RuntimeError("tracklet-viterbi runtime was not installed correctly")
    return _ORIGINAL_RUNNER(
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
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
    )


def _config_from_environment() -> TrackletViterbiAssociationConfig:
    """Read optional ``RAFT_UAV_TRACKLET_*`` tuning values."""

    range_gate = _env_float("RAFT_UAV_TRACKLET_RANGE_GATE_M", 850.0)
    return TrackletViterbiAssociationConfig(
        max_candidates_per_frame=int(_env_float("RAFT_UAV_TRACKLET_MAX_CANDIDATES", 8.0)),
        missed_detection_cost=_env_float("RAFT_UAV_TRACKLET_MISSED_DETECTION_COST", 7.0),
        consecutive_miss_cost=_env_float("RAFT_UAV_TRACKLET_CONSECUTIVE_MISS_COST", 1.0),
        track_switch_cost=_env_float("RAFT_UAV_TRACKLET_TRACK_SWITCH_COST", 8.0),
        missing_track_id_cost=_env_float("RAFT_UAV_TRACKLET_MISSING_TRACK_ID_COST", 1.0),
        catprob_weight=_env_float("RAFT_UAV_TRACKLET_CATPROB_WEIGHT", 2.5),
        anchor_nis_weight=_env_float("RAFT_UAV_TRACKLET_ANCHOR_NIS_WEIGHT", 0.35),
        transition_nis_weight=_env_float("RAFT_UAV_TRACKLET_TRANSITION_NIS_WEIGHT", 1.0),
        velocity_nis_weight=_env_float("RAFT_UAV_TRACKLET_VELOCITY_NIS_WEIGHT", 0.15),
        transition_position_std_m=_env_float("RAFT_UAV_TRACKLET_TRANSITION_POSITION_STD_M", 40.0),
        transition_speed_std_mps=_env_float("RAFT_UAV_TRACKLET_TRANSITION_SPEED_STD_MPS", 18.0),
        velocity_std_mps=_env_float("RAFT_UAV_TRACKLET_VELOCITY_STD_MPS", 12.0),
        max_speed_mps=_env_float("RAFT_UAV_TRACKLET_MAX_SPEED_MPS", 55.0),
        max_speed_penalty=_env_float("RAFT_UAV_TRACKLET_MAX_SPEED_PENALTY", 10.0),
        range_gate_m=None if range_gate <= 0.0 else range_gate,
        range_gate_slack_m=_env_float("RAFT_UAV_TRACKLET_RANGE_GATE_SLACK_M", 150.0),
        range_penalty=_env_float("RAFT_UAV_TRACKLET_RANGE_PENALTY", 10.0),
        use_rf_anchor=_env_bool("RAFT_UAV_TRACKLET_USE_RF_ANCHOR", True),
    )


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}
