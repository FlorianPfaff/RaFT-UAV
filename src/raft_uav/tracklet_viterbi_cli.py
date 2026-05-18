"""Command-line wrapper that enables the tracklet-Viterbi radar association mode.

This module reuses :mod:`raft_uav.cli` and patches only its in-process radar
association dispatcher before argument parsing.  It keeps the default
``raft-uav`` entry point unchanged while exposing the experimental
sequence-level association method through ``raft-uav-tracklet-viterbi`` or
``python -m raft_uav.tracklet_viterbi_cli``.

Controlled ablation runs can select the base or retention-aware implementation
through environment variables. This avoids invasive changes to the shared
``raft_uav.cli`` parser while still letting experiment scripts isolate the
effect of retention, soft class-probability handling, and track-support priors.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import os

import pandas as pd

from raft_uav import cli as _base_cli
from raft_uav.baselines import tracklet_viterbi as _base_tracklet_viterbi
from raft_uav.baselines import tracklet_viterbi_retention as _retention_tracklet_viterbi
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import (
    RADAR_ASSOCIATION_MODES as _BASE_RADAR_ASSOCIATION_MODES,
    run_async_cv_baseline_with_radar_association as _base_radar_association_runner,
)
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig

_TRACKLET_MODE = "tracklet-viterbi"
_TRACKLET_VARIANT_ENV = "RAFT_UAV_TRACKLET_VARIANT"
_CATPROB_MODE_ENV = "RAFT_UAV_TRACKLET_CATPROB_RETENTION_MODE"
_BELOW_CATPROB_PENALTY_ENV = "RAFT_UAV_TRACKLET_BELOW_CATPROB_THRESHOLD_PENALTY"
_TRACK_SUPPORT_WEIGHT_ENV = "RAFT_UAV_TRACKLET_SUPPORT_WEIGHT"
_MAX_TRACK_SUPPORT_REWARD_ENV = "RAFT_UAV_TRACKLET_MAX_SUPPORT_REWARD"
_MAX_CANDIDATE_POOL_ENV = "RAFT_UAV_TRACKLET_MAX_CANDIDATE_POOL_PER_FRAME"
_MAX_CANDIDATES_PER_TRACK_ENV = "RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_TRACK_ID"
_TRACKLET_VARIANTS = ("base", "retention")


class _TrackletConfigOverlay:
    """Expose base Viterbi config fields plus experiment-only extension fields."""

    def __init__(self, base: TrackletViterbiAssociationConfig, **overrides: object) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


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
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Dispatch to the experimental tracklet-Viterbi runner when requested."""

    if association == _TRACKLET_MODE:
        del truth, track_switch_nis_ratio, geometry_velocity_std_mps
        del geometry_velocity_weight, geometry_switch_penalty, geometry_catprob_weight
        del pda_nis_temperature, pda_catprob_exponent, track_bank_max_hypotheses
        del track_bank_max_assignments, track_bank_max_candidates, track_bank_gate_probability
        del track_bank_detection_probability, track_bank_clutter_intensity
        del track_bank_prune_log_weight_delta, truth_gate_m, truth_time_gate_s
        runner = _tracklet_runner_from_environment()
        config = _tracklet_config_from_environment()
        return runner(
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
            config=config,
        )

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
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
    )


def _tracklet_runner_from_environment() -> Callable[
    ..., tuple[list[dict[str, object]], pd.DataFrame]
]:
    variant = os.environ.get(_TRACKLET_VARIANT_ENV, "retention").strip().lower()
    if variant == "base":
        return _base_tracklet_viterbi.run_async_cv_baseline_with_tracklet_viterbi_association
    if variant == "retention":
        return _retention_tracklet_viterbi.run_async_cv_baseline_with_tracklet_viterbi_association
    raise ValueError(
        f"{_TRACKLET_VARIANT_ENV} must be one of {_TRACKLET_VARIANTS}; got {variant!r}"
    )


def _tracklet_config_from_environment() -> _TrackletConfigOverlay:
    return _TrackletConfigOverlay(
        TrackletViterbiAssociationConfig(),
        catprob_retention_mode=_env_str(_CATPROB_MODE_ENV, "soft"),
        below_catprob_threshold_penalty=_env_float(_BELOW_CATPROB_PENALTY_ENV, 3.0),
        track_support_weight=_env_float(_TRACK_SUPPORT_WEIGHT_ENV, 0.45),
        max_track_support_reward=_env_float(_MAX_TRACK_SUPPORT_REWARD_ENV, 4.0),
        max_candidate_pool_per_frame=_env_int(_MAX_CANDIDATE_POOL_ENV, 24),
        max_candidates_per_track_id=_env_int(_MAX_CANDIDATES_PER_TRACK_ENV, 1),
    )


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def main(argv: list[str] | None = None) -> int:
    """Run the standard CLI with the experimental association mode enabled."""

    modes = tuple(dict.fromkeys((*_BASE_RADAR_ASSOCIATION_MODES, _TRACKLET_MODE)))
    _base_cli.RADAR_ASSOCIATION_MODES = modes
    _base_cli.run_async_cv_baseline_with_radar_association = (
        run_async_cv_baseline_with_radar_association
    )
    return _base_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
