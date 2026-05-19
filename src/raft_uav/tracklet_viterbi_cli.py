"""Canonical command-line entry point with tracklet-Viterbi association enabled.

The installed ``raft-uav`` command routes through this module.  It reuses
:mod:`raft_uav.cli`, registers ``tracklet-viterbi`` as an additional radar
association mode, and forwards all non-tracklet modes to the base dispatcher.
The ``raft-uav-tracklet-viterbi`` command remains as a compatibility alias for
older experiment notes.

Controlled ablation runs can select the base, retention-aware,
range-covariance-aware, or fixed-lag implementation through wrapper-only
command-line arguments or matching environment variables. The wrapper strips
those ``--tracklet-*`` arguments before forwarding to the shared base CLI
parser.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import pandas as pd

from raft_uav import cli as _base_cli
from raft_uav.baselines import tracklet_viterbi_fixed_lag as _fixed_lag_tracklet_viterbi
from raft_uav.baselines import (
    tracklet_viterbi_range_covariance as _range_covariance_tracklet_viterbi,
)
from raft_uav.baselines import tracklet_viterbi_retention as _retention_tracklet_viterbi
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import (
    RADAR_ASSOCIATION_MODES as _BASE_RADAR_ASSOCIATION_MODES,
    RADAR_MEASUREMENT_MODELS,
    run_async_cv_baseline_with_radar_association as _base_radar_association_runner,
)
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_result import (
    TrackletViterbiResult,
    run_async_cv_baseline_with_tracklet_viterbi_result as _run_base_tracklet_viterbi_result,
)

_TRACKLET_MODE = "tracklet-viterbi"
_TRACKLET_VARIANT_ENV = "RAFT_UAV_TRACKLET_VARIANT"
_CATPROB_MODE_ENV = "RAFT_UAV_TRACKLET_CATPROB_RETENTION_MODE"
_BELOW_CATPROB_PENALTY_ENV = "RAFT_UAV_TRACKLET_BELOW_CATPROB_THRESHOLD_PENALTY"
_TRACK_SUPPORT_WEIGHT_ENV = "RAFT_UAV_TRACKLET_SUPPORT_WEIGHT"
_MAX_TRACK_SUPPORT_REWARD_ENV = "RAFT_UAV_TRACKLET_MAX_SUPPORT_REWARD"
_MAX_CANDIDATES_PER_FRAME_ENV = "RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_FRAME"
_PATH_BEAM_WIDTH_ENV = "RAFT_UAV_TRACKLET_PATH_BEAM_WIDTH"
_REPLAY_NIS_WEIGHT_ENV = "RAFT_UAV_TRACKLET_REPLAY_NIS_WEIGHT"
_REPLAY_REJECTION_COST_ENV = "RAFT_UAV_TRACKLET_REPLAY_REJECTION_COST"
_REPLAY_ROUGHNESS_WEIGHT_ENV = "RAFT_UAV_TRACKLET_REPLAY_ROUGHNESS_WEIGHT"
_MAX_CANDIDATE_POOL_ENV = "RAFT_UAV_TRACKLET_MAX_CANDIDATE_POOL_PER_FRAME"
_MAX_CANDIDATES_PER_TRACK_ENV = "RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_TRACK_ID"
_VITERBI_LAG_S_ENV = "RAFT_UAV_TRACKLET_VITERBI_LAG_S"
_RF_ANCHOR_MODE_ENV = "RAFT_UAV_TRACKLET_RF_ANCHOR_MODE"
_RADAR_MEASUREMENT_MODEL_ENV = "RAFT_UAV_RADAR_MEASUREMENT_MODEL"
_TRACKLET_VARIANTS = ("base", "retention", "range-covariance", "fixed-lag")
_CATPROB_RETENTION_MODES = ("soft", "hard")
_RF_ANCHOR_MODES = ("causal", "smoothed")
_LAST_TRACKLET_VITERBI_RESULT: TrackletViterbiResult | None = None


class _TrackletConfigOverlay:
    """Expose base Viterbi config fields plus experiment-only extension fields."""

    def __init__(self, base: TrackletViterbiAssociationConfig, **overrides: object) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def enabled_radar_association_modes() -> tuple[str, ...]:
    """Return base radar association modes plus the canonical tracklet mode."""

    return tuple(dict.fromkeys((*_BASE_RADAR_ASSOCIATION_MODES, _TRACKLET_MODE)))


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
    rf_anchor_weight: float = 0.35,
    rf_anchor_time_gate_s: float = 2.0,
    rf_anchor_nis_cap: float = 25.0,
    rf_anchor_gate_nis: float = 25.0,
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
    stable_segment_interpolation_max_gap_s: float | None = 5.0,
    stable_segment_interpolation_max_speed_mps: float | None = 65.0,
    stable_segment_interpolation_std_scale: float = 2.0,
    stable_segment_interpolation_gap_std_mps: float = 12.0,
    stable_segment_rf_score_weight: float = 1.0,
    stable_segment_rf_time_gate_s: float = 2.0,
    stable_segment_rf_nis_cap: float = 25.0,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
    tracker_factory: Callable[..., object] | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Dispatch to the tracklet-Viterbi runner when requested."""

    if association == _TRACKLET_MODE:
        global _LAST_TRACKLET_VITERBI_RESULT
        _LAST_TRACKLET_VITERBI_RESULT = None
        del truth, track_switch_nis_ratio, geometry_velocity_std_mps
        del geometry_velocity_weight, geometry_switch_penalty, geometry_catprob_weight
        del rf_anchor_weight, rf_anchor_time_gate_s, rf_anchor_nis_cap
        del rf_anchor_gate_nis
        del pda_nis_temperature, pda_catprob_exponent, track_bank_max_hypotheses
        del track_bank_max_assignments, track_bank_max_candidates, track_bank_gate_probability
        del track_bank_detection_probability, track_bank_clutter_intensity
        del track_bank_prune_log_weight_delta, stable_segment_min_frames
        del stable_segment_max_transition_speed_mps, stable_segment_range_gate_m
        del stable_segment_interpolation_max_gap_s
        del stable_segment_interpolation_max_speed_mps
        del stable_segment_interpolation_std_scale
        del stable_segment_interpolation_gap_std_mps
        del stable_segment_rf_score_weight, stable_segment_rf_time_gate_s
        del stable_segment_rf_nis_cap
        del truth_gate_m, truth_time_gate_s
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
            tracker_factory=tracker_factory,
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
        rf_anchor_weight=rf_anchor_weight,
        rf_anchor_time_gate_s=rf_anchor_time_gate_s,
        rf_anchor_nis_cap=rf_anchor_nis_cap,
        rf_anchor_gate_nis=rf_anchor_gate_nis,
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
        stable_segment_interpolation_max_gap_s=stable_segment_interpolation_max_gap_s,
        stable_segment_interpolation_max_speed_mps=stable_segment_interpolation_max_speed_mps,
        stable_segment_interpolation_std_scale=stable_segment_interpolation_std_scale,
        stable_segment_interpolation_gap_std_mps=stable_segment_interpolation_gap_std_mps,
        stable_segment_rf_score_weight=stable_segment_rf_score_weight,
        stable_segment_rf_time_gate_s=stable_segment_rf_time_gate_s,
        stable_segment_rf_nis_cap=stable_segment_rf_nis_cap,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
        tracker_factory=tracker_factory,
    )


def _tracklet_runner_from_environment() -> Callable[
    ..., tuple[list[dict[str, object]], pd.DataFrame]
]:
    variant = os.environ.get(_TRACKLET_VARIANT_ENV, "fixed-lag").strip().lower()
    if variant == "base":
        return _run_base_tracklet_viterbi_association
    if variant == "retention":
        return _retention_tracklet_viterbi.run_async_cv_baseline_with_tracklet_viterbi_association
    if variant == "range-covariance":
        return _range_covariance_tracklet_viterbi.run_async_cv_baseline_with_tracklet_viterbi_association
    if variant == "fixed-lag":
        return _run_fixed_lag_tracklet_viterbi_association
    raise ValueError(
        f"{_TRACKLET_VARIANT_ENV} must be one of {_TRACKLET_VARIANTS}; got {variant!r}"
    )


def _run_base_tracklet_viterbi_association(
    **kwargs: object,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    global _LAST_TRACKLET_VITERBI_RESULT

    result = _run_base_tracklet_viterbi_result(**kwargs)
    _LAST_TRACKLET_VITERBI_RESULT = result
    return result.records, result.accepted_radar


def _run_fixed_lag_tracklet_viterbi_association(
    **kwargs: object,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    global _LAST_TRACKLET_VITERBI_RESULT

    result = (
        _fixed_lag_tracklet_viterbi
        .run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_result(
            lag_s=_env_float(_VITERBI_LAG_S_ENV, 20.0),
            **kwargs,
        )
    )
    _LAST_TRACKLET_VITERBI_RESULT = result
    return result.records, result.accepted_radar


def _tracklet_config_from_environment() -> _TrackletConfigOverlay:
    base = TrackletViterbiAssociationConfig()
    return _TrackletConfigOverlay(
        base,
        max_candidates_per_frame=_env_int(
            _MAX_CANDIDATES_PER_FRAME_ENV,
            int(base.max_candidates_per_frame),
        ),
        path_beam_width=_env_int(_PATH_BEAM_WIDTH_ENV, int(base.path_beam_width)),
        replay_nis_weight=_env_float(_REPLAY_NIS_WEIGHT_ENV, float(base.replay_nis_weight)),
        replay_rejection_cost=_env_float(_REPLAY_REJECTION_COST_ENV, float(base.replay_rejection_cost)),
        replay_roughness_weight=_env_float(_REPLAY_ROUGHNESS_WEIGHT_ENV, float(base.replay_roughness_weight)),
        catprob_retention_mode=_env_str(_CATPROB_MODE_ENV, "soft"),
        below_catprob_threshold_penalty=_env_float(_BELOW_CATPROB_PENALTY_ENV, 3.0),
        track_support_weight=_env_float(_TRACK_SUPPORT_WEIGHT_ENV, 0.45),
        max_track_support_reward=_env_float(_MAX_TRACK_SUPPORT_REWARD_ENV, 4.0),
        max_candidate_pool_per_frame=_env_int(_MAX_CANDIDATE_POOL_ENV, 24),
        max_candidates_per_track_id=_env_int(_MAX_CANDIDATES_PER_TRACK_ENV, 1),
        rf_anchor_mode=_env_str(_RF_ANCHOR_MODE_ENV, str(base.rf_anchor_mode)),
    )


def _tracklet_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tracklet-variant", choices=_TRACKLET_VARIANTS)
    parser.add_argument("--radar-measurement-model", choices=RADAR_MEASUREMENT_MODELS)
    parser.add_argument("--tracklet-catprob-retention-mode", choices=_CATPROB_RETENTION_MODES)
    parser.add_argument(
        "--tracklet-below-catprob-threshold-penalty",
        type=_nonnegative_float,
    )
    parser.add_argument("--tracklet-track-support-weight", type=_nonnegative_float)
    parser.add_argument("--tracklet-max-track-support-reward", type=_nonnegative_float)
    parser.add_argument("--tracklet-max-candidates-per-frame", type=_positive_int)
    parser.add_argument("--tracklet-path-beam-width", type=_positive_int)
    parser.add_argument("--tracklet-replay-nis-weight", type=_nonnegative_float)
    parser.add_argument("--tracklet-replay-rejection-cost", type=_nonnegative_float)
    parser.add_argument("--tracklet-replay-roughness-weight", type=_nonnegative_float)
    parser.add_argument("--tracklet-max-candidate-pool-per-frame", type=_positive_int)
    parser.add_argument("--tracklet-max-candidates-per-track-id", type=_positive_int)
    parser.add_argument("--tracklet-viterbi-lag-s", type=_positive_float)
    parser.add_argument("--tracklet-rf-anchor-mode", choices=_RF_ANCHOR_MODES)
    return parser


def _extract_tracklet_args(argv: list[str] | None) -> tuple[list[str], dict[str, str]]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    namespace, remaining = _tracklet_parser().parse_known_args(raw_argv)
    updates = _environment_updates_from_namespace(namespace)
    return remaining, updates


def _environment_updates_from_namespace(namespace: argparse.Namespace) -> dict[str, str]:
    updates: dict[str, str] = {}
    _maybe_add(updates, _TRACKLET_VARIANT_ENV, namespace.tracklet_variant)
    _maybe_add(updates, _RADAR_MEASUREMENT_MODEL_ENV, namespace.radar_measurement_model)
    _maybe_add(updates, _CATPROB_MODE_ENV, namespace.tracklet_catprob_retention_mode)
    _maybe_add(
        updates,
        _BELOW_CATPROB_PENALTY_ENV,
        namespace.tracklet_below_catprob_threshold_penalty,
    )
    _maybe_add(updates, _TRACK_SUPPORT_WEIGHT_ENV, namespace.tracklet_track_support_weight)
    _maybe_add(updates, _MAX_TRACK_SUPPORT_REWARD_ENV, namespace.tracklet_max_track_support_reward)
    _maybe_add(updates, _MAX_CANDIDATES_PER_FRAME_ENV, namespace.tracklet_max_candidates_per_frame)
    _maybe_add(updates, _PATH_BEAM_WIDTH_ENV, namespace.tracklet_path_beam_width)
    _maybe_add(updates, _REPLAY_NIS_WEIGHT_ENV, namespace.tracklet_replay_nis_weight)
    _maybe_add(updates, _REPLAY_REJECTION_COST_ENV, namespace.tracklet_replay_rejection_cost)
    _maybe_add(updates, _REPLAY_ROUGHNESS_WEIGHT_ENV, namespace.tracklet_replay_roughness_weight)
    _maybe_add(updates, _MAX_CANDIDATE_POOL_ENV, namespace.tracklet_max_candidate_pool_per_frame)
    _maybe_add(updates, _MAX_CANDIDATES_PER_TRACK_ENV, namespace.tracklet_max_candidates_per_track_id)
    _maybe_add(updates, _VITERBI_LAG_S_ENV, namespace.tracklet_viterbi_lag_s)
    _maybe_add(updates, _RF_ANCHOR_MODE_ENV, namespace.tracklet_rf_anchor_mode)
    return updates


def _maybe_add(updates: dict[str, str], key: str, value: object | None) -> None:
    if value is not None:
        updates[key] = str(value)


@contextmanager
def _temporary_environment(updates: Mapping[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


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


def _run_baseline_output_dir(argv: list[str]) -> Path | None:
    """Return the output directory for a forwarded run-baseline invocation."""

    if len(argv) < 2 or argv[0] != "run-baseline":
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command")
    parser.add_argument("dataset_root")
    parser.add_argument("--flight")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    namespace, _ = parser.parse_known_args(argv)
    if namespace.flight is None:
        return None
    return namespace.output_dir / str(namespace.flight)


def _write_last_tracklet_viterbi_artifacts(argv: list[str]) -> None:
    """Write replay-preserving artifacts for the canonical tracklet runner."""

    result = _LAST_TRACKLET_VITERBI_RESULT
    output_dir = _run_baseline_output_dir(argv)
    if result is None or output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_radar_path = output_dir / "accepted_radar.csv"
    viterbi_selected_radar_path = output_dir / "viterbi_selected_radar.csv"
    radar_candidate_ledger_path = output_dir / "radar_candidate_ledger.csv"
    artifact_summary_path = output_dir / "tracklet_viterbi_artifacts.json"

    result.accepted_radar.to_csv(accepted_radar_path, index=False)
    result.viterbi_selected_radar.to_csv(viterbi_selected_radar_path, index=False)
    result.radar_candidate_ledger.to_csv(radar_candidate_ledger_path, index=False)
    artifact_summary_path.write_text(
        json.dumps(
            {
                "accepted_radar_rows": int(len(result.accepted_radar)),
                "viterbi_selected_radar_rows": int(len(result.viterbi_selected_radar)),
                "radar_candidate_ledger_rows": int(len(result.radar_candidate_ledger)),
                "radar_candidate_ledger_selected_rows": _selected_ledger_rows(
                    result.radar_candidate_ledger
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"accepted_radar_csv={accepted_radar_path}")
    print(f"viterbi_selected_radar_csv={viterbi_selected_radar_path}")
    print(f"radar_candidate_ledger_csv={radar_candidate_ledger_path}")
    print(f"tracklet_viterbi_artifacts_json={artifact_summary_path}")


def _selected_ledger_rows(ledger: pd.DataFrame) -> int:
    if "association_viterbi_selected" not in ledger.columns:
        return 0
    return int(ledger["association_viterbi_selected"].fillna(False).astype(bool).sum())


def main(argv: list[str] | None = None) -> int:
    """Run the standard CLI with tracklet-Viterbi association enabled."""

    filtered_argv, env_updates = _extract_tracklet_args(argv)
    _base_cli.RADAR_ASSOCIATION_MODES = enabled_radar_association_modes()
    _base_cli.run_async_cv_baseline_with_radar_association = (
        run_async_cv_baseline_with_radar_association
    )
    with _temporary_environment(env_updates):
        return_code = _base_cli.main(filtered_argv)
    if return_code == 0:
        _write_last_tracklet_viterbi_artifacts(filtered_argv)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
