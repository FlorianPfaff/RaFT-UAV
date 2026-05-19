"""Heteroscedastic radar covariance for tracklet-Viterbi association.

The base tracklet-Viterbi implementation uses one Cartesian covariance for all
selected radar rows.  This wrapper keeps the existing retention-aware Viterbi
path but patches the replay and RF-anchor scoring hooks so each radar row uses
a candidate-specific ENU covariance.

Rows are annotated through :mod:`raft_uav.baselines.radar_covariance`, whose
``range-angle`` model projects independent range, azimuth, and elevation noise
through the spherical-to-ENU Jacobian.  This makes angular uncertainty grow with
range and preserves orientation-dependent ENU correlations.  The historical
Cartesian covariance and configurable horizontal/vertical floors remain
conservative lower bounds for Kalman gating.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines import radar_association as _radar_association
from raft_uav.baselines import tracklet_viterbi as _base
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_covariance import (
    RADAR_COVARIANCE_COLUMNS,
    RadarCovarianceConfig,
    append_radar_covariance_columns,
    row_radar_covariance,
)
from raft_uav.baselines.tracklet_viterbi_retention import (
    run_async_cv_baseline_with_tracklet_viterbi_association as _run_retention_association,
)

TrackletViterbiAssociationConfig = _base.TrackletViterbiAssociationConfig
DEFAULT_USE_RANGE_ADAPTIVE_RADAR_COVARIANCE = True
DEFAULT_RADAR_RANGE_STD_M = 5.0
DEFAULT_RADAR_RANGE_XY_FLOOR_STD_M = 20.0
DEFAULT_RADAR_RANGE_Z_FLOOR_STD_M = 30.0
# Backwards-compatible legacy names: these are now interpreted as angular stds
# in radians, not as post-hoc Cartesian range multipliers.
DEFAULT_RADAR_RANGE_XY_SCALE = 0.035
DEFAULT_RADAR_RANGE_Z_SCALE = 0.050
DEFAULT_RADAR_AZIMUTH_STD_RAD = DEFAULT_RADAR_RANGE_XY_SCALE
DEFAULT_RADAR_ELEVATION_STD_RAD = DEFAULT_RADAR_RANGE_Z_SCALE
DEFAULT_RADAR_AZIMUTH_STD_DEG = float(np.degrees(DEFAULT_RADAR_AZIMUTH_STD_RAD))
DEFAULT_RADAR_ELEVATION_STD_DEG = float(np.degrees(DEFAULT_RADAR_ELEVATION_STD_RAD))
DEFAULT_RADAR_COVARIANCE_MIN_STD_M = 3.0
DEFAULT_RADAR_COVARIANCE_MAX_STD_M = 250.0


def run_async_cv_baseline_with_tracklet_viterbi_association(
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
    tracker_factory: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run retention-aware Viterbi with heteroscedastic radar covariance."""

    cfg = config or TrackletViterbiAssociationConfig()
    covariance_config = _radar_covariance_config(
        cfg,
        np.diag([float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]),
    )
    if _use_range_adaptive_radar_covariance(cfg) and covariance_config.mode != "fixed":
        radar = append_radar_covariance_columns(radar, covariance_config)
    with _range_adaptive_covariance_hooks(cfg, covariance_config):
        return _run_retention_association(
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
            config=cfg,
            tracker_factory=tracker_factory,
        )


@contextmanager
def _range_adaptive_covariance_hooks(
    config: TrackletViterbiAssociationConfig,
    covariance_config: RadarCovarianceConfig | None = None,
):
    original_candidate_cost_terms = _base._candidate_cost_terms
    original_radar_row_to_measurement = _radar_association._radar_row_to_measurement

    def candidate_cost_terms_with_adaptive_covariance(
        *,
        row: pd.Series,
        position: np.ndarray,
        anchor: _base._AnchorState | None,
        covariance: np.ndarray,
        config: TrackletViterbiAssociationConfig,
    ) -> tuple[float, float, float]:
        row_covariance = _radar_row_covariance(row, covariance, config, covariance_config)
        return original_candidate_cost_terms(
            row=row,
            position=position,
            anchor=anchor,
            covariance=row_covariance,
            config=config,
        )

    def radar_row_to_measurement_with_adaptive_covariance(
        row: pd.Series,
        covariance: np.ndarray,
    ) -> TrackingMeasurement:
        row_covariance = _radar_row_covariance(row, covariance, config, covariance_config)
        _write_radar_covariance_diagnostics(row, row_covariance, covariance)
        return original_radar_row_to_measurement(row, row_covariance)

    _base._candidate_cost_terms = candidate_cost_terms_with_adaptive_covariance
    _radar_association._radar_row_to_measurement = radar_row_to_measurement_with_adaptive_covariance
    try:
        yield
    finally:
        _base._candidate_cost_terms = original_candidate_cost_terms
        _radar_association._radar_row_to_measurement = original_radar_row_to_measurement


def _radar_row_covariance(
    row: pd.Series,
    default_covariance: np.ndarray,
    config: Any,
    covariance_config: RadarCovarianceConfig | None = None,
) -> np.ndarray:
    """Return heteroscedastic ENU radar covariance for one radar row.

    The row covariance comes from the shared ``radar_covariance`` helper, so the
    tracklet-Viterbi path uses the same range-angle model as the runtime radar
    association code.  The projected covariance is floored by the original
    Cartesian covariance and by configurable horizontal/vertical standard
    deviations, preserving the historical conservative lower bound.  If the row
    geometry is unavailable, the fixed covariance is returned unchanged.
    """

    default_covariance = np.asarray(default_covariance, dtype=float)
    if not _use_range_adaptive_radar_covariance(config):
        return default_covariance

    radar_config = covariance_config or _radar_covariance_config(config, default_covariance)
    if radar_config.mode == "fixed":
        return default_covariance

    row_with_covariance = _annotated_row(row, radar_config)
    row_covariance = row_radar_covariance(row_with_covariance, default_covariance)
    if row_covariance is None:
        return default_covariance

    floored_covariance = _apply_cartesian_covariance_floor(
        row_covariance,
        default_covariance,
        config,
    )
    return _symmetric_covariance_or_default(floored_covariance, default_covariance)


def _use_range_adaptive_radar_covariance(config: Any) -> bool:
    return bool(
        getattr(
            config,
            "use_range_adaptive_radar_covariance",
            DEFAULT_USE_RANGE_ADAPTIVE_RADAR_COVARIANCE,
        )
    )


def _annotated_row(row: pd.Series, config: RadarCovarianceConfig) -> pd.Series:
    if all(column in row.index for column in RADAR_COVARIANCE_COLUMNS):
        return row
    frame = pd.DataFrame([row.to_dict()])
    annotated = append_radar_covariance_columns(frame, config)
    if annotated.empty:
        return row
    return annotated.iloc[0]


def _radar_covariance_config(config: Any, default_covariance: np.ndarray) -> RadarCovarianceConfig:
    base = RadarCovarianceConfig.from_environment()
    default_covariance = np.asarray(default_covariance, dtype=float)
    default_xy_std_m = float(
        np.sqrt(max(default_covariance[0, 0], default_covariance[1, 1], 0.0))
    )
    default_z_std_m = float(np.sqrt(max(default_covariance[2, 2], 0.0)))
    return RadarCovarianceConfig(
        mode=str(getattr(config, "radar_covariance_mode", base.mode)),
        xy_std_m=default_xy_std_m,
        z_std_m=default_z_std_m,
        range_std_m=_positive_config_float(config, "radar_range_std_m", base.range_std_m),
        azimuth_std_deg=_angular_config_std_deg(
            config,
            degree_name="radar_azimuth_std_deg",
            radian_name="radar_azimuth_std_rad",
            legacy_name="radar_range_xy_scale",
            default_deg=base.azimuth_std_deg,
        ),
        elevation_std_deg=_angular_config_std_deg(
            config,
            degree_name="radar_elevation_std_deg",
            radian_name="radar_elevation_std_rad",
            legacy_name="radar_range_z_scale",
            default_deg=base.elevation_std_deg,
        ),
        min_std_m=_positive_config_float(
            config,
            "radar_covariance_min_std_m",
            base.min_std_m,
        ),
        max_std_m=_positive_config_float(
            config,
            "radar_covariance_max_std_m",
            base.max_std_m,
        ),
        origin_east_m=_finite_config_float(config, "radar_origin_east_m", base.origin_east_m),
        origin_north_m=_finite_config_float(config, "radar_origin_north_m", base.origin_north_m),
        origin_up_m=_finite_config_float(config, "radar_origin_up_m", base.origin_up_m),
    )


def _apply_cartesian_covariance_floor(
    covariance: np.ndarray,
    default_covariance: np.ndarray,
    config: Any,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    default_covariance = np.asarray(default_covariance, dtype=float)
    horizontal_floor_std_m = _nonnegative_config_float(
        config,
        "radar_range_xy_floor_std_m",
        DEFAULT_RADAR_RANGE_XY_FLOOR_STD_M,
    )
    vertical_floor_std_m = _nonnegative_config_float(
        config,
        "radar_range_z_floor_std_m",
        DEFAULT_RADAR_RANGE_Z_FLOOR_STD_M,
    )
    floor_variances = np.array(
        [
            max(float(default_covariance[0, 0]), horizontal_floor_std_m**2),
            max(float(default_covariance[1, 1]), horizontal_floor_std_m**2),
            max(float(default_covariance[2, 2]), vertical_floor_std_m**2),
        ],
        dtype=float,
    )
    missing_variance = np.maximum(floor_variances - np.diag(covariance), 0.0)
    return covariance + np.diag(missing_variance)


def _symmetric_covariance_or_default(
    covariance: np.ndarray,
    default_covariance: np.ndarray,
) -> np.ndarray:
    covariance = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    if not np.isfinite(covariance).all():
        return np.asarray(default_covariance, dtype=float)
    return covariance


def _positive_config_float(config: Any, name: str, default: float) -> float:
    raw_value = getattr(config, name, default)
    if raw_value is None:
        raw_value = default
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_config_float(config: Any, name: str, default: float) -> float:
    raw_value = getattr(config, name, default)
    if raw_value is None:
        raw_value = default
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite_config_float(config: Any, name: str, default: float) -> float:
    raw_value = getattr(config, name, default)
    if raw_value is None:
        raw_value = default
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _angular_config_std_deg(
    config: Any,
    *,
    degree_name: str,
    radian_name: str,
    legacy_name: str,
    default_deg: float,
) -> float:
    degree_value = getattr(config, degree_name, None)
    radian_value = getattr(config, radian_name, None)
    legacy_value = getattr(config, legacy_name, None)
    if degree_value is not None:
        return _positive_config_float(config, degree_name, default_deg)
    if radian_value is not None:
        return float(np.degrees(_positive_config_float(config, radian_name, np.radians(default_deg))))
    if legacy_value is not None:
        return float(np.degrees(_positive_config_float(config, legacy_name, np.radians(default_deg))))
    return float(default_deg)


def _write_radar_covariance_diagnostics(
    row: pd.Series,
    row_covariance: np.ndarray,
    default_covariance: np.ndarray,
) -> None:
    """Attach selected-row covariance diagnostics for ablation analysis."""

    row_covariance = np.asarray(row_covariance, dtype=float)
    default_covariance = np.asarray(default_covariance, dtype=float)
    row["association_radar_std_east_m"] = float(np.sqrt(max(row_covariance[0, 0], 0.0)))
    row["association_radar_std_north_m"] = float(np.sqrt(max(row_covariance[1, 1], 0.0)))
    row["association_radar_std_up_m"] = float(np.sqrt(max(row_covariance[2, 2], 0.0)))
    row["association_radar_xy_std_m"] = max(
        float(row["association_radar_std_east_m"]),
        float(row["association_radar_std_north_m"]),
    )
    row["association_radar_z_std_m"] = float(row["association_radar_std_up_m"])
    row["association_radar_cov_en_m2"] = float(row_covariance[0, 1])
    row["association_radar_cov_eu_m2"] = float(row_covariance[0, 2])
    row["association_radar_cov_nu_m2"] = float(row_covariance[1, 2])
    row["association_radar_covariance_model"] = "range-angle"
    row["association_radar_covariance_adaptive"] = bool(
        not np.allclose(row_covariance, default_covariance)
    )
