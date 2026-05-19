"""NIS-based covariance calibration utilities.

The asynchronous baseline already writes per-update normalized innovation
squared (NIS) diagnostics.  If the assumed innovation covariance is calibrated,
accepted NIS values with measurement dimension ``d`` should follow a
chi-square distribution with ``d`` degrees of freedom.  Systematic deviations
therefore provide a direct scale estimate for source-specific measurement
covariances.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass(frozen=True)
class NisCalibrationSettings:
    """Configuration for robust source-wise NIS calibration."""

    group_columns: tuple[str, ...] = ("source", "measurement_dim")
    nis_column: str = "nis"
    accepted_column: str | None = "accepted"
    min_samples: int = 8
    calibration_quantile: float = 0.5
    report_quantiles: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95)
    min_scale: float = 0.2
    max_scale: float = 25.0

    def __post_init__(self) -> None:
        if not self.group_columns:
            raise ValueError("group_columns must not be empty")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if not 0.0 < self.calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be in (0, 1)")
        if not self.report_quantiles:
            raise ValueError("report_quantiles must not be empty")
        if any(not 0.0 < quantile < 1.0 for quantile in self.report_quantiles):
            raise ValueError("all report_quantiles must be in (0, 1)")
        if not 0.0 < self.min_scale <= self.max_scale:
            raise ValueError("expected 0 < min_scale <= max_scale")


@dataclass(frozen=True)
class NisCalibrationGroup:
    """Estimated covariance scale for one diagnostic group."""

    key: dict[str, str | int | float]
    samples: int
    measurement_dim: int
    mean_nis: float
    observed_quantile: float
    expected_quantile: float
    covariance_scale: float
    quantiles: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "samples": self.samples,
            "measurement_dim": self.measurement_dim,
            "mean_nis": self.mean_nis,
            "observed_quantile": self.observed_quantile,
            "expected_quantile": self.expected_quantile,
            "covariance_scale": self.covariance_scale,
            "quantiles": self.quantiles,
        }


def estimate_nis_covariance_scales(
    diagnostics: pd.DataFrame,
    *,
    settings: NisCalibrationSettings | None = None,
) -> list[NisCalibrationGroup]:
    """Estimate multiplicative covariance scales from NIS diagnostics.

    A scale larger than one means the innovation covariance was too small on
    average and should be inflated.  A scale smaller than one means it was too
    conservative.  The estimator is quantile based by default because it is less
    sensitive to a small number of unmodelled clutter associations than the mean.
    """

    settings = settings or NisCalibrationSettings()
    _validate_diagnostics_frame(diagnostics, settings)

    frame = diagnostics.copy()
    frame[settings.nis_column] = pd.to_numeric(frame[settings.nis_column], errors="coerce")
    frame = frame[np.isfinite(frame[settings.nis_column])]
    frame = frame[frame[settings.nis_column] >= 0.0]

    if settings.accepted_column is not None and settings.accepted_column in frame.columns:
        frame = frame[frame[settings.accepted_column].astype(bool)]

    groups: list[NisCalibrationGroup] = []
    grouped = frame.groupby(list(settings.group_columns), dropna=False, sort=True)
    for raw_key, group in grouped:
        values = group[settings.nis_column].to_numpy(dtype=float)
        if values.size < settings.min_samples:
            continue

        measurement_dim = _measurement_dimension(raw_key, group, settings.group_columns)
        observed = float(np.quantile(values, settings.calibration_quantile))
        expected = float(chi2.ppf(settings.calibration_quantile, df=measurement_dim))
        if not np.isfinite(expected) or expected <= 0.0:
            continue

        raw_scale = observed / expected
        scale = float(np.clip(raw_scale, settings.min_scale, settings.max_scale))
        quantiles = {
            f"q{int(100 * quantile):02d}": float(np.quantile(values, quantile))
            for quantile in settings.report_quantiles
        }
        groups.append(
            NisCalibrationGroup(
                key=_group_key_dict(raw_key, settings.group_columns),
                samples=int(values.size),
                measurement_dim=measurement_dim,
                mean_nis=float(np.mean(values)),
                observed_quantile=observed,
                expected_quantile=expected,
                covariance_scale=scale,
                quantiles=quantiles,
            )
        )
    return groups


def make_nis_calibration_report(
    diagnostics: pd.DataFrame,
    *,
    settings: NisCalibrationSettings | None = None,
) -> dict[str, object]:
    """Return a JSON-serializable calibration report."""

    settings = settings or NisCalibrationSettings()
    groups = estimate_nis_covariance_scales(diagnostics, settings=settings)
    by_source = _source_level_summary(groups)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": "nis_quantile_covariance_scale",
        "settings": {
            "group_columns": list(settings.group_columns),
            "nis_column": settings.nis_column,
            "accepted_column": settings.accepted_column,
            "min_samples": settings.min_samples,
            "calibration_quantile": settings.calibration_quantile,
            "min_scale": settings.min_scale,
            "max_scale": settings.max_scale,
        },
        "source_covariance_scales": by_source,
        "groups": [group.as_dict() for group in groups],
    }


def load_diagnostics(paths: Iterable[str]) -> pd.DataFrame:
    """Load and concatenate one or more baseline diagnostics CSV files."""

    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("at least one diagnostics path is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def _validate_diagnostics_frame(
    diagnostics: pd.DataFrame,
    settings: NisCalibrationSettings,
) -> None:
    missing = [
        column
        for column in (*settings.group_columns, settings.nis_column)
        if column not in diagnostics.columns
    ]
    if missing:
        raise ValueError(f"diagnostics frame is missing required columns: {missing}")


def _measurement_dimension(
    raw_key: object,
    group: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> int:
    key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
    if "measurement_dim" in group_columns:
        dim = key[group_columns.index("measurement_dim")]
    else:
        dim = group["measurement_dim"].iloc[0]
    measurement_dim = int(dim)
    if measurement_dim < 1:
        raise ValueError(f"measurement_dim must be positive, got {measurement_dim}")
    return measurement_dim


def _group_key_dict(
    raw_key: object,
    group_columns: tuple[str, ...],
) -> dict[str, str | int | float]:
    key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
    return {column: _json_scalar(value) for column, value in zip(group_columns, key, strict=True)}


def _json_scalar(value: object) -> str | int | float:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def _source_level_summary(groups: list[NisCalibrationGroup]) -> dict[str, float]:
    weighted: dict[str, tuple[float, int]] = {}
    for group in groups:
        source = str(group.key.get("source", "all"))
        scale_sum, sample_sum = weighted.get(source, (0.0, 0))
        weighted[source] = (
            scale_sum + group.covariance_scale * group.samples,
            sample_sum + group.samples,
        )
    return {
        source: float(scale_sum / sample_sum)
        for source, (scale_sum, sample_sum) in sorted(weighted.items())
        if sample_sum > 0
    }
