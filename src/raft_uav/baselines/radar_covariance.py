"""Range-dependent radar covariance helpers for Fortem track rows.

The AERPAW/Fortem radar measurement geometry is not well represented by a
single ENU diagonal covariance: range uncertainty is radial, while angular
uncertainty grows with distance.  These helpers encode a per-row ENU covariance
using the same ``association_cov_*`` columns that the radar-association code
already accepts for Kalman updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from raft_uav.calibration.bias import BIAS_RESIDUAL_STD_COLUMN_PREFIX

RADAR_COVARIANCE_COLUMNS = (
    "association_cov_ee",
    "association_cov_nn",
    "association_cov_uu",
    "association_cov_en",
    "association_cov_eu",
    "association_cov_nu",
)
GENERIC_RADAR_COVARIANCE_COLUMNS = (
    "cov_ee",
    "cov_nn",
    "cov_uu",
    "cov_en",
    "cov_eu",
    "cov_nu",
)
BIAS_RESIDUAL_COVARIANCE_COLUMNS = (
    f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}east_m",
    f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}north_m",
    f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}up_m",
)
BIAS_RESIDUAL_INCLUDED_COLUMN = "association_cov_includes_bias_residual"


@dataclass(frozen=True)
class RadarCovarianceConfig:
    """Configuration for radar position covariance construction.

    ``mode='fixed'`` preserves the historical diagonal ENU covariance.
    ``mode='range-angle'`` projects a spherical range/azimuth/elevation error
    model into ENU coordinates.  The default values match the Fortem-style
    uncertainty scale used in the paper discussion: a few metres of range noise
    and degree-level angular uncertainty.
    """

    mode: str = "range-angle"
    xy_std_m: float = 25.0
    z_std_m: float = 35.0
    range_std_m: float = 5.0
    azimuth_std_deg: float = 2.0
    elevation_std_deg: float = 2.0
    min_std_m: float = 3.0
    max_std_m: float = 250.0
    origin_east_m: float = 0.0
    origin_north_m: float = 0.0
    origin_up_m: float = 0.0

    @classmethod
    def from_environment(cls) -> "RadarCovarianceConfig":
        """Read optional runtime settings from ``RAFT_UAV_RADAR_*`` variables."""

        return cls(
            mode=os.environ.get("RAFT_UAV_RADAR_COVARIANCE_MODE", cls.mode),
            xy_std_m=_env_float("RAFT_UAV_RADAR_XY_STD_M", cls.xy_std_m),
            z_std_m=_env_float("RAFT_UAV_RADAR_Z_STD_M", cls.z_std_m),
            range_std_m=_env_float("RAFT_UAV_RADAR_RANGE_STD_M", cls.range_std_m),
            azimuth_std_deg=_env_float(
                "RAFT_UAV_RADAR_AZIMUTH_STD_DEG", cls.azimuth_std_deg
            ),
            elevation_std_deg=_env_float(
                "RAFT_UAV_RADAR_ELEVATION_STD_DEG", cls.elevation_std_deg
            ),
            min_std_m=_env_float("RAFT_UAV_RADAR_COVARIANCE_MIN_STD_M", cls.min_std_m),
            max_std_m=_env_float("RAFT_UAV_RADAR_COVARIANCE_MAX_STD_M", cls.max_std_m),
            origin_east_m=_env_float("RAFT_UAV_RADAR_ORIGIN_EAST_M", cls.origin_east_m),
            origin_north_m=_env_float(
                "RAFT_UAV_RADAR_ORIGIN_NORTH_M", cls.origin_north_m
            ),
            origin_up_m=_env_float("RAFT_UAV_RADAR_ORIGIN_UP_M", cls.origin_up_m),
        )

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "range-angle"}:
            raise ValueError("mode must be 'fixed' or 'range-angle'")
        for name in (
            "xy_std_m",
            "z_std_m",
            "range_std_m",
            "azimuth_std_deg",
            "elevation_std_deg",
            "min_std_m",
            "max_std_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.max_std_m) < float(self.min_std_m):
            raise ValueError("max_std_m must be >= min_std_m")
        for name in ("origin_east_m", "origin_north_m", "origin_up_m"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

    def fixed_covariance(self) -> np.ndarray:
        """Return the fixed fallback radar covariance."""

        return fixed_radar_covariance(self.xy_std_m, self.z_std_m)

    def origin_vector(self) -> np.ndarray:
        """Return the configured radar origin as an ENU vector."""

        return np.array(
            [float(self.origin_east_m), float(self.origin_north_m), float(self.origin_up_m)],
            dtype=float,
        )


def fixed_radar_covariance(xy_std_m: float = 25.0, z_std_m: float = 35.0) -> np.ndarray:
    """Return the historical fixed diagonal radar position covariance."""

    xy_std = float(xy_std_m)
    z_std = float(z_std_m)
    if not np.isfinite(xy_std) or xy_std <= 0.0:
        raise ValueError("xy_std_m must be finite and positive")
    if not np.isfinite(z_std) or z_std <= 0.0:
        raise ValueError("z_std_m must be finite and positive")
    return np.diag([xy_std**2, xy_std**2, z_std**2])


def append_radar_covariance_columns(
    radar: pd.DataFrame,
    config: RadarCovarianceConfig | None = None,
) -> pd.DataFrame:
    """Annotate radar rows with per-row ENU covariance columns when possible."""

    cfg = config or RadarCovarianceConfig.from_environment()
    if radar.empty or cfg.mode == "fixed":
        return radar
    required = {"east_m", "north_m", "up_m"}
    if not required.issubset(radar.columns):
        return radar

    out = radar.copy()
    covariances: list[np.ndarray] = []
    used_ranges: list[float] = []
    includes_bias_residual: list[bool] = []
    for _, row in out.iterrows():
        covariance, used_range_m = range_angle_radar_covariance(row, cfg)
        covariance, included_bias = _add_bias_residual_uncertainty(
            row,
            covariance,
            min_std_m=1.0e-6,
            max_std_m=float(cfg.max_std_m),
        )
        covariances.append(covariance)
        used_ranges.append(float(used_range_m))
        includes_bias_residual.append(bool(included_bias))

    out["association_cov_ee"] = [float(cov[0, 0]) for cov in covariances]
    out["association_cov_nn"] = [float(cov[1, 1]) for cov in covariances]
    out["association_cov_uu"] = [float(cov[2, 2]) for cov in covariances]
    out["association_cov_en"] = [float(cov[0, 1]) for cov in covariances]
    out["association_cov_eu"] = [float(cov[0, 2]) for cov in covariances]
    out["association_cov_nu"] = [float(cov[1, 2]) for cov in covariances]
    out["association_covariance_mode"] = cfg.mode
    out["association_cov_range_m"] = used_ranges
    out["association_cov_trace_m2"] = [float(np.trace(cov)) for cov in covariances]
    out[BIAS_RESIDUAL_INCLUDED_COLUMN] = includes_bias_residual
    return out


def range_angle_radar_covariance(
    row: pd.Series,
    config: RadarCovarianceConfig | None = None,
) -> tuple[np.ndarray, float]:
    """Project one radar row's range/angle uncertainty into ENU covariance."""

    cfg = config or RadarCovarianceConfig.from_environment()
    fallback = cfg.fixed_covariance()
    position = _row_position(row)
    if position is None:
        return fallback, float("nan")

    los = position - cfg.origin_vector()
    geometric_range_m = float(np.linalg.norm(los))
    logged_range_m = _positive_float(row.get("range_m"))
    used_range_m = logged_range_m if logged_range_m is not None else geometric_range_m
    if geometric_range_m <= 1.0e-6 or not np.isfinite(used_range_m) or used_range_m <= 0.0:
        return fallback, float(used_range_m)

    direction = los / geometric_range_m
    east_unit, north_unit, up_unit = direction
    azimuth_rad = float(np.arctan2(east_unit, north_unit))
    elevation_rad = float(np.arcsin(np.clip(up_unit, -1.0, 1.0)))

    cos_el = float(np.cos(elevation_rad))
    sin_el = float(np.sin(elevation_rad))
    sin_az = float(np.sin(azimuth_rad))
    cos_az = float(np.cos(azimuth_rad))
    range_for_angles = float(used_range_m)

    # ENU convention:
    # east=r cos(el) sin(az), north=r cos(el) cos(az), up=r sin(el)
    jacobian = np.array(
        [
            [
                cos_el * sin_az,
                range_for_angles * cos_el * cos_az,
                -range_for_angles * sin_el * sin_az,
            ],
            [
                cos_el * cos_az,
                -range_for_angles * cos_el * sin_az,
                -range_for_angles * sin_el * cos_az,
            ],
            [sin_el, 0.0, range_for_angles * cos_el],
        ],
        dtype=float,
    )
    spherical_covariance = np.diag(
        [
            float(cfg.range_std_m) ** 2,
            np.deg2rad(float(cfg.azimuth_std_deg)) ** 2,
            np.deg2rad(float(cfg.elevation_std_deg)) ** 2,
        ]
    )
    covariance = jacobian @ spherical_covariance @ jacobian.T
    return _regularized_covariance(
        covariance,
        min_std_m=float(cfg.min_std_m),
        max_std_m=float(cfg.max_std_m),
    ), float(used_range_m)


def row_radar_covariance(
    row: pd.Series,
    fallback_covariance: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return covariance encoded in a radar row, or ``fallback_covariance``.

    Bias-corrected radar rows may carry learned residual standard deviations.
    These are added as independent per-axis variance terms unless the encoded
    association covariance already declares that it includes them.
    """

    fallback = None if fallback_covariance is None else np.asarray(fallback_covariance, dtype=float)
    generic_covariance = _generic_row_covariance(row)
    if generic_covariance is not None:
        if _truthy(row.get(BIAS_RESIDUAL_INCLUDED_COLUMN, False)):
            return generic_covariance
        covariance, _ = _add_bias_residual_uncertainty(
            row,
            generic_covariance,
            min_std_m=1.0e-6,
            max_std_m=1.0e9,
        )
        return covariance
    if not all(column in row for column in RADAR_COVARIANCE_COLUMNS):
        if fallback is None:
            return fallback
        covariance, _ = _add_bias_residual_uncertainty(
            row,
            fallback,
            min_std_m=1.0e-6,
            max_std_m=1.0e9,
        )
        return covariance
    try:
        ee, nn, uu, en, eu, nu = [float(row[column]) for column in RADAR_COVARIANCE_COLUMNS]
    except (TypeError, ValueError):
        if fallback is None:
            return fallback
        covariance, _ = _add_bias_residual_uncertainty(
            row,
            fallback,
            min_std_m=1.0e-6,
            max_std_m=1.0e9,
        )
        return covariance
    values = np.array([ee, nn, uu, en, eu, nu], dtype=float)
    if not np.isfinite(values).all():
        if fallback is None:
            return fallback
        covariance, _ = _add_bias_residual_uncertainty(
            row,
            fallback,
            min_std_m=1.0e-6,
            max_std_m=1.0e9,
        )
        return covariance
    covariance = np.array([[ee, en, eu], [en, nn, nu], [eu, nu, uu]], dtype=float)
    covariance = _regularized_covariance(covariance, min_std_m=1.0e-6, max_std_m=1.0e9)
    if _truthy(row.get(BIAS_RESIDUAL_INCLUDED_COLUMN, False)):
        return covariance
    covariance, _ = _add_bias_residual_uncertainty(
        row,
        covariance,
        min_std_m=1.0e-6,
        max_std_m=1.0e9,
    )
    return covariance


def _generic_row_covariance(row: pd.Series) -> np.ndarray | None:
    diagonal = [_positive_float(row.get(column)) for column in GENERIC_RADAR_COVARIANCE_COLUMNS[:3]]
    if not all(value is not None for value in diagonal):
        return None
    ee, nn, uu = [float(value) for value in diagonal]
    en, eu, nu = [
        _finite_float(row.get(column), default=0.0)
        for column in GENERIC_RADAR_COVARIANCE_COLUMNS[3:]
    ]
    covariance = np.array([[ee, en, eu], [en, nn, nu], [eu, nu, uu]], dtype=float)
    return _regularized_covariance(covariance, min_std_m=1.0e-6, max_std_m=1.0e9)


def candidate_radar_covariances(
    candidates: pd.DataFrame,
    fallback_covariance: np.ndarray,
) -> np.ndarray:
    """Return one 3x3 covariance per candidate row."""

    fallback = np.asarray(fallback_covariance, dtype=float).reshape(3, 3)
    if candidates.empty:
        return np.empty((0, 3, 3), dtype=float)
    covariances = [
        np.asarray(row_radar_covariance(row, fallback), dtype=float)
        for _, row in candidates.iterrows()
    ]
    return np.stack(covariances, axis=0)


def _row_position(row: pd.Series) -> np.ndarray | None:
    try:
        position = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
    except (KeyError, TypeError, ValueError):
        return None
    return position if np.isfinite(position).all() else None


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0.0 else None


def _finite_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def _bias_residual_variance(row: pd.Series) -> np.ndarray | None:
    variances: list[float] = []
    has_finite_value = False
    for column in BIAS_RESIDUAL_COVARIANCE_COLUMNS:
        value = _positive_float(row.get(column))
        if value is None:
            variances.append(0.0)
            continue
        has_finite_value = True
        variances.append(float(value) ** 2)
    if not has_finite_value:
        return None
    return np.asarray(variances, dtype=float)


def _add_bias_residual_uncertainty(
    row: pd.Series,
    covariance: np.ndarray,
    *,
    min_std_m: float,
    max_std_m: float,
) -> tuple[np.ndarray, bool]:
    variances = _bias_residual_variance(row)
    if variances is None:
        return np.asarray(covariance, dtype=float), False
    inflated = np.asarray(covariance, dtype=float).reshape(3, 3) + np.diag(variances)
    return _regularized_covariance(inflated, min_std_m=min_std_m, max_std_m=max_std_m), True


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return False


def _regularized_covariance(
    covariance: np.ndarray, *, min_std_m: float, max_std_m: float
) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    if symmetric.shape != (3, 3) or not np.isfinite(symmetric).all():
        raise ValueError("radar covariance must be a finite 3x3 matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, float(min_std_m) ** 2, float(max_std_m) ** 2)
    regularized = (eigenvectors * eigenvalues.reshape(1, -1)) @ eigenvectors.T
    return 0.5 * (regularized + regularized.T)
