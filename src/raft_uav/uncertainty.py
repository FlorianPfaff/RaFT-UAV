"""Learned heteroscedastic RF/radar measurement uncertainty.

The module keeps the model deliberately small and dependency-light.  It fits a
ridge-regularized log-linear variance model for each measured axis and writes
row-wise covariance columns that tracking code can consume as measurement
covariances.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

RF_FEATURES = ("intercept", "log1p_cep", "log1p_rho", "valid_sensor_fraction")
RADAR_FEATURES = (
    "intercept",
    "log1p_range",
    "log1p_abs_radial_velocity",
    "log1p_num_inliers",
    "cat_prob_uav",
    "velocity_norm",
)
SOURCE_DIMS = {"rf": ("east", "north"), "radar": ("east", "north", "up")}
COORD_COL = {"east": "east_m", "north": "north_m", "up": "up_m"}
COV_SUFFIX = {"east": "ee", "north": "nn", "up": "uu"}
DEFAULT_MIN_STD = {"rf": {"east": 10.0, "north": 10.0}, "radar": {"east": 3.0, "north": 3.0, "up": 5.0}}
DEFAULT_MAX_STD = {"rf": {"east": 500.0, "north": 500.0}, "radar": {"east": 300.0, "north": 300.0, "up": 500.0}}


@dataclass(frozen=True)
class VarianceHead:
    """One log-linear variance head for one source axis."""

    source: str
    dimension: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    min_std_m: float
    max_std_m: float
    training_rows: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = feature_matrix(frame, self.source, self.feature_names)
        beta = np.asarray(self.coefficients, dtype=float)
        if x.shape[1] != beta.size:
            raise ValueError("feature/coefficient dimension mismatch")
        min_var = float(self.min_std_m) ** 2
        max_var = float(self.max_std_m) ** 2
        log_var = np.clip(x @ beta, np.log(min_var), np.log(max_var))
        return np.clip(np.exp(log_var), min_var, max_var)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dimension": self.dimension,
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "min_std_m": self.min_std_m,
            "max_std_m": self.max_std_m,
            "training_rows": self.training_rows,
        }

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> "VarianceHead":
        return cls(
            source=str(item["source"]),
            dimension=str(item["dimension"]),
            feature_names=tuple(str(v) for v in item["feature_names"]),
            coefficients=tuple(float(v) for v in item["coefficients"]),
            min_std_m=float(item["min_std_m"]),
            max_std_m=float(item["max_std_m"]),
            training_rows=int(item.get("training_rows", 0)),
        )


@dataclass(frozen=True)
class HeteroscedasticUncertaintyModel:
    """Small container for source-specific variance heads."""

    heads: tuple[VarianceHead, ...]
    metadata: Mapping[str, Any]

    def apply_rf(self, rf: pd.DataFrame) -> pd.DataFrame:
        return self.apply(rf, source="rf")

    def apply_radar(self, radar: pd.DataFrame) -> pd.DataFrame:
        return self.apply(radar, source="radar")

    def apply(self, frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
        out = frame.copy()
        if out.empty:
            return out
        for head in self._heads(source):
            variance = head.predict(out)
            suffix = COV_SUFFIX[head.dimension]
            out[f"cov_{suffix}"] = variance
            out[f"std_{head.dimension}_m"] = np.sqrt(variance)
        out["cov_en"] = 0.0
        if source == "radar":
            out["cov_eu"] = 0.0
            out["cov_nu"] = 0.0
        out["uncertainty_model"] = "heteroscedastic-loglinear"
        return out

    def _heads(self, source: str) -> tuple[VarianceHead, ...]:
        heads = tuple(head for head in self.heads if head.source == source)
        if not heads:
            raise ValueError(f"model has no heads for source {source!r}")
        return heads

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_type": "heteroscedastic-loglinear-variance",
            "metadata": dict(self.metadata),
            "heads": [head.to_dict() for head in self.heads],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeteroscedasticUncertaintyModel":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported uncertainty schema {payload.get('schema_version')!r}")
        return cls(
            heads=tuple(VarianceHead.from_dict(item) for item in payload["heads"]),
            metadata=dict(payload.get("metadata", {})),
        )


def load_uncertainty_model(path: Path) -> HeteroscedasticUncertaintyModel:
    return HeteroscedasticUncertaintyModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_heteroscedastic_uncertainty_model(
    *,
    rf: pd.DataFrame | None,
    radar: pd.DataFrame | None,
    truth: pd.DataFrame,
    ridge_lambda: float = 1.0,
    max_time_delta_s: float = 2.0,
    min_std_m: Mapping[str, Mapping[str, float]] | None = None,
    max_std_m: Mapping[str, Mapping[str, float]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HeteroscedasticUncertaintyModel:
    """Fit source/axis log-variance models from residuals to truth."""

    min_std = _nested(DEFAULT_MIN_STD, min_std_m)
    max_std = _nested(DEFAULT_MAX_STD, max_std_m)
    heads: list[VarianceHead] = []
    for source, frame in (("rf", rf), ("radar", radar)):
        if frame is None or frame.empty:
            continue
        aligned = _aligned_residuals(frame, truth, max_time_delta_s=max_time_delta_s)
        if aligned.empty:
            continue
        features = RF_FEATURES if source == "rf" else RADAR_FEATURES
        x = feature_matrix(aligned, source, features)
        for dim in SOURCE_DIMS[source]:
            residual = aligned[f"residual_{dim}_m"].to_numpy(dtype=float)
            target = np.log(np.clip(residual**2, min_std[source][dim] ** 2, max_std[source][dim] ** 2))
            heads.append(
                VarianceHead(
                    source=source,
                    dimension=dim,
                    feature_names=tuple(features),
                    coefficients=tuple(float(v) for v in _fit_ridge(x, target, ridge_lambda)),
                    min_std_m=float(min_std[source][dim]),
                    max_std_m=float(max_std[source][dim]),
                    training_rows=int(np.isfinite(target).sum()),
                )
            )
    if not heads:
        raise ValueError("no RF or radar residuals available for uncertainty fitting")
    return HeteroscedasticUncertaintyModel(
        heads=tuple(heads),
        metadata={
            "ridge_lambda": float(ridge_lambda),
            "max_time_delta_s": float(max_time_delta_s),
            **dict(metadata or {}),
        },
    )


def feature_matrix(frame: pd.DataFrame, source: str, feature_names: Sequence[str]) -> np.ndarray:
    values = _feature_frame(frame, source)
    columns: list[np.ndarray] = []
    for feature in feature_names:
        if feature == "intercept":
            columns.append(np.ones(len(values), dtype=float))
        elif feature in values.columns:
            columns.append(pd.to_numeric(values[feature], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        else:
            columns.append(np.zeros(len(values), dtype=float))
    return np.column_stack(columns)


def covariance_from_row(
    row: pd.Series,
    dim: int,
    fallback: np.ndarray,
    *,
    prefixes: Sequence[str] = ("association_cov", "cov"),
) -> np.ndarray:
    """Read row-wise covariance columns with a safe fallback."""

    fallback = np.asarray(fallback, dtype=float)
    if dim == 2:
        names = ("ee", "nn")
        cross = ((0, 1, "en"),)
    elif dim == 3:
        names = ("ee", "nn", "uu")
        cross = ((0, 1, "en"), (0, 2, "eu"), (1, 2, "nu"))
    else:
        raise ValueError("dim must be 2 or 3")
    for prefix in prefixes:
        diagonal = [_positive(row.get(f"{prefix}_{name}")) for name in names]
        if all(value is not None for value in diagonal):
            cov = np.diag([float(value) for value in diagonal])
            for i, j, suffix in cross:
                value = _finite(row.get(f"{prefix}_{suffix}"))
                if value is not None:
                    cov[i, j] = cov[j, i] = value
            return cov
    return fallback.copy()


def _feature_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    if source == "rf":
        cep = _num(frame, ("CEP", "cep", "cep_m"), 0.0)
        rho = _num(frame, ("RHO", "rho"), 0.0)
        total = _num(frame, ("TotalSensors", "Total Sensors", "total_sensors"), 0.0)
        valid = _num(frame, ("ValidSensors", "Valid Sensors", "valid_sensors"), 0.0)
        out["log1p_cep"] = np.log1p(np.maximum(cep, 0.0))
        out["log1p_rho"] = np.log1p(np.maximum(rho, 0.0))
        out["valid_sensor_fraction"] = np.divide(valid, np.maximum(total, 1.0))
        return out
    if source == "radar":
        range_m = _num(frame, ("range_m", "range", "Range"), 0.0)
        radial = _num(frame, ("radial_velocity_mps", "radialVelocity"), 0.0)
        inliers = _num(frame, ("num_inliers", "numInliers"), 0.0)
        catprob = _num(frame, ("cat_prob_uav", "catProbUav"), 0.0)
        ve = _num(frame, ("velocity_east_mps", "v_east_mps"), 0.0)
        vn = _num(frame, ("velocity_north_mps", "v_north_mps"), 0.0)
        vd = _num(frame, ("velocity_down_mps", "v_down_mps"), 0.0)
        out["log1p_range"] = np.log1p(np.maximum(range_m, 0.0))
        out["log1p_abs_radial_velocity"] = np.log1p(np.abs(radial))
        out["log1p_num_inliers"] = np.log1p(np.maximum(inliers, 0.0))
        out["cat_prob_uav"] = np.clip(catprob, 0.0, 1.0)
        out["velocity_norm"] = np.sqrt(ve**2 + vn**2 + vd**2)
        return out
    raise ValueError(f"unknown source {source!r}")


def _aligned_residuals(frame: pd.DataFrame, truth: pd.DataFrame, *, max_time_delta_s: float) -> pd.DataFrame:
    required = ["time_s", "east_m", "north_m"]
    if not all(column in frame.columns for column in required):
        return frame.iloc[0:0].copy()
    truth_times = truth["time_s"].to_numpy(dtype=float)
    query_times = frame["time_s"].to_numpy(dtype=float)
    if truth_times.size == 0 or query_times.size == 0:
        return frame.iloc[0:0].copy()
    truth_idx = _nearest_time_indices(truth_times, query_times)
    dt = np.abs(truth_times[truth_idx] - query_times)
    out = frame.loc[dt <= float(max_time_delta_s)].copy()
    if out.empty:
        return out
    truth_rows = truth.iloc[truth_idx[dt <= float(max_time_delta_s)]].reset_index(drop=True)
    out = out.reset_index(drop=True)
    for dim in ("east", "north", "up"):
        col = COORD_COL[dim]
        if col in out.columns and col in truth_rows.columns:
            out[f"residual_{dim}_m"] = out[col].to_numpy(dtype=float) - truth_rows[col].to_numpy(dtype=float)
    return out


def _nearest_time_indices(reference_times_s: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_times_s, dtype=float)
    query = np.asarray(query_times_s, dtype=float)
    insertion = np.searchsorted(reference, query)
    right = np.clip(insertion, 0, reference.size - 1)
    left = np.clip(insertion - 1, 0, reference.size - 1)
    use_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(use_right, right, left)


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        raise ValueError("cannot fit a variance head without finite samples")
    penalty = float(ridge_lambda) * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    normal = x.T @ x + penalty
    rhs = x.T @ y
    try:
        return np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(normal) @ rhs


def _num(frame: pd.DataFrame, aliases: Sequence[str], default: float) -> np.ndarray:
    for alias in aliases:
        if alias in frame.columns:
            values = pd.to_numeric(frame[alias], errors="coerce").to_numpy(dtype=float)
            return np.where(np.isfinite(values), values, float(default))
    return np.full(len(frame), float(default), dtype=float)


def _nested(
    base: Mapping[str, Mapping[str, float]],
    override: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    out = {source: dict(values) for source, values in base.items()}
    for source, values in dict(override or {}).items():
        out.setdefault(source, {}).update({key: float(value) for key, value in values.items()})
    return out


def _positive(value: object) -> float | None:
    value = _finite(value)
    return value if value is not None and value > 0.0 else None


def _finite(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
