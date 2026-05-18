"""Learned RF/radar measurement bias correction.

The model learns residuals

    bias = measured_ENU_position - nearest_truth_ENU_position

from normalized training flights and subtracts the predicted bias before the
Kalman update.  It deliberately uses only NumPy/Pandas to avoid adding a new
runtime dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

BIAS_MODEL_VERSION = 1
RF_TARGET_COLUMNS = ("east_m", "north_m")
RADAR_TARGET_COLUMNS = ("east_m", "north_m", "up_m")
TARGET_COLUMNS_BY_SOURCE = {"rf": RF_TARGET_COLUMNS, "radar": RADAR_TARGET_COLUMNS}
BIAS_RESIDUAL_STD_COLUMN_PREFIX = "bias_residual_std_"

_DEFAULT_FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "rf": ("time_s", "east_m", "north_m", "std_m", "CEP"),
    "radar": (
        "time_s",
        "east_m",
        "north_m",
        "up_m",
        "range_m",
        "radial_velocity_mps",
        "num_inliers",
        "cat_prob_uav",
        "velocity_north_mps",
        "velocity_east_mps",
        "velocity_down_mps",
    ),
}


@dataclass(frozen=True)
class SensorBiasCorrectionModel:
    """Ridge-linear bias model for one measurement source."""

    source: str
    target_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    intercept: np.ndarray
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    residual_std: np.ndarray
    training_rows: int
    ridge_alpha: float
    time_gate_s: float

    def __post_init__(self) -> None:
        targets = tuple(str(column) for column in self.target_columns)
        features = tuple(str(column) for column in self.feature_columns)
        intercept = np.asarray(self.intercept, dtype=float).reshape(len(targets))
        coefficients = np.asarray(self.coefficients, dtype=float).reshape(len(features), len(targets))
        feature_mean = np.asarray(self.feature_mean, dtype=float).reshape(len(features))
        feature_scale = np.asarray(self.feature_scale, dtype=float).reshape(len(features))
        residual_std = np.asarray(self.residual_std, dtype=float).reshape(len(targets))
        feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 0.0), feature_scale, 1.0)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "target_columns", targets)
        object.__setattr__(self, "feature_columns", features)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)
        object.__setattr__(self, "residual_std", residual_std)
        object.__setattr__(self, "training_rows", int(self.training_rows))
        object.__setattr__(self, "ridge_alpha", float(self.ridge_alpha))
        object.__setattr__(self, "time_gate_s", float(self.time_gate_s))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict ENU residual bias for rows in ``frame``."""

        if frame.empty:
            return np.empty((0, len(self.target_columns)), dtype=float)
        features = _feature_matrix(
            frame,
            self.feature_columns,
            self.feature_mean,
            raw_target_columns=self.target_columns,
        )
        standardized = (features - self.feature_mean) / self.feature_scale if features.size else features
        return self.intercept.reshape(1, -1) + standardized @ self.coefficients

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with predicted bias subtracted from ENU target columns."""

        out = frame.copy()
        if out.empty:
            return out
        _require_columns(out, self.target_columns, context=f"{self.source} bias correction")
        for column in self.target_columns:
            raw_column = _raw_column_name(column)
            if raw_column not in out.columns:
                out[raw_column] = out[column]
        predicted = self.predict(out)
        for index, column in enumerate(self.target_columns):
            raw_column = _raw_column_name(column)
            bias_column = _bias_column_name(column)
            residual_std_column = _bias_residual_std_column_name(column)
            raw = pd.to_numeric(out[raw_column], errors="coerce").to_numpy(dtype=float)
            bias = predicted[:, index]
            corrected = raw - bias
            valid = np.isfinite(raw) & np.isfinite(bias)
            residual_std = float(self.residual_std[index])
            if not np.isfinite(residual_std) or residual_std < 0.0:
                residual_std = float("nan")
            out[bias_column] = bias
            out[residual_std_column] = residual_std
            out.loc[valid, column] = corrected[valid]
        out["bias_correction_source"] = self.source
        out["bias_correction_training_rows"] = int(self.training_rows)
        out["bias_correction_features"] = ",".join(self.feature_columns)
        return out

    # Compatibility with the earlier bank-style API.
    def predict_bias(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict(frame)

    def correct_frame(self, frame: pd.DataFrame, *, keep_uncorrected: bool = True) -> pd.DataFrame:
        del keep_uncorrected
        return self.apply(frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": BIAS_MODEL_VERSION,
            "source": self.source,
            "target_columns": list(self.target_columns),
            "target_axes": list(self.target_columns),
            "feature_columns": list(self.feature_columns),
            "intercept": self.intercept.tolist(),
            "coefficients": self.coefficients.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "residual_std": self.residual_std.tolist(),
            "training_rows": int(self.training_rows),
            "training_count": int(self.training_rows),
            "ridge_alpha": float(self.ridge_alpha),
            "time_gate_s": float(self.time_gate_s),
            "fit_rmse_by_axis_m": {
                axis: float(value) for axis, value in zip(self.target_columns, self.residual_std)
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SensorBiasCorrectionModel":
        version = int(payload.get("version", BIAS_MODEL_VERSION))
        if version != BIAS_MODEL_VERSION:
            raise ValueError(f"unsupported bias model version {version}")
        target_columns = payload.get("target_columns", payload.get("target_axes"))
        training_rows = payload.get("training_rows", payload.get("training_count", 0))
        residual_std = payload.get("residual_std")
        if residual_std is None:
            rmse = payload.get("fit_rmse_by_axis_m", {})
            residual_std = [float(rmse.get(axis, 0.0)) for axis in target_columns]
        return cls(
            source=str(payload["source"]),
            target_columns=tuple(str(column) for column in target_columns),
            feature_columns=tuple(str(column) for column in payload["feature_columns"]),
            intercept=np.asarray(payload["intercept"], dtype=float),
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            residual_std=np.asarray(residual_std, dtype=float),
            training_rows=int(training_rows),
            ridge_alpha=float(payload.get("ridge_alpha", 0.0)),
            time_gate_s=float(payload.get("time_gate_s", 2.0)),
        )


BiasCorrectionModel = SensorBiasCorrectionModel


@dataclass(frozen=True)
class BiasCorrectionBank:
    """Container for source-specific RF/radar bias models."""

    models: Mapping[str, SensorBiasCorrectionModel]

    def correct_frame(self, frame: pd.DataFrame, source: str) -> pd.DataFrame:
        model = self.models.get(str(source))
        return frame if model is None else model.apply(frame)

    def save(self, path: Path) -> None:
        save_bias_correction_models(self.models, path)

    def summary(self, model_path: Path | None = None) -> dict[str, Any]:
        summary: dict[str, Any] = {"enabled": True, "models": bias_correction_summary(self.models)}
        if model_path is not None:
            summary["model_path"] = str(model_path)
        return summary

    @classmethod
    def load(cls, path: Path) -> "BiasCorrectionBank":
        return cls(load_bias_correction_models(path))


def fit_sensor_bias_correction(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    feature_columns: Sequence[str] | None = None,
    time_gate_s: float = 2.0,
    ridge_alpha: float = 1.0e-2,
    min_samples: int = 4,
) -> SensorBiasCorrectionModel:
    examples = make_bias_training_examples(
        measurements,
        truth,
        source=source,
        target_columns=target_columns,
        time_gate_s=time_gate_s,
    )
    return fit_sensor_bias_correction_from_examples(
        examples,
        source=source,
        target_columns=target_columns,
        feature_columns=feature_columns,
        time_gate_s=time_gate_s,
        ridge_alpha=ridge_alpha,
        min_samples=min_samples,
    )


def fit_sensor_bias_correction_from_examples(
    examples: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    feature_columns: Sequence[str] | None = None,
    time_gate_s: float = 2.0,
    ridge_alpha: float = 1.0e-2,
    min_samples: int = 4,
) -> SensorBiasCorrectionModel:
    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be nonnegative")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    targets = tuple(str(column) for column in target_columns)
    bias_columns = tuple(_bias_column_name(column) for column in targets)
    _require_columns(examples, bias_columns, context="bias examples")
    y = examples.loc[:, bias_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid_y = np.isfinite(y).all(axis=1)
    examples = examples.loc[valid_y].reset_index(drop=True)
    y = y[valid_y]
    if y.shape[0] == 0:
        raise ValueError(f"no finite {source} bias training examples")

    selected_features = _select_feature_columns(str(source), examples, feature_columns, min_samples)
    x = _feature_matrix(examples, selected_features, None, raw_target_columns=targets)
    valid_x = np.isfinite(x).all(axis=1) if x.shape[1] else np.ones(y.shape[0], dtype=bool)
    y = y[valid_x]
    x = x[valid_x]
    if y.shape[0] == 0:
        raise ValueError(f"no finite {source} feature rows")
    if x.shape[0] < int(min_samples):
        selected_features = tuple()
        x = np.empty((y.shape[0], 0), dtype=float)

    feature_mean = _nanmean_or_zero(x)
    feature_scale = np.nanstd(x, axis=0) if x.shape[1] else np.empty(0, dtype=float)
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1.0e-12), feature_scale, 1.0)
    standardized = (x - feature_mean) / feature_scale if x.shape[1] else x
    design = np.column_stack([np.ones(y.shape[0]), standardized])
    regularizer = np.eye(design.shape[1]) * float(ridge_alpha)
    regularizer[0, 0] = 0.0
    lhs = design.T @ design + regularizer
    rhs = design.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    residual = y - design @ beta
    residual_std = np.std(residual, axis=0) if residual.size else np.zeros(len(targets))
    return SensorBiasCorrectionModel(
        source=str(source),
        target_columns=targets,
        feature_columns=selected_features,
        intercept=beta[0],
        coefficients=beta[1:],
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        residual_std=residual_std,
        training_rows=int(y.shape[0]),
        ridge_alpha=float(ridge_alpha),
        time_gate_s=float(time_gate_s),
    )


def make_bias_training_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    time_gate_s: float = 2.0,
) -> pd.DataFrame:
    if time_gate_s < 0.0:
        raise ValueError("time_gate_s must be nonnegative")
    targets = tuple(str(column) for column in target_columns)
    if measurements.empty or truth.empty:
        return pd.DataFrame()
    _require_columns(measurements, ("time_s", *targets), context=f"{source} measurements")
    _require_columns(truth, ("time_s", *targets), context="truth")
    truth_sorted = truth.sort_values("time_s").reset_index(drop=True)
    truth_times = pd.to_numeric(truth_sorted["time_s"], errors="coerce").to_numpy(dtype=float)
    valid_truth = np.isfinite(truth_times)
    truth_sorted = truth_sorted.loc[valid_truth].reset_index(drop=True)
    truth_times = truth_times[valid_truth]
    rows = measurements.copy().reset_index(drop=True)
    query_times = pd.to_numeric(rows["time_s"], errors="coerce").to_numpy(dtype=float)
    valid_measurement = np.isfinite(query_times)
    rows = rows.loc[valid_measurement].reset_index(drop=True)
    query_times = query_times[valid_measurement]
    if not len(truth_times) or not len(query_times):
        return pd.DataFrame()
    nearest = _nearest_time_indices(truth_times, query_times)
    delta_s = np.abs(truth_times[nearest] - query_times)
    keep = delta_s <= float(time_gate_s)
    if not np.any(keep):
        return pd.DataFrame()
    rows = rows.loc[keep].reset_index(drop=True)
    matched_truth = truth_sorted.iloc[nearest[keep]].reset_index(drop=True)
    for column in targets:
        measurement_values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
        truth_values = pd.to_numeric(matched_truth[column], errors="coerce").to_numpy(dtype=float)
        rows[_bias_column_name(column)] = measurement_values - truth_values
    rows["bias_truth_time_delta_s"] = delta_s[keep]
    rows["bias_source"] = str(source)
    return rows


def bias_training_rows(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    max_time_delta_s: float = 2.0,
    max_position_error_m: float | None = None,
) -> pd.DataFrame:
    targets = TARGET_COLUMNS_BY_SOURCE[str(source)]
    rows = make_bias_training_examples(
        measurements,
        truth,
        source=source,
        target_columns=targets,
        time_gate_s=max_time_delta_s,
    )
    if max_position_error_m is not None and not rows.empty:
        bias_columns = [_bias_column_name(column) for column in targets]
        distances = np.linalg.norm(rows[bias_columns].to_numpy(dtype=float), axis=1)
        rows = rows.loc[distances <= float(max_position_error_m)].reset_index(drop=True)
    for column in targets:
        rows[f"target_bias_{column}"] = rows[_bias_column_name(column)]
    return rows


def fit_bias_correction_model(
    rows: pd.DataFrame,
    *,
    source: str,
    ridge_alpha: float = 1.0,
    min_samples: int = 5,
    feature_columns: Sequence[str] | None = None,
) -> SensorBiasCorrectionModel:
    targets = TARGET_COLUMNS_BY_SOURCE[str(source)]
    return fit_sensor_bias_correction_from_examples(
        rows,
        source=source,
        target_columns=targets,
        feature_columns=feature_columns,
        ridge_alpha=ridge_alpha,
        min_samples=min_samples,
    )


def fit_bias_correction_bank(
    rows_by_source: Mapping[str, pd.DataFrame],
    *,
    ridge_alpha: float = 1.0,
    min_samples: int = 5,
) -> BiasCorrectionBank:
    models: dict[str, SensorBiasCorrectionModel] = {}
    for source, rows in rows_by_source.items():
        if rows is not None and not rows.empty:
            models[str(source)] = fit_bias_correction_model(
                rows,
                source=str(source),
                ridge_alpha=ridge_alpha,
                min_samples=min_samples,
            )
    if not models:
        raise ValueError("no bias correction models could be fitted")
    return BiasCorrectionBank(models)


def fit_bias_correction_models(
    *,
    rf: pd.DataFrame | None,
    radar: pd.DataFrame | None,
    truth: pd.DataFrame,
    time_gate_s: float = 2.0,
    ridge_alpha: float = 1.0e-2,
    min_samples: int = 4,
) -> dict[str, SensorBiasCorrectionModel]:
    models: dict[str, SensorBiasCorrectionModel] = {}
    if rf is not None and not rf.empty:
        models["rf"] = fit_sensor_bias_correction(
            rf,
            truth,
            source="rf",
            target_columns=RF_TARGET_COLUMNS,
            time_gate_s=time_gate_s,
            ridge_alpha=ridge_alpha,
            min_samples=min_samples,
        )
    if radar is not None and not radar.empty:
        models["radar"] = fit_sensor_bias_correction(
            radar,
            truth,
            source="radar",
            target_columns=RADAR_TARGET_COLUMNS,
            time_gate_s=time_gate_s,
            ridge_alpha=ridge_alpha,
            min_samples=min_samples,
        )
    return models


def save_bias_correction_models(models: Mapping[str, SensorBiasCorrectionModel], path: Path) -> None:
    payload = {
        "version": BIAS_MODEL_VERSION,
        "models": {source: model.to_dict() for source, model in sorted(models.items())},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_bias_correction_models(path: Path) -> dict[str, SensorBiasCorrectionModel]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("version", BIAS_MODEL_VERSION))
    if version != BIAS_MODEL_VERSION:
        raise ValueError(f"unsupported bias model bundle version {version}")
    models = payload.get("models", {})
    if not isinstance(models, Mapping):
        raise ValueError("bias model bundle must contain a models mapping")
    return {str(source): SensorBiasCorrectionModel.from_dict(model) for source, model in models.items()}


def load_bias_correction_bank(path: Path) -> BiasCorrectionBank:
    return BiasCorrectionBank(load_bias_correction_models(path))


def apply_bias_correction_models(
    *,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    models: Mapping[str, SensorBiasCorrectionModel],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected_rf = models["rf"].apply(rf) if "rf" in models and not rf.empty else rf
    corrected_radar = models["radar"].apply(radar) if "radar" in models and not radar.empty else radar
    return corrected_rf, corrected_radar


def bias_correction_summary(models: Mapping[str, SensorBiasCorrectionModel]) -> dict[str, Any]:
    return {
        source: {
            "target_columns": list(model.target_columns),
            "feature_columns": list(model.feature_columns),
            "training_rows": int(model.training_rows),
            "training_count": int(model.training_rows),
            "ridge_alpha": float(model.ridge_alpha),
            "time_gate_s": float(model.time_gate_s),
            "intercept_m": model.intercept.tolist(),
            "residual_std_m": model.residual_std.tolist(),
            "fit_rmse_by_axis_m": {
                axis: float(value) for axis, value in zip(model.target_columns, model.residual_std)
            },
        }
        for source, model in sorted(models.items())
    }


def _select_feature_columns(
    source: str,
    frame: pd.DataFrame,
    requested: Sequence[str] | None,
    min_samples: int,
) -> tuple[str, ...]:
    candidates = requested if requested is not None else _DEFAULT_FEATURE_COLUMNS.get(source, ("time_s",))
    selected: list[str] = []
    for column in candidates:
        column = str(column)
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).sum() >= min_samples:
            selected.append(column)
    return tuple(dict.fromkeys(selected))


def _feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    default_values: np.ndarray | Sequence[float] | None,
    *,
    raw_target_columns: Sequence[str],
) -> np.ndarray:
    if not feature_columns:
        return np.empty((len(frame), 0), dtype=float)
    defaults = None if default_values is None else np.asarray(default_values, dtype=float).reshape(-1)
    raw_targets = {str(column): _raw_column_name(str(column)) for column in raw_target_columns}
    columns: list[np.ndarray] = []
    for index, column in enumerate(feature_columns):
        column = str(column)
        raw_column = raw_targets.get(column)
        source_column = raw_column if raw_column is not None and raw_column in frame.columns else column
        if source_column not in frame.columns:
            fill = 0.0 if defaults is None else float(defaults[index])
            columns.append(np.full(len(frame), fill, dtype=float))
            continue
        values = pd.to_numeric(frame[source_column], errors="coerce").to_numpy(dtype=float)
        if defaults is not None:
            values = np.where(np.isfinite(values), values, float(defaults[index]))
        columns.append(values)
    return np.column_stack(columns).astype(float, copy=False)


def _nearest_time_indices(reference_times_s: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    insertion = np.searchsorted(reference, query)
    right = np.clip(insertion, 0, reference.size - 1)
    left = np.clip(insertion - 1, 0, reference.size - 1)
    use_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(use_right, right, left)


def _nanmean_or_zero(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 0:
        return np.empty(0, dtype=float)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=0)
    return np.where(np.isfinite(mean), mean, 0.0)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = [str(column) for column in columns if str(column) not in frame.columns]
    if missing:
        raise KeyError(f"{context} requires columns: {', '.join(missing)}")


def _raw_column_name(column: str) -> str:
    return f"raw_{column}"


def _bias_column_name(column: str) -> str:
    return f"bias_{column}"


def _bias_residual_std_column_name(column: str) -> str:
    return f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}{column}"
