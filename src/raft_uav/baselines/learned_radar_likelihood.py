"""Learned radar association likelihoods.

The model is intentionally dependency-light: a standardized logistic likelihood
trained on truth-labeled radar candidates. It predicts the probability that a
candidate row is the UAV and can be used to replace hand-tuned association
scores while leaving the Kalman update unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODEL_TYPE = "raft-uav.learned-radar-association-logistic-v1"

DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "log1p_association_nis",
    "sqrt_association_nis",
    "position_residual_norm_m",
    "horizontal_residual_norm_m",
    "abs_vertical_residual_m",
    "candidate_count",
    "candidate_rank_by_nis",
    "cat_prob_uav",
    "neg_log_cat_prob_uav",
    "missing_cat_prob_uav",
    "track_id_changed",
    "same_track_id",
    "missing_track_id",
    "velocity_residual_norm_mps",
    "speed_mps",
    "missing_velocity",
    "log1p_range_m",
    "abs_radial_velocity_mps",
    "missing_radial_velocity",
    "log1p_num_inliers",
    "missing_num_inliers",
)


@dataclass(frozen=True)
class LearnedRadarAssociationModel:
    """Standardized logistic model for radar association."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.feature_names)
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        scale = np.asarray(self.scale, dtype=float).reshape(-1)
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if not names:
            raise ValueError("feature_names must not be empty")
        if mean.size != len(names) or scale.size != len(names) or weights.size != len(names):
            raise ValueError("mean, scale, and weights must match feature_names")
        scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "intercept", float(self.intercept))

    def logits_from_features(self, features: pd.DataFrame) -> np.ndarray:
        matrix = _raw_feature_matrix(features, self.feature_names)
        filled = np.where(np.isfinite(matrix), matrix, self.mean.reshape(1, -1))
        standardized = (filled - self.mean.reshape(1, -1)) / self.scale.reshape(1, -1)
        return standardized @ self.weights + self.intercept

    def predict_proba_features(self, features: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.logits_from_features(features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": MODEL_TYPE,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "intercept": self.intercept,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LearnedRadarAssociationModel":
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"unsupported learned radar association model type {payload.get('model_type')!r}")
        return cls(
            feature_names=tuple(payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            weights=np.asarray(payload["weights"], dtype=float),
            intercept=float(payload["intercept"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRadarAssociationModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def radar_association_feature_frame(
    candidates: pd.DataFrame,
    *,
    tracker_state: np.ndarray,
    current_track_id: int | None,
) -> pd.DataFrame:
    """Build learned-association features for one radar frame."""

    if candidates.empty:
        return pd.DataFrame(columns=list(DEFAULT_FEATURE_NAMES))
    state = np.asarray(tracker_state, dtype=float).reshape(6)
    positions = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    residuals = positions - state[:3].reshape(1, 3)
    horizontal_residual = np.linalg.norm(residuals[:, :2], axis=1)
    position_residual = np.linalg.norm(residuals, axis=1)

    nis = _numeric_column(candidates, "association_nis")
    finite_nis = np.where(np.isfinite(nis), nis, np.inf)
    ranks = pd.Series(finite_nis).rank(method="average").to_numpy(dtype=float) - 1.0
    ranks = np.where(np.isfinite(finite_nis), ranks, float(len(candidates)))

    cat_prob = _numeric_column(candidates, "cat_prob_uav")
    missing_cat_prob = ~np.isfinite(cat_prob)
    clipped_cat_prob = np.clip(np.where(missing_cat_prob, 0.5, cat_prob), 1.0e-6, 1.0)

    velocity, missing_velocity = _velocity_matrix_enu(candidates)
    velocity_residual = np.linalg.norm(velocity - state[3:6].reshape(1, 3), axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    velocity_residual = np.where(missing_velocity, np.nan, velocity_residual)
    speed = np.where(missing_velocity, np.nan, speed)

    track_id = _numeric_column(candidates, "track_id")
    missing_track_id = ~np.isfinite(track_id)
    if current_track_id is None:
        same_track_id = np.zeros(len(candidates), dtype=float)
        track_id_changed = np.zeros(len(candidates), dtype=float)
    else:
        finite = np.isfinite(track_id)
        same = finite & (track_id.astype(float) == float(current_track_id))
        same_track_id = same.astype(float)
        track_id_changed = (~same & finite).astype(float)

    range_m = _numeric_column(candidates, "range_m")
    radial_velocity = _numeric_column(candidates, "radial_velocity_mps")
    num_inliers = _numeric_column(candidates, "num_inliers")
    return pd.DataFrame(
        {
            "log1p_association_nis": np.log1p(np.where(np.isfinite(nis), np.maximum(nis, 0.0), np.nan)),
            "sqrt_association_nis": np.sqrt(np.where(np.isfinite(nis), np.maximum(nis, 0.0), np.nan)),
            "position_residual_norm_m": position_residual,
            "horizontal_residual_norm_m": horizontal_residual,
            "abs_vertical_residual_m": np.abs(residuals[:, 2]),
            "candidate_count": np.full(len(candidates), float(len(candidates))),
            "candidate_rank_by_nis": ranks,
            "cat_prob_uav": np.where(missing_cat_prob, np.nan, clipped_cat_prob),
            "neg_log_cat_prob_uav": -np.log(clipped_cat_prob),
            "missing_cat_prob_uav": missing_cat_prob.astype(float),
            "track_id_changed": track_id_changed,
            "same_track_id": same_track_id,
            "missing_track_id": missing_track_id.astype(float),
            "velocity_residual_norm_mps": velocity_residual,
            "speed_mps": speed,
            "missing_velocity": missing_velocity.astype(float),
            "log1p_range_m": np.log1p(np.where(np.isfinite(range_m), np.maximum(range_m, 0.0), np.nan)),
            "abs_radial_velocity_mps": np.abs(radial_velocity),
            "missing_radial_velocity": (~np.isfinite(radial_velocity)).astype(float),
            "log1p_num_inliers": np.log1p(np.where(np.isfinite(num_inliers), np.maximum(num_inliers, 0.0), np.nan)),
            "missing_num_inliers": (~np.isfinite(num_inliers)).astype(float),
        }
    )


def score_radar_candidates_with_learned_likelihood(
    candidates: pd.DataFrame,
    *,
    model: LearnedRadarAssociationModel,
    tracker_state: np.ndarray,
    current_track_id: int | None,
) -> pd.DataFrame:
    """Append learned probability and NLL association score columns."""

    features = radar_association_feature_frame(
        candidates,
        tracker_state=tracker_state,
        current_track_id=current_track_id,
    )
    probabilities = np.clip(model.predict_proba_features(features), 1.0e-12, 1.0)
    scored = candidates.copy()
    scored["association_mode"] = "learned-likelihood"
    scored["association_action"] = "learned_likelihood"
    scored["association_learned_probability"] = probabilities
    scored["association_score"] = -np.log(probabilities)
    scored["association_candidate_rows"] = int(len(scored))
    return scored


def fit_learned_radar_association_model(
    examples: pd.DataFrame,
    *,
    feature_names: Iterable[str] = DEFAULT_FEATURE_NAMES,
    label_column: str = "label",
    l2: float = 1.0e-3,
    max_iter: int = 500,
    balance_classes: bool = True,
    metadata: dict[str, Any] | None = None,
) -> LearnedRadarAssociationModel:
    """Fit a standardized logistic association model."""

    names = tuple(str(name) for name in feature_names)
    labels = pd.to_numeric(examples[label_column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(labels)
    labels = (labels[keep] > 0.0).astype(float)
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        raise ValueError("training examples must contain both positive and negative labels")
    raw = _raw_feature_matrix(examples.loc[keep], names)
    mean = np.nanmean(raw, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    filled = np.where(np.isfinite(raw), raw, mean.reshape(1, -1))
    scale = np.std(filled, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    matrix = (filled - mean.reshape(1, -1)) / scale.reshape(1, -1)

    sample_weight = np.ones(labels.size, dtype=float)
    if balance_classes:
        positives = float(labels.sum())
        negatives = float(labels.size - positives)
        sample_weight = np.where(
            labels > 0.0,
            labels.size / max(2.0 * positives, 1.0),
            labels.size / max(2.0 * negatives, 1.0),
        )
    sample_weight /= max(float(sample_weight.sum()), 1.0)

    prior = np.clip(float(labels.mean()), 1.0e-6, 1.0 - 1.0e-6)
    initial = np.zeros(len(names) + 1, dtype=float)
    initial[-1] = np.log(prior / (1.0 - prior))
    result = minimize(
        _loss_and_grad,
        initial,
        args=(matrix, labels, sample_weight, float(l2)),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter)},
    )
    if not result.success:
        raise RuntimeError(f"learned radar association fit failed: {result.message}")
    return LearnedRadarAssociationModel(
        feature_names=names,
        mean=mean,
        scale=scale,
        weights=np.asarray(result.x[:-1], dtype=float),
        intercept=float(result.x[-1]),
        metadata=dict(metadata or {}),
    )


def _loss_and_grad(params: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    weights = params[:-1]
    intercept = params[-1]
    logits = x @ weights + intercept
    prob = _sigmoid(logits)
    eps = 1.0e-12
    loss = -np.sum(w * (y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps)))
    loss += 0.5 * l2 * float(weights @ weights)
    diff = w * (prob - y)
    grad_w = x.T @ diff + l2 * weights
    grad_b = float(np.sum(diff))
    return float(loss), np.concatenate([grad_w, np.array([grad_b])])


def _raw_feature_matrix(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    columns = []
    for name in names:
        if name in frame.columns:
            columns.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
        else:
            columns.append(np.full(len(frame), np.nan, dtype=float))
    return np.column_stack(columns) if columns else np.empty((len(frame), 0), dtype=float)


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _velocity_matrix_enu(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps")
    if not all(column in frame.columns for column in required):
        return np.zeros((len(frame), 3), dtype=float), np.ones(len(frame), dtype=bool)
    velocity = np.column_stack(
        [
            _numeric_column(frame, "velocity_east_mps"),
            _numeric_column(frame, "velocity_north_mps"),
            -_numeric_column(frame, "velocity_down_mps"),
        ]
    )
    missing = ~np.isfinite(velocity).all(axis=1)
    return np.where(np.isfinite(velocity), velocity, 0.0), missing


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return np.where(value >= 0.0, 1.0 / (1.0 + np.exp(-value)), np.exp(value) / (1.0 + np.exp(value)))
