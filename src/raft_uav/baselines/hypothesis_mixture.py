"""Soft output fusion for weighted tracking hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GaussianHypothesis:
    """One weighted Gaussian tracking hypothesis."""

    mean: np.ndarray
    covariance: np.ndarray
    log_weight: float = 0.0
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance must match mean dimension")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "log_weight", float(self.log_weight))


def moment_match_hypotheses(hypotheses: list[GaussianHypothesis]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return moment-matched mean/covariance and normalized weights."""

    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    weights = normalize_log_weights([hypothesis.log_weight for hypothesis in hypotheses])
    means = np.stack([hypothesis.mean for hypothesis in hypotheses], axis=0)
    mean = weights @ means
    covariance = np.zeros((mean.size, mean.size), dtype=float)
    for weight, hypothesis in zip(weights, hypotheses):
        diff = hypothesis.mean - mean
        covariance += float(weight) * (hypothesis.covariance + np.outer(diff, diff))
    return mean, _symmetrized(covariance), weights


def normalize_log_weights(log_weights: list[float] | np.ndarray) -> np.ndarray:
    """Normalize log weights to probabilities."""

    values = np.asarray(log_weights, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("log_weights must not be empty")
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return np.full(values.size, 1.0 / values.size)
    weights = np.exp(values - maximum)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(values.size, 1.0 / values.size)
    return weights / total


def position_mixture_from_association_rows(
    rows: pd.DataFrame,
    *,
    covariance_columns: tuple[str, str, str, str, str, str] = (
        "association_cov_ee",
        "association_cov_nn",
        "association_cov_uu",
        "association_cov_en",
        "association_cov_eu",
        "association_cov_nu",
    ),
    score_column: str = "association_score",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moment-match candidate rows into one soft position update."""

    if rows.empty:
        raise ValueError("rows must not be empty")
    positions = rows[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    scores = pd.to_numeric(rows.get(score_column, pd.Series(0.0, index=rows.index)), errors="coerce")
    log_weights = -scores.fillna(scores.max() if scores.notna().any() else 0.0).to_numpy(dtype=float)
    weights = normalize_log_weights(log_weights)
    mean = weights @ positions
    covariance = np.zeros((3, 3), dtype=float)
    has_covariance = all(column in rows.columns for column in covariance_columns)
    for index, (_, row) in enumerate(rows.iterrows()):
        row_covariance = _covariance_from_row(row, covariance_columns) if has_covariance else np.diag([25.0**2, 25.0**2, 35.0**2])
        diff = positions[index] - mean
        covariance += float(weights[index]) * (row_covariance + np.outer(diff, diff))
    return mean, _symmetrized(covariance), weights


def _covariance_from_row(row: pd.Series, columns: tuple[str, str, str, str, str, str]) -> np.ndarray:
    ee, nn, uu, en, eu, nu = [float(row[column]) for column in columns]
    covariance = np.array([[ee, en, eu], [en, nn, nu], [eu, nu, uu]], dtype=float)
    if not np.isfinite(covariance).all():
        return np.diag([25.0**2, 25.0**2, 35.0**2])
    return _symmetrized(covariance)


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
