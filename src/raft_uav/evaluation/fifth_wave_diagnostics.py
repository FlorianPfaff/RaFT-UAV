"""Fifth-wave evaluation and safety diagnostics for RaFT-UAV.

The helpers in this module are intentionally dependency-light and additive.  They
are meant to make future tracker improvements easier to trust: paired
comparisons, block-bootstrap uncertainty, do-no-harm update decisions, recovery
metrics, ambiguity diagnostics, calibration-transfer checks, deterministic
artifact comparisons, and conservative leaderboard ranking.

None of the functions assumes access to the raw dataset.  They operate on the
CSV artifacts already written by the RaFT-UAV runners whenever possible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd


MetricFunction = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class BootstrapInterval:
    """A block-bootstrap confidence interval for one scalar statistic."""

    metric: str
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    block_size: int
    resamples: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "metric": self.metric,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "samples": self.samples,
            "block_size": self.block_size,
            "resamples": self.resamples,
        }


@dataclass(frozen=True)
class DoNoHarmDecision:
    """Decision made by a conservative radar-update safety controller."""

    action: str
    risk_score: float
    reasons: tuple[str, ...]
    covariance_scale: float = 1.0
    defer_lag_s: float = 0.0

    @property
    def should_apply(self) -> bool:
        return self.action in {"apply", "soften"}


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Runtime and memory instrumentation for one experiment block."""

    label: str
    wall_time_s: float
    cpu_time_s: float
    peak_memory_mb: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "label": self.label,
            "wall_time_s": self.wall_time_s,
            "cpu_time_s": self.cpu_time_s,
            "peak_memory_mb": self.peak_memory_mb,
        }


class RuntimeMonitor(AbstractContextManager["RuntimeMonitor"]):
    """Context manager that records wall time, CPU time, and peak memory.

    Example
    -------
    >>> with RuntimeMonitor("tracklet") as monitor:
    ...     _ = sum(range(10))
    >>> monitor.snapshot.wall_time_s >= 0.0
    True
    """

    def __init__(self, label: str = "run") -> None:
        self.label = str(label)
        self.snapshot = RuntimeSnapshot(self.label, 0.0, 0.0, 0.0)
        self._wall_start = 0.0
        self._cpu_start = 0.0
        self._started_trace = False

    def __enter__(self) -> "RuntimeMonitor":
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_trace = True
        else:
            self._started_trace = False
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        _, peak = tracemalloc.get_traced_memory()
        if self._started_trace:
            tracemalloc.stop()
        self.snapshot = RuntimeSnapshot(
            label=self.label,
            wall_time_s=float(time.perf_counter() - self._wall_start),
            cpu_time_s=float(time.process_time() - self._cpu_start),
            peak_memory_mb=float(peak / (1024.0**2)),
        )
        return False


def block_bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    *,
    metric: str | MetricFunction = "mean",
    block_size: int = 50,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> BootstrapInterval:
    """Return a block-bootstrap interval for autocorrelated trajectory errors.

    Values are split into contiguous non-overlapping blocks and blocks are sampled
    with replacement.  This is a conservative alternative to treating every
    timestamp as independent.
    """

    x = _finite_vector(values)
    if x.size == 0:
        return BootstrapInterval(_metric_name(metric), np.nan, np.nan, np.nan, confidence, 0, block_size, resamples)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    fn = _metric_function(metric)
    estimate = fn(x)
    blocks = _contiguous_blocks(x, min(int(block_size), x.size))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(resamples), dtype=float)
    for idx in range(int(resamples)):
        sampled = [blocks[int(i)] for i in rng.integers(0, len(blocks), size=len(blocks))]
        draws[idx] = fn(np.concatenate(sampled)[: x.size])
    alpha = 1.0 - float(confidence)
    return BootstrapInterval(
        metric=_metric_name(metric),
        estimate=float(estimate),
        lower=float(np.percentile(draws, 100.0 * alpha / 2.0)),
        upper=float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0))),
        confidence=float(confidence),
        samples=int(x.size),
        block_size=int(block_size),
        resamples=int(resamples),
    )


def paired_error_delta_frame(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
    dimensions: int = 3,
    label_a: str = "method_a",
    label_b: str = "method_b",
) -> pd.DataFrame:
    """Return per-truth-timestamp paired error deltas for two methods.

    The delta is ``error_a - error_b``.  Negative values mean method A is better
    at that truth timestamp.
    """

    _validate_position_frame(method_a, "method_a")
    _validate_position_frame(method_b, "method_b")
    _validate_position_frame(truth, "truth")
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_xyz = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    a_idx, a_dt = _nearest_indices_and_deltas(method_a["time_s"].to_numpy(dtype=float), truth_times)
    b_idx, b_dt = _nearest_indices_and_deltas(method_b["time_s"].to_numpy(dtype=float), truth_times)
    keep = (a_idx >= 0) & (b_idx >= 0) & (a_dt <= float(max_time_delta_s)) & (b_dt <= float(max_time_delta_s))
    a_pos = method_a[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[a_idx[keep]]
    b_pos = method_b[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[b_idx[keep]]
    ref = truth_xyz[keep]
    error_a = np.linalg.norm(a_pos[:, :dimensions] - ref[:, :dimensions], axis=1)
    error_b = np.linalg.norm(b_pos[:, :dimensions] - ref[:, :dimensions], axis=1)
    return pd.DataFrame(
        {
            "time_s": truth_times[keep],
            "error_a_m": error_a,
            "error_b_m": error_b,
            "delta_error_m": error_a - error_b,
            "abs_delta_error_m": np.abs(error_a - error_b),
            "method_a": label_a,
            "method_b": label_b,
            "a_time_delta_s": a_dt[keep],
            "b_time_delta_s": b_dt[keep],
        }
    )


def paired_delta_summary(
    delta_frame: pd.DataFrame,
    *,
    block_size: int = 50,
    resamples: int = 2000,
    seed: int | None = 0,
) -> dict[str, object]:
    """Summarize paired error deltas with block-bootstrap intervals."""

    if delta_frame.empty:
        return {"paired_samples": 0}
    delta = pd.to_numeric(delta_frame["delta_error_m"], errors="coerce").to_numpy(dtype=float)
    mean_ci = block_bootstrap_interval(delta, metric="mean", block_size=block_size, resamples=resamples, seed=seed)
    median_ci = block_bootstrap_interval(delta, metric="median", block_size=block_size, resamples=resamples, seed=seed)
    return {
        "paired_samples": int(np.isfinite(delta).sum()),
        "mean_delta_error_m": mean_ci.estimate,
        "mean_delta_lower_m": mean_ci.lower,
        "mean_delta_upper_m": mean_ci.upper,
        "median_delta_error_m": median_ci.estimate,
        "median_delta_lower_m": median_ci.lower,
        "median_delta_upper_m": median_ci.upper,
        "fraction_a_better": float(np.mean(delta < 0.0)),
        "fraction_b_better": float(np.mean(delta > 0.0)),
    }


def do_no_harm_radar_decision(
    *,
    association_nis: float | None = None,
    gate_threshold: float | None = None,
    association_confidence: float | None = None,
    candidate_entropy: float | None = None,
    top1_top2_margin: float | None = None,
    rf_anchor_nis: float | None = None,
    rf_anchor_gate_nis: float | None = None,
    track_switch_risk: float | None = None,
    miss_streak: int = 0,
    recent_recovery_mode: bool = False,
) -> DoNoHarmDecision:
    """Choose whether a radar update should be applied, softened, deferred, or skipped."""

    risk = 0.0
    reasons: list[str] = []
    if association_nis is not None and gate_threshold is not None and gate_threshold > 0.0:
        ratio = float(association_nis) / float(gate_threshold)
        if ratio > 1.0:
            risk += min(2.0, ratio - 1.0)
            reasons.append("nis_above_gate")
        elif ratio > 0.75:
            risk += 0.35
            reasons.append("nis_near_gate")
    if association_confidence is not None and float(association_confidence) < 0.35:
        risk += 0.75
        reasons.append("low_association_confidence")
    if candidate_entropy is not None and float(candidate_entropy) > 1.0:
        risk += min(1.0, 0.4 * float(candidate_entropy))
        reasons.append("high_candidate_entropy")
    if top1_top2_margin is not None and float(top1_top2_margin) < 0.25:
        risk += 0.6
        reasons.append("small_top1_top2_margin")
    if rf_anchor_nis is not None and rf_anchor_gate_nis is not None and rf_anchor_gate_nis > 0.0:
        if float(rf_anchor_nis) > float(rf_anchor_gate_nis):
            risk += 1.0
            reasons.append("rf_anchor_disagreement")
    if track_switch_risk is not None and float(track_switch_risk) > 0.5:
        risk += float(track_switch_risk)
        reasons.append("track_switch_risk")
    if miss_streak >= 3:
        risk += 0.4
        reasons.append("long_miss_streak_reacquisition")
    if recent_recovery_mode:
        risk += 0.5
        reasons.append("recent_recovery_mode")

    if risk >= 2.5:
        return DoNoHarmDecision("skip", float(risk), tuple(reasons), covariance_scale=float("inf"))
    if risk >= 1.5:
        return DoNoHarmDecision("defer", float(risk), tuple(reasons), covariance_scale=4.0, defer_lag_s=5.0)
    if risk >= 0.75:
        return DoNoHarmDecision("soften", float(risk), tuple(reasons), covariance_scale=1.0 + risk)
    return DoNoHarmDecision("apply", float(risk), tuple(reasons), covariance_scale=1.0)


def candidate_ambiguity_index(
    candidates: pd.DataFrame,
    *,
    score_column: str = "association_score",
    lower_score_is_better: bool = True,
) -> dict[str, float | int]:
    """Return frame-level ambiguity diagnostics from candidate scores."""

    if candidates.empty:
        return {
            "candidate_count": 0,
            "top1_top2_score_margin": np.nan,
            "candidate_entropy": np.nan,
            "effective_candidate_count": 0.0,
        }
    if score_column in candidates.columns:
        scores = pd.to_numeric(candidates[score_column], errors="coerce").to_numpy(dtype=float)
    elif "association_nis" in candidates.columns:
        scores = pd.to_numeric(candidates["association_nis"], errors="coerce").to_numpy(dtype=float)
    else:
        scores = np.arange(len(candidates), dtype=float)
    probabilities = _softmax(-scores if lower_score_is_better else scores)
    ordered = np.sort(scores[np.isfinite(scores)])
    margin = float(ordered[1] - ordered[0]) if ordered.size >= 2 else float("inf")
    entropy = _entropy(probabilities)
    return {
        "candidate_count": int(len(candidates)),
        "top1_top2_score_margin": margin,
        "candidate_entropy": entropy,
        "effective_candidate_count": float(1.0 / np.sum(probabilities**2)) if probabilities.size else 0.0,
        "top_candidate_probability": float(np.max(probabilities)) if probabilities.size else np.nan,
    }


def ambiguity_by_frame(
    radar_candidates: pd.DataFrame,
    *,
    frame_column: str = "frame_index",
    score_column: str = "association_score",
) -> pd.DataFrame:
    """Compute candidate ambiguity diagnostics for every radar frame."""

    if radar_candidates.empty:
        return pd.DataFrame()
    group_column = frame_column if frame_column in radar_candidates.columns else "time_s"
    rows = []
    for key, group in radar_candidates.groupby(group_column, sort=True):
        row = {group_column: key, **candidate_ambiguity_index(group, score_column=score_column)}
        if "time_s" in group.columns:
            row["time_s"] = float(pd.to_numeric(group["time_s"], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def recovery_events(
    times_s: Sequence[float] | np.ndarray,
    errors_m: Sequence[float] | np.ndarray,
    *,
    threshold_m: float,
) -> pd.DataFrame:
    """Return contiguous catastrophic-error events and their recovery durations."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    if times.size != errors.size:
        raise ValueError("times_s and errors_m must have the same length")
    finite = np.isfinite(times) & np.isfinite(errors)
    times = times[finite]
    errors = errors[finite]
    if times.size == 0:
        return pd.DataFrame(columns=["start_time_s", "end_time_s", "duration_s", "max_error_m"])
    bad = errors > float(threshold_m)
    rows: list[dict[str, float | int]] = []
    start: int | None = None
    for idx, is_bad in enumerate(bad):
        if is_bad and start is None:
            start = idx
        if start is not None and (not is_bad or idx == len(bad) - 1):
            end = idx - 1 if not is_bad else idx
            rows.append(
                {
                    "start_index": int(start),
                    "end_index": int(end),
                    "start_time_s": float(times[start]),
                    "end_time_s": float(times[end]),
                    "duration_s": float(times[end] - times[start]),
                    "max_error_m": float(np.max(errors[start : end + 1])),
                }
            )
            start = None
    return pd.DataFrame.from_records(rows)


def bad_segment_table(
    times_s: Sequence[float] | np.ndarray,
    errors_m: Sequence[float] | np.ndarray,
    *,
    window_s: float = 20.0,
    stride_s: float = 5.0,
    top_k: int = 10,
) -> pd.DataFrame:
    """Mine the worst contiguous time windows by RMSE and p95 error."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    finite = np.isfinite(times) & np.isfinite(errors)
    times = times[finite]
    errors = errors[finite]
    if times.size == 0:
        return pd.DataFrame()
    rows: list[dict[str, float | int]] = []
    start = float(times[0])
    end_time = float(times[-1])
    while start <= end_time:
        end = start + float(window_s)
        keep = (times >= start) & (times < end)
        if np.any(keep):
            e = errors[keep]
            rows.append(
                {
                    "start_time_s": start,
                    "end_time_s": end,
                    "sample_count": int(e.size),
                    "rmse_m": float(np.sqrt(np.mean(e**2))),
                    "mae_m": float(np.mean(np.abs(e))),
                    "p95_m": float(np.percentile(e, 95)),
                    "max_m": float(np.max(e)),
                }
            )
        start += float(stride_s)
    return pd.DataFrame.from_records(rows).sort_values(
        ["p95_m", "rmse_m"], ascending=[False, False]
    ).head(int(top_k))


def calibration_transfer_summary(
    train_diagnostics: pd.DataFrame,
    heldout_diagnostics: pd.DataFrame,
    *,
    source_column: str = "source",
    nis_column: str = "nis",
) -> pd.DataFrame:
    """Compare innovation calibration statistics between training and held-out runs."""

    train = _nis_group_summary(train_diagnostics, source_column=source_column, nis_column=nis_column)
    heldout = _nis_group_summary(heldout_diagnostics, source_column=source_column, nis_column=nis_column)
    merged = train.merge(heldout, on=source_column, how="outer", suffixes=("_train", "_heldout"))
    for column in ("mean_nis", "p95_nis"):
        merged[f"{column}_transfer_delta"] = merged[f"{column}_heldout"] - merged[f"{column}_train"]
    return merged


def error_attribution_by_source_sequence(
    estimates: pd.DataFrame,
    errors_m: Sequence[float] | np.ndarray,
    *,
    source_column: str = "source",
    gap_threshold_s: float = 5.0,
) -> pd.DataFrame:
    """Summarize errors conditioned on recent measurement-source history."""

    if estimates.empty:
        return pd.DataFrame()
    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    n = min(len(estimates), errors.size)
    frame = estimates.iloc[:n].copy()
    frame["error_m"] = errors[:n]
    source = frame[source_column].astype(str) if source_column in frame.columns else pd.Series(["unknown"] * n)
    dt = frame["time_s"].diff().fillna(0.0) if "time_s" in frame.columns else pd.Series([0.0] * n)
    previous_source = source.shift(1).fillna("start")
    labels = []
    for idx in range(n):
        if dt.iloc[idx] > gap_threshold_s:
            labels.append("after_long_gap")
        elif source.iloc[idx] == "radar" and previous_source.iloc[idx] != "radar":
            labels.append("first_radar_after_nonradar")
        elif source.iloc[idx] == "rf" and previous_source.iloc[idx] != "rf":
            labels.append("first_rf_after_nonrf")
        else:
            labels.append(f"last_{source.iloc[idx]}")
    frame["source_sequence_label"] = labels
    return _group_error_summary(frame, "source_sequence_label", "error_m")


def vertical_horizontal_error_summary(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
) -> dict[str, float | int]:
    """Return separate horizontal and vertical error metrics."""

    aligned = _aligned_error_components(estimates, truth, max_time_delta_s=max_time_delta_s)
    if aligned.empty:
        return {"matched_count": 0}
    h = aligned["horizontal_error_m"].to_numpy(dtype=float)
    v = np.abs(aligned["vertical_error_m"].to_numpy(dtype=float))
    return {
        "matched_count": int(len(aligned)),
        "horizontal_rmse_m": float(np.sqrt(np.mean(h**2))),
        "horizontal_p95_m": float(np.percentile(h, 95)),
        "up_rmse_m": float(np.sqrt(np.mean(v**2))),
        "up_p95_m": float(np.percentile(v, 95)),
        "vertical_to_horizontal_rmse_ratio": float(np.sqrt(np.mean(v**2)) / max(np.sqrt(np.mean(h**2)), 1e-12)),
    }


def track_purity_summary(selected_radar: pd.DataFrame, *, track_column: str = "track_id") -> dict[str, float | int | None]:
    """Return selected-track purity and entropy diagnostics."""

    if selected_radar.empty or track_column not in selected_radar.columns:
        return {
            "selected_radar_rows": int(len(selected_radar)),
            "dominant_track_id": None,
            "dominant_track_fraction": np.nan,
            "selected_track_entropy": np.nan,
            "selected_track_count": 0,
        }
    tracks = pd.to_numeric(selected_radar[track_column], errors="coerce").dropna().astype(int)
    if tracks.empty:
        return {
            "selected_radar_rows": int(len(selected_radar)),
            "dominant_track_id": None,
            "dominant_track_fraction": np.nan,
            "selected_track_entropy": np.nan,
            "selected_track_count": 0,
        }
    counts = tracks.value_counts(sort=True)
    probs = counts.to_numpy(dtype=float) / float(counts.sum())
    return {
        "selected_radar_rows": int(len(selected_radar)),
        "dominant_track_id": int(counts.index[0]),
        "dominant_track_fraction": float(probs[0]),
        "selected_track_entropy": _entropy(probs),
        "selected_track_count": int(len(counts)),
    }


def conservative_leaderboard_rank(
    rows: pd.DataFrame,
    *,
    objective: str = "p95_3d_error_m",
    constraints: Mapping[str, tuple[str, float]] | None = None,
) -> pd.DataFrame:
    """Rank methods by a robust objective after applying hard constraints.

    Constraint operators are ``ge``, ``gt``, ``le``, ``lt``, and ``eq``.
    Ineligible rows remain visible with ``eligible=False`` and no robust rank.
    """

    if rows.empty:
        return rows.copy()
    out = rows.copy()
    eligible = np.ones(len(out), dtype=bool)
    for column, (op, value) in (constraints or {}).items():
        if column not in out.columns:
            eligible &= False
            continue
        series = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        eligible &= _constraint_mask(series, op, float(value))
    out["eligible"] = eligible
    out["robust_rank"] = np.nan
    if objective not in out.columns:
        return out
    subset = out.loc[eligible].copy()
    if subset.empty:
        return out
    order = subset.sort_values(objective, ascending=True).index
    out.loc[order, "robust_rank"] = np.arange(1, len(order) + 1)
    return out


def deterministic_artifact_summary(
    estimates_a: pd.DataFrame,
    estimates_b: pd.DataFrame,
    *,
    selected_a: pd.DataFrame | None = None,
    selected_b: pd.DataFrame | None = None,
    atol: float = 1.0e-9,
) -> dict[str, object]:
    """Compare two runs of the same method for deterministic artifacts."""

    summary: dict[str, object] = {
        "estimate_rows_a": int(len(estimates_a)),
        "estimate_rows_b": int(len(estimates_b)),
        "estimate_row_count_equal": bool(len(estimates_a) == len(estimates_b)),
    }
    common = [c for c in ("time_s", "east_m", "north_m", "up_m") if c in estimates_a.columns and c in estimates_b.columns]
    if common and len(estimates_a) == len(estimates_b):
        diff = estimates_a[common].to_numpy(dtype=float) - estimates_b[common].to_numpy(dtype=float)
        summary["estimate_max_abs_delta"] = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
        summary["estimates_nearly_equal"] = bool(np.allclose(diff, 0.0, atol=float(atol), rtol=0.0))
    if selected_a is not None and selected_b is not None:
        summary["selected_rows_a"] = int(len(selected_a))
        summary["selected_rows_b"] = int(len(selected_b))
        summary["selected_row_count_equal"] = bool(len(selected_a) == len(selected_b))
        key_cols = [c for c in ("time_s", "track_id", "east_m", "north_m", "up_m") if c in selected_a.columns and c in selected_b.columns]
        if key_cols and len(selected_a) == len(selected_b):
            summary["selected_rows_equal"] = bool(selected_a[key_cols].reset_index(drop=True).equals(selected_b[key_cols].reset_index(drop=True)))
    return summary


def pseudo_label_candidates(
    candidates: pd.DataFrame,
    *,
    min_catprob_positive: float = 0.9,
    max_anchor_nis_positive: float = 4.0,
    max_catprob_negative: float = 0.1,
    min_anchor_nis_negative: float = 25.0,
) -> pd.DataFrame:
    """Create conservative oracle-free pseudo-labels for weak-label expansion."""

    frame = candidates.copy()
    cat = pd.to_numeric(frame.get("cat_prob_uav", pd.Series(np.nan, index=frame.index)), errors="coerce")
    anchor = pd.to_numeric(frame.get("association_anchor_nis", pd.Series(np.nan, index=frame.index)), errors="coerce")
    labels = np.full(len(frame), np.nan)
    positive = (cat >= float(min_catprob_positive)) & (anchor <= float(max_anchor_nis_positive))
    negative = (cat <= float(max_catprob_negative)) | (anchor >= float(min_anchor_nis_negative))
    labels[positive.to_numpy(dtype=bool)] = 1.0
    labels[negative.to_numpy(dtype=bool)] = 0.0
    frame["pseudo_label"] = labels
    frame["pseudo_label_source"] = np.where(np.isfinite(labels), "high_confidence_rules", "unlabeled")
    return frame


def adaptive_smoothing_lag_s(
    *,
    base_lag_s: float = 2.0,
    max_lag_s: float = 20.0,
    candidate_entropy: float | None = None,
    association_confidence: float | None = None,
    miss_streak: int = 0,
    recovery_mode: bool = False,
) -> float:
    """Return an adaptive fixed-lag horizon from confidence diagnostics."""

    lag = float(base_lag_s)
    if candidate_entropy is not None:
        lag += 2.0 * max(0.0, float(candidate_entropy) - 0.5)
    if association_confidence is not None:
        lag += 5.0 * max(0.0, 0.5 - float(association_confidence))
    lag += min(8.0, 1.5 * max(0, int(miss_streak)))
    if recovery_mode:
        lag += 5.0
    return float(np.clip(lag, float(base_lag_s), float(max_lag_s)))


def residual_whiteness_summary(
    diagnostics: pd.DataFrame,
    *,
    value_column: str = "nis",
    max_lag: int = 10,
) -> pd.DataFrame:
    """Compute simple residual/NIS autocorrelation and Ljung-Box-style diagnostics."""

    if diagnostics.empty or value_column not in diagnostics.columns:
        return pd.DataFrame()
    source = diagnostics["source"] if "source" in diagnostics.columns else pd.Series(["all"] * len(diagnostics))
    rows = []
    for name, group in diagnostics.groupby(source, sort=True):
        values = _finite_vector(group[value_column].to_numpy(dtype=float))
        autocorr = _autocorrelation(values, max_lag=max_lag)
        q = _ljung_box_q(values, autocorr)
        row = {"source": name, "sample_count": int(values.size), "ljung_box_q": q}
        for lag, value in enumerate(autocorr, start=1):
            row[f"autocorr_lag_{lag}"] = float(value)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def method_family_ensemble_decision(
    *,
    stable_segment_available: bool = False,
    learned_confidence: float | None = None,
    rf_radar_disagreement: float | None = None,
    recovery_mode: bool = False,
    ambiguity_entropy: float | None = None,
) -> str:
    """Rule-based method-family selector for heterogeneous tracker regimes."""

    if recovery_mode or (rf_radar_disagreement is not None and rf_radar_disagreement > 25.0):
        return "rf_fallback_or_recovery"
    if stable_segment_available and (ambiguity_entropy is None or ambiguity_entropy < 0.75):
        return "stable_segments"
    if learned_confidence is not None and learned_confidence >= 0.65:
        return "learned_tracklet"
    if ambiguity_entropy is not None and ambiguity_entropy >= 1.25:
        return "pda_or_mht"
    return "tracklet_viterbi"


def accuracy_runtime_pareto(
    rows: pd.DataFrame,
    *,
    error_column: str = "p95_3d_error_m",
    runtime_column: str = "wall_time_s",
) -> pd.DataFrame:
    """Mark the accuracy/runtime Pareto front."""

    if rows.empty or error_column not in rows.columns or runtime_column not in rows.columns:
        out = rows.copy()
        out["pareto_accuracy_runtime"] = False
        return out
    out = rows.copy()
    errors = pd.to_numeric(out[error_column], errors="coerce").to_numpy(dtype=float)
    runtimes = pd.to_numeric(out[runtime_column], errors="coerce").to_numpy(dtype=float)
    front = np.zeros(len(out), dtype=bool)
    for i in range(len(out)):
        if not np.isfinite(errors[i]) or not np.isfinite(runtimes[i]):
            continue
        dominated = np.any(
            (errors <= errors[i])
            & (runtimes <= runtimes[i])
            & ((errors < errors[i]) | (runtimes < runtimes[i]))
        )
        front[i] = not dominated
    out["pareto_accuracy_runtime"] = front
    return out


def oracle_replay_realistic_gap(
    real_errors_m: Sequence[float] | np.ndarray,
    oracle_replay_errors_m: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Compare a real method against oracle association under the same filtering path."""

    real = _finite_vector(real_errors_m)
    oracle = _finite_vector(oracle_replay_errors_m)
    n = min(real.size, oracle.size)
    if n == 0:
        return {"paired_samples": 0}
    real = real[:n]
    oracle = oracle[:n]
    return {
        "paired_samples": int(n),
        "real_rmse_m": float(np.sqrt(np.mean(real**2))),
        "oracle_replay_rmse_m": float(np.sqrt(np.mean(oracle**2))),
        "association_gap_rmse_m": float(np.sqrt(np.mean(real**2)) - np.sqrt(np.mean(oracle**2))),
        "real_p95_m": float(np.percentile(real, 95)),
        "oracle_replay_p95_m": float(np.percentile(oracle, 95)),
    }


def _aligned_error_components(estimates: pd.DataFrame, truth: pd.DataFrame, *, max_time_delta_s: float) -> pd.DataFrame:
    _validate_position_frame(estimates, "estimates")
    _validate_position_frame(truth, "truth")
    truth_times = truth["time_s"].to_numpy(dtype=float)
    estimate_times = estimates["time_s"].to_numpy(dtype=float)
    idx, dt = _nearest_indices_and_deltas(estimate_times, truth_times)
    keep = (idx >= 0) & (dt <= float(max_time_delta_s))
    est = estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[idx[keep]]
    ref = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[keep]
    residual = est - ref
    return pd.DataFrame(
        {
            "time_s": truth_times[keep],
            "horizontal_error_m": np.linalg.norm(residual[:, :2], axis=1),
            "vertical_error_m": residual[:, 2],
            "time_delta_s": dt[keep],
        }
    )


def _validate_position_frame(frame: pd.DataFrame, name: str) -> None:
    required = {"time_s", "east_m", "north_m", "up_m"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _nearest_indices_and_deltas(source_times_s: np.ndarray, query_times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_times_s, dtype=float).reshape(-1)
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    if source.size == 0:
        return np.full(query.size, -1, dtype=int), np.full(query.size, np.inf)
    order = np.argsort(source)
    sorted_source = source[order]
    insertion = np.searchsorted(sorted_source, query)
    right = np.clip(insertion, 0, sorted_source.size - 1)
    left = np.clip(insertion - 1, 0, sorted_source.size - 1)
    choose_right = np.abs(sorted_source[right] - query) < np.abs(sorted_source[left] - query)
    sorted_idx = np.where(choose_right, right, left)
    idx = order[sorted_idx]
    return idx.astype(int), np.abs(source[idx] - query)


def _finite_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _metric_function(metric: str | MetricFunction) -> MetricFunction:
    if callable(metric):
        return metric
    if metric == "mean":
        return lambda x: float(np.mean(x))
    if metric == "median":
        return lambda x: float(np.median(x))
    if metric == "rmse":
        return lambda x: float(np.sqrt(np.mean(x**2)))
    if metric == "mae":
        return lambda x: float(np.mean(np.abs(x)))
    if metric == "p95":
        return lambda x: float(np.percentile(x, 95))
    raise ValueError(f"unknown metric {metric!r}")


def _metric_name(metric: str | MetricFunction) -> str:
    return metric if isinstance(metric, str) else getattr(metric, "__name__", "custom")


def _contiguous_blocks(values: np.ndarray, block_size: int) -> list[np.ndarray]:
    return [values[start : start + block_size] for start in range(0, values.size, block_size)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float).reshape(-1)
    x = np.where(np.isfinite(x), x, -np.inf)
    if x.size == 0 or not np.isfinite(x).any():
        return np.full(x.size, 1.0 / max(x.size, 1))
    shifted = x - np.max(x)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    return weights / total if total > 0.0 else np.full(x.size, 1.0 / x.size)


def _entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    p = p[np.isfinite(p) & (p > 0.0)]
    return float(-np.sum(p * np.log(p))) if p.size else 0.0


def _nis_group_summary(frame: pd.DataFrame, *, source_column: str, nis_column: str) -> pd.DataFrame:
    if frame.empty or nis_column not in frame.columns:
        return pd.DataFrame(columns=[source_column, "count", "mean_nis", "p95_nis"])
    source = frame[source_column] if source_column in frame.columns else pd.Series(["all"] * len(frame))
    rows = []
    for name, group in frame.groupby(source, sort=True):
        values = _finite_vector(group[nis_column].to_numpy(dtype=float))
        rows.append(
            {
                source_column: name,
                "count": int(values.size),
                "mean_nis": float(np.mean(values)) if values.size else np.nan,
                "p95_nis": float(np.percentile(values, 95)) if values.size else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _group_error_summary(frame: pd.DataFrame, group_column: str, error_column: str) -> pd.DataFrame:
    rows = []
    for name, group in frame.groupby(group_column, sort=True):
        values = _finite_vector(group[error_column].to_numpy(dtype=float))
        if values.size == 0:
            continue
        rows.append(
            {
                group_column: name,
                "count": int(values.size),
                "rmse_m": float(np.sqrt(np.mean(values**2))),
                "mae_m": float(np.mean(np.abs(values))),
                "p95_m": float(np.percentile(values, 95)),
                "max_m": float(np.max(values)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _constraint_mask(values: np.ndarray, op: str, target: float) -> np.ndarray:
    if op == "ge":
        return values >= target
    if op == "gt":
        return values > target
    if op == "le":
        return values <= target
    if op == "lt":
        return values < target
    if op == "eq":
        return np.isclose(values, target)
    raise ValueError(f"unknown constraint operator {op!r}")


def _autocorrelation(values: np.ndarray, *, max_lag: int) -> np.ndarray:
    x = _finite_vector(values)
    if x.size <= 1:
        return np.full(max(0, int(max_lag)), np.nan)
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.zeros(max(0, int(max_lag)))
    out = []
    for lag in range(1, int(max_lag) + 1):
        if lag >= x.size:
            out.append(np.nan)
        else:
            out.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(out, dtype=float)


def _ljung_box_q(values: np.ndarray, autocorr: np.ndarray) -> float:
    n = _finite_vector(values).size
    if n <= 1:
        return np.nan
    q = 0.0
    for k, rho in enumerate(autocorr, start=1):
        if np.isfinite(rho) and n > k:
            q += float(rho**2 / (n - k))
    return float(n * (n + 2) * q)


def estimate_error_frame(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
) -> pd.DataFrame:
    """Align estimates to nearest truth samples and return per-estimate errors."""

    _validate_position_frame(estimates, "estimates")
    _validate_position_frame(truth, "truth")
    estimate_times = estimates["time_s"].to_numpy(dtype=float)
    truth_times = truth["time_s"].to_numpy(dtype=float)
    idx, dt = _nearest_indices_and_deltas(truth_times, estimate_times)
    keep = (idx >= 0) & (dt <= float(max_time_delta_s))
    est = estimates.loc[keep, ["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    ref = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[idx[keep]]
    residual = est - ref
    out = estimates.loc[keep].copy().reset_index(drop=True)
    out["truth_time_delta_s"] = dt[keep]
    out["error_2d_m"] = np.linalg.norm(residual[:, :2], axis=1)
    out["error_3d_m"] = np.linalg.norm(residual, axis=1)
    out["vertical_error_m"] = residual[:, 2]
    return out
