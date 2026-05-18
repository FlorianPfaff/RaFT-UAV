"""Paper-style RF/radar/fusion diagnostic tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_association import run_async_cv_baseline_with_radar_association
from raft_uav.baselines.smoothing import smooth_tracking_records
from raft_uav.cli import _records_to_frame
from raft_uav.diagnostics.time_offset import (
    catprob_candidate_pool,
    highest_catprob_candidate,
    nearest_candidate_to_truth,
    radar_frame_groups,
    truth_position_at_time,
    truth_positions_at_times,
)
from raft_uav.io.aerpaw import (
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
)

RADAR_SELECTIONS = (
    "radar-highest-catprob",
    "radar-longest-track",
    "radar-oracle-nearest-truth",
    "radar-catprob-oracle-nearest",
)
FUSION_ASSOCIATIONS = ("prediction-nis", "tracklet-viterbi")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-diagnose-paper-table",
        description="write paper-style RF/radar/fusion diagnostic tables",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper-table"))
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.4)
    parser.add_argument("--truth-time-gate-s", type=float, default=2.0)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument(
        "--fusion-association",
        action="append",
        choices=FUSION_ASSOCIATIONS,
        default=None,
    )
    args = parser.parse_args(argv)
    result = run_paper_table_diagnostic(
        dataset_root=args.dataset_root,
        flight_name=args.flight,
        output_dir=args.output_dir,
        radar_catprob_threshold=args.radar_catprob_threshold,
        truth_time_gate_s=args.truth_time_gate_s,
        acceleration_std_mps2=args.acceleration_std,
        smoother_lag_s=args.smoother_lag_s,
        include_fusion=not args.skip_fusion,
        fusion_associations=tuple(args.fusion_association or FUSION_ASSOCIATIONS),
    )
    print(f"flight={result['flight']}")
    print(f"rows={result['rows']}")
    print(f"table_csv={result['table_csv']}")
    print(f"summary_json={result['summary_json']}")
    return 0


def run_paper_table_diagnostic(
    *,
    dataset_root: Path,
    flight_name: str,
    output_dir: Path = Path("outputs/paper-table"),
    radar_catprob_threshold: float = 0.4,
    truth_time_gate_s: float = 2.0,
    acceleration_std_mps2: float = 4.0,
    smoother_lag_s: float = 20.0,
    include_fusion: bool = True,
    fusion_associations: tuple[str, ...] = FUSION_ASSOCIATIONS,
) -> dict[str, Any]:
    """Build and write a paper-style comparison table for one flight."""

    flight = select_flight(Path(dataset_root), flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)
    truth = truth.sort_values("time_s").reset_index(drop=True)

    rf = pd.DataFrame()
    rf_measurements = []
    if flight.rf_csv is not None:
        rf = _inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time),
            truth,
        )
        rf_measurements = rf_measurements_to_enu(rf)

    radar = pd.DataFrame()
    if flight.radar_json is not None:
        radar = _inside_truth_window(
            normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
            truth,
        )

    rows: list[dict[str, Any]] = []
    if not rf.empty:
        rows.append(
            metric_row(
                method="RF raw",
                modality="rf",
                times_s=rf["time_s"].to_numpy(dtype=float),
                positions_m=rf[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
                truth=truth,
                candidate_count=len(rf),
                selected_count=len(rf),
                max_time_delta_s=truth_time_gate_s,
                track_ids=None,
            )
        )
    if not radar.empty:
        for selection in RADAR_SELECTIONS:
            selected = select_radar_for_table(
                radar=radar,
                truth=truth,
                selection=selection,
                catprob_threshold=radar_catprob_threshold,
                max_time_delta_s=truth_time_gate_s,
            )
            rows.append(
                metric_row(
                    method=selection,
                    modality="radar",
                    times_s=selected["time_s"].to_numpy(dtype=float)
                    if not selected.empty
                    else np.empty(0),
                    positions_m=selected[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
                    if not selected.empty
                    else np.empty((0, 3)),
                    truth=truth,
                    candidate_count=len(radar_frame_groups(radar)),
                    selected_count=len(selected),
                    max_time_delta_s=truth_time_gate_s,
                    track_ids=_track_ids(selected),
                )
            )
    if include_fusion and not radar.empty:
        for association in fusion_associations:
            rows.extend(
                fusion_rows(
                    association=association,
                    rf_measurements=rf_measurements,
                    radar=radar,
                    truth=truth,
                    radar_catprob_threshold=radar_catprob_threshold,
                    acceleration_std_mps2=acceleration_std_mps2,
                    smoother_lag_s=smoother_lag_s,
                    max_time_delta_s=truth_time_gate_s,
                )
            )

    table = pd.DataFrame.from_records(rows)
    flight_output = Path(output_dir) / flight.name
    flight_output.mkdir(parents=True, exist_ok=True)
    table_csv = flight_output / "paper_table.csv"
    summary_json = flight_output / "paper_table_summary.json"
    table.to_csv(table_csv, index=False)
    payload = {
        "flight": flight.name,
        "rows": int(len(table)),
        "radar_catprob_threshold": float(radar_catprob_threshold),
        "truth_time_gate_s": float(truth_time_gate_s),
        "table_csv": str(table_csv),
        "methods": table["method"].tolist() if "method" in table else [],
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "summary_json": str(summary_json)}


def fusion_rows(
    *,
    association: str,
    rf_measurements: list[Any],
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    radar_catprob_threshold: float,
    acceleration_std_mps2: float,
    smoother_lag_s: float,
    max_time_delta_s: float,
) -> list[dict[str, Any]]:
    """Run one fusion association and return unsmoothed/smoothed metric rows."""

    try:
        records, selected = run_async_cv_baseline_with_radar_association(
            rf_measurements=rf_measurements,
            radar=radar,
            association=association,
            truth=truth,
            acceleration_std_mps2=acceleration_std_mps2,
            candidate_catprob_threshold=radar_catprob_threshold,
        )
    except Exception as exc:
        return [_failed_row(f"fusion-{association}", exc)]
    if not records:
        return [_failed_row(f"fusion-{association}", "no-records")]

    rows = []
    estimate = _records_to_frame(records)
    rows.append(
        metric_row(
            method=f"fusion-{association}",
            modality="fusion",
            times_s=estimate["time_s"].to_numpy(dtype=float),
            positions_m=estimate[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
            truth=truth,
            candidate_count=len(estimate),
            selected_count=len(estimate),
            max_time_delta_s=max_time_delta_s,
            track_ids=_track_ids(selected),
        )
    )
    smoothed_records = smooth_tracking_records(
        records,
        method="fixed-lag",
        acceleration_std_mps2=acceleration_std_mps2,
        lag_s=smoother_lag_s,
    )
    smoothed = _records_to_frame(smoothed_records)
    rows.append(
        metric_row(
            method=f"fusion-{association}-fixed-lag",
            modality="fusion",
            times_s=smoothed["time_s"].to_numpy(dtype=float),
            positions_m=smoothed[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
            truth=truth,
            candidate_count=len(smoothed),
            selected_count=len(smoothed),
            max_time_delta_s=max_time_delta_s,
            track_ids=_track_ids(selected),
        )
    )
    return rows


def select_radar_for_table(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    selection: str,
    catprob_threshold: float,
    max_time_delta_s: float,
) -> pd.DataFrame:
    """Select one radar row per frame for paper-table diagnostics."""

    groups = radar_frame_groups(radar)
    longest_track = _longest_track_id(radar) if selection == "radar-longest-track" else None
    selected_rows: list[pd.Series] = []
    for group in groups:
        time_s = float(group["time_s"].median())
        truth_xyz = truth_position_at_time(truth, time_s, max_delta_s=max_time_delta_s)
        if selection == "radar-highest-catprob":
            selected = highest_catprob_candidate(group)
        elif selection == "radar-longest-track":
            if longest_track is None or "track_id" not in group:
                selected = None
            else:
                track_rows = group.loc[
                    pd.to_numeric(group["track_id"], errors="coerce") == longest_track
                ]
                selected = highest_catprob_candidate(track_rows)
        elif selection == "radar-oracle-nearest-truth":
            selected = nearest_candidate_to_truth(group, truth_xyz)
        elif selection == "radar-catprob-oracle-nearest":
            selected = nearest_candidate_to_truth(catprob_candidate_pool(group, catprob_threshold), truth_xyz)
        else:
            raise ValueError(f"unknown radar table selection {selection!r}")
        if selected is not None:
            selected_rows.append(selected.copy())
    if not selected_rows:
        return radar.iloc[0:0].copy()
    sort_columns = [
        column for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    return pd.DataFrame(selected_rows).sort_values(sort_columns).reset_index(drop=True)


def metric_row(
    *,
    method: str,
    modality: str,
    times_s: np.ndarray,
    positions_m: np.ndarray,
    truth: pd.DataFrame,
    candidate_count: int,
    selected_count: int,
    max_time_delta_s: float,
    track_ids: list[int] | None,
) -> dict[str, Any]:
    """Return one paper-style metric row."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    positions = np.asarray(positions_m, dtype=float)
    if positions.size == 0:
        positions = np.empty((0, 3))
    truth_positions, mask = truth_positions_at_times(truth, times, max_delta_s=max_time_delta_s)
    finite = mask & np.isfinite(positions[:, :3]).all(axis=1) if times.size else np.array([])
    errors_2d = (
        np.linalg.norm(positions[finite, :2] - truth_positions[finite, :2], axis=1)
        if times.size
        else np.empty(0)
    )
    errors_3d = (
        np.linalg.norm(positions[finite, :3] - truth_positions[finite, :3], axis=1)
        if times.size
        else np.empty(0)
    )
    row: dict[str, Any] = {
        "method": method,
        "modality": modality,
        "status": "ok",
        "candidate_count": int(candidate_count),
        "selected_count": int(selected_count),
        "matched_count": int(errors_3d.size),
        "coverage": _safe_ratio(int(errors_3d.size), int(candidate_count)),
        "track_switches": int(_track_switches(track_ids or [])),
        "track_ids": ",".join(str(value) for value in (track_ids or [])),
    }
    row.update(_error_summary(errors_2d, prefix="error_2d"))
    row.update(_error_summary(errors_3d, prefix="error_3d"))
    return row


def _error_summary(errors_m: np.ndarray, *, prefix: str) -> dict[str, float]:
    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return {
            f"{prefix}_mean_m": float("nan"),
            f"{prefix}_std_m": float("nan"),
            f"{prefix}_rmse_m": float("nan"),
            f"{prefix}_p50_m": float("nan"),
            f"{prefix}_p95_m": float("nan"),
            f"{prefix}_max_m": float("nan"),
        }
    return {
        f"{prefix}_mean_m": float(np.mean(errors)),
        f"{prefix}_std_m": float(np.std(errors)),
        f"{prefix}_rmse_m": float(np.sqrt(np.mean(errors**2))),
        f"{prefix}_p50_m": float(np.percentile(errors, 50.0)),
        f"{prefix}_p95_m": float(np.percentile(errors, 95.0)),
        f"{prefix}_max_m": float(np.max(errors)),
    }


def _failed_row(method: str, error: object) -> dict[str, Any]:
    return {
        "method": method,
        "modality": "fusion",
        "status": f"failed: {error}",
        "candidate_count": 0,
        "selected_count": 0,
        "matched_count": 0,
        "coverage": 0.0,
    }


def _track_ids(frame: pd.DataFrame) -> list[int]:
    if frame.empty or "track_id" not in frame.columns:
        return []
    values = pd.to_numeric(frame["track_id"], errors="coerce").dropna()
    return [int(value) for value in values.to_numpy(dtype=float)]


def _track_switches(track_ids: list[int]) -> int:
    if not track_ids:
        return 0
    switches = 0
    previous = int(track_ids[0])
    for value in track_ids[1:]:
        current = int(value)
        if current != previous:
            switches += 1
        previous = current
    return switches


def _longest_track_id(radar: pd.DataFrame) -> int | None:
    if radar.empty or "track_id" not in radar.columns:
        return None
    values = pd.to_numeric(radar["track_id"], errors="coerce").dropna()
    if values.empty:
        return None
    return int(values.astype(int).value_counts().idxmax())


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    truth_min = float(truth["time_s"].min())
    truth_max = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= truth_min) & (frame["time_s"] <= truth_max)].copy()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
