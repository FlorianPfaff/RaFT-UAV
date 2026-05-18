"""Paper-style RF/radar/fusion diagnostic tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
    TrackingUpdateDiagnostics,
    gate_threshold_from_probability,
)
from raft_uav.baselines.radar_association import (
    _events,
    _catprob_penalty,
    _nis_scored_candidates,
    _optional_float,
    _optional_track_id,
    _radar_row_to_measurement,
    _selected_rows_frame,
    _track_switch_penalty,
    _weight_entropy,
    run_async_cv_baseline_with_radar_association,
)
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
    "radar-longest-track-range-gated",
    "radar-longest-continuous-track-range-gated",
    "radar-longest-track-range-gated-interpolated",
    "radar-stable-segments-range-gated",
    "radar-stable-segments-range-gated-interpolated",
    "radar-oracle-nearest-truth",
    "radar-catprob-oracle-nearest",
)
RANGE_GATED_RADAR_SELECTIONS = (
    "radar-longest-track-range-gated",
    "radar-longest-continuous-track-range-gated",
    "radar-longest-track-range-gated-interpolated",
    "radar-stable-segments-range-gated",
    "radar-stable-segments-range-gated-interpolated",
)
FUSION_ASSOCIATIONS = (
    "prediction-nis",
    "tracklet-viterbi",
    "paper-compatible",
    "paper-longest-track",
)


@dataclass(frozen=True)
class _TrackSegment:
    frame: pd.DataFrame
    track_id: int
    start_time_s: float
    end_time_s: float
    start_position_m: np.ndarray
    end_position_m: np.ndarray
    frames: int
    mean_catprob: float

    @property
    def score(self) -> float:
        return float(self.frames) * max(self.mean_catprob, 0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-diagnose-paper-table",
        description="write paper-style RF/radar/fusion diagnostic tables",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper-table"))
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.4)
    parser.add_argument("--radar-range-gate-m", type=float, default=800.0)
    parser.add_argument("--stable-segment-min-frames", type=int, default=100)
    parser.add_argument("--stable-segment-max-transition-speed-mps", type=float, default=65.0)
    parser.add_argument("--fusion-nis-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-nis-gate-prob", type=float, default=0.99)
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
        radar_range_gate_m=None if args.radar_range_gate_m <= 0.0 else args.radar_range_gate_m,
        stable_segment_min_frames=args.stable_segment_min_frames,
        stable_segment_max_transition_speed_mps=args.stable_segment_max_transition_speed_mps,
        fusion_nis_gate_prob=args.fusion_nis_gate_prob,
        rf_nis_gate_prob=args.rf_nis_gate_prob,
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
    radar_range_gate_m: float | None = 800.0,
    stable_segment_min_frames: int = 100,
    stable_segment_max_transition_speed_mps: float = 65.0,
    fusion_nis_gate_prob: float = 0.99,
    rf_nis_gate_prob: float = 0.99,
    truth_time_gate_s: float = 2.0,
    acceleration_std_mps2: float = 4.0,
    smoother_lag_s: float = 20.0,
    include_fusion: bool = True,
    fusion_associations: tuple[str, ...] = FUSION_ASSOCIATIONS,
) -> dict[str, Any]:
    """Build and write a paper-style comparison table for one flight."""

    if stable_segment_min_frames < 1:
        raise ValueError("stable_segment_min_frames must be positive")
    if stable_segment_max_transition_speed_mps <= 0.0:
        raise ValueError("stable_segment_max_transition_speed_mps must be positive")

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
                range_gate_m=radar_range_gate_m if selection in RANGE_GATED_RADAR_SELECTIONS else None,
                stable_segment_min_frames=stable_segment_min_frames,
                stable_segment_max_transition_speed_mps=stable_segment_max_transition_speed_mps,
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
                    radar_range_gate_m=radar_range_gate_m,
                    fusion_nis_gate_prob=fusion_nis_gate_prob,
                    rf_nis_gate_prob=rf_nis_gate_prob,
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
        "radar_range_gate_m": None if radar_range_gate_m is None else float(radar_range_gate_m),
        "stable_segment_min_frames": int(stable_segment_min_frames),
        "stable_segment_max_transition_speed_mps": float(stable_segment_max_transition_speed_mps),
        "fusion_nis_gate_prob": float(fusion_nis_gate_prob),
        "rf_nis_gate_prob": float(rf_nis_gate_prob),
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
    radar_range_gate_m: float | None,
    fusion_nis_gate_prob: float,
    rf_nis_gate_prob: float,
    acceleration_std_mps2: float,
    smoother_lag_s: float,
    max_time_delta_s: float,
) -> list[dict[str, Any]]:
    """Run one fusion association and return unsmoothed/smoothed metric rows."""

    try:
        if association == "paper-compatible":
            records, selected = run_paper_compatible_cv_fusion(
                rf_measurements=rf_measurements,
                radar=radar,
                acceleration_std_mps2=acceleration_std_mps2,
                radar_range_gate_m=radar_range_gate_m,
                radar_catprob_threshold=radar_catprob_threshold,
                nis_gate_probability=fusion_nis_gate_prob,
                rf_nis_gate_probability=rf_nis_gate_prob,
            )
        elif association == "paper-longest-track":
            records, selected = run_paper_longest_track_cv_fusion(
                rf_measurements=rf_measurements,
                radar=radar,
                acceleration_std_mps2=acceleration_std_mps2,
                radar_range_gate_m=radar_range_gate_m,
                nis_gate_probability=fusion_nis_gate_prob,
                rf_nis_gate_probability=rf_nis_gate_prob,
            )
        else:
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
    range_gate_m: float | None = None,
    stable_segment_min_frames: int = 100,
    stable_segment_max_transition_speed_mps: float = 65.0,
    max_time_delta_s: float,
) -> pd.DataFrame:
    """Select one radar row per frame for paper-table diagnostics."""

    if selection == "radar-longest-track-range-gated-interpolated":
        anchors = select_radar_for_table(
            radar=radar,
            truth=truth,
            selection="radar-longest-track-range-gated",
            catprob_threshold=catprob_threshold,
            range_gate_m=range_gate_m,
            stable_segment_min_frames=stable_segment_min_frames,
            stable_segment_max_transition_speed_mps=stable_segment_max_transition_speed_mps,
            max_time_delta_s=max_time_delta_s,
        )
        return _interpolate_selected_radar_to_frame_times(
            radar,
            anchors,
            association_mode="radar-longest-track-range-gated-interpolated",
        )
    if selection == "radar-stable-segments-range-gated-interpolated":
        anchors = select_radar_for_table(
            radar=radar,
            truth=truth,
            selection="radar-stable-segments-range-gated",
            catprob_threshold=catprob_threshold,
            range_gate_m=range_gate_m,
            stable_segment_min_frames=stable_segment_min_frames,
            stable_segment_max_transition_speed_mps=stable_segment_max_transition_speed_mps,
            max_time_delta_s=max_time_delta_s,
        )
        return _interpolate_selected_radar_to_frame_times(
            radar,
            anchors,
            association_mode="radar-stable-segments-range-gated-interpolated",
        )
    if selection == "radar-stable-segments-range-gated":
        return select_stable_radar_segments(
            radar,
            range_gate_m=range_gate_m,
            catprob_threshold=catprob_threshold,
            min_segment_frames=stable_segment_min_frames,
            max_transition_speed_mps=stable_segment_max_transition_speed_mps,
        )

    groups = radar_frame_groups(radar)
    longest_track_source = _range_candidate_pool(radar, range_gate_m) if selection == "radar-longest-continuous-track-range-gated" else radar
    longest_track = (
        _longest_continuous_track_id(longest_track_source)
        if selection == "radar-longest-continuous-track-range-gated"
        else _longest_track_id(longest_track_source)
        if selection
        in {
            "radar-longest-track",
            "radar-longest-track-range-gated",
            "radar-longest-continuous-track-range-gated",
        }
        else None
    )
    selected_rows: list[pd.Series] = []
    for group in groups:
        group = _range_candidate_pool(group, range_gate_m)
        if group.empty:
            continue
        time_s = float(group["time_s"].median())
        truth_xyz = (
            truth_position_at_time(truth, time_s, max_delta_s=max_time_delta_s)
            if selection in {"radar-oracle-nearest-truth", "radar-catprob-oracle-nearest"}
            else None
        )
        if selection == "radar-highest-catprob":
            selected = highest_catprob_candidate(group)
        elif selection in {
            "radar-longest-track",
            "radar-longest-track-range-gated",
            "radar-longest-continuous-track-range-gated",
        }:
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


def run_paper_compatible_cv_fusion(
    *,
    rf_measurements: list[TrackingMeasurement],
    radar: pd.DataFrame,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    radar_range_gate_m: float | None = 800.0,
    radar_catprob_threshold: float | None = 0.4,
    nis_gate_probability: float = 0.99,
    rf_nis_gate_probability: float = 0.99,
    track_switch_cost: float = 3.0,
    catprob_weight: float = 2.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run a paper-compatible CV fusion baseline with explicit radar coasting.

    Radar frames are filtered before the Kalman update by range, UAV class
    probability, and predicted-state NIS.  When no candidate survives, the
    tracker predicts to the radar timestamp and records a missed-detection
    posterior instead of forcing a bad radar update.
    """

    if radar_catprob_threshold is not None and not 0.0 <= float(radar_catprob_threshold) <= 1.0:
        raise ValueError("radar_catprob_threshold must be in [0, 1] or None")
    if radar_range_gate_m is not None and float(radar_range_gate_m) <= 0.0:
        raise ValueError("radar_range_gate_m must be positive or None")
    if not 0.0 < float(nis_gate_probability) < 1.0:
        raise ValueError("nis_gate_probability must be in (0, 1)")
    if not 0.0 < float(rf_nis_gate_probability) < 1.0:
        raise ValueError("rf_nis_gate_probability must be in (0, 1)")
    if track_switch_cost < 0.0:
        raise ValueError("track_switch_cost must be nonnegative")
    if catprob_weight < 0.0:
        raise ValueError("catprob_weight must be nonnegative")

    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    events = _events(list(rf_measurements), radar)
    if not events:
        return [], _selected_rows_frame(radar, [])

    initial = _initial_paper_measurement(
        events,
        covariance=covariance,
        radar_range_gate_m=radar_range_gate_m,
        radar_catprob_threshold=radar_catprob_threshold,
    )
    if initial is None:
        return [], _selected_rows_frame(radar, [])

    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial.vector,
        initial_time_s=initial.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []
    current_track_id: int | None = None
    longest_track_id = _longest_track_id(radar)
    nis_threshold = gate_threshold_from_probability(float(nis_gate_probability), 3)
    rf_gate_threshold = gate_threshold_from_probability(float(rf_nis_gate_probability), 2)
    assert nis_threshold is not None
    assert rf_gate_threshold is not None

    for event in events:
        time_s = float(event["time_s"])
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            diagnostics = tracker.update(measurement, gate_threshold=rf_gate_threshold)
            records.append(_tracking_record(measurement, tracker, diagnostics))
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        tracker.predict_to(time_s)
        selected = select_paper_compatible_candidate(
            candidates,
            tracker=tracker,
            covariance=covariance,
            longest_track_id=longest_track_id,
            current_track_id=current_track_id,
            radar_range_gate_m=radar_range_gate_m,
            radar_catprob_threshold=radar_catprob_threshold,
            nis_gate_threshold=nis_threshold,
            track_switch_cost=track_switch_cost,
            catprob_weight=catprob_weight,
        )
        if selected is None:
            records.append(_coast_record(time_s=time_s, tracker=tracker, source="radar"))
            continue

        measurement = _radar_row_to_measurement(selected, covariance)
        diagnostics = tracker.update(measurement, gate_threshold=nis_threshold)
        if diagnostics.accepted:
            current_track_id = _optional_track_id(selected)
            selected_rows.append(selected)
        records.append(
            _tracking_record(
                measurement,
                tracker,
                diagnostics,
                track_id=_optional_track_id(selected),
                association_nis=_optional_float(selected.get("association_nis")),
                association_score=_optional_float(selected.get("association_score")),
                association_mode="paper-compatible",
            )
        )

    return records, _selected_rows_frame(radar, selected_rows)


def run_paper_longest_track_cv_fusion(
    *,
    rf_measurements: list[TrackingMeasurement],
    radar: pd.DataFrame,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    radar_range_gate_m: float | None = 800.0,
    nis_gate_probability: float = 0.99,
    rf_nis_gate_probability: float = 0.99,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion using only stable longest-track radar anchors."""

    if radar_range_gate_m is not None and float(radar_range_gate_m) <= 0.0:
        raise ValueError("radar_range_gate_m must be positive or None")
    if not 0.0 < float(nis_gate_probability) < 1.0:
        raise ValueError("nis_gate_probability must be in (0, 1)")
    if not 0.0 < float(rf_nis_gate_probability) < 1.0:
        raise ValueError("rf_nis_gate_probability must be in (0, 1)")

    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    events = _events(list(rf_measurements), radar)
    if not events:
        return [], _selected_rows_frame(radar, [])

    anchors = select_radar_for_table(
        radar=radar,
        truth=pd.DataFrame(),
        selection="radar-longest-track-range-gated",
        catprob_threshold=0.0,
        range_gate_m=radar_range_gate_m,
        max_time_delta_s=float("inf"),
    )
    anchor_by_key = {_radar_row_key(row): row for _, row in anchors.iterrows()}
    initial = _initial_anchor_measurement(events, anchor_by_key=anchor_by_key, covariance=covariance)
    if initial is None:
        return [], _selected_rows_frame(radar, [])

    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial.vector,
        initial_time_s=initial.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []
    nis_threshold = gate_threshold_from_probability(float(nis_gate_probability), 3)
    rf_gate_threshold = gate_threshold_from_probability(float(rf_nis_gate_probability), 2)
    assert nis_threshold is not None
    assert rf_gate_threshold is not None

    for event in events:
        time_s = float(event["time_s"])
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            diagnostics = tracker.update(measurement, gate_threshold=rf_gate_threshold)
            records.append(_tracking_record(measurement, tracker, diagnostics))
            continue

        tracker.predict_to(time_s)
        selected = anchor_by_key.get(_radar_event_key(event))
        if selected is None:
            records.append(_coast_record(time_s=time_s, tracker=tracker, source="radar"))
            continue
        measurement = _radar_row_to_measurement(selected, covariance)
        diagnostics = tracker.update(measurement, gate_threshold=nis_threshold)
        if diagnostics.accepted:
            selected_rows.append(selected)
        records.append(
            _tracking_record(
                measurement,
                tracker,
                diagnostics,
                track_id=_optional_track_id(selected),
                association_nis=_optional_float(selected.get("association_nis")),
                association_mode="paper-longest-track",
            )
        )

    return records, _selected_rows_frame(radar, selected_rows)


def select_stable_radar_segments(
    radar: pd.DataFrame,
    *,
    range_gate_m: float | None = 800.0,
    catprob_threshold: float | None = 0.4,
    min_segment_frames: int = 100,
    max_transition_speed_mps: float = 65.0,
) -> pd.DataFrame:
    """Select multiple stable, high-confidence radar track segments."""

    if min_segment_frames < 1:
        raise ValueError("min_segment_frames must be positive")
    if max_transition_speed_mps <= 0.0:
        raise ValueError("max_transition_speed_mps must be positive")
    pool = _catprob_hard_candidate_pool(_range_candidate_pool(radar, range_gate_m), catprob_threshold)
    if pool.empty or "track_id" not in pool.columns:
        return radar.iloc[0:0].copy()

    segments = _stable_track_segments(pool, min_segment_frames=min_segment_frames)
    if not segments:
        return radar.iloc[0:0].copy()
    selected_segments = _stitch_segments(
        segments,
        max_transition_speed_mps=float(max_transition_speed_mps),
    )
    if not selected_segments:
        return radar.iloc[0:0].copy()

    selected = pd.concat([segment.frame for segment in selected_segments], ignore_index=True)
    selected["association_mode"] = "radar-stable-segments-range-gated"
    selected["association_segment_count"] = int(len(selected_segments))
    sort_columns = [
        column for column in ("time_s", "frame_index", "track_id", "track_index") if column in selected.columns
    ]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def select_paper_compatible_candidate(
    candidates: pd.DataFrame,
    *,
    tracker: AsyncConstantVelocityKalmanTracker,
    covariance: np.ndarray,
    longest_track_id: int | None,
    current_track_id: int | None,
    radar_range_gate_m: float | None,
    radar_catprob_threshold: float | None,
    nis_gate_threshold: float,
    track_switch_cost: float,
    catprob_weight: float,
) -> pd.Series | None:
    """Return the best hard-gated paper-compatible candidate, or ``None`` to coast."""

    pool = _range_candidate_pool(candidates, radar_range_gate_m)
    if pool.empty:
        return None
    if longest_track_id is not None and "track_id" in pool.columns:
        track_ids = pd.to_numeric(pool["track_id"], errors="coerce")
        longest_pool = pool.loc[track_ids == int(longest_track_id)]
        if not longest_pool.empty:
            pool = longest_pool
    pool = _catprob_hard_candidate_pool(pool, radar_catprob_threshold)
    if pool.empty:
        return None

    scored = _nis_scored_candidates(pool, tracker=tracker, covariance=covariance)
    scored = scored.loc[
        pd.to_numeric(scored["association_nis"], errors="coerce") <= float(nis_gate_threshold)
    ].copy()
    if scored.empty:
        return None

    scored["association_track_switch_penalty"] = _track_switch_penalty(
        scored,
        current_track_id=current_track_id,
        switch_penalty=float(track_switch_cost),
    )
    scored["association_catprob_penalty"] = _catprob_penalty(scored, catprob_weight)
    scored["association_score"] = (
        scored["association_nis"]
        + scored["association_track_switch_penalty"]
        + scored["association_catprob_penalty"]
    )
    selected = scored.loc[scored["association_score"].idxmin()].copy()
    selected["association_mode"] = "paper-compatible"
    selected["association_action"] = "hard_gated_update"
    selected["association_effective_candidates"] = int(len(scored))
    selected["association_weight_entropy"] = _paper_candidate_entropy(scored)
    return selected


def _initial_paper_measurement(
    events: list[dict[str, object]],
    *,
    covariance: np.ndarray,
    radar_range_gate_m: float | None,
    radar_catprob_threshold: float | None,
) -> TrackingMeasurement | None:
    for event in events:
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            return measurement
        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        pool = _catprob_hard_candidate_pool(
            _range_candidate_pool(candidates, radar_range_gate_m),
            radar_catprob_threshold,
        )
        selected = highest_catprob_candidate(pool)
        if selected is not None:
            return _radar_row_to_measurement(selected, covariance)
    return None


def _initial_anchor_measurement(
    events: list[dict[str, object]],
    *,
    anchor_by_key: dict[object, pd.Series],
    covariance: np.ndarray,
) -> TrackingMeasurement | None:
    for event in events:
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            return measurement
        selected = anchor_by_key.get(_radar_event_key(event))
        if selected is not None:
            return _radar_row_to_measurement(selected, covariance)
    return None


def _stable_track_segments(
    radar: pd.DataFrame,
    *,
    min_segment_frames: int,
) -> list[_TrackSegment]:
    segments: list[_TrackSegment] = []
    for track_id, track_rows in radar.groupby("track_id", sort=True):
        ordered = track_rows.sort_values(["frame_index" if "frame_index" in track_rows.columns else "time_s", "time_s"])
        frame_values = (
            pd.to_numeric(ordered["frame_index"], errors="coerce").to_numpy(dtype=float)
            if "frame_index" in ordered.columns
            else ordered["time_s"].to_numpy(dtype=float)
        )
        splits = np.r_[0, np.where(np.diff(frame_values) > _segment_gap_threshold(frame_values))[0] + 1, len(ordered)]
        for start, end in zip(splits[:-1], splits[1:]):
            frame = ordered.iloc[int(start) : int(end)].copy()
            if len(frame) < int(min_segment_frames):
                continue
            positions = frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
            times = frame["time_s"].to_numpy(dtype=float)
            catprob = (
                pd.to_numeric(frame["cat_prob_uav"], errors="coerce").to_numpy(dtype=float)
                if "cat_prob_uav" in frame.columns
                else np.ones(len(frame), dtype=float)
            )
            mean_catprob = float(np.nanmean(catprob))
            if not np.isfinite(mean_catprob):
                mean_catprob = 0.0
            segments.append(
                _TrackSegment(
                    frame=frame,
                    track_id=int(track_id),
                    start_time_s=float(times[0]),
                    end_time_s=float(times[-1]),
                    start_position_m=positions[0],
                    end_position_m=positions[-1],
                    frames=int(len(frame)),
                    mean_catprob=mean_catprob,
                )
            )
    return sorted(segments, key=lambda item: (item.start_time_s, -item.score))


def _segment_gap_threshold(frame_values: np.ndarray) -> float:
    values = np.sort(np.asarray(frame_values, dtype=float).reshape(-1))
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("inf")
    diffs = np.diff(values)
    positive = diffs[diffs > 1.0e-9]
    if positive.size == 0:
        return float("inf")
    if _integer_like(values):
        return 1.5
    return 1.5 * float(np.median(positive))


def _stitch_segments(
    segments: list[_TrackSegment],
    *,
    max_transition_speed_mps: float,
) -> list[_TrackSegment]:
    ordered = sorted(segments, key=lambda item: (item.start_time_s, item.end_time_s))
    best_paths: list[list[_TrackSegment]] = []
    best_scores: list[float] = []
    for segment in ordered:
        best_path = [segment]
        best_score = segment.score
        for index, previous in enumerate(ordered[: len(best_paths)]):
            if not _segments_can_follow(previous, segment, max_transition_speed_mps=max_transition_speed_mps):
                continue
            score = best_scores[index] + segment.score
            if score > best_score:
                best_score = score
                best_path = [*best_paths[index], segment]
        best_paths.append(best_path)
        best_scores.append(best_score)
    if not best_paths:
        return []
    return best_paths[int(np.argmax(best_scores))]


def _segments_can_follow(
    previous: _TrackSegment,
    current: _TrackSegment,
    *,
    max_transition_speed_mps: float,
) -> bool:
    if current.start_time_s <= previous.end_time_s:
        return False
    dt_s = current.start_time_s - previous.end_time_s
    if dt_s <= 0.0:
        return False
    distance_m = float(np.linalg.norm(current.start_position_m - previous.end_position_m))
    return distance_m / dt_s <= float(max_transition_speed_mps)


def _radar_event_key(event: dict[str, object]) -> object:
    candidates = event["candidates"]
    assert isinstance(candidates, pd.DataFrame)
    if "frame_index" in candidates.columns:
        values = pd.to_numeric(candidates["frame_index"], errors="coerce").dropna()
        if not values.empty:
            return ("frame_index", int(values.iloc[0]))
    return ("time_s", round(float(event["time_s"]), 9))


def _radar_row_key(row: pd.Series) -> object:
    if "frame_index" in row.index and np.isfinite(float(row["frame_index"])):
        return ("frame_index", int(row["frame_index"]))
    return ("time_s", round(float(row["time_s"]), 9))


def _range_candidate_pool(candidates: pd.DataFrame, range_gate_m: float | None) -> pd.DataFrame:
    if candidates.empty or range_gate_m is None:
        return candidates
    ranges = _candidate_ranges_m(candidates)
    pool = candidates.loc[np.isfinite(ranges) & (ranges <= float(range_gate_m))].copy()
    pool["association_range_gate_m"] = float(range_gate_m)
    return pool


def _interpolate_selected_radar_to_frame_times(
    radar: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    association_mode: str,
) -> pd.DataFrame:
    if radar.empty or selected.empty:
        return radar.iloc[0:0].copy()
    frame_times = np.array(
        [float(group["time_s"].median()) for group in radar_frame_groups(radar)],
        dtype=float,
    )
    if frame_times.size == 0:
        return radar.iloc[0:0].copy()

    anchors = (
        selected.sort_values("time_s")
        .drop_duplicates(subset=["time_s"], keep="last")
        .reset_index(drop=True)
    )
    anchor_times = anchors["time_s"].to_numpy(dtype=float)
    if anchor_times.size == 0:
        return radar.iloc[0:0].copy()

    out = pd.DataFrame({"time_s": frame_times})
    for column in ("east_m", "north_m", "up_m"):
        out[column] = np.interp(
            frame_times,
            anchor_times,
            anchors[column].to_numpy(dtype=float),
        )
    out["association_mode"] = association_mode
    out["association_interpolated"] = True
    if "track_id" in anchors.columns:
        track_ids = pd.to_numeric(anchors["track_id"], errors="coerce").dropna()
        if not track_ids.empty:
            out["track_id"] = int(track_ids.astype(int).mode().iloc[0])
    return out


def _catprob_hard_candidate_pool(
    candidates: pd.DataFrame,
    threshold: float | None,
) -> pd.DataFrame:
    if candidates.empty or threshold is None or "cat_prob_uav" not in candidates.columns:
        return candidates
    scores = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce")
    pool = candidates.loc[scores >= float(threshold)].copy()
    pool["association_catprob_threshold"] = float(threshold)
    return pool


def _candidate_ranges_m(candidates: pd.DataFrame) -> np.ndarray:
    if "range_m" in candidates.columns:
        ranges = pd.to_numeric(candidates["range_m"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(ranges).any():
            return ranges
    positions = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    return np.linalg.norm(positions, axis=1)


def _paper_candidate_entropy(scored: pd.DataFrame) -> float:
    if len(scored) <= 1 or "association_score" not in scored.columns:
        return 0.0
    scores = pd.to_numeric(scored["association_score"], errors="coerce").to_numpy(dtype=float)
    scores = np.where(np.isfinite(scores), scores, np.inf)
    finite = np.isfinite(scores)
    if not finite.any():
        return 0.0
    weights = np.exp(-(scores[finite] - float(np.min(scores[finite]))))
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    return _weight_entropy(weights / total)


def _tracking_record(
    measurement: TrackingMeasurement,
    tracker: AsyncConstantVelocityKalmanTracker,
    diagnostics: TrackingUpdateDiagnostics,
    *,
    track_id: int | None = None,
    association_nis: float | None = None,
    association_score: float | None = None,
    association_mode: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "time_s": float(measurement.time_s),
        "source": measurement.source,
        "state": tracker.state.copy(),
        "covariance": tracker.covariance_matrix.copy(),
        **diagnostics.to_record(),
    }
    if track_id is not None:
        record["track_id"] = track_id
    if association_nis is not None:
        record["association_nis"] = association_nis
    if association_score is not None:
        record["association_score"] = association_score
    if association_mode is not None:
        record["association_mode"] = association_mode
    return record


def _coast_record(
    *,
    time_s: float,
    tracker: AsyncConstantVelocityKalmanTracker,
    source: str,
) -> dict[str, object]:
    diagnostics = TrackingUpdateDiagnostics(
        time_s=float(time_s),
        source=source,
        measurement_dim=3,
        accepted=False,
        update_action="missed_detection",
        nis=float("nan"),
        gate_threshold=None,
        safety_gate_threshold=None,
        residual_gate_threshold_m=None,
        covariance_scale=1.0,
        inflation_alpha=None,
        residual_norm_m=float("nan"),
    )
    return {
        "time_s": float(time_s),
        "source": source,
        "state": tracker.state.copy(),
        "covariance": tracker.covariance_matrix.copy(),
        **diagnostics.to_record(),
        "association_mode": "paper-compatible",
    }


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


def _longest_continuous_track_id(radar: pd.DataFrame) -> int | None:
    if radar.empty or "track_id" not in radar.columns:
        return None
    if "frame_index" in radar.columns:
        frame_values = pd.to_numeric(radar["frame_index"], errors="coerce")
    else:
        frame_values = pd.to_numeric(radar["time_s"], errors="coerce")
    track_values = pd.to_numeric(radar["track_id"], errors="coerce")
    finite = np.isfinite(frame_values) & np.isfinite(track_values)
    frame = pd.DataFrame(
        {
            "track_id": track_values.loc[finite].astype(int),
            "frame": frame_values.loc[finite].to_numpy(dtype=float),
            "cat_prob_uav": pd.to_numeric(
                radar.loc[finite, "cat_prob_uav"]
                if "cat_prob_uav" in radar.columns
                else pd.Series(np.nan, index=radar.index[finite]),
                errors="coerce",
            ).to_numpy(dtype=float),
        }
    )
    if frame.empty:
        return None

    best_track: int | None = None
    best_score: tuple[float, float, float, float] | None = None
    for track_id, group in frame.groupby("track_id", sort=True):
        positions = np.sort(np.unique(group["frame"].to_numpy(dtype=float)))
        longest_run = _longest_run_length(positions)
        total = float(len(positions))
        mean_catprob = float(np.nanmean(group["cat_prob_uav"].to_numpy(dtype=float)))
        if not np.isfinite(mean_catprob):
            mean_catprob = 0.0
        score = (float(longest_run), total, mean_catprob, -float(track_id))
        if best_score is None or score > best_score:
            best_score = score
            best_track = int(track_id)
    return best_track


def _longest_run_length(values: np.ndarray) -> int:
    ordered = np.sort(np.asarray(values, dtype=float).reshape(-1))
    ordered = ordered[np.isfinite(ordered)]
    if ordered.size == 0:
        return 0
    if ordered.size == 1:
        return 1
    diffs = np.diff(ordered)
    positive_diffs = diffs[diffs > 1.0e-9]
    max_gap = 1.5 if _integer_like(ordered) else 1.5 * float(np.median(positive_diffs))
    if not np.isfinite(max_gap) or max_gap <= 0.0:
        max_gap = 1.5
    longest = 1
    current = 1
    for gap in diffs:
        if gap <= max_gap:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def _integer_like(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return bool(finite.size and np.allclose(finite, np.round(finite)))


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
