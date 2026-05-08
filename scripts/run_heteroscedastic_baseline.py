"""Run the CV baseline with learned row-wise RF/radar measurement covariance."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.baselines.kalman import TrackingMeasurement, run_async_cv_baseline  # noqa: E402
from raft_uav.baselines.smoothing import SMOOTHER_MODES, smooth_tracking_records  # noqa: E402
from raft_uav.evaluation.metrics import position_errors_m, summarize_errors  # noqa: E402
from raft_uav.io.aerpaw import (  # noqa: E402
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_flight,
    select_radar_measurement_rows,
)
from raft_uav.uncertainty import covariance_from_row, load_uncertainty_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--uncertainty-model", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/heteroscedastic_baseline"))
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument(
        "--radar-selection",
        choices=["catprob", "truth-gated", "all", "none"],
        default="catprob",
    )
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.5)
    parser.add_argument("--truth-gate-m", type=float, default=150.0)
    parser.add_argument("--truth-time-gate-s", type=float, default=1.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--smoother", choices=SMOOTHER_MODES, default="none")
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    args = parser.parse_args()

    flight = select_flight(args.dataset_root, args.flight)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")

    model = load_uncertainty_model(args.uncertainty_model)
    truth, projector, truth_origin_time = normalize_truth(read_truth(flight.truth_txt))

    rf = pd.DataFrame()
    radar = pd.DataFrame()
    selected_radar = pd.DataFrame()
    measurements: list[TrackingMeasurement] = []

    if flight.rf_csv is not None:
        rf = _inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time), truth
        )
        rf = model.apply_rf(rf)
        measurements.extend(_rf_measurements_to_enu(rf))

    if flight.radar_json is not None:
        radar = _inside_truth_window(
            normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
            truth,
        )
        radar = model.apply_radar(radar)
        selected_radar = select_radar_measurement_rows(
            radar,
            selection=args.radar_selection,
            truth=truth,
            catprob_threshold=args.radar_catprob_threshold,
            truth_gate_m=args.truth_gate_m,
            truth_time_gate_s=args.truth_time_gate_s,
        )
        measurements.extend(_radar_measurements_to_enu(selected_radar))

    records = run_async_cv_baseline(measurements, acceleration_std_mps2=args.acceleration_std)
    if not records:
        raise RuntimeError(f"{flight.name} produced no heteroscedastic baseline records")
    records = smooth_tracking_records(
        records,
        method=args.smoother,
        acceleration_std_mps2=args.acceleration_std,
        lag_s=args.smoother_lag_s,
    )

    estimate_frame = _records_to_frame(records)
    metrics = _metrics(
        flight_name=flight.name,
        truth=truth,
        rf=rf,
        radar=radar,
        selected_radar=selected_radar,
        estimate_frame=estimate_frame,
        uncertainty_model=args.uncertainty_model,
        max_eval_time_delta_s=args.max_eval_time_delta_s,
        acceleration_std=args.acceleration_std,
        smoother=args.smoother,
        smoother_lag_s=args.smoother_lag_s,
    )

    output_dir = args.output_dir / flight.name
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate_frame.to_csv(output_dir / "estimates.csv", index=False)
    selected_radar.to_csv(output_dir / "selected_radar.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"flight={flight.name}")
    print(f"uncertainty_model={args.uncertainty_model}")
    print(f"measurements={len(measurements)}")
    print(f"posterior_records={len(records)}")
    print(f"rf_rows={len(rf)}")
    print(f"radar_rows={len(radar)}")
    print(f"selected_radar_rows={len(selected_radar)}")
    print(f"metrics_json={output_dir / 'metrics.json'}")
    print(f"rmse_2d_m={metrics['position_error_2d']['rmse_m']:.3f}")
    print(f"rmse_3d_m={metrics['position_error_3d']['rmse_m']:.3f}")
    return 0


def _rf_measurements_to_enu(rf: pd.DataFrame, default_std_m: float = 75.0) -> list[TrackingMeasurement]:
    measurements: list[TrackingMeasurement] = []
    for _, row in rf.iterrows():
        std_m = _positive_float(row.get("std_m")) or float(default_std_m)
        fallback = np.diag([std_m**2, std_m**2])
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=np.array([float(row["east_m"]), float(row["north_m"])]),
                covariance=covariance_from_row(row, 2, fallback),
                source="rf",
            )
        )
    return measurements


def _radar_measurements_to_enu(
    radar: pd.DataFrame,
    default_xy_std_m: float = 25.0,
    default_z_std_m: float = 35.0,
) -> list[TrackingMeasurement]:
    fallback = np.diag([default_xy_std_m**2, default_xy_std_m**2, default_z_std_m**2])
    measurements: list[TrackingMeasurement] = []
    for _, row in radar.iterrows():
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])]),
                covariance=covariance_from_row(row, 3, fallback),
                source="radar",
            )
        )
    return measurements


def _records_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        state = np.asarray(record["state"], dtype=float).reshape(6)
        rows.append(
            {
                "time_s": float(record["time_s"]),
                "source": str(record["source"]),
                "east_m": state[0],
                "north_m": state[1],
                "up_m": state[2],
                "v_east_mps": state[3],
                "v_north_mps": state[4],
                "v_up_mps": state[5],
                "accepted": bool(record.get("accepted", True)),
                "update_action": str(record.get("update_action", "updated")),
                "nis": _optional_float(record.get("nis")),
                "covariance_scale": _optional_float(record.get("covariance_scale")),
                "residual_norm_m": _optional_float(record.get("residual_norm_m")),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("time_s").reset_index(drop=True)


def _metrics(
    *,
    flight_name: str,
    truth: pd.DataFrame,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    selected_radar: pd.DataFrame,
    estimate_frame: pd.DataFrame,
    uncertainty_model: Path,
    max_eval_time_delta_s: float,
    acceleration_std: float,
    smoother: str,
    smoother_lag_s: float,
) -> dict[str, Any]:
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    estimate_times = estimate_frame["time_s"].to_numpy(dtype=float)
    estimate_positions = estimate_frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    source_counts = Counter(str(value) for value in estimate_frame["source"])
    return {
        "flight": flight_name,
        "uncertainty_model": str(uncertainty_model),
        "state": ["east", "north", "up", "v_east", "v_north", "v_up"],
        "acceleration_std_mps2": float(acceleration_std),
        "rf_covariance": "heteroscedastic learned cov_* columns",
        "radar_covariance": "heteroscedastic learned cov_* columns",
        "smoother": {
            "method": smoother,
            "lag_s": float(smoother_lag_s) if smoother == "fixed-lag" else None,
        },
        "truth_rows": int(len(truth)),
        "rf_rows": int(len(rf)),
        "radar_rows": int(len(radar)),
        "selected_radar_rows": int(len(selected_radar)),
        "posterior_records": int(len(estimate_frame)),
        "source_counts": {key: int(value) for key, value in sorted(source_counts.items())},
        "position_error_2d": summarize_errors(
            position_errors_m(
                estimate_times,
                estimate_positions,
                truth_times,
                truth_positions,
                max_time_delta_s=max_eval_time_delta_s,
                dimensions=2,
            )
        ),
        "position_error_3d": summarize_errors(
            position_errors_m(
                estimate_times,
                estimate_positions,
                truth_times,
                truth_positions,
                max_time_delta_s=max_eval_time_delta_s,
                dimensions=3,
            )
        ),
    }


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    return frame.loc[
        (frame["time_s"] >= float(truth["time_s"].min()))
        & (frame["time_s"] <= float(truth["time_s"].max()))
    ].copy()


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0.0 else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
