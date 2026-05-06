"""Command-line entry points for RaFT-UAV experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import run_async_cv_baseline
from raft_uav.evaluation.metrics import position_errors_m, summarize_errors
from raft_uav.io.aerpaw import (
    discover_flights,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    radar_measurements_to_enu,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
    select_radar_measurement_rows,
    summarize_flight_schema,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raft-uav")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="list discovered AERPAW flights")
    inspect_parser.add_argument("dataset_root", type=Path)
    inspect_parser.add_argument(
        "--flight",
        action="append",
        help="inspect only this flight; can be passed multiple times",
    )

    baseline_parser = subparsers.add_parser(
        "run-baseline", help="run the initial CV fusion baseline"
    )
    baseline_parser.add_argument("dataset_root", type=Path)
    baseline_parser.add_argument("--flight", required=True)
    baseline_parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    baseline_parser.add_argument("--acceleration-std", type=float, default=4.0)
    baseline_parser.add_argument(
        "--radar-selection",
        choices=["catprob", "truth-gated", "all", "none"],
        default="catprob",
        help="trackData row selection used before fusion",
    )
    baseline_parser.add_argument("--radar-catprob-threshold", type=float, default=0.5)
    baseline_parser.add_argument("--truth-gate-m", type=float, default=150.0)
    baseline_parser.add_argument("--truth-time-gate-s", type=float, default=1.0)
    baseline_parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    baseline_parser.add_argument(
        "--enable-gating",
        action="store_true",
        help="enable normalized-innovation-squared Mahalanobis gates before updates",
    )
    baseline_parser.add_argument(
        "--rf-gate-prob",
        type=float,
        default=0.99,
        help="chi-square gate probability for 2D RF updates when gating is enabled",
    )
    baseline_parser.add_argument(
        "--radar-gate-prob",
        type=float,
        default=0.99,
        help="chi-square gate probability for 3D radar updates when gating is enabled",
    )

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.dataset_root, args.flight)
    if args.command == "run-baseline":
        return _run_baseline(
            args.dataset_root,
            args.flight,
            args.output_dir,
            args.acceleration_std,
            args.radar_selection,
            args.radar_catprob_threshold,
            args.truth_gate_m,
            args.truth_time_gate_s,
            args.max_eval_time_delta_s,
            args.enable_gating,
            args.rf_gate_prob,
            args.radar_gate_prob,
        )
    raise ValueError(args.command)


def _inspect(dataset_root: Path, requested_flights: list[str] | None) -> int:
    if requested_flights:
        flights = [select_flight(dataset_root, name) for name in requested_flights]
        discovered_count = len(discover_flights(dataset_root))
    else:
        flights = discover_flights(dataset_root)
        discovered_count = len(flights)

    print(f"discovered_flights={discovered_count}")
    for flight in flights:
        summary = summarize_flight_schema(flight)
        print(f"\nflight={summary['flight']}")
        for modality in ("truth", "rf", "radar"):
            _print_modality_summary(modality, summary.get(modality))
    return 0


def _run_baseline(
    dataset_root: Path,
    flight_name: str,
    output_dir: Path,
    acceleration_std: float,
    radar_selection: str,
    radar_catprob_threshold: float,
    truth_gate_m: float,
    truth_time_gate_s: float,
    max_eval_time_delta_s: float,
    enable_gating: bool,
    rf_gate_prob: float,
    radar_gate_prob: float,
) -> int:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")

    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)

    rf = pd.DataFrame()
    radar = pd.DataFrame()
    selected_radar = pd.DataFrame()
    measurements = []
    if flight.rf_csv is not None:
        rf = _inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time), truth
        )
        measurements.extend(rf_measurements_to_enu(rf))
    if flight.radar_json is not None:
        radar = _inside_truth_window(
            normalize_radar(
                read_radar_tracks_json(flight.radar_json), projector, truth_origin_time
            ),
            truth,
        )
        selected_radar = select_radar_measurement_rows(
            radar,
            selection=radar_selection,
            truth=truth,
            catprob_threshold=radar_catprob_threshold,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        measurements.extend(radar_measurements_to_enu(selected_radar))

    gate_probabilities = None
    if enable_gating:
        gate_probabilities = {"rf": rf_gate_prob, "radar": radar_gate_prob}

    records = run_async_cv_baseline(
        measurements,
        acceleration_std_mps2=acceleration_std,
        gate_probabilities_by_source=gate_probabilities,
    )
    if not records:
        raise RuntimeError(f"{flight.name} produced no baseline posterior records")

    estimate_frame = _records_to_frame(records)
    diagnostics_columns = [
        "time_s",
        "source",
        "measurement_dim",
        "accepted",
        "nis",
        "gate_threshold",
        "residual_norm_m",
    ]
    diagnostics_frame = estimate_frame[diagnostics_columns].copy()

    flight_output = output_dir / flight.name
    flight_output.mkdir(parents=True, exist_ok=True)

    estimates_path = flight_output / "estimates.csv"
    diagnostics_path = flight_output / "diagnostics.csv"
    metrics_path = flight_output / "metrics.json"
    plot_path = flight_output / "trajectory.png"
    estimate_frame.to_csv(estimates_path, index=False)
    diagnostics_frame.to_csv(diagnostics_path, index=False)

    metrics = _baseline_metrics(
        flight_name=flight.name,
        flight=flight,
        truth=truth,
        rf=rf,
        radar=radar,
        selected_radar=selected_radar,
        estimate_frame=estimate_frame,
        acceleration_std=acceleration_std,
        radar_selection=radar_selection,
        radar_catprob_threshold=radar_catprob_threshold,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
        max_eval_time_delta_s=max_eval_time_delta_s,
        enable_gating=enable_gating,
        rf_gate_prob=rf_gate_prob,
        radar_gate_prob=radar_gate_prob,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_trajectory_plot(plot_path, truth, rf, selected_radar, estimate_frame, flight.name)

    print(f"flight={flight.name}")
    print(f"measurements={len(measurements)}")
    print(f"posterior_records={len(records)}")
    print(f"accepted_measurements={metrics['accepted_measurements']}")
    print(f"rejected_measurements={metrics['rejected_measurements']}")
    print(f"rf_rows={len(rf)}")
    print(f"radar_rows={len(radar)}")
    print(f"selected_radar_rows={len(selected_radar)}")
    print(f"selected_radar_track_ids={metrics['selected_radar_track_ids']}")
    print(f"metrics_json={metrics_path}")
    print(f"estimates_csv={estimates_path}")
    print(f"diagnostics_csv={diagnostics_path}")
    print(f"trajectory_png={plot_path}")
    print(f"rmse_2d_m={metrics['position_error_2d']['rmse_m']:.3f}")
    print(f"rmse_3d_m={metrics['position_error_3d']['rmse_m']:.3f}")
    return 0


def _records_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        state = np.asarray(record["state"], dtype=float).reshape(6)
        rows.append(
            {
                "time_s": float(record["time_s"]),
                "source": str(record["source"]),
                "measurement_dim": int(record.get("measurement_dim", 0)),
                "accepted": bool(record.get("accepted", True)),
                "nis": _optional_float(record.get("nis")),
                "gate_threshold": _optional_float(record.get("gate_threshold")),
                "residual_norm_m": _optional_float(record.get("residual_norm_m")),
                "east_m": state[0],
                "north_m": state[1],
                "up_m": state[2],
                "v_east_mps": state[3],
                "v_north_mps": state[4],
                "v_up_mps": state[5],
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("time_s").reset_index(drop=True)


def _baseline_metrics(
    *,
    flight_name: str,
    flight: Any,
    truth: pd.DataFrame,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    selected_radar: pd.DataFrame,
    estimate_frame: pd.DataFrame,
    acceleration_std: float,
    radar_selection: str,
    radar_catprob_threshold: float,
    truth_gate_m: float,
    truth_time_gate_s: float,
    max_eval_time_delta_s: float,
    enable_gating: bool,
    rf_gate_prob: float,
    radar_gate_prob: float,
) -> dict[str, Any]:
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    estimate_times = estimate_frame["time_s"].to_numpy(dtype=float)
    estimate_positions = estimate_frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    error_2d = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_eval_time_delta_s,
        dimensions=2,
    )
    error_3d = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_eval_time_delta_s,
        dimensions=3,
    )
    source_counts = Counter(str(value) for value in estimate_frame["source"])
    accepted_mask = estimate_frame["accepted"].astype(bool)
    accepted_by_source = Counter(
        str(value) for value in estimate_frame.loc[accepted_mask, "source"]
    )
    rejected_by_source = Counter(
        str(value) for value in estimate_frame.loc[~accepted_mask, "source"]
    )

    selected_ids = []
    if "track_id" in selected_radar.columns:
        selected_ids = sorted(int(value) for value in selected_radar["track_id"].dropna().unique())

    return {
        "flight": flight_name,
        "files": {
            "truth": flight.truth_txt.name if flight.truth_txt else None,
            "rf": flight.rf_csv.name if flight.rf_csv else None,
            "radar": flight.radar_json.name if flight.radar_json else None,
        },
        "state": ["east", "north", "up", "v_east", "v_north", "v_up"],
        "acceleration_std_mps2": float(acceleration_std),
        "rf_covariance": "diag(CEP^2, CEP^2), default std 75 m",
        "radar_covariance": "diag(25^2, 25^2, 35^2) m^2",
        "radar_selection": radar_selection,
        "radar_catprob_threshold": float(radar_catprob_threshold),
        "truth_gate_m": float(truth_gate_m),
        "truth_time_gate_s": float(truth_time_gate_s),
        "max_eval_time_delta_s": float(max_eval_time_delta_s),
        "gating": {
            "enabled": bool(enable_gating),
            "test_statistic": "normalized innovation squared",
            "rf_gate_probability": float(rf_gate_prob) if enable_gating else None,
            "radar_gate_probability": float(radar_gate_prob) if enable_gating else None,
        },
        "truth_rows": int(len(truth)),
        "rf_rows": int(len(rf)),
        "radar_rows": int(len(radar)),
        "selected_radar_rows": int(len(selected_radar)),
        "selected_radar_track_ids": selected_ids,
        "posterior_records": int(len(estimate_frame)),
        "accepted_measurements": int(accepted_mask.sum()),
        "rejected_measurements": int((~accepted_mask).sum()),
        "source_counts": {key: int(value) for key, value in sorted(source_counts.items())},
        "accepted_by_source": {
            key: int(value) for key, value in sorted(accepted_by_source.items())
        },
        "rejected_by_source": {
            key: int(value) for key, value in sorted(rejected_by_source.items())
        },
        "nis_by_source": _summarize_nis_by_source(estimate_frame),
        "time_range_s": {
            "truth_min": float(truth["time_s"].min()),
            "truth_max": float(truth["time_s"].max()),
            "estimate_min": float(estimate_frame["time_s"].min()),
            "estimate_max": float(estimate_frame["time_s"].max()),
        },
        "position_error_2d": summarize_errors(error_2d),
        "position_error_3d": summarize_errors(error_3d),
    }


def _summarize_nis_by_source(estimate_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for source, group in estimate_frame.groupby("source"):
        values = pd.to_numeric(group["nis"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            summaries[str(source)] = {
                "count": 0.0,
                "mean": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
            }
            continue
        summaries[str(source)] = {
            "count": float(values.size),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    return summaries


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    truth_min = float(truth["time_s"].min())
    truth_max = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= truth_min) & (frame["time_s"] <= truth_max)].copy()


def _write_trajectory_plot(
    path: Path,
    truth: pd.DataFrame,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    estimates: pd.DataFrame,
    flight_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    ax.plot(truth["east_m"], truth["north_m"], color="black", linewidth=1.8, label="truth")
    if not rf.empty:
        ax.scatter(
            rf["east_m"],
            rf["north_m"],
            s=14,
            color="#d95f02",
            alpha=0.55,
            linewidths=0,
            label="RF",
        )
    if not radar.empty:
        ax.scatter(
            radar["east_m"],
            radar["north_m"],
            s=10,
            color="#1b9e77",
            alpha=0.45,
            linewidths=0,
            label="radar",
        )
    ax.plot(
        estimates["east_m"],
        estimates["north_m"],
        color="#386cb0",
        linewidth=1.2,
        alpha=0.9,
        label="CV fusion",
    )
    rejected = estimates.loc[~estimates["accepted"].astype(bool)]
    if not rejected.empty:
        ax.scatter(
            rejected["east_m"],
            rejected["north_m"],
            s=22,
            marker="x",
            color="#7570b3",
            alpha=0.8,
            label="rejected update state",
        )
    ax.set_title(f"{flight_name} ENU trajectory sanity check")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    _set_trajectory_limits(ax, truth, radar, estimates)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _set_trajectory_limits(
    ax: Any,
    truth: pd.DataFrame,
    radar: pd.DataFrame,
    estimates: pd.DataFrame,
) -> None:
    frames = [truth, estimates]
    if not radar.empty:
        frames.append(radar)
    xy = np.vstack([frame[["east_m", "north_m"]].to_numpy(dtype=float) for frame in frames])
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.size == 0:
        return
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x_pad = max(25.0, 0.08 * (x_max - x_min))
    y_pad = max(25.0, 0.08 * (y_max - y_min))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def _print_modality_summary(modality: str, summary: dict[str, Any] | None) -> None:
    if summary is None:
        print(f"  {modality}: missing")
        return
    columns = ",".join(summary["columns"])
    print(f"  {modality}: file={summary['file']} rows={summary['rows']}")
    print(f"    columns={columns}")
    if "raw_time_min" in summary or "time_s_min" in summary:
        print(
            "    time="
            f"raw[{summary.get('raw_time_min')} -> {summary.get('raw_time_max')}] "
            f"s[{_fmt(summary.get('time_s_min'))} -> {_fmt(summary.get('time_s_max'))}]"
        )
    if modality == "radar" and summary.get("track_ids_count") is not None:
        print(
            f"    track_ids_count={summary['track_ids_count']} "
            f"track_ids_sample={summary['track_ids_sample']}"
        )


def _fmt(value: object) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
