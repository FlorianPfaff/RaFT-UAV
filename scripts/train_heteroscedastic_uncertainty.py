"""Train heteroscedastic RF/radar measurement uncertainty models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.io.aerpaw import (  # noqa: E402
    discover_flights,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_flight,
)
from raft_uav.uncertainty import fit_heteroscedastic_uncertainty_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--flight",
        action="append",
        help="training flight name or substring; can be repeated; defaults to all flights",
    )
    parser.add_argument(
        "--exclude-flight",
        action="append",
        default=[],
        help="flight name or substring to leave out, e.g. for leave-one-flight-out evaluation",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/uncertainty/model.json"))
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    args = parser.parse_args()

    flights = _training_flights(args.dataset_root, args.flight, args.exclude_flight)
    rf_parts: list[pd.DataFrame] = []
    radar_parts: list[pd.DataFrame] = []
    truth_parts: list[pd.DataFrame] = []
    time_offset_s = 0.0
    used_flights: list[str] = []

    for flight in flights:
        if flight.truth_txt is None:
            continue
        truth, projector, origin_time = normalize_truth(read_truth(flight.truth_txt))
        truth = truth.copy()
        truth["time_s"] = truth["time_s"] + time_offset_s
        truth_parts.append(truth)
        used_flights.append(flight.name)

        if flight.rf_csv is not None:
            rf = normalize_rf(read_rf_csv(flight.rf_csv), projector, origin_time)
            rf = _inside_truth_window(rf, truth, offset_s=time_offset_s)
            rf_parts.append(rf)
        if flight.radar_json is not None:
            radar = normalize_radar(read_radar_tracks_json(flight.radar_json), projector, origin_time)
            radar = _inside_truth_window(radar, truth, offset_s=time_offset_s)
            radar_parts.append(radar)

        time_offset_s = float(truth["time_s"].max()) + 10_000.0

    if not truth_parts:
        raise RuntimeError("no flights with truth telemetry were available for training")

    truth_all = pd.concat(truth_parts, ignore_index=True)
    rf_all = pd.concat(rf_parts, ignore_index=True) if rf_parts else pd.DataFrame()
    radar_all = pd.concat(radar_parts, ignore_index=True) if radar_parts else pd.DataFrame()
    model = fit_heteroscedastic_uncertainty_model(
        rf=rf_all,
        radar=radar_all,
        truth=truth_all,
        ridge_lambda=args.ridge_lambda,
        max_time_delta_s=args.max_time_delta_s,
        metadata={
            "training_flights": used_flights,
            "rf_rows": int(len(rf_all)),
            "radar_rows": int(len(radar_all)),
            "truth_rows": int(len(truth_all)),
        },
    )
    model.save(args.output)

    print(f"output={args.output}")
    print(f"training_flights={','.join(used_flights)}")
    print(f"rf_rows={len(rf_all)}")
    print(f"radar_rows={len(radar_all)}")
    print(f"truth_rows={len(truth_all)}")
    return 0


def _training_flights(dataset_root: Path, requested: list[str] | None, excluded: list[str]) -> list[Any]:
    if requested:
        flights = [select_flight(dataset_root, name) for name in requested]
    else:
        flights = discover_flights(dataset_root)
    excluded_names = {select_flight(dataset_root, name).name for name in excluded}
    return [flight for flight in flights if flight.name not in excluded_names]


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame, *, offset_s: float) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    shifted = frame.copy()
    shifted["time_s"] = shifted["time_s"] + float(offset_s)
    return shifted.loc[
        (shifted["time_s"] >= float(truth["time_s"].min()))
        & (shifted["time_s"] <= float(truth["time_s"].max()))
    ].copy()


if __name__ == "__main__":
    raise SystemExit(main())
