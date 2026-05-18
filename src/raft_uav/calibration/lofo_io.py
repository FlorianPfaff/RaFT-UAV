"""I/O helpers for LOFO calibration scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.io.aerpaw import (
    discover_flights,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_flight,
)


def requested_flights(dataset_root: Path, requested: list[str] | None) -> list[str]:
    """Resolve requested flights, defaulting to Opt1/2/3 when present."""

    if requested:
        return [select_flight(dataset_root, name).name for name in requested]
    discovered = [f.name for f in discover_flights(dataset_root) if f.truth_txt and f.radar_json]
    opt = [
        name
        for name in ("Opt1", "Opt2", "Opt3")
        if any(name.lower() in item.lower() for item in discovered)
    ]
    return opt or discovered


def load_flight_frames(dataset_root: Path, flight_name: str) -> dict[str, pd.DataFrame]:
    """Load normalized truth/RF/radar frames for one flight."""

    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, projector, origin_time = normalize_truth(read_truth(flight.truth_txt))
    rf = pd.DataFrame()
    if flight.rf_csv is not None:
        rf = inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, origin_time),
            truth,
        )
    radar = pd.DataFrame()
    if flight.radar_json is not None:
        radar = inside_truth_window(
            normalize_radar(read_radar_tracks_json(flight.radar_json), projector, origin_time),
            truth,
        )
    return {"truth": truth, "rf": rf, "radar": radar}


def inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Keep rows inside the truth time span."""

    if frame.empty or truth.empty or "time_s" not in frame.columns:
        return frame
    return frame.loc[
        (frame["time_s"] >= truth["time_s"].min())
        & (frame["time_s"] <= truth["time_s"].max())
    ].copy()


def jsonable(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-safe objects."""

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
