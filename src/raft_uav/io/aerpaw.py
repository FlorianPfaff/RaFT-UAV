"""AERPAW Dataset-28 / Dryad RF Sensor and Radar loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.coordinates import LocalENUProjector

DEFAULT_SENSOR_CLOCK_OFFSET_S = -4.0 * 60.0 * 60.0
DEFAULT_RF_CLOCK_OFFSET_S = DEFAULT_SENSOR_CLOCK_OFFSET_S
DEFAULT_RADAR_CLOCK_OFFSET_S = DEFAULT_SENSOR_CLOCK_OFFSET_S
# Backward-compatible alias for callers that imported the historical shared name.
SENSOR_CLOCK_OFFSET_S = DEFAULT_SENSOR_CLOCK_OFFSET_S
RADAR_SELECTION_MODES = ("catprob", "catprob-all", "truth-gated", "all", "none")
TRUTH_COLUMNS = [
    "sample",
    "longitude",
    "latitude",
    "altitude_m",
    "attitude_raw",
    "velocity_raw",
    "battery",
    "timestamp_raw",
    "field9",
    "field10",
]


@dataclass(frozen=True)
class FlightPaths:
    """Paths for one RF/radar tracking flight."""

    name: str
    root: Path
    rf_csv: Path | None
    radar_json: Path | None
    truth_txt: Path | None


def find_rf_sensor_and_radar_root(dataset_root: Path) -> Path:
    """Find the RF Sensor and Radar directory under an extracted Dryad dataset."""

    root = Path(dataset_root)
    if root.is_dir() and root.name in {"RF Sensor and Radar", "RF_Sensor_and_Radar"}:
        return root
    candidates = [root / "RF Sensor and Radar", root / "RF_Sensor_and_Radar"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name == "RF Sensor and Radar":
            return candidate
    raise FileNotFoundError(f"Could not find RF Sensor and Radar folder under {root}")


def discover_flights(dataset_root: Path) -> list[FlightPaths]:
    """Discover candidate flights with RF, radar, and ground-truth files."""

    rf_radar_root = find_rf_sensor_and_radar_root(dataset_root)
    flights: list[FlightPaths] = []
    for folder in sorted(path for path in rf_radar_root.iterdir() if path.is_dir()):
        rf_files = sorted(folder.glob("*.csv"))
        radar_files = sorted(folder.glob("radar_data*.json"))
        truth_files = sorted(folder.glob("*vehicleOut*.txt"))
        if not (rf_files or radar_files or truth_files):
            continue
        flights.append(
            FlightPaths(
                name=folder.name,
                root=folder,
                rf_csv=_preferred_variant(rf_files),
                radar_json=_preferred_variant(radar_files),
                truth_txt=_preferred_variant(truth_files),
            )
        )
    return flights


def select_flight(dataset_root: Path, name: str) -> FlightPaths:
    """Select a discovered flight by exact name or case-insensitive substring."""

    flights = discover_flights(dataset_root)
    exact = [flight for flight in flights if flight.name == name]
    if len(exact) == 1:
        return exact[0]
    partial = [flight for flight in flights if name.lower() in flight.name.lower()]
    if len(partial) == 1:
        return partial[0]
    available = ", ".join(flight.name for flight in flights[:40])
    raise ValueError(f"Could not uniquely select flight {name!r}. Available: {available}")


def read_rf_csv(path: Path) -> pd.DataFrame:
    """Read Keysight RF sensor localization outputs."""

    frame = pd.read_csv(path, index_col=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def read_truth(path: Path) -> pd.DataFrame:
    """Read no-header UAV telemetry exported as comma-separated text."""

    frame = pd.read_csv(path, header=None, names=TRUTH_COLUMNS)
    frame["timestamp_raw"] = frame["timestamp_raw"].astype(str)
    return frame


def read_radar_tracks_json(path: Path) -> pd.DataFrame:
    """Read Fortem newline-delimited radar JSON logs into one row per track."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for frame_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            line_number = frame_index + 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid radar JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                continue

            params = payload.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                params = {}

            track_data = payload.get("trackData") or []
            if not isinstance(track_data, list):
                continue
            for track_index, track in enumerate(track_data):
                if not isinstance(track, dict):
                    continue
                records.append(_flatten_track(frame_index, track_index, track, params))
    return pd.DataFrame.from_records(records)


def read_radar_json(path: Path) -> pd.DataFrame:
    """Compatibility wrapper for the track-level radar JSON reader."""

    return read_radar_tracks_json(path)


def normalize_truth(truth: pd.DataFrame) -> tuple[pd.DataFrame, LocalENUProjector, pd.Timestamp]:
    """Normalize truth telemetry timestamps and append local ENU coordinates."""

    out = truth.copy()
    out["timestamp"] = pd.to_datetime(
        out["timestamp_raw"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    out = out.dropna(subset=["timestamp", "latitude", "longitude", "altitude_m"]).copy()
    out = out[np.isfinite(out[["latitude", "longitude", "altitude_m"]].to_numpy()).all(axis=1)]
    out = out.sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        raise ValueError("truth telemetry contains no valid timestamped LLA rows")

    origin_time = pd.Timestamp(out["timestamp"].iloc[0])
    projector = projector_from_truth(out)
    out["time_s"] = (out["timestamp"] - origin_time).dt.total_seconds()
    return truth_to_enu(out, projector), projector, origin_time


def normalize_rf(
    rf: pd.DataFrame,
    projector: LocalENUProjector,
    truth_origin_time: pd.Timestamp,
    default_std_m: float = 75.0,
    clock_offset_s: float = DEFAULT_RF_CLOCK_OFFSET_S,
) -> pd.DataFrame:
    """Normalize RF rows to truth-relative time and local ENU coordinates.

    ``clock_offset_s`` is added to each RF timestamp before converting to the
    truth-relative timeline. RF and radar offsets are intentionally independent
    because the raw sensors can encode time in different reference frames.
    """

    out = rf.copy()
    out["timestamp_raw"] = out["Time"].astype(str)
    out["timestamp"] = pd.to_datetime(out["Time"], errors="coerce") + pd.to_timedelta(
        clock_offset_s, unit="s"
    )
    out["time_s"] = (out["timestamp"] - truth_origin_time).dt.total_seconds()

    numeric_cols = ["Latitude", "Longitude", "Elevation", "CEP"]
    for column in numeric_cols:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    valid = (
        out["timestamp"].notna()
        & np.isfinite(out["Latitude"])
        & np.isfinite(out["Longitude"])
        & (out["Latitude"] != 0.0)
        & (out["Longitude"] != 0.0)
    )
    out = out.loc[valid].sort_values("time_s").reset_index(drop=True)
    if out.empty:
        out[["east_m", "north_m", "up_m", "std_m"]] = np.empty((0, 4))
        return out

    enu = projector.transform_many(
        out["Latitude"].to_numpy(),
        out["Longitude"].to_numpy(),
        np.full(len(out), projector.origin_altitude_m),
    )
    out[["east_m", "north_m", "up_m"]] = enu

    cep = pd.to_numeric(out["CEP"], errors="coerce") if "CEP" in out.columns else np.nan
    std = np.asarray(cep, dtype=float)
    std = np.where(np.isfinite(std) & (std > 0.0), std, float(default_std_m))
    out["std_m"] = std
    return out


def normalize_radar(
    radar: pd.DataFrame,
    projector: LocalENUProjector,
    truth_origin_time: pd.Timestamp,
    clock_offset_s: float = DEFAULT_RADAR_CLOCK_OFFSET_S,
) -> pd.DataFrame:
    """Normalize radar track rows to truth-relative time and local ENU coordinates.

    ``clock_offset_s`` is added to each Fortem ``globalTime`` timestamp before
    converting to the truth-relative timeline. This offset is independent of the
    RF offset to avoid silently coupling two different sensor clocks.
    """

    out = radar.copy()
    if out.empty:
        out[["timestamp", "time_s", "east_m", "north_m", "up_m"]] = np.empty((0, 5))
        return out

    out["timestamp_raw"] = out["global_time_raw_s"]
    out["timestamp"] = pd.to_datetime(out["global_time_raw_s"], unit="s", errors="coerce")
    out["timestamp"] = out["timestamp"] + pd.to_timedelta(clock_offset_s, unit="s")
    out["time_s"] = (out["timestamp"] - truth_origin_time).dt.total_seconds()

    for column in ["latitude", "longitude", "altitude_m", "cat_prob_uav", "track_id"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    valid = (
        out["timestamp"].notna()
        & np.isfinite(out["latitude"])
        & np.isfinite(out["longitude"])
        & np.isfinite(out["altitude_m"])
    )
    out = out.loc[valid].sort_values(["time_s", "track_id"]).reset_index(drop=True)
    if out.empty:
        out[["east_m", "north_m", "up_m"]] = np.empty((0, 3))
        return out

    enu = projector.transform_many(
        out["latitude"].to_numpy(),
        out["longitude"].to_numpy(),
        out["altitude_m"].to_numpy(),
    )
    out[["east_m", "north_m", "up_m"]] = enu
    return out


def select_radar_measurement_rows(
    radar: pd.DataFrame,
    *,
    selection: str = "catprob",
    truth: pd.DataFrame | None = None,
    catprob_threshold: float = 0.5,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
) -> pd.DataFrame:
    """Select radar track rows for the single-UAV baseline.

    ``catprob`` is an online single-target selector: for each radar frame it
    keeps at most the highest-probability UAV candidate above the threshold.
    Use ``catprob-all`` only to reproduce the legacy behavior that fed every
    above-threshold candidate in the same frame into the single-target filter.
    """

    if radar.empty or selection == "none":
        return radar.iloc[0:0].copy()
    if selection == "all":
        return radar.copy()
    if selection == "catprob":
        return _catprob_best_per_frame_rows(radar, catprob_threshold)
    if selection == "catprob-all":
        return _catprob_threshold_rows(radar, catprob_threshold)
    if selection == "truth-gated":
        if truth is None:
            raise ValueError("truth-gated radar selection requires normalized truth")
        return _truth_gated_rows(radar, truth, truth_gate_m, truth_time_gate_s)
    raise ValueError(f"unknown radar selection {selection!r}")


def projector_from_truth(truth: pd.DataFrame) -> LocalENUProjector:
    """Create a local ENU projector from the first valid truth latitude/longitude row."""

    first = truth[["latitude", "longitude", "altitude_m"]].dropna().iloc[0]
    return LocalENUProjector(
        origin_latitude_deg=float(first["latitude"]),
        origin_longitude_deg=float(first["longitude"]),
        origin_altitude_m=float(first["altitude_m"]),
    )


def truth_to_enu(truth: pd.DataFrame, projector: LocalENUProjector) -> pd.DataFrame:
    """Append ENU columns to a truth dataframe."""

    out = truth.copy()
    enu = projector.transform_many(
        out["latitude"].to_numpy(), out["longitude"].to_numpy(), out["altitude_m"].to_numpy()
    )
    out[["east_m", "north_m", "up_m"]] = enu
    return out


def rf_measurements_to_enu(
    rf: pd.DataFrame,
    projector: LocalENUProjector | None = None,
    truth_origin_time: pd.Timestamp | None = None,
    default_std_m: float = 75.0,
    clock_offset_s: float = DEFAULT_RF_CLOCK_OFFSET_S,
) -> list[TrackingMeasurement]:
    """Convert RF localization rows to 2D ENU measurements."""

    frame = rf
    if "east_m" not in frame.columns:
        if projector is None or truth_origin_time is None:
            raise ValueError("raw RF rows require projector and truth_origin_time")
        frame = normalize_rf(
            frame,
            projector,
            truth_origin_time,
            default_std_m=default_std_m,
            clock_offset_s=clock_offset_s,
        )

    measurements: list[TrackingMeasurement] = []
    for _, row in frame.iterrows():
        std_m = float(row["std_m"]) if np.isfinite(row["std_m"]) else float(default_std_m)
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=np.array([float(row["east_m"]), float(row["north_m"])]),
                covariance=np.diag([std_m**2, std_m**2]),
                source="rf",
            )
        )
    return measurements


def radar_measurements_to_enu(
    radar: pd.DataFrame,
    projector: LocalENUProjector | None = None,
    truth_origin_time: pd.Timestamp | None = None,
    default_xy_std_m: float = 25.0,
    default_z_std_m: float = 35.0,
    default_velocity_std_mps: float = 12.0,
    include_velocity: bool = False,
    clock_offset_s: float = DEFAULT_RADAR_CLOCK_OFFSET_S,
) -> list[TrackingMeasurement]:
    """Convert radar track rows to ENU measurements.

    The default is position-only, even when Fortem velocity components are
    present. This keeps the legacy radar-selection baseline and the radar
    association baselines on the same 3D measurement model. Set
    ``include_velocity=True`` only for deliberate 6D position-plus-velocity
    ablations.
    """

    frame = radar
    if "east_m" not in frame.columns:
        if projector is None or truth_origin_time is None:
            raise ValueError("raw radar rows require projector and truth_origin_time")
        frame = normalize_radar(
            frame,
            projector,
            truth_origin_time,
            clock_offset_s=clock_offset_s,
        )

    position_covariance = np.diag(
        [default_xy_std_m**2, default_xy_std_m**2, default_z_std_m**2]
    )
    measurements: list[TrackingMeasurement] = []
    for _, row in frame.iterrows():
        position = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
        velocity = _radar_velocity_vector_enu(row) if include_velocity else None
        if velocity is None:
            vector = position
            covariance = position_covariance
        else:
            vector = np.concatenate([position, velocity])
            covariance = np.zeros((6, 6), dtype=float)
            covariance[:3, :3] = position_covariance
            covariance[3:, 3:] = np.diag([default_velocity_std_mps**2] * 3)
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=vector,
                covariance=covariance,
                source="radar",
            )
        )
    return measurements


def _radar_velocity_vector_enu(row: pd.Series) -> np.ndarray | None:
    """Return Fortem NED velocity as ENU velocity when all components are finite."""

    required = ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps")
    if not all(column in row.index for column in required):
        return None
    velocity = np.array(
        [
            float(row["velocity_east_mps"]),
            float(row["velocity_north_mps"]),
            -float(row["velocity_down_mps"]),
        ],
        dtype=float,
    )
    return velocity if np.isfinite(velocity).all() else None


def summarize_flight_schema(
    flight: FlightPaths,
    *,
    rf_clock_offset_s: float = DEFAULT_RF_CLOCK_OFFSET_S,
    radar_clock_offset_s: float = DEFAULT_RADAR_CLOCK_OFFSET_S,
) -> dict[str, Any]:
    """Return row counts, columns, and time ranges for one discovered flight."""

    summary: dict[str, Any] = {"flight": flight.name, "root": str(flight.root)}
    truth: pd.DataFrame | None = None
    projector: LocalENUProjector | None = None
    origin_time: pd.Timestamp | None = None
    if flight.truth_txt is not None:
        truth_raw = read_truth(flight.truth_txt)
        truth, projector, origin_time = normalize_truth(truth_raw)
        summary["truth"] = _frame_summary(flight.truth_txt, truth, "timestamp_raw")
    else:
        summary["truth"] = None

    if flight.rf_csv is not None:
        rf_raw = read_rf_csv(flight.rf_csv)
        rf = (
            normalize_rf(
                rf_raw,
                projector,
                origin_time,
                clock_offset_s=rf_clock_offset_s,
            )
            if projector and origin_time
            else rf_raw
        )
        summary["rf"] = _frame_summary(flight.rf_csv, rf, "timestamp_raw")
    else:
        summary["rf"] = None

    if flight.radar_json is not None:
        radar_raw = read_radar_tracks_json(flight.radar_json)
        radar = (
            normalize_radar(
                radar_raw,
                projector,
                origin_time,
                clock_offset_s=radar_clock_offset_s,
            )
            if projector and origin_time
            else radar_raw
        )
        summary["radar"] = _frame_summary(flight.radar_json, radar, "timestamp_raw")
        if "track_id" in radar.columns:
            ids = sorted(int(value) for value in radar["track_id"].dropna().unique())
            summary["radar"]["track_ids_count"] = len(ids)
            summary["radar"]["track_ids_sample"] = ids[:12]
    else:
        summary["radar"] = None
    return summary


def _preferred_variant(paths: Iterable[Path]) -> Path | None:
    files = list(paths)
    if not files:
        return None
    rerun = [path for path in files if "rerun" in path.name.lower()]
    return sorted(rerun or files)[0]


def _flatten_track(
    frame_index: int,
    track_index: int,
    track: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    lla = track.get("lla") or [np.nan, np.nan, np.nan]
    velocity_ned = track.get("velocityNed") or [np.nan, np.nan, np.nan]
    cat_prob = track.get("catProb") or []
    return {
        "frame_index": frame_index,
        "track_index": track_index,
        "track_id": track.get("id"),
        "latitude": _list_get(lla, 0),
        "longitude": _list_get(lla, 1),
        "altitude_m": _list_get(lla, 2),
        "velocity_north_mps": _list_get(velocity_ned, 0),
        "velocity_east_mps": _list_get(velocity_ned, 1),
        "velocity_down_mps": _list_get(velocity_ned, 2),
        "gps_week": track.get("gpsWeek", params.get("gpsWeek")),
        "gps_seconds": track.get("gpsSeconds", params.get("gpsSeconds")),
        "global_time_raw_s": track.get("globalTime", params.get("globalTime")),
        "range_m": track.get("range"),
        "radial_velocity_mps": track.get("radialVelocity"),
        "num_inliers": track.get("numInliers"),
        "cat_prob_uav": _list_get(cat_prob, 0),
        "cat_prob_raw": cat_prob,
    }


def _catprob_threshold_rows(radar: pd.DataFrame, catprob_threshold: float) -> pd.DataFrame:
    if "cat_prob_uav" not in radar.columns:
        raise KeyError("radar catprob selection requires cat_prob_uav")
    return radar.loc[radar["cat_prob_uav"] >= float(catprob_threshold)].copy()


def _catprob_best_per_frame_rows(
    radar: pd.DataFrame, catprob_threshold: float
) -> pd.DataFrame:
    candidates = _catprob_threshold_rows(radar, catprob_threshold)
    if candidates.empty:
        return candidates

    frame_column = _radar_frame_group_column(candidates)
    ranked = _catprob_ranked_rows(candidates, frame_column)
    if frame_column is None:
        return ranked.head(1).sort_index()
    return ranked.groupby(frame_column, sort=False, dropna=False).head(1).sort_index()


def _catprob_ranked_rows(
    candidates: pd.DataFrame, frame_column: str | None = None
) -> pd.DataFrame:
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if frame_column is not None:
        sort_columns.append(frame_column)
        ascending.append(True)
    sort_columns.append("cat_prob_uav")
    ascending.append(False)
    for tie_breaker in ("track_id", "track_index"):
        if tie_breaker in candidates.columns:
            sort_columns.append(tie_breaker)
            ascending.append(True)
    return candidates.sort_values(sort_columns, ascending=ascending, kind="mergesort")


def _radar_frame_group_column(radar: pd.DataFrame) -> str | None:
    for column in ("frame_index", "timestamp", "time_s"):
        if column in radar.columns:
            return column
    return None


def _truth_gated_rows(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> pd.DataFrame:
    truth_times = truth["time_s"].to_numpy(dtype=float)
    query_times = radar["time_s"].to_numpy(dtype=float)
    truth_indices = _nearest_time_indices(truth_times, query_times)
    dt = np.abs(truth_times[truth_indices] - query_times)
    radar_xyz = radar[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    truth_xyz = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[truth_indices]
    errors = np.linalg.norm(radar_xyz - truth_xyz, axis=1)
    keep = (dt <= float(truth_time_gate_s)) & (errors <= float(truth_gate_m))
    return radar.loc[keep].copy()


def _nearest_time_indices(reference_times_s: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    insertion = np.searchsorted(reference, query)
    right = np.clip(insertion, 0, reference.size - 1)
    left = np.clip(insertion - 1, 0, reference.size - 1)
    use_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(use_right, right, left)


def _frame_summary(path: Path, frame: pd.DataFrame, raw_time_col: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "file": path.name,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }
    if raw_time_col in frame.columns and len(frame):
        raw = frame[raw_time_col].dropna()
        summary["raw_time_min"] = str(raw.iloc[0]) if len(raw) else None
        summary["raw_time_max"] = str(raw.iloc[-1]) if len(raw) else None
    if "time_s" in frame.columns and len(frame):
        times = frame["time_s"].dropna()
        summary["time_s_min"] = float(times.min()) if len(times) else None
        summary["time_s_max"] = float(times.max()) if len(times) else None
    return summary


def _list_get(values: Any, index: int) -> Any:
    if isinstance(values, (list, tuple)) and index < len(values):
        return values[index]
    return np.nan
