"""CLI-exposed runtime configuration for experimental variants."""

from __future__ import annotations

import argparse
import os
from typing import Any

RADAR_COVARIANCE_MODES = ("fixed", "range-angle")
RADAR_UPDATE_DIMENSIONS = ("position", "position-velocity")


def add_runtime_configuration_arguments(parser: argparse.ArgumentParser) -> None:
    """Register reproducibility arguments shared by baseline-style CLIs."""

    radar = parser.add_argument_group("radar covariance runtime configuration")
    radar.add_argument("--radar-covariance-mode", choices=RADAR_COVARIANCE_MODES, default="range-angle")
    radar.add_argument("--radar-xy-std-m", type=float, default=25.0)
    radar.add_argument("--radar-z-std-m", type=float, default=35.0)
    radar.add_argument("--radar-range-std-m", type=float, default=5.0)
    radar.add_argument("--radar-azimuth-std-deg", type=float, default=2.0)
    radar.add_argument("--radar-elevation-std-deg", type=float, default=2.0)
    radar.add_argument("--radar-covariance-min-std-m", type=float, default=3.0)
    radar.add_argument("--radar-covariance-max-std-m", type=float, default=250.0)
    radar.add_argument("--radar-origin-east-m", type=float, default=0.0)
    radar.add_argument("--radar-origin-north-m", type=float, default=0.0)
    radar.add_argument("--radar-origin-up-m", type=float, default=0.0)

    radar_update = parser.add_argument_group("radar measurement update runtime configuration")
    radar_update.add_argument(
        "--radar-update-dimension",
        choices=RADAR_UPDATE_DIMENSIONS,
        default="position",
        help=(
            "radar Kalman update vector; position preserves the historical 3D "
            "update, position-velocity also conditions on Fortem velocityNed"
        ),
    )
    radar_update.add_argument(
        "--radar-velocity-std-mps",
        type=float,
        default=12.0,
        help="per-axis standard deviation for Fortem velocity components in 6D radar updates",
    )

    tracklet = parser.add_argument_group("tracklet-viterbi runtime configuration")
    tracklet.add_argument("--tracklet-max-candidates", type=int, default=8)
    tracklet.add_argument("--tracklet-missed-detection-cost", type=float, default=7.0)
    tracklet.add_argument("--tracklet-consecutive-miss-cost", type=float, default=1.0)
    tracklet.add_argument("--tracklet-track-switch-cost", type=float, default=8.0)
    tracklet.add_argument("--tracklet-missing-track-id-cost", type=float, default=1.0)
    tracklet.add_argument("--tracklet-catprob-weight", type=float, default=2.5)
    tracklet.add_argument("--tracklet-anchor-nis-weight", type=float, default=0.35)
    tracklet.add_argument("--tracklet-transition-nis-weight", type=float, default=1.0)
    tracklet.add_argument("--tracklet-velocity-nis-weight", type=float, default=0.15)
    tracklet.add_argument("--tracklet-transition-position-std-m", type=float, default=40.0)
    tracklet.add_argument("--tracklet-transition-speed-std-mps", type=float, default=18.0)
    tracklet.add_argument("--tracklet-velocity-std-mps", type=float, default=12.0)
    tracklet.add_argument("--tracklet-max-speed-mps", type=float, default=55.0)
    tracklet.add_argument("--tracklet-max-speed-penalty", type=float, default=10.0)
    tracklet.add_argument("--tracklet-range-gate-m", type=float, default=850.0)
    tracklet.add_argument("--tracklet-range-gate-slack-m", type=float, default=150.0)
    tracklet.add_argument("--tracklet-range-penalty", type=float, default=10.0)
    tracklet.add_argument("--disable-tracklet-rf-anchor", action="store_true")


def parse_runtime_config(argv: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Parse known runtime flags and return the remaining original CLI args."""

    parser = argparse.ArgumentParser(add_help=False)
    add_runtime_configuration_arguments(parser)
    args, remaining = parser.parse_known_args(argv)
    return runtime_config_from_args(args), remaining


def runtime_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build a JSON-serializable runtime configuration from parsed args."""

    radar = {
        "mode": str(args.radar_covariance_mode),
        "xy_std_m": _positive_float(args.radar_xy_std_m, "radar_xy_std_m"),
        "z_std_m": _positive_float(args.radar_z_std_m, "radar_z_std_m"),
        "range_std_m": _positive_float(args.radar_range_std_m, "radar_range_std_m"),
        "azimuth_std_deg": _positive_float(args.radar_azimuth_std_deg, "radar_azimuth_std_deg"),
        "elevation_std_deg": _positive_float(args.radar_elevation_std_deg, "radar_elevation_std_deg"),
        "min_std_m": _positive_float(args.radar_covariance_min_std_m, "radar_covariance_min_std_m"),
        "max_std_m": _positive_float(args.radar_covariance_max_std_m, "radar_covariance_max_std_m"),
        "origin_east_m": _finite_float(args.radar_origin_east_m, "radar_origin_east_m"),
        "origin_north_m": _finite_float(args.radar_origin_north_m, "radar_origin_north_m"),
        "origin_up_m": _finite_float(args.radar_origin_up_m, "radar_origin_up_m"),
    }
    if radar["max_std_m"] < radar["min_std_m"]:
        raise ValueError("radar_covariance_max_std_m must be >= radar_covariance_min_std_m")

    radar_update = {
        "dimension": str(args.radar_update_dimension),
        "velocity_std_mps": _positive_float(args.radar_velocity_std_mps, "radar_velocity_std_mps"),
    }
    if radar_update["dimension"] not in RADAR_UPDATE_DIMENSIONS:
        raise ValueError(f"radar_update_dimension must be one of {RADAR_UPDATE_DIMENSIONS}")

    tracklet = {
        "max_candidates": _positive_int(args.tracklet_max_candidates, "tracklet_max_candidates"),
        "missed_detection_cost": _positive_float(args.tracklet_missed_detection_cost, "tracklet_missed_detection_cost"),
        "consecutive_miss_cost": _positive_float(args.tracklet_consecutive_miss_cost, "tracklet_consecutive_miss_cost"),
        "track_switch_cost": _positive_float(args.tracklet_track_switch_cost, "tracklet_track_switch_cost"),
        "missing_track_id_cost": _positive_float(args.tracklet_missing_track_id_cost, "tracklet_missing_track_id_cost"),
        "catprob_weight": _nonnegative_float(args.tracklet_catprob_weight, "tracklet_catprob_weight"),
        "anchor_nis_weight": _nonnegative_float(args.tracklet_anchor_nis_weight, "tracklet_anchor_nis_weight"),
        "transition_nis_weight": _nonnegative_float(args.tracklet_transition_nis_weight, "tracklet_transition_nis_weight"),
        "velocity_nis_weight": _nonnegative_float(args.tracklet_velocity_nis_weight, "tracklet_velocity_nis_weight"),
        "transition_position_std_m": _positive_float(args.tracklet_transition_position_std_m, "tracklet_transition_position_std_m"),
        "transition_speed_std_mps": _positive_float(args.tracklet_transition_speed_std_mps, "tracklet_transition_speed_std_mps"),
        "velocity_std_mps": _positive_float(args.tracklet_velocity_std_mps, "tracklet_velocity_std_mps"),
        "max_speed_mps": _positive_float(args.tracklet_max_speed_mps, "tracklet_max_speed_mps"),
        "max_speed_penalty": _nonnegative_float(args.tracklet_max_speed_penalty, "tracklet_max_speed_penalty"),
        "range_gate_m": None if float(args.tracklet_range_gate_m) <= 0.0 else _positive_float(args.tracklet_range_gate_m, "tracklet_range_gate_m"),
        "range_gate_slack_m": _nonnegative_float(args.tracklet_range_gate_slack_m, "tracklet_range_gate_slack_m"),
        "range_penalty": _nonnegative_float(args.tracklet_range_penalty, "tracklet_range_penalty"),
        "use_rf_anchor": not bool(args.disable_tracklet_rf_anchor),
    }
    return {"radar_covariance": radar, "radar_update": radar_update, "tracklet_viterbi": tracklet}


def apply_runtime_environment(runtime_config: dict[str, Any]) -> None:
    """Apply CLI runtime config to the existing RAFT_UAV_* runtime layer."""

    radar = runtime_config["radar_covariance"]
    radar_update = runtime_config["radar_update"]
    tracklet = runtime_config["tracklet_viterbi"]
    mapping = {
        "RAFT_UAV_RADAR_COVARIANCE_MODE": radar["mode"],
        "RAFT_UAV_RADAR_XY_STD_M": radar["xy_std_m"],
        "RAFT_UAV_RADAR_Z_STD_M": radar["z_std_m"],
        "RAFT_UAV_RADAR_RANGE_STD_M": radar["range_std_m"],
        "RAFT_UAV_RADAR_AZIMUTH_STD_DEG": radar["azimuth_std_deg"],
        "RAFT_UAV_RADAR_ELEVATION_STD_DEG": radar["elevation_std_deg"],
        "RAFT_UAV_RADAR_COVARIANCE_MIN_STD_M": radar["min_std_m"],
        "RAFT_UAV_RADAR_COVARIANCE_MAX_STD_M": radar["max_std_m"],
        "RAFT_UAV_RADAR_ORIGIN_EAST_M": radar["origin_east_m"],
        "RAFT_UAV_RADAR_ORIGIN_NORTH_M": radar["origin_north_m"],
        "RAFT_UAV_RADAR_ORIGIN_UP_M": radar["origin_up_m"],
        "RAFT_UAV_RADAR_UPDATE_DIMENSION": radar_update["dimension"],
        "RAFT_UAV_RADAR_VELOCITY_STD_MPS": radar_update["velocity_std_mps"],
        "RAFT_UAV_TRACKLET_MAX_CANDIDATES": tracklet["max_candidates"],
        "RAFT_UAV_TRACKLET_MISSED_DETECTION_COST": tracklet["missed_detection_cost"],
        "RAFT_UAV_TRACKLET_CONSECUTIVE_MISS_COST": tracklet["consecutive_miss_cost"],
        "RAFT_UAV_TRACKLET_TRACK_SWITCH_COST": tracklet["track_switch_cost"],
        "RAFT_UAV_TRACKLET_MISSING_TRACK_ID_COST": tracklet["missing_track_id_cost"],
        "RAFT_UAV_TRACKLET_CATPROB_WEIGHT": tracklet["catprob_weight"],
        "RAFT_UAV_TRACKLET_ANCHOR_NIS_WEIGHT": tracklet["anchor_nis_weight"],
        "RAFT_UAV_TRACKLET_TRANSITION_NIS_WEIGHT": tracklet["transition_nis_weight"],
        "RAFT_UAV_TRACKLET_VELOCITY_NIS_WEIGHT": tracklet["velocity_nis_weight"],
        "RAFT_UAV_TRACKLET_TRANSITION_POSITION_STD_M": tracklet["transition_position_std_m"],
        "RAFT_UAV_TRACKLET_TRANSITION_SPEED_STD_MPS": tracklet["transition_speed_std_mps"],
        "RAFT_UAV_TRACKLET_VELOCITY_STD_MPS": tracklet["velocity_std_mps"],
        "RAFT_UAV_TRACKLET_MAX_SPEED_MPS": tracklet["max_speed_mps"],
        "RAFT_UAV_TRACKLET_MAX_SPEED_PENALTY": tracklet["max_speed_penalty"],
        "RAFT_UAV_TRACKLET_RANGE_GATE_M": 0.0 if tracklet["range_gate_m"] is None else tracklet["range_gate_m"],
        "RAFT_UAV_TRACKLET_RANGE_GATE_SLACK_M": tracklet["range_gate_slack_m"],
        "RAFT_UAV_TRACKLET_RANGE_PENALTY": tracklet["range_penalty"],
        "RAFT_UAV_TRACKLET_USE_RF_ANCHOR": "1" if tracklet["use_rf_anchor"] else "0",
    }
    for name, value in mapping.items():
        os.environ[name] = str(value)


def _finite_float(value: object, name: str) -> float:
    number = float(value)
    if number != number or abs(number) == float("inf"):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _positive_int(value: object, name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number
