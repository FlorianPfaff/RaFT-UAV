"""CLI for fitting learned radar association likelihoods."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.baselines.learned_radar_likelihood import (
    DEFAULT_FEATURE_NAMES,
    fit_learned_radar_association_model,
)
from raft_uav.baselines.radar_likelihood_training import collect_radar_association_training_frame
from raft_uav.io.aerpaw import (
    discover_flights,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-train-radar-association",
        description="fit a learned radar association likelihood from training flights",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--flight",
        action="append",
        default=None,
        help=(
            "training flight name/substr; repeat for multiple flights; "
            "defaults to all usable flights"
        ),
    )
    parser.add_argument(
        "--exclude-flight",
        action="append",
        default=[],
        help="flight name/substr to leave out, e.g. for leave-one-flight-out validation",
    )
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-examples", type=Path, default=None)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--radar-xy-std", type=float, default=25.0)
    parser.add_argument("--radar-z-std", type=float, default=35.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.5)
    parser.add_argument("--disable-radar-catprob-threshold", action="store_true")
    parser.add_argument("--positive-gate-m", type=float, default=50.0)
    parser.add_argument("--truth-gate-m", type=float, default=150.0)
    parser.add_argument("--truth-time-gate-s", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1.0e-3)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--no-balance-classes", action="store_true")
    args = parser.parse_args(argv)

    flights = (
        [select_flight(args.dataset_root, name) for name in args.flight]
        if args.flight
        else discover_flights(args.dataset_root)
    )
    if args.exclude_flight:
        excluded = {select_flight(args.dataset_root, name).name for name in args.exclude_flight}
        flights = [flight for flight in flights if flight.name not in excluded]
    frames: list[pd.DataFrame] = []
    used_flights: list[str] = []
    for flight in flights:
        if flight.truth_txt is None or flight.radar_json is None:
            continue
        truth_raw = read_truth(flight.truth_txt)
        truth, projector, truth_origin_time = normalize_truth(truth_raw)
        rf_measurements = []
        if flight.rf_csv is not None:
            rf = _inside_truth_window(
                normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time),
                truth,
            )
            rf_measurements = rf_measurements_to_enu(rf)
        radar = _inside_truth_window(
            normalize_radar(
                read_radar_tracks_json(flight.radar_json),
                projector,
                truth_origin_time,
            ),
            truth,
        )
        examples = collect_radar_association_training_frame(
            rf_measurements=rf_measurements,
            radar=radar,
            truth=truth,
            flight_name=flight.name,
            acceleration_std_mps2=args.acceleration_std,
            radar_xy_std_m=args.radar_xy_std,
            radar_z_std_m=args.radar_z_std,
            candidate_catprob_threshold=None
            if args.disable_radar_catprob_threshold
            else args.radar_catprob_threshold,
            positive_gate_m=args.positive_gate_m,
            truth_gate_m=args.truth_gate_m,
            truth_time_gate_s=args.truth_time_gate_s,
        )
        if not examples.empty:
            frames.append(examples)
            used_flights.append(flight.name)

    if not frames:
        raise RuntimeError("no training examples were collected")
    examples = pd.concat(frames, ignore_index=True)
    model = fit_learned_radar_association_model(
        examples,
        feature_names=DEFAULT_FEATURE_NAMES,
        label_column="label",
        l2=args.l2,
        max_iter=args.max_iter,
        balance_classes=not args.no_balance_classes,
        metadata={
            "training_flights": used_flights,
            "excluded_flights": list(args.exclude_flight or []),
            "positive_gate_m": float(args.positive_gate_m),
            "truth_gate_m": float(args.truth_gate_m),
            "truth_time_gate_s": float(args.truth_time_gate_s),
            "radar_catprob_threshold": None
            if args.disable_radar_catprob_threshold
            else float(args.radar_catprob_threshold),
            "radar_xy_std_m": float(args.radar_xy_std),
            "radar_z_std_m": float(args.radar_z_std),
            "acceleration_std_mps2": float(args.acceleration_std),
        },
    )
    model.save(args.output_model)
    if args.output_examples is not None:
        args.output_examples.parent.mkdir(parents=True, exist_ok=True)
        examples.to_csv(args.output_examples, index=False)

    positives = int(examples["label"].sum())
    negatives = int(len(examples) - positives)
    print(f"training_flights={len(used_flights)}")
    print(f"examples={len(examples)}")
    print(f"positive_examples={positives}")
    print(f"negative_examples={negatives}")
    print(f"features={','.join(model.feature_names)}")
    print(f"model_json={args.output_model}")
    if args.output_examples is not None:
        print(f"examples_csv={args.output_examples}")
    return 0


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    start = float(truth["time_s"].min())
    end = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= start) & (frame["time_s"] <= end)].copy()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
