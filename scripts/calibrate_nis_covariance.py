#!/usr/bin/env python3
"""Estimate source-wise covariance scales from RaFT-UAV diagnostics.csv files.

Example:
    python scripts/calibrate_nis_covariance.py \
        outputs/baseline/Opt*/diagnostics.csv \
        --output outputs/calibration/nis_covariance_scales.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raft_uav.calibration.nis import (
    NisCalibrationSettings,
    load_diagnostics,
    make_nis_calibration_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate covariance multipliers from accepted NIS diagnostics."
    )
    parser.add_argument(
        "diagnostics",
        nargs="+",
        help="One or more diagnostics.csv files produced by raft-uav run-baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/calibration/nis_covariance_scales.json"),
        help="JSON report path.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=8,
        help="Minimum accepted NIS values required for a source/dimension group.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="Chi-square quantile used to estimate the covariance multiplier.",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.2,
        help="Lower clipping bound for suggested covariance multipliers.",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=25.0,
        help="Upper clipping bound for suggested covariance multipliers.",
    )
    args = parser.parse_args()

    settings = NisCalibrationSettings(
        min_samples=args.min_samples,
        calibration_quantile=args.quantile,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )
    diagnostics = load_diagnostics(args.diagnostics)
    report = make_nis_calibration_report(diagnostics, settings=settings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {args.output}")
    source_scales = report["source_covariance_scales"]
    if source_scales:
        print("suggested source covariance multipliers:")
        for source, scale in source_scales.items():
            print(f"  {source}: {scale:.3f}")
    else:
        print("no groups met the sample-count criterion")


if __name__ == "__main__":
    main()
