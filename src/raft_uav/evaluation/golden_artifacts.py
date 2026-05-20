"""Golden-artifact checks for small RaFT-UAV regression runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_REQUIRED_FILES = ("metrics.json", "estimates.csv", "diagnostics.csv", "selected_radar.csv")


def check_run_artifacts(
    run_dir: Path,
    *,
    required_files: Sequence[str] = DEFAULT_REQUIRED_FILES,
    max_nan_fraction: float = 0.05,
) -> list[dict[str, Any]]:
    """Return check results for one run output directory."""

    results: list[dict[str, Any]] = []
    for name in required_files:
        path = run_dir / name
        results.append(
            {
                "check": "required_file_exists",
                "file": name,
                "passed": path.exists(),
                "message": "" if path.exists() else f"missing {path}",
            }
        )
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        results.extend(_check_metrics(metrics_path))
    for csv_name in ("estimates.csv", "diagnostics.csv", "selected_radar.csv"):
        path = run_dir / csv_name
        if path.exists():
            results.extend(_check_csv(path, max_nan_fraction=max_nan_fraction))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-nan-fraction", type=float, default=0.05)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        for row in check_run_artifacts(run_dir, max_nan_fraction=args.max_nan_fraction):
            rows.append({"run_dir": str(run_dir), **row})
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    failures = [row for row in rows if not row.get("passed", False)]
    print(f"checks_json={args.output_json}")
    print(f"failed_checks={len(failures)}")
    if args.fail_on_error and failures:
        return 1
    return 0


def _check_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [{"check": "metrics_json_parse", "file": str(path), "passed": False, "message": str(exc)}]
    rows.append({"check": "metrics_json_parse", "file": str(path), "passed": True, "message": ""})
    for key in ("posterior_records", "accepted_measurements", "position_error_3d"):
        rows.append(
            {
                "check": "metrics_required_key",
                "file": str(path),
                "key": key,
                "passed": key in metrics,
                "message": "" if key in metrics else f"missing key {key}",
            }
        )
    return rows


def _check_csv(path: Path, *, max_nan_fraction: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return [{"check": "csv_parse", "file": str(path), "passed": False, "message": str(exc)}]
    rows.append({"check": "csv_parse", "file": str(path), "passed": True, "message": ""})
    rows.append(
        {
            "check": "csv_nonempty_or_allowed",
            "file": str(path),
            "passed": len(frame) > 0 or path.name == "selected_radar.csv",
            "message": "" if len(frame) > 0 or path.name == "selected_radar.csv" else "CSV has no rows",
        }
    )
    if frame.empty:
        return rows
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty:
        nan_fraction = float(numeric.isna().to_numpy().mean())
        rows.append(
            {
                "check": "numeric_nan_fraction",
                "file": str(path),
                "passed": nan_fraction <= float(max_nan_fraction),
                "value": nan_fraction,
                "message": "" if nan_fraction <= float(max_nan_fraction) else "too many NaNs",
            }
        )
    if "time_s" in frame.columns:
        times = pd.to_numeric(frame["time_s"], errors="coerce").dropna().to_numpy(dtype=float)
        monotonic = bool(np.all(np.diff(times) >= -1.0e-9)) if times.size else True
        rows.append(
            {
                "check": "time_monotonic",
                "file": str(path),
                "passed": monotonic,
                "message": "" if monotonic else "time_s is not sorted",
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
