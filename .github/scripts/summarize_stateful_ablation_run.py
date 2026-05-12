#!/usr/bin/env python3
"""Write a compact per-run summary for stateful learned association ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def count_actions(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "update_action" not in frame.columns:
        return {}
    return {
        str(action): int(count)
        for action, count in frame["update_action"].astype(str).value_counts().sort_index().items()
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    metrics = load_json_if_exists(args.metrics_path)
    diagnostic_summary = load_json_if_exists(args.diagnostic_summary_path)
    summary: dict[str, Any] = {
        "flight": args.flight,
        "variant": args.variant,
        "gating": args.gating,
        "association_safety_gate_enabled": args.gating == "on",
        "beam_track_switch_cost": float(args.beam_track_switch_cost),
        "beam_lag_s": float(args.beam_lag_s),
        "radar_inflation_alpha": float(args.radar_inflation_alpha),
        "rf_inflation_alpha": float(args.rf_inflation_alpha),
        "status": "missing_metrics",
        "metrics_path": str(args.metrics_path),
        "diagnostic_summary_path": str(args.diagnostic_summary_path),
        "diagnostics_path": str(args.diagnostics_path),
    }
    if metrics is None:
        return summary

    action_counts = count_actions(args.diagnostics_path)
    summary.update(
        {
            "status": "ok",
            "radar_association": metrics.get("radar_association"),
            "learned_radar_association_mode": metrics.get("learned_radar_association_mode"),
            "selected_radar_rows": metrics.get("selected_radar_rows"),
            "posterior_records": metrics.get("posterior_records"),
            "accepted_measurements": metrics.get("accepted_measurements"),
            "rejected_measurements": metrics.get("rejected_measurements"),
            "reweighted_measurements": metrics.get("reweighted_measurements"),
            "rmse_2d_m": (metrics.get("position_error_2d") or {}).get("rmse_m"),
            "p95_2d_m": (metrics.get("position_error_2d") or {}).get("p95_m"),
            "rmse_3d_m": (metrics.get("position_error_3d") or {}).get("rmse_m"),
            "p95_3d_m": (metrics.get("position_error_3d") or {}).get("p95_m"),
            "update_action_counts": action_counts,
            "missed_detection_count": int(action_counts.get("missed_detection", 0)),
            "rejected_count": int(action_counts.get("rejected", 0)),
            "inflated_count": int(action_counts.get("inflated", 0)),
        }
    )
    if isinstance(diagnostic_summary, dict):
        summary["track_switch_count"] = (
            (diagnostic_summary.get("track_switches") or {})
            .get("selected_radar", {})
            .get("count")
        )
        summary["covariance_inflation_count"] = (
            (diagnostic_summary.get("covariance_inflation") or {}).get("count")
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gating", choices=["on", "off"], required=True)
    parser.add_argument("--beam-track-switch-cost", type=float, required=True)
    parser.add_argument("--beam-lag-s", type=float, required=True)
    parser.add_argument("--radar-inflation-alpha", type=float, required=True)
    parser.add_argument("--rf-inflation-alpha", type=float, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--diagnostic-summary-path", type=Path, required=True)
    parser.add_argument("--diagnostics-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_summary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
