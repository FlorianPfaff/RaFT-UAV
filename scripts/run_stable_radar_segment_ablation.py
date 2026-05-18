"""Run fast stable-radar-segment diagnostic ablations."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ablation_common as common  # noqa: E402
from raft_uav.diagnostics.paper_table import run_paper_table_diagnostic  # noqa: E402

RADAR_SELECTIONS = (
    "radar-longest-track-range-gated",
    "radar-stable-segments-range-gated",
    "radar-stable-segments-range-gated-interpolated",
)
SUMMARY_COLUMNS = (
    "flight",
    "method",
    "radar_catprob_threshold",
    "radar_range_gate_m",
    "stable_segment_min_frames",
    "stable_segment_max_transition_speed_mps",
    "candidate_count",
    "selected_count",
    "matched_count",
    "coverage",
    "track_switches",
    "error_3d_mean_m",
    "error_3d_rmse_m",
    "error_3d_p95_m",
    "error_3d_max_m",
    "error_2d_mean_m",
    "error_2d_rmse_m",
    "error_2d_p95_m",
    "table_path",
)


@dataclass(frozen=True)
class StableSegmentConfig:
    """One stable-radar-segment diagnostic configuration."""

    name: str
    radar_catprob_threshold: float
    radar_range_gate_m: float
    stable_segment_min_frames: int
    stable_segment_max_transition_speed_mps: float


def main() -> int:
    parser = argparse.ArgumentParser()
    common.add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/stable_radar_segment_ablation"),
        default_summary_output=Path("outputs/stable_radar_segment_ablation.csv"),
    )
    parser.add_argument("--catprob-thresholds", nargs="*", type=float, default=[0.4])
    parser.add_argument("--range-gates-m", nargs="*", type=float, default=[800.0])
    parser.add_argument("--min-segment-frames", nargs="*", type=int, default=[75, 100, 150])
    parser.add_argument("--max-transition-speeds-mps", nargs="*", type=float, default=[35.0, 65.0, 100.0])
    parser.add_argument("--truth-time-gate-s", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    _validate_args(args)

    rows: list[dict[str, object]] = []
    for config in _configs(args):
        run_dir = args.output_dir / config.name
        for flight in args.flights:
            table_path = run_dir / flight / "paper_table.csv"
            if not (args.skip_existing and table_path.exists()):
                run_paper_table_diagnostic(
                    dataset_root=args.dataset_root,
                    flight_name=flight,
                    output_dir=run_dir,
                    radar_catprob_threshold=config.radar_catprob_threshold,
                    radar_range_gate_m=config.radar_range_gate_m,
                    stable_segment_min_frames=config.stable_segment_min_frames,
                    stable_segment_max_transition_speed_mps=config.stable_segment_max_transition_speed_mps,
                    radar_selections=RADAR_SELECTIONS,
                    truth_time_gate_s=args.truth_time_gate_s,
                    include_fusion=False,
                )
            rows.extend(_rows_from_table(config, table_path))
    _write_summary(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "catprob_thresholds",
        "range_gates_m",
        "min_segment_frames",
        "max_transition_speeds_mps",
    ):
        if not getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} must not be empty")
    if any(value < 1 for value in args.min_segment_frames):
        raise SystemExit("--min-segment-frames values must be positive")
    if any(value <= 0.0 for value in args.range_gates_m):
        raise SystemExit("--range-gates-m values must be positive")
    if any(value <= 0.0 for value in args.max_transition_speeds_mps):
        raise SystemExit("--max-transition-speeds-mps values must be positive")


def _configs(args: argparse.Namespace) -> list[StableSegmentConfig]:
    configs: list[StableSegmentConfig] = []
    for catprob, range_gate, min_frames, max_speed in itertools.product(
        args.catprob_thresholds,
        args.range_gates_m,
        args.min_segment_frames,
        args.max_transition_speeds_mps,
    ):
        configs.append(
            StableSegmentConfig(
                name=_config_name(catprob, range_gate, min_frames, max_speed),
                radar_catprob_threshold=float(catprob),
                radar_range_gate_m=float(range_gate),
                stable_segment_min_frames=int(min_frames),
                stable_segment_max_transition_speed_mps=float(max_speed),
            )
        )
    return configs


def _config_name(
    catprob: float,
    range_gate_m: float,
    min_frames: int,
    max_transition_speed_mps: float,
) -> str:
    return (
        f"stable_cat{common.slug(catprob, precision=2)}"
        f"_rg{common.slug(range_gate_m, precision=0)}"
        f"_min{int(min_frames)}"
        f"_v{common.slug(max_transition_speed_mps, precision=0)}"
    )


def _rows_from_table(
    config: StableSegmentConfig,
    table_path: Path,
) -> list[dict[str, object]]:
    table = pd.read_csv(table_path)
    rows: list[dict[str, object]] = []
    for _, item in table.iterrows():
        method = str(item.get("method", ""))
        if method not in RADAR_SELECTIONS:
            continue
        row: dict[str, object] = {
            "flight": table_path.parent.name,
            "method": method,
            "radar_catprob_threshold": config.radar_catprob_threshold,
            "radar_range_gate_m": config.radar_range_gate_m,
            "stable_segment_min_frames": config.stable_segment_min_frames,
            "stable_segment_max_transition_speed_mps": config.stable_segment_max_transition_speed_mps,
            "table_path": str(table_path),
        }
        for column in SUMMARY_COLUMNS:
            if column not in row and column in item.index:
                row[column] = _csv_value(item[column])
        rows.append({column: row.get(column, "") for column in SUMMARY_COLUMNS})
    return rows


def _csv_value(value: Any) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return round(float(value), 3)
    return value


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No stable radar segment ablation rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS)).to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
