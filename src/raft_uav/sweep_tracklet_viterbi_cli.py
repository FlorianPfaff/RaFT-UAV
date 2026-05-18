"""Grid-search tracklet-Viterbi association settings over one or more flights."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class TrackletSweepConfig:
    """One tracklet-Viterbi parameter combination."""

    track_switch_cost: float
    anchor_nis_weight: float
    missed_detection_cost: float
    max_candidates: int

    @property
    def config_id(self) -> str:
        return (
            f"sw{self.track_switch_cost:g}_"
            f"anc{self.anchor_nis_weight:g}_"
            f"miss{self.missed_detection_cost:g}_"
            f"cand{self.max_candidates:d}"
        ).replace(".", "p")

    def environment(self) -> dict[str, str]:
        """Return RAFT_UAV_TRACKLET_* settings for this configuration."""

        return {
            "RAFT_UAV_TRACKLET_TRACK_SWITCH_COST": str(float(self.track_switch_cost)),
            "RAFT_UAV_TRACKLET_ANCHOR_NIS_WEIGHT": str(float(self.anchor_nis_weight)),
            "RAFT_UAV_TRACKLET_MISSED_DETECTION_COST": str(float(self.missed_detection_cost)),
            "RAFT_UAV_TRACKLET_MAX_CANDIDATES": str(int(self.max_candidates)),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-sweep-tracklet-viterbi",
        description="grid-search tracklet-Viterbi parameters via raft-uav run-baseline",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracklet-viterbi-sweep"))
    parser.add_argument("--track-switch-cost", default="6,8,12,16,24")
    parser.add_argument("--anchor-nis-weight", default="0.2,0.35,0.5,0.8")
    parser.add_argument("--missed-detection-cost", default="4,7,10")
    parser.add_argument("--max-candidates", default="6,8,12")
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.4)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--smoother", default="fixed-lag")
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--robust-update", default="none")
    parser.add_argument("--enable-gating", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="continue the sweep if one run fails",
    )
    args = parser.parse_args(argv)

    configs = build_sweep_configs(
        track_switch_costs=parse_float_grid(args.track_switch_cost),
        anchor_nis_weights=parse_float_grid(args.anchor_nis_weight),
        missed_detection_costs=parse_float_grid(args.missed_detection_cost),
        max_candidates=parse_int_grid(args.max_candidates),
    )
    results = run_sweep(
        dataset_root=args.dataset_root,
        flights=tuple(args.flight),
        output_dir=args.output_dir,
        configs=configs,
        radar_catprob_threshold=args.radar_catprob_threshold,
        acceleration_std=args.acceleration_std,
        smoother=args.smoother,
        smoother_lag_s=args.smoother_lag_s,
        max_eval_time_delta_s=args.max_eval_time_delta_s,
        robust_update=args.robust_update,
        enable_gating=args.enable_gating,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )
    write_sweep_outputs(results, args.output_dir, opt1_name="Opt1")
    print(f"configs={len(configs)}")
    print(f"runs={len(results)}")
    print(f"sweep_results_csv={args.output_dir / 'sweep_results.csv'}")
    print(f"sweep_summary_csv={args.output_dir / 'sweep_summary.csv'}")
    return 0


def parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("float grid must not be empty")
    return values


def parse_int_grid(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("integer grid must not be empty")
    if any(value < 1 for value in values):
        raise ValueError("integer grid values must be positive")
    return values


def build_sweep_configs(
    *,
    track_switch_costs: Iterable[float],
    anchor_nis_weights: Iterable[float],
    missed_detection_costs: Iterable[float],
    max_candidates: Iterable[int],
) -> list[TrackletSweepConfig]:
    return [
        TrackletSweepConfig(
            track_switch_cost=float(track_switch_cost),
            anchor_nis_weight=float(anchor_nis_weight),
            missed_detection_cost=float(missed_detection_cost),
            max_candidates=int(max_candidate_count),
        )
        for track_switch_cost, anchor_nis_weight, missed_detection_cost, max_candidate_count in itertools.product(
            track_switch_costs,
            anchor_nis_weights,
            missed_detection_costs,
            max_candidates,
        )
    ]


def run_sweep(
    *,
    dataset_root: Path,
    flights: tuple[str, ...],
    output_dir: Path,
    configs: list[TrackletSweepConfig],
    radar_catprob_threshold: float,
    acceleration_std: float,
    smoother: str,
    smoother_lag_s: float,
    max_eval_time_delta_s: float,
    robust_update: str,
    enable_gating: bool,
    dry_run: bool,
    continue_on_error: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for flight in flights:
            run_dir = output_dir / "runs" / config.config_id
            command = baseline_command(
                dataset_root=dataset_root,
                flight=flight,
                output_dir=run_dir,
                radar_catprob_threshold=radar_catprob_threshold,
                acceleration_std=acceleration_std,
                smoother=smoother,
                smoother_lag_s=smoother_lag_s,
                max_eval_time_delta_s=max_eval_time_delta_s,
                robust_update=robust_update,
                enable_gating=enable_gating,
            )
            row = {
                "config_id": config.config_id,
                "flight": flight,
                **asdict(config),
                "command": " ".join(command),
            }
            if dry_run:
                rows.append({**row, "status": "dry-run"})
                continue
            env = {**os.environ, **config.environment()}
            completed = subprocess.run(command, env=env, check=False)
            if completed.returncode != 0:
                failed = {**row, "status": "failed", "returncode": completed.returncode}
                rows.append(failed)
                if not continue_on_error:
                    raise RuntimeError(f"sweep run failed: {failed}")
                continue
            metrics_path = run_dir / flight / "metrics.json"
            rows.append({**row, **flatten_metrics(metrics_path), "status": "ok", "returncode": 0})
    return rows


def baseline_command(
    *,
    dataset_root: Path,
    flight: str,
    output_dir: Path,
    radar_catprob_threshold: float,
    acceleration_std: float,
    smoother: str,
    smoother_lag_s: float,
    max_eval_time_delta_s: float,
    robust_update: str,
    enable_gating: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "raft_uav.cli",
        "run-baseline",
        str(dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        "--radar-association",
        "tracklet-viterbi",
        "--radar-catprob-threshold",
        str(float(radar_catprob_threshold)),
        "--acceleration-std",
        str(float(acceleration_std)),
        "--smoother",
        smoother,
        "--smoother-lag-s",
        str(float(smoother_lag_s)),
        "--max-eval-time-delta-s",
        str(float(max_eval_time_delta_s)),
        "--robust-update",
        robust_update,
    ]
    if enable_gating:
        command.append("--enable-gating")
    return command


def flatten_metrics(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    error_2d = metrics.get("position_error_2d", {})
    error_3d = metrics.get("position_error_3d", {})
    return {
        "metrics_json": str(metrics_path),
        "rmse_2d_m": error_2d.get("rmse_m"),
        "p95_2d_m": error_2d.get("p95_m"),
        "rmse_3d_m": error_3d.get("rmse_m"),
        "p95_3d_m": error_3d.get("p95_m"),
        "accepted_measurements": metrics.get("accepted_measurements"),
        "rejected_measurements": metrics.get("rejected_measurements"),
        "selected_radar_rows": metrics.get("selected_radar_rows"),
        "selected_radar_track_ids": ",".join(
            str(value) for value in metrics.get("selected_radar_track_ids", [])
        ),
    }


def write_sweep_outputs(
    rows: list[dict[str, Any]], output_dir: Path, *, opt1_name: str = "Opt1"
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame.from_records(rows)
    results.to_csv(output_dir / "sweep_results.csv", index=False)
    if results.empty or "rmse_3d_m" not in results:
        return
    ok = results.loc[results["status"] == "ok"].copy()
    if ok.empty:
        return
    summary = summarize_by_config(ok)
    summary.to_csv(output_dir / "sweep_summary.csv", index=False)
    write_best_json(summary, output_dir / "best_by_rmse.json", "mean_rmse_3d_m")
    write_best_json(summary, output_dir / "best_by_p95.json", "mean_p95_3d_m")
    if "worst_p95_3d_m" in summary:
        write_best_json(summary, output_dir / "best_by_worst_p95.json", "worst_p95_3d_m")
    opt1 = ok.loc[ok["flight"] == opt1_name]
    if not opt1.empty:
        write_best_json(opt1, output_dir / "best_by_opt1_p95.json", "p95_3d_m")


def summarize_by_config(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(
        [
            "config_id",
            "track_switch_cost",
            "anchor_nis_weight",
            "missed_detection_cost",
            "max_candidates",
        ],
        dropna=False,
    )
    summary = grouped.agg(
        flights=("flight", "nunique"),
        mean_rmse_3d_m=("rmse_3d_m", "mean"),
        mean_p95_3d_m=("p95_3d_m", "mean"),
        worst_p95_3d_m=("p95_3d_m", "max"),
        mean_selected_radar_rows=("selected_radar_rows", "mean"),
    )
    return summary.reset_index().sort_values(["mean_rmse_3d_m", "mean_p95_3d_m"])


def write_best_json(frame: pd.DataFrame, path: Path, column: str) -> None:
    if column not in frame or frame.empty:
        return
    best = frame.sort_values(column, kind="mergesort").iloc[0].to_dict()
    path.write_text(json.dumps(best, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
