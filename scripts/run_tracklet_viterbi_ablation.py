"""Run sequence-level tracklet-Viterbi radar association ablations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import ablation_common as common
from raft_uav.tracklet_viterbi_cli import main as tracklet_viterbi_main


@dataclass(frozen=True)
class _Config:
    name: str
    threshold: float


def main() -> int:
    parser = argparse.ArgumentParser()
    common.add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/tracklet_viterbi_ablation"),
        default_summary_output=Path("outputs/tracklet_viterbi_ablation.csv"),
    )
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.3, 0.4, 0.5])
    common.add_fixed_lag_argument(parser)
    common.add_soft_update_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    configs = [_Config(_threshold_name(threshold), threshold) for threshold in args.thresholds]
    rows = common.run_named_config_experiments(args, configs, _run_one, _candidate_row)
    common.write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _candidate_row(
    config: _Config,
    metrics_path: Path,
    metrics: dict[str, object],
) -> dict[str, object]:
    return common.tracking_summary_row(
        config.name,
        metrics_path,
        metrics,
        extra_fields={"radar_catprob_threshold": metrics.get("radar_catprob_threshold", "")},
        include_selected_track_ids=True,
        include_reweighted=True,
        include_inflation=True,
    )


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    config: _Config,
) -> None:
    cli_args = [
        "run-baseline",
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        "--radar-association",
        "tracklet-viterbi",
        "--radar-catprob-threshold",
        str(config.threshold),
        *[str(option) for option in common.robust_update_options(args)],
        *[str(option) for option in common.smoother_options("fixed-lag", args.fixed_lag_s)],
    ]
    print("raft-uav-tracklet-viterbi " + " ".join(cli_args), flush=True)
    status = tracklet_viterbi_main(cli_args)
    if status != 0:
        raise RuntimeError(f"tracklet-viterbi run failed with status {status}")


def _threshold_name(threshold: float) -> str:
    return f"tracklet_viterbi_t{common.slug(threshold, precision=2)}"


if __name__ == "__main__":
    raise SystemExit(main())
