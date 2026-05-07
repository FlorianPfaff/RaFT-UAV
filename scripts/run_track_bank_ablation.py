"""Run PyRecEst MHT track-bank association ablations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import ablation_common as common


@dataclass(frozen=True)
class _Config:
    name: str
    association: str


def main() -> int:
    parser = argparse.ArgumentParser()
    common.add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/track_bank_ablation"),
        default_summary_output=Path("outputs/track_bank_ablation_opt1_opt3.csv"),
    )
    parser.add_argument("--candidate-threshold", type=float, default=0.4)
    common.add_fixed_lag_argument(parser)
    common.add_soft_update_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    configs = [
        _Config("prediction_nis_t0p40", "prediction-nis"),
        _Config("track_bank_t0p40", "track-bank"),
    ]
    rows = common.run_named_config_experiments(args, configs, _run_one, _row)
    common.write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _row(config: _Config, metrics_path: Path, metrics: dict[str, object]) -> dict[str, object]:
    track_bank = metrics.get("track_bank_association") or {}
    return common.tracking_summary_row(
        config.name,
        metrics_path,
        metrics,
        extra_fields={
            "radar_catprob_threshold": metrics.get("radar_catprob_threshold", ""),
            "track_bank_max_hypotheses": _metric_value(track_bank, "max_hypotheses"),
            "track_bank_gate_probability": _metric_value(track_bank, "gate_probability"),
            "track_bank_detection_probability": _metric_value(
                track_bank, "detection_probability"
            ),
            "track_bank_clutter_intensity": _metric_value(track_bank, "clutter_intensity"),
        },
        include_selected_track_ids=True,
    )


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    config: _Config,
) -> None:
    options: list[object] = ["--radar-catprob-threshold", args.candidate_threshold]
    options.extend(common.robust_update_options(args))
    options.extend(common.smoother_options("fixed-lag", args.fixed_lag_s))
    common.run_baseline(
        dataset_root=args.dataset_root,
        flight=flight,
        output_dir=output_dir,
        association=config.association,
        extra_options=options,
    )


def _metric_value(mapping: object, key: str) -> object:
    return common.empty_if_none(mapping.get(key)) if isinstance(mapping, dict) else ""


if __name__ == "__main__":
    raise SystemExit(main())
