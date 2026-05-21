"""One-command, leakage-safe result-improvement workflow for RaFT-UAV.

The suite is intentionally an orchestration layer.  It wires together the
existing diagnostics, calibration, tuning, SOTA evaluation, oracle-gap reports,
and constrained ranking so result improvements are evaluated in a repeatable
leave-one-flight-out workflow rather than by hand-tuning one flight.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_METHODS: tuple[str, ...] = (
    "cv_catprob",
    "imm_catprob",
    "imm_catprob_fixed_lag",
    "imm_tracklet_viterbi_fixed_lag",
    "imm_learned_tracklet_viterbi_fixed_lag",
    "hetero_imm_tracklet_viterbi_fixed_lag",
    "hetero_imm_learned_tracklet_viterbi_fixed_lag",
    "hetero_cv_lofo_nis_fixed_lag",
)
DEFAULT_FLIGHTS: tuple[str, ...] = ("Opt1", "Opt2", "Opt3")
DEFAULT_RUNTIME_ENV: dict[str, str] = {
    # Full candidate-retention path: keep ambiguous low-catProb candidates, but
    # penalize them softly instead of pruning them before sequence decoding.
    "RAFT_UAV_TRACKLET_CATPROB_RETENTION_MODE": "soft",
    "RAFT_UAV_TRACKLET_MAX_CANDIDATES": "12",
    "RAFT_UAV_TRACKLET_MAX_CANDIDATE_POOL_PER_FRAME": "36",
    "RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_TRACK_ID": "2",
    "RAFT_UAV_TRACKLET_SUPPORT_WEIGHT": "0.45",
    "RAFT_UAV_TRACKLET_MAX_SUPPORT_REWARD": "4.0",
    # Prefer the physically motivated range/angle covariance; LOFO covariance
    # tuning can override these values in follow-up runs.
    "RAFT_UAV_RADAR_COVARIANCE_MODE": "range-angle",
    "RAFT_UAV_RADAR_RANGE_STD_M": "5.0",
    "RAFT_UAV_RADAR_AZIMUTH_STD_DEG": "2.0",
    "RAFT_UAV_RADAR_ELEVATION_STD_DEG": "2.0",
    "RAFT_UAV_RADAR_COVARIANCE_MIN_STD_M": "3.0",
    "RAFT_UAV_RADAR_COVARIANCE_MAX_STD_M": "250.0",
    # Enable runtime hooks used by learned/heteroscedastic tracklet variants.
    "RAFT_UAV_SOFT_CATPROB_RETENTION": "1",
    "RAFT_UAV_RADAR_UPDATE_USES_VELOCITY": "1",
}


@dataclass(frozen=True)
class CommandSpec:
    """A subprocess command plus environment overrides."""

    name: str
    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: Path | None = None

    def shell_line(self) -> str:
        """Return a copy-pasteable shell command."""

        prefix = " ".join(
            f"{shlex.quote(str(key))}={shlex.quote(str(value))}"
            for key, value in sorted(self.env.items())
        )
        command = shlex.join(tuple(str(part) for part in self.argv))
        return f"{prefix} {command}".strip()


@dataclass(frozen=True)
class ImprovementSuiteConfig:
    """Configuration for the integrated result-improvement suite."""

    dataset_root: Path
    output_dir: Path = Path("outputs/result_improvement_suite")
    flights: tuple[str, ...] = DEFAULT_FLIGHTS
    methods: tuple[str, ...] = DEFAULT_METHODS
    candidate_threshold: float = 0.4
    fixed_lag_s: float = 20.0
    skip_existing: bool = False
    include_sota: bool = True
    include_nested_tuning: bool = True
    include_covariance_tuning: bool = True
    include_time_offset_calibration: bool = True
    include_oracle_gap: bool = True
    include_constrained_ranking: bool = True
    runtime_env: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_RUNTIME_ENV))

    @property
    def sota_dir(self) -> Path:
        return self.output_dir / "leave_flight_out_sota"

    @property
    def diagnostics_dir(self) -> Path:
        return self.output_dir / "diagnostics"


def build_improvement_suite_plan(config: ImprovementSuiteConfig) -> list[CommandSpec]:
    """Return the ordered command plan for all recommended improvements."""

    commands: list[CommandSpec] = []
    if config.include_time_offset_calibration:
        commands.append(_time_offset_command(config))
    if config.include_covariance_tuning:
        commands.append(_radar_covariance_tuning_command(config))
    if config.include_nested_tuning:
        commands.append(_nested_lofo_tuning_command(config))
    if config.include_sota:
        commands.append(_sota_command(config))
    if config.include_oracle_gap:
        commands.extend(_oracle_gap_commands(config))
    if config.include_constrained_ranking:
        commands.append(_constrained_ranking_command(config))
    return commands


def execute_command_plan(commands: Sequence[CommandSpec], *, dry_run: bool = False) -> None:
    """Execute a command plan in order."""

    for spec in commands:
        print(f"[{spec.name}] {spec.shell_line()}", flush=True)
        if dry_run:
            continue
        env = None
        if spec.env:
            import os

            env = os.environ.copy()
            env.update({str(key): str(value) for key, value in spec.env.items()})
        subprocess.run(list(spec.argv), check=True, cwd=spec.cwd, env=env)


def write_improvement_suite_manifest(
    path: Path,
    *,
    config: ImprovementSuiteConfig,
    commands: Sequence[CommandSpec],
) -> None:
    """Write a JSON manifest with resolved commands and runtime settings."""

    payload = {
        "dataset_root": str(config.dataset_root),
        "output_dir": str(config.output_dir),
        "flights": list(config.flights),
        "methods": list(config.methods),
        "candidate_threshold": float(config.candidate_threshold),
        "fixed_lag_s": float(config.fixed_lag_s),
        "skip_existing": bool(config.skip_existing),
        "runtime_env": dict(config.runtime_env),
        "commands": [
            {
                "name": spec.name,
                "argv": list(spec.argv),
                "env": dict(spec.env),
                "cwd": None if spec.cwd is None else str(spec.cwd),
                "shell": spec.shell_line(),
            }
            for spec in commands
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _time_offset_command(config: ImprovementSuiteConfig) -> CommandSpec:
    argv = [
        sys.executable,
        "scripts/run_lofo_time_offset_calibration.py",
        str(config.dataset_root),
        "--output-dir",
        str(config.output_dir / "lofo_time_offset"),
        "--offset-min-s",
        "-10",
        "--offset-max-s",
        "10",
        "--offset-step-s",
        "0.25",
    ]
    _append_flights(argv, config.flights, flag="--flight")
    # The time-offset CLI currently has no --skip-existing switch; the suite
    # still records skip_existing in the manifest for the other steps.
    return CommandSpec("lofo_time_offset_calibration", tuple(argv), dict(config.runtime_env))


def _radar_covariance_tuning_command(config: ImprovementSuiteConfig) -> CommandSpec:
    argv = [
        sys.executable,
        "scripts/run_lofo_radar_covariance_tuning.py",
        str(config.dataset_root),
        "--output-dir",
        str(config.output_dir / "lofo_radar_covariance"),
    ]
    _append_flights(argv, config.flights, flag="--flight")
    if config.skip_existing:
        argv.append("--skip-existing")
    return CommandSpec("lofo_radar_covariance_tuning", tuple(argv), dict(config.runtime_env))


def _nested_lofo_tuning_command(config: ImprovementSuiteConfig) -> CommandSpec:
    argv = [
        sys.executable,
        "scripts/run_nested_lofo_tuning.py",
        str(config.dataset_root),
        "--output-dir",
        str(config.output_dir / "nested_lofo_tuning"),
    ]
    if config.flights:
        argv.append("--flights")
        argv.extend(config.flights)
    if config.skip_existing:
        argv.append("--skip-existing")
    return CommandSpec("nested_lofo_tuning", tuple(argv), dict(config.runtime_env))


def _sota_command(config: ImprovementSuiteConfig) -> CommandSpec:
    argv = [
        sys.executable,
        "scripts/run_leave_flight_out_sota.py",
        str(config.dataset_root),
        "--output-dir",
        str(config.sota_dir),
        "--candidate-threshold",
        f"{config.candidate_threshold:g}",
        "--fixed-lag-s",
        f"{config.fixed_lag_s:g}",
        "--methods",
        *config.methods,
        "--enable-soft-catprob-retention",
        "--enable-radar-velocity-update",
    ]
    if config.flights:
        argv.append("--flights")
        argv.extend(config.flights)
    if config.skip_existing:
        argv.append("--skip-existing")
    return CommandSpec("leave_flight_out_sota", tuple(argv), dict(config.runtime_env))


def _oracle_gap_commands(config: ImprovementSuiteConfig) -> list[CommandSpec]:
    if not config.flights or not config.methods:
        return []
    commands: list[CommandSpec] = []
    for method in config.methods:
        for flight in config.flights:
            run_dir = config.sota_dir / f"heldout_{_slug(flight)}" / method
            output_dir = config.diagnostics_dir / "oracle_gap" / method / flight
            commands.append(
                CommandSpec(
                    f"oracle_gap_{method}_{flight}",
                    (
                        sys.executable,
                        "scripts/run_oracle_gap_decomposition.py",
                        str(config.dataset_root),
                        "--run-dir",
                        str(run_dir),
                        "--output-dir",
                        str(output_dir),
                        "--flights",
                        flight,
                    ),
                    dict(config.runtime_env),
                )
            )
    return commands


def _constrained_ranking_command(config: ImprovementSuiteConfig) -> CommandSpec:
    return CommandSpec(
        "constrained_leaderboard_ranking",
        (
            sys.executable,
            "scripts/run_constrained_ablation_optimizer.py",
            str(config.sota_dir / "fold_summary.csv"),
            "--output-csv",
            str(config.sota_dir / "constrained_rank.csv"),
            "--objective",
            "error_3d_rmse_m",
            "--constraint",
            "truth_coverage_rate:>=:0.95",
            "--constraint",
            "track_switch_count:<=:10",
            "--pareto-minimize",
            "error_3d_rmse_m",
            "error_3d_p95_m",
            "--pareto-maximize",
            "truth_coverage_rate",
        ),
        dict(config.runtime_env),
    )


def _append_flights(argv: list[str], flights: Sequence[str], *, flag: str) -> None:
    for flight in flights:
        argv.extend([flag, str(flight)])


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/result_improvement_suite"))
    parser.add_argument("--flights", nargs="*", default=list(DEFAULT_FLIGHTS))
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--candidate-threshold", type=float, default=0.4)
    parser.add_argument("--fixed-lag-s", type=float, default=20.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sota", action="store_true")
    parser.add_argument("--no-nested-tuning", action="store_true")
    parser.add_argument("--no-covariance-tuning", action="store_true")
    parser.add_argument("--no-time-offset-calibration", action="store_true")
    parser.add_argument("--no-oracle-gap", action="store_true")
    parser.add_argument("--no-constrained-ranking", action="store_true")
    parser.add_argument(
        "--runtime-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="additional RAFT_UAV_* runtime override; can be repeated",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    runtime_env = dict(DEFAULT_RUNTIME_ENV)
    for assignment in args.runtime_env:
        if "=" not in assignment:
            raise ValueError(f"runtime env override must be NAME=VALUE, got {assignment!r}")
        name, value = assignment.split("=", 1)
        runtime_env[name] = value

    config = ImprovementSuiteConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        flights=tuple(args.flights or ()),
        methods=tuple(args.methods or ()),
        candidate_threshold=args.candidate_threshold,
        fixed_lag_s=args.fixed_lag_s,
        skip_existing=args.skip_existing,
        include_sota=not args.no_sota,
        include_nested_tuning=not args.no_nested_tuning,
        include_covariance_tuning=not args.no_covariance_tuning,
        include_time_offset_calibration=not args.no_time_offset_calibration,
        include_oracle_gap=not args.no_oracle_gap,
        include_constrained_ranking=not args.no_constrained_ranking,
        runtime_env=runtime_env,
    )
    commands = build_improvement_suite_plan(config)
    write_improvement_suite_manifest(
        config.output_dir / "result_improvement_suite_manifest.json",
        config=config,
        commands=commands,
    )
    execute_command_plan(commands, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
