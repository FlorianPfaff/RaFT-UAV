from pathlib import Path

from raft_uav.research.result_improvement_suite import (
    CommandSpec,
    ImprovementSuiteConfig,
    build_improvement_suite_plan,
)


def test_command_shell_line_includes_environment_prefix() -> None:
    spec = CommandSpec("demo", ("python", "script.py", "a b"), {"B": "two words", "A": "1"})

    line = spec.shell_line()

    assert line.startswith("A=1 B='two words'")
    assert "python script.py 'a b'" in line


def test_suite_plan_contains_all_major_workflow_steps() -> None:
    cfg = ImprovementSuiteConfig(
        dataset_root=Path("data"),
        flights=("Opt1",),
        methods=("imm_tracklet_viterbi_fixed_lag",),
    )

    names = [spec.name for spec in build_improvement_suite_plan(cfg)]

    assert "lofo_time_offset_calibration" in names
    assert "lofo_radar_covariance_tuning" in names
    assert "nested_lofo_tuning" in names
    assert "leave_flight_out_sota" in names
    assert "oracle_gap_imm_tracklet_viterbi_fixed_lag_Opt1" in names
    assert "constrained_leaderboard_ranking" in names


def test_suite_plan_can_disable_diagnostic_steps() -> None:
    cfg = ImprovementSuiteConfig(
        dataset_root=Path("data"),
        include_oracle_gap=False,
        include_constrained_ranking=False,
    )

    names = [spec.name for spec in build_improvement_suite_plan(cfg)]

    assert not any(name.startswith("oracle_gap_") for name in names)
    assert "constrained_leaderboard_ranking" not in names
