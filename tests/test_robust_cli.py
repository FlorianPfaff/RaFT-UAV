from raft_uav import cli as base_cli
from raft_uav import robust_cli, tracklet_viterbi_cli


def _baseline_args(mode: str) -> list[str]:
    return [
        "run-baseline",
        "/tmp/nonexistent-dataset",
        "--flight",
        "dummy-flight",
        "--robust-update",
        mode,
    ]


def test_legacy_cli_accepts_student_t_and_huber(monkeypatch):
    seen_modes: list[str] = []

    def fake_run_baseline(*args):
        seen_modes.extend(value for value in args if value in {"student-t", "huber"})
        return 0

    monkeypatch.setattr(base_cli, "_run_baseline", fake_run_baseline)

    assert robust_cli.main(_baseline_args("student-t")) == 0
    assert robust_cli.main(_baseline_args("huber")) == 0

    assert seen_modes == ["student-t", "huber"]


def test_tracklet_cli_accepts_student_t_and_huber(monkeypatch):
    seen_modes: list[str] = []

    def fake_run_baseline(*args):
        seen_modes.extend(value for value in args if value in {"student-t", "huber"})
        return 0

    monkeypatch.setattr(base_cli, "_run_baseline", fake_run_baseline)

    assert tracklet_viterbi_cli.main(_baseline_args("student-t")) == 0
    assert tracklet_viterbi_cli.main(_baseline_args("huber")) == 0

    assert seen_modes == ["student-t", "huber"]
