from __future__ import annotations

from raft_uav import cli
from raft_uav.io.aerpaw import DEFAULT_RADAR_CLOCK_OFFSET_S, DEFAULT_RF_CLOCK_OFFSET_S


def test_run_baseline_forwards_default_clock_offsets(monkeypatch, tmp_path):
    captured: dict[str, tuple[object, ...]] = {}

    def fake_run_baseline(*args: object) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_baseline", fake_run_baseline)

    assert cli.main(["run-baseline", str(tmp_path), "--flight", "Opt1"]) == 0

    assert captured["args"][-2:] == (
        DEFAULT_RF_CLOCK_OFFSET_S,
        DEFAULT_RADAR_CLOCK_OFFSET_S,
    )


def test_run_baseline_forwards_independent_clock_offsets(monkeypatch, tmp_path):
    captured: dict[str, tuple[object, ...]] = {}

    def fake_run_baseline(*args: object) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_baseline", fake_run_baseline)

    assert (
        cli.main(
            [
                "run-baseline",
                str(tmp_path),
                "--flight",
                "Opt1",
                "--rf-clock-offset-s",
                "0",
                "--radar-clock-offset-s",
                "-14400",
            ]
        )
        == 0
    )

    assert captured["args"][-2:] == (0.0, -14400.0)
