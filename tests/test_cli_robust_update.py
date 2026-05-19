"""CLI regression tests for robust Kalman update modes."""

from __future__ import annotations

import inspect

from raft_uav import cli


def test_run_baseline_cli_exposes_all_robust_update_modes(monkeypatch, tmp_path) -> None:
    """The command line should expose every robust mode implemented downstream."""

    parameter_names = list(inspect.signature(cli._run_baseline).parameters)
    captured: list[dict[str, object]] = []

    def fake_run_baseline(*args: object) -> int:
        captured.append(dict(zip(parameter_names, args)))
        return 0

    monkeypatch.setattr(cli, "_run_baseline", fake_run_baseline)

    for mode in ("nis-inflate", "student-t", "huber"):
        assert (
            cli.main(
                [
                    "run-baseline",
                    str(tmp_path),
                    "--flight",
                    "synthetic-flight",
                    "--robust-update",
                    mode,
                ]
            )
            == 0
        )
        assert captured[-1]["robust_update"] == mode
