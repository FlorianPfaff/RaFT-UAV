import inspect

import pytest

from raft_uav import cli


@pytest.mark.parametrize("mode", ["student-t", "huber"])
def test_run_baseline_cli_accepts_pyrecest_robust_update_modes(monkeypatch, mode):
    captured = {}
    parameter_names = list(inspect.signature(cli._run_baseline).parameters)

    def fake_run_baseline(*args):
        captured.update(dict(zip(parameter_names, args, strict=True)))
        return 0

    monkeypatch.setattr(cli, "_run_baseline", fake_run_baseline)

    assert (
        cli.main(
            [
                "run-baseline",
                "/tmp/aerpaw",
                "--flight",
                "Opt1",
                "--robust-update",
                mode,
            ]
        )
        == 0
    )
    assert captured["robust_update"] == mode
