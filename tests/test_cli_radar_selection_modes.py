import inspect
from pathlib import Path

from raft_uav import cli


def test_run_baseline_cli_accepts_legacy_catprob_all_radar_selection(monkeypatch):
    captured = {}
    signature = inspect.signature(cli._run_baseline)

    def fake_run_baseline(*args):
        captured.update(signature.bind(*args).arguments)
        return 0

    monkeypatch.setattr(cli, "_run_baseline", fake_run_baseline)

    argv = [
        "run-baseline",
        "dataset-root",
        "--flight",
        "flight-1",
        "--radar-selection",
        "catprob-all",
    ]

    assert cli.main(argv) == 0
    assert captured["dataset_root"] == Path("dataset-root")
    assert captured["legacy_radar_selection"] == "catprob-all"
