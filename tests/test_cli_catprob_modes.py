from pathlib import Path

from raft_uav import cli


def test_run_baseline_accepts_catprob_all_association(monkeypatch):
    captured = {}

    def fake_run_baseline(
        dataset_root,
        flight_name,
        output_dir,
        acceleration_std,
        uncertainty_model_path,
        radar_association,
        legacy_radar_selection,
        *args,
    ):
        del output_dir, acceleration_std, uncertainty_model_path, args
        captured["dataset_root"] = dataset_root
        captured["flight_name"] = flight_name
        captured["radar_association"] = radar_association
        captured["legacy_radar_selection"] = legacy_radar_selection
        return 0

    monkeypatch.setattr(cli, "_run_baseline", fake_run_baseline)

    exit_code = cli.main(
        [
            "run-baseline",
            str(Path("dataset")),
            "--flight",
            "flight-a",
            "--radar-association",
            "catprob-all",
        ]
    )

    assert exit_code == 0
    assert captured["radar_association"] == "catprob-all"
    assert captured["legacy_radar_selection"] is None


def test_run_baseline_accepts_catprob_all_legacy_selection(monkeypatch):
    captured = {}

    def fake_run_baseline(
        dataset_root,
        flight_name,
        output_dir,
        acceleration_std,
        uncertainty_model_path,
        radar_association,
        legacy_radar_selection,
        *args,
    ):
        del dataset_root, flight_name, output_dir, acceleration_std, uncertainty_model_path, args
        captured["radar_association"] = radar_association
        captured["legacy_radar_selection"] = legacy_radar_selection
        return 0

    monkeypatch.setattr(cli, "_run_baseline", fake_run_baseline)

    exit_code = cli.main(
        [
            "run-baseline",
            str(Path("dataset")),
            "--flight",
            "flight-a",
            "--radar-selection",
            "catprob-all",
        ]
    )

    assert exit_code == 0
    assert captured["radar_association"] == "catprob"
    assert captured["legacy_radar_selection"] == "catprob-all"
