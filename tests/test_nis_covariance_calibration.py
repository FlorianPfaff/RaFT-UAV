import json

import numpy as np
import pandas as pd

from raft_uav.calibration.nis_covariance import (
    ENV_NIS_COVARIANCE_CALIBRATION_JSON,
    covariance_scale_for_source_dim,
    fit_nis_covariance_calibration_from_frame,
    scale_covariance_for_calibrated_source,
    write_nis_covariance_calibration,
)


def test_mean_nis_calibration_fits_source_dimension_groups():
    diagnostics = pd.DataFrame(
        {
            "source": ["rf"] * 4 + ["radar"] * 4,
            "measurement_dim": [2] * 4 + [3] * 4,
            "accepted": [True] * 8,
            "nis": [4.0, 4.0, 4.0, 4.0, 9.0, 9.0, 9.0, 9.0],
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        method="mean",
        min_samples=1,
        min_scale=0.1,
        max_scale=10.0,
    )

    assert payload["groups"]["rf:2"]["applied_scale"] == 2.0
    assert payload["groups"]["radar:3"]["applied_scale"] == 3.0
    assert covariance_scale_for_source_dim(payload, "rf", 2) == 2.0
    assert covariance_scale_for_source_dim(payload, "radar", 3) == 3.0


def test_calibration_ignores_rejected_updates_by_default():
    diagnostics = pd.DataFrame(
        {
            "source": ["radar", "radar", "radar"],
            "measurement_dim": [3, 3, 3],
            "accepted": [True, True, False],
            "nis": [3.0, 3.0, 300.0],
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        method="mean",
        min_samples=1,
    )

    assert payload["groups"]["radar:3"]["count"] == 2
    assert payload["groups"]["radar:3"]["applied_scale"] == 1.0


def test_too_few_samples_disable_group_without_changing_runtime_scale():
    diagnostics = pd.DataFrame(
        {
            "source": ["rf"],
            "measurement_dim": [2],
            "accepted": [True],
            "nis": [20.0],
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        method="mean",
        min_samples=2,
    )

    group = payload["groups"]["rf:2"]
    assert group["enabled"] is False
    assert group["applied_scale"] == 1.0
    assert covariance_scale_for_source_dim(payload, "rf", 2) == 1.0


def test_runtime_calibration_scales_covariance_from_environment(tmp_path, monkeypatch):
    payload = {
        "schema": "raft-uav-nis-covariance-calibration-v1",
        "groups": {
            "radar:3": {
                "source": "radar",
                "measurement_dim": 3,
                "count": 10,
                "method": "mean",
                "statistic": 6.0,
                "target": 3.0,
                "raw_scale": 2.0,
                "applied_scale": 2.0,
                "enabled": True,
                "accepted_only": True,
                "quantile": None,
            }
        },
    }
    path = write_nis_covariance_calibration(payload, tmp_path / "calibration.json")
    monkeypatch.setenv(ENV_NIS_COVARIANCE_CALIBRATION_JSON, str(path))

    covariance = np.eye(3)
    scaled = scale_covariance_for_calibrated_source("radar", 3, covariance)

    np.testing.assert_allclose(scaled, 2.0 * covariance)


def test_written_payload_is_json_and_round_trippable(tmp_path):
    diagnostics = pd.DataFrame(
        {
            "source": ["rf", "rf"],
            "measurement_dim": [2, 2],
            "accepted": [True, True],
            "nis": [2.0, 2.0],
        }
    )
    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        min_samples=1,
    )

    path = write_nis_covariance_calibration(payload, tmp_path / "calibration.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["schema"] == "raft-uav-nis-covariance-calibration-v1"
    assert loaded["groups"]["rf:2"]["applied_scale"] == 1.0
