from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.bias_model_cli import _extract_bias_model
from raft_uav.calibration.bias import BiasCorrectionBank, SensorBiasCorrectionModel
from raft_uav.calibration import bias_runtime
from raft_uav.calibration.bias_runtime import BIAS_MODEL_ENV


def _constant_rf_model() -> SensorBiasCorrectionModel:
    return SensorBiasCorrectionModel(
        source="rf",
        target_columns=("east_m", "north_m"),
        feature_columns=(),
        intercept=np.array([10.0, -5.0]),
        coefficients=np.empty((0, 2), dtype=float),
        feature_mean=np.empty(0, dtype=float),
        feature_scale=np.empty(0, dtype=float),
        residual_std=np.zeros(2, dtype=float),
        training_rows=12,
        ridge_alpha=0.0,
        time_gate_s=2.0,
    )


def test_runtime_bias_correction_applies_configured_model(tmp_path, monkeypatch):
    path = tmp_path / "bias_model.json"
    BiasCorrectionBank({"rf": _constant_rf_model()}).save(path)
    monkeypatch.setenv(BIAS_MODEL_ENV, str(path))
    monkeypatch.setattr(bias_runtime, "_CACHED_MODEL_PATH", None)
    monkeypatch.setattr(bias_runtime, "_CACHED_BANK", None)
    frame = pd.DataFrame({"east_m": [12.0], "north_m": [3.0], "std_m": [75.0]})

    corrected = bias_runtime._apply_runtime_bias(frame, "rf")

    assert corrected["east_m"].tolist() == [2.0]
    assert corrected["north_m"].tolist() == [8.0]
    assert corrected["raw_east_m"].tolist() == [12.0]
    assert corrected["raw_north_m"].tolist() == [3.0]
    assert corrected["bias_model_path"].tolist() == [str(path)]


def test_runtime_bias_correction_is_noop_without_model(monkeypatch):
    monkeypatch.delenv(BIAS_MODEL_ENV, raising=False)
    frame = pd.DataFrame({"east_m": [1.0], "north_m": [2.0]})

    corrected = bias_runtime._apply_runtime_bias(frame, "rf")

    assert corrected is frame


def test_bias_wrapper_extracts_model_path_and_preserves_remaining_args():
    path, remaining = _extract_bias_model(
        ["--bias-model", "model.json", "run-baseline", "data", "--flight", "Opt1"]
    )

    assert path == Path("model.json")
    assert remaining == ["run-baseline", "data", "--flight", "Opt1"]


def test_bias_wrapper_supports_equals_form():
    path, remaining = _extract_bias_model(
        ["--bias-model=model.json", "run-baseline", "data", "--flight", "Opt1"]
    )

    assert path == Path("model.json")
    assert remaining == ["run-baseline", "data", "--flight", "Opt1"]
