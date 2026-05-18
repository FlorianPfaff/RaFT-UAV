"""Runtime wiring for optional learned RF/radar bias correction.

The hook is controlled by the ``RAFT_UAV_BIAS_MODEL`` environment variable. It
wraps the AERPAW normalization functions so every runner that uses the standard
loaders can apply the same learned bias model without duplicating calibration
logic in each experiment script.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.calibration.bias import BiasCorrectionBank

BIAS_MODEL_ENV = "RAFT_UAV_BIAS_MODEL"
BIAS_MODEL_PATH_ENV = "RAFT_UAV_BIAS_MODEL_PATH"

_INSTALLED = False
_ORIGINAL_NORMALIZE_RF: Any = None
_ORIGINAL_NORMALIZE_RADAR: Any = None
_CACHED_MODEL_PATH: Path | None = None
_CACHED_BANK: BiasCorrectionBank | None = None


def install() -> None:
    """Install RF/radar normalization wrappers once."""

    global _INSTALLED, _ORIGINAL_NORMALIZE_RF, _ORIGINAL_NORMALIZE_RADAR
    if _INSTALLED:
        return

    from raft_uav.io import aerpaw

    _ORIGINAL_NORMALIZE_RF = aerpaw.normalize_rf
    _ORIGINAL_NORMALIZE_RADAR = aerpaw.normalize_radar
    aerpaw.normalize_rf = _normalize_rf_with_bias
    aerpaw.normalize_radar = _normalize_radar_with_bias
    _INSTALLED = True


def configured_bias_model_path() -> Path | None:
    """Return the configured model path, if any."""

    raw = os.environ.get(BIAS_MODEL_ENV) or os.environ.get(BIAS_MODEL_PATH_ENV)
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


def bias_correction_enabled() -> bool:
    """Return whether a bias model path is configured."""

    return configured_bias_model_path() is not None


def bias_correction_summary() -> dict[str, object]:
    """Return metadata for the active runtime bias correction."""

    path = configured_bias_model_path()
    if path is None:
        return {"enabled": False}
    bank = _load_bank(path)
    return bank.summary(path)


def _normalize_rf_with_bias(*args: Any, **kwargs: Any) -> pd.DataFrame:
    assert _ORIGINAL_NORMALIZE_RF is not None
    frame = _ORIGINAL_NORMALIZE_RF(*args, **kwargs)
    return _apply_runtime_bias(frame, "rf")


def _normalize_radar_with_bias(*args: Any, **kwargs: Any) -> pd.DataFrame:
    assert _ORIGINAL_NORMALIZE_RADAR is not None
    frame = _ORIGINAL_NORMALIZE_RADAR(*args, **kwargs)
    return _apply_runtime_bias(frame, "radar")


def _apply_runtime_bias(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    path = configured_bias_model_path()
    if path is None:
        return frame
    bank = _load_bank(path)
    corrected = bank.correct_frame(frame, source)
    if corrected.empty:
        return corrected
    if "bias_correction_source" in corrected.columns:
        corrected = corrected.copy()
        corrected["bias_model_path"] = str(path)
    return corrected


def _load_bank(path: Path) -> BiasCorrectionBank:
    global _CACHED_MODEL_PATH, _CACHED_BANK
    resolved = path.resolve()
    if _CACHED_BANK is None or _CACHED_MODEL_PATH != resolved:
        _CACHED_BANK = BiasCorrectionBank.load(resolved)
        _CACHED_MODEL_PATH = resolved
    return _CACHED_BANK
