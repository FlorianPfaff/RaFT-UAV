from __future__ import annotations

import importlib.util
from pathlib import Path


def load_find_dataset_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "actions"
        / "ensure-aadm2025dryad-dataset"
        / "find_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("find_dataset", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_dataset_ignores_transient_cache_dirs(tmp_path: Path) -> None:
    module = load_find_dataset_module()
    transient_rf_root = (
        tmp_path
        / "AADM2025Dryad.tmp.123.1.Opt3"
        / "RF Sensor and Radar"
    )
    transient_rf_root.mkdir(parents=True)

    assert module.find_rf_root(tmp_path, max_depth=8) is None
    assert module.find_rf_root(transient_rf_root.parent, max_depth=8) is None
    assert (
        module.find_rf_root(
            transient_rf_root.parent,
            max_depth=8,
            allow_transient=True,
        )
        == transient_rf_root
    )


def test_find_dataset_accepts_persistent_cache_dir(tmp_path: Path) -> None:
    module = load_find_dataset_module()
    rf_root = tmp_path / "AADM2025Dryad" / "RF Sensor and Radar"
    rf_root.mkdir(parents=True)

    assert module.find_rf_root(tmp_path, max_depth=8) == rf_root
