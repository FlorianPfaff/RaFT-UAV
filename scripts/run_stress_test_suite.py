#!/usr/bin/env python3
"""Generate deterministic RF/radar stress-test artifacts from normalized CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.stress.perturbations import PerturbationConfig, perturb_radar, perturb_rf  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar-csv", type=Path, required=True)
    parser.add_argument("--rf-csv", type=Path, default=None)
    parser.add_argument("--configs-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stress_test_suite"))
    args = parser.parse_args(argv)

    radar = pd.read_csv(args.radar_csv)
    rf = pd.read_csv(args.rf_csv) if args.rf_csv is not None and args.rf_csv.exists() else pd.DataFrame()
    configs = _load_configs(args.configs_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for config in configs:
        run_dir = args.output_dir / config.name
        run_dir.mkdir(parents=True, exist_ok=True)
        radar_out = perturb_radar(radar, config)
        radar_path = run_dir / "radar_perturbed.csv"
        radar_out.to_csv(radar_path, index=False)
        rf_path = None
        if not rf.empty:
            rf_out = perturb_rf(rf, config)
            rf_path = run_dir / "rf_perturbed.csv"
            rf_out.to_csv(rf_path, index=False)
        manifest.append(
            {
                "name": config.name,
                "radar_csv": str(radar_path),
                "rf_csv": "" if rf_path is None else str(rf_path),
                **config.__dict__,
            }
        )
    (args.output_dir / "stress_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame.from_records(manifest).to_csv(args.output_dir / "stress_manifest.csv", index=False)
    print(f"manifest_json={args.output_dir / 'stress_manifest.json'}")
    return 0


def _load_configs(path: Path | None) -> list[PerturbationConfig]:
    if path is None:
        return [
            PerturbationConfig(name="drop_radar_20pct", radar_drop_rate=0.2, seed=101),
            PerturbationConfig(name="timestamp_jitter_500ms", timestamp_jitter_std_s=0.5, seed=102),
            PerturbationConfig(name="false_tracks_2", false_tracks_per_frame=2, seed=103),
            PerturbationConfig(name="velocity_noise_8mps", velocity_noise_std_mps=8.0, seed=104),
            PerturbationConfig(name="combined_hard", radar_drop_rate=0.2, timestamp_jitter_std_s=0.5, false_tracks_per_frame=2, velocity_noise_std_mps=8.0, seed=105),
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("stress configs JSON must be a list")
    return [PerturbationConfig.from_mapping(item) for item in payload]


if __name__ == "__main__":
    raise SystemExit(main())
