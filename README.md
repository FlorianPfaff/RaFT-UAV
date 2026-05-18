# RaFT-UAV

Radar-RF Fusion Tracking for UAVs.

This repository contains implementation and evaluation code for tracking UAVs
with AERPAW Dataset-28 / Dryad DOI `10.5061/dryad.7d7wm3898`. The initial
baseline is an asynchronous constant-velocity Kalman fusion tracker built on
PyRecEst.

Large datasets and generated bulk artifacts are intentionally not stored in
this repository.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Data Layout

Download and extract the Dryad archive outside git, for example:

```text
data/raw/AADM2025Dryad/
  RF Sensor and Radar/
    <flight>/
      AADM*.csv
      radar_data_*.json
      date_time_vehicleOut.txt
```

The loader starts with the RF Sensor and Radar folder because it contains the
modalities needed for the tracking work.

## First Commands

Inspect available flights:

```bash
python -m raft_uav.cli inspect data/raw/AADM2025Dryad
```

Run the initial fusion baseline on one flight:

```bash
python -m raft_uav.cli run-baseline data/raw/AADM2025Dryad --flight Opt2
```

Run the normalized-innovation-squared gated baseline on one flight:

```bash
python -m raft_uav.cli run-baseline data/raw/AADM2025Dryad --flight Opt2 --enable-gating --rf-gate-prob 0.99 --radar-gate-prob 0.99
```

Run the soft NIS covariance-inflation baseline on one flight:

```bash
python -m raft_uav.cli run-baseline data/raw/AADM2025Dryad --flight Opt2 --robust-update nis-inflate --rf-gate-prob 0.99 --radar-gate-prob 0.99
```

Tune source-specific inflation strength:

```bash
python -m raft_uav.cli run-baseline data/raw/AADM2025Dryad --flight Opt2 --robust-update nis-inflate --rf-inflation-alpha 1.5 --radar-inflation-alpha 0.5
```

Run the Opt1-Opt3 source-specific inflation grid:

```bash
python scripts/run_source_specific_inflation_grid.py data/raw/AADM2025Dryad
```

Run the Opt1-Opt3 radar association ablation:

```bash
python scripts/run_radar_association_ablation.py data/raw/AADM2025Dryad
```

Run the Opt1-Opt3 smoothing ablation:

```bash
python scripts/run_smoothing_ablation.py data/raw/AADM2025Dryad
```

Analyze online radar association against the diagnostic oracle on Opt1:

```bash
python scripts/analyze_association_failures.py data/raw/AADM2025Dryad --flight Opt1
```

Run the Opt1-Opt3 radar candidate class-probability threshold ablation:

```bash
python scripts/run_candidate_threshold_ablation.py data/raw/AADM2025Dryad --thresholds 0.4 0.5
```

Run the Opt1-Opt3 PDA-mixture association ablation:

```bash
python scripts/run_pda_association_ablation.py data/raw/AADM2025Dryad
```

Run the Opt1-Opt3 PyRecEst MHT track-bank ablation:

```bash
python scripts/run_track_bank_ablation.py data/raw/AADM2025Dryad
```

The first baseline is deliberately conservative. It is meant to reproduce the
published constant-velocity Kalman fusion setup before adding robust gating,
learned sensor uncertainties, maneuvering models, and smoothing.

Baseline runs write gitignored per-flight artifacts under `outputs/baseline/`:

- `estimates.csv`
- `diagnostics.csv`
- `selected_radar.csv`
- `metrics.json`
- `trajectory.png`

Radar JSON frames contain many Fortem `trackData` entries. The default
reproducibility baseline uses `--radar-association catprob`, which selects the
highest-UAV-probability radar row per JSON frame among rows whose UAV class
probability `catProb[0]` is at least `0.5`; this gives the single-UAV filter at
most one radar update per frame. Use `--radar-selection all` only for deliberate
multi-row radar stress tests. Use `--radar-association oracle-nearest-truth`
only as a diagnostic upper bound because it uses ground truth. The online
alternatives are `--radar-association prediction-nis`,
`--radar-association track-continuity`, and the experimental
`--radar-association geometry-score` mode, which adds velocity consistency,
track-switch, and UAV class-probability terms to the NIS score. The
experimental `--radar-association pda-mixture` mode keeps multiple candidates
inside one radar update by using NIS/class-probability weights and adding the
candidate spread to the measurement covariance. The experimental
`--radar-association track-bank` mode uses PyRecEst's
`MultiHypothesisTracker` to keep multiple single-UAV association hypotheses
alive across radar frames. Baseline runs also write `hypotheses.csv` for modes
that expose per-hypothesis diagnostics.
The legacy `--radar-selection truth-gated` mode is retained for schema
debugging.

Use `--smoother fixed-lag --smoother-lag-s 20` to apply a 20-second RTS
fixed-lag pass before metrics and plots are written. `--smoother rts` runs the
full offline RTS smoother and is mainly useful as an upper-bound diagnostic.
