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

After installation, prefer the `raft-uav` console script for experiments. It
routes through the canonical tracklet-Viterbi wrapper, registers the
`tracklet-viterbi` radar-association mode, and exposes the wrapper-only
`--tracklet-*` options. Use `python -m raft_uav.cli` only for legacy base-CLI
debugging.

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
raft-uav inspect data/raw/AADM2025Dryad
```

Run the canonical range-covariance tracklet-Viterbi fusion baseline on one
flight:

```bash
raft-uav run-baseline data/raw/AADM2025Dryad \
  --flight Opt2 \
  --radar-association tracklet-viterbi \
  --tracklet-variant range-covariance
```

For a maneuver-aware replay of the same association path, add
`--tracklet-replay-tracker imm`. For a strictly legacy CV/catProb baseline,
spell it out explicitly:

```bash
raft-uav run-baseline data/raw/AADM2025Dryad --flight Opt2 --radar-association catprob
```

Run the normalized-innovation-squared gated baseline on one flight:

```bash
raft-uav run-baseline data/raw/AADM2025Dryad --flight Opt2 --enable-gating --rf-gate-prob 0.99 --radar-gate-prob 0.99
```

Run the soft NIS covariance-inflation baseline on one flight:

```bash
raft-uav run-baseline data/raw/AADM2025Dryad --flight Opt2 --robust-update nis-inflate --rf-gate-prob 0.99 --radar-gate-prob 0.99
```

Tune source-specific inflation strength:

```bash
raft-uav run-baseline data/raw/AADM2025Dryad --flight Opt2 --robust-update nis-inflate --rf-inflation-alpha 1.5 --radar-inflation-alpha 0.5
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

Calibrate residual RF/radar time offsets on training flights, then apply the
held-out offsets explicitly in the baseline run:

```bash
raft-uav-lofo-time-offset data/raw/AADM2025Dryad \
  --flight Opt1 --flight Opt2 --flight Opt3 \
  --output-dir outputs/lofo_time_offset

python -m raft_uav.cli run-baseline data/raw/AADM2025Dryad \
  --flight Opt2 \
  --rf-time-offset-correction-s <rf_offset_s_from_train_folds> \
  --radar-time-offset-correction-s <radar_offset_s_from_train_folds>
```

The loader-level `--rf-clock-offset-s` and `--radar-clock-offset-s` arguments
control conversion from raw sensor timestamps to the truth timeline. The
`--*-time-offset-correction-s` arguments are residual calibrated corrections
applied after normalization, which matches the output convention of
`raft-uav-lofo-time-offset`.

Build a paper-style comparison table with the paper-compatible hard-gated
fusion row:

```bash
raft-uav-diagnose-paper-table data/raw/AADM2025Dryad \
  --flight Opt1 \
  --fusion-association paper-compatible \
  --stable-segment-min-frames 100 \
  --stable-segment-max-transition-speed-mps 65
```

The `paper-compatible` fusion path applies an 800 m radar range gate, radar
class-probability thresholding, NIS gates for RF/radar updates, and records a
radar `missed_detection` posterior when no radar candidate passes the hard
preselector. The table also includes stable range-gated radar segment rows,
including an interpolated full-frame diagnostic, to separate clean radar
coverage from long-gap fill behavior. The stable-segment knobs control how long
a same-track run must be before it is trusted and how aggressively separate
segments may be stitched across radar ID changes.

Run the Opt1-Opt3 radar candidate class-probability threshold ablation:

```bash
python scripts/run_candidate_threshold_ablation.py data/raw/AADM2025Dryad --thresholds 0.4 0.5
```

Sweep the stable radar segment diagnostic without running fusion or oracle rows:

```bash
python scripts/run_stable_radar_segment_ablation.py data/raw/AADM2025Dryad \
  --min-segment-frames 75 100 150 \
  --max-transition-speeds-mps 35 65 100
```

The script writes per-flight rows plus aggregate rows to the summary CSV, and a
separate ranking CSV next to it for quickly identifying the best knob setting.
Ranking defaults to `--ranking-min-coverage 0.95`, so low-coverage rows remain
visible but are not treated as recommendation-eligible. Ranking rows also
include coverage-penalized error columns and a Pareto-front flag for comparing
coverage/error tradeoffs. A compact recommendation JSON is written next to the
summary and ranking CSVs for workflow and paper-note automation.

Run the Opt1-Opt3 PDA-mixture association ablation:

```bash
python scripts/run_pda_association_ablation.py data/raw/AADM2025Dryad
```

Run the Opt1-Opt3 PyRecEst MHT track-bank ablation:

```bash
python scripts/run_track_bank_ablation.py data/raw/AADM2025Dryad
```

Run the current best non-oracle preset on one flight:

```bash
raft-uav-best-non-oracle data/raw/AADM2025Dryad --flight Opt2
```

Run the integrated result-improvement suite.  This executes the leakage-safe
calibration/tuning/evaluation workflow used for result-oriented development:
LOFO time-offset calibration, LOFO radar covariance tuning, nested LOFO tuning,
leave-flight-out SOTA evaluation, oracle-gap diagnostics, and constrained
leaderboard ranking.

```bash
raft-uav-result-improvement-suite data/raw/AADM2025Dryad \
  --flights Opt1 Opt2 Opt3 \
  --skip-existing
```

Use `--dry-run` to write the manifest and print the exact commands without
executing them.

The preset expands to range-covariance tracklet-Viterbi association, IMM replay,
Student-t robust updates, and 20-second fixed-lag RTS smoothing. It deliberately
does not use truth-gated or nearest-truth radar association; truth is used only
for the same post-run metrics already produced by `run-baseline`.

The first baseline is deliberately conservative. It is meant to reproduce the
published constant-velocity Kalman fusion setup before adding robust gating,
learned sensor uncertainties, maneuvering models, and smoothing.

Baseline runs write gitignored per-flight artifacts under `outputs/baseline/`:

- `estimates.csv`
- `diagnostics.csv`
- `selected_radar.csv`
- `metrics.json`
- `trajectory.png`

Radar JSON frames contain many Fortem `trackData` entries. The lower-level
base-CLI default remains `--radar-association catprob`, which keeps radar rows
whose UAV class probability `catProb[0]` is at least `0.5`. For
result-oriented reproduction runs, prefer the explicit tracklet-Viterbi path:

```bash
raft-uav run-baseline data/raw/AADM2025Dryad \
  --flight Opt2 \
  --radar-association tracklet-viterbi \
  --tracklet-variant range-covariance
```

The installed wrapper defaults the tracklet implementation to
`range-covariance` when `tracklet-viterbi` is selected, but commands in this
README spell it out to avoid accidental regressions to the base dispatcher.
Use `--radar-association oracle-nearest-truth` only as a diagnostic upper bound
because it uses ground truth. The online alternatives are
`--radar-association prediction-nis`,
`--radar-association track-continuity`, and the experimental
`--radar-association geometry-score` mode, which adds velocity consistency,
track-switch, and UAV class-probability terms to the NIS score. The
experimental `--radar-association pda-mixture` mode keeps multiple candidates
inside one radar update by using NIS/class-probability weights and adding the
candidate spread to the measurement covariance. The experimental
`--radar-association track-bank` mode uses PyRecEst's
`MultiHypothesisTracker` to keep multiple single-UAV association hypotheses
alive across radar frames. Baseline runs also write `hypotheses.csv` for modes
that expose per-hypothesis diagnostics. Legacy `--radar-selection` modes are
retained for schema debugging and reproducibility; use
`--radar-selection catprob-all` only to reproduce the historical behavior that
feeds every above-threshold radar candidate in a frame into the single-target
filter.

Use `--smoother fixed-lag --smoother-lag-s 20` to apply a 20-second RTS
fixed-lag pass before metrics and plots are written. `--smoother rts` runs the
full offline RTS smoother and is mainly useful as an upper-bound diagnostic.

The IMM runner supports the same post-filter smoothing switches, so online IMM,
bounded-latency IMM, and offline upper-bound IMM rows can be compared directly:

```bash
raft-uav-imm data/raw/AADM2025Dryad --flight Opt2 --tracker imm

raft-uav-imm data/raw/AADM2025Dryad --flight Opt2 --tracker imm \
  --smoother fixed-lag --smoother-lag-s 20

raft-uav-imm data/raw/AADM2025Dryad --flight Opt2 --tracker imm \
  --smoother rts
```

Run the leave-one-flight-out SOTA protocol with the explicit online/fixed-lag/RTS
IMM rows and the current best non-oracle tracklet replay row:

```bash
python scripts/run_leave_flight_out_sota.py data/raw/AADM2025Dryad \
  --methods cv_catprob imm_catprob imm_catprob_fixed_lag imm_catprob_rts \
  imm_tracklet_viterbi_fixed_lag
```

For the leakage-safe calibrated heteroscedastic CV row, use:

```bash
python scripts/run_leave_flight_out_sota.py data/raw/AADM2025Dryad \
  --methods hetero_cv_lofo_nis_fixed_lag
```
