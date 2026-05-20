# Advanced result-improvement utilities

This patch adds scaffolding for the next set of RaFT-UAV result-improvement experiments.  The modules are designed to be leakage-safe when used after a tracking run or inside a nested leave-one-flight-out protocol.

## Oracle-gap decomposition

```bash
python scripts/run_oracle_gap_decomposition.py data/raw/AADM2025Dryad \
  --run-dir outputs/leave_flight_out_sota/heldout_Opt1/imm_tracklet_viterbi_fixed_lag \
  --flights Opt1
```

The report separates missing/plausible radar candidates, wrong association, filter rejection, and posterior drift after a correct association. It also writes identity-stability metrics such as track-switch count and dominant-track fraction.

## Nested LOFO tuning

```bash
python scripts/run_nested_lofo_tuning.py data/raw/AADM2025Dryad \
  --flights Opt1 Opt2 Opt3 \
  --skip-existing
```

The runner selects method/hyperparameter candidates on the training flights for each held-out fold, then evaluates the selected candidate once on the held-out flight. Candidate grids can be supplied as JSON.

## Typed experiment provenance

`raft_uav.experiments.config.write_resolved_experiment_config(...)` records command-line arguments, Python/platform metadata, git commit/dirty state, and `RAFT_UAV_*` environment overrides.

## Additional reusable modules

- `raft_uav.baselines.delayed_initialization`: delayed multi-hypothesis initialization helpers.
- `raft_uav.baselines.radar_track_features`: causal Fortem track-level features for learned association.
- `raft_uav.baselines.adaptive_process_noise`: NIS-driven process-noise adaptation heuristics.
- `raft_uav.calibration.time_offset_state`: online scalar time-offset state update.
- `raft_uav.baselines.hypothesis_mixture`: moment matching for soft output fusion.
- `raft_uav.stress.perturbations`: deterministic perturbations for stress-test artifacts.

These modules are intentionally small and dependency-light so they can be wired into the existing baseline paths incrementally.
