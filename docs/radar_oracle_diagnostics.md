# Radar oracle and time-offset diagnostics

This diagnostic is truth-based and is not a publishable tracking method. It is intended to answer two questions before more association or learning work:

1. What is the nearest-candidate oracle error if the correct Fortem radar row is selected in each frame?
2. Does a constant radar timestamp offset reduce that oracle error?

Run:

```bash
python scripts/run_radar_oracle_diagnostics.py data/raw/AADM2025Dryad \
  --flight Opt1 \
  --offset-min-s -10 \
  --offset-max-s 10 \
  --offset-step-s 0.25
```

Outputs per flight:

- `time_offset_sweep.csv`: paper-style mean/std/max, RMSE, P95, and coverage for each tested offset.
- `nearest_candidate_oracle_offset0.csv`: nearest radar candidate per frame with no time correction.
- `nearest_candidate_oracle_best_offset.csv`: nearest radar candidate per frame at the best constant offset.
- `oracle_diagnostics.json`: compact summary.

Interpretation:

- If the best-offset oracle is still high, the bottleneck is likely radar geometry/calibration or the flight itself, not association.
- If the best-offset oracle drops sharply, timestamp alignment is a high-priority fix.
- If the nearest-candidate oracle is good but the tracker is poor, the bottleneck is association/filtering.
