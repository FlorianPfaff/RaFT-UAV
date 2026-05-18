# LOFO time-offset calibration

This workflow estimates constant RF/radar timestamp corrections on training
flights only, applies those offsets to a held-out flight, and optionally runs
the tracklet-Viterbi fusion baseline on the corrected measurement timelines.

The held-out flight's truth is used only for final evaluation, not for choosing
its timestamp correction.

Run the default Opt1/Opt2/Opt3 leave-one-flight-out evaluation:

```bash
python scripts/run_lofo_time_offset_calibration.py data/raw/AADM2025Dryad \
  --offset-min-s -10 \
  --offset-max-s 10 \
  --offset-step-s 0.25
```

For each held-out flight, the script writes:

- `radar_time_offset_sweep.csv`: radar nearest-candidate oracle sweep on training flights.
- `rf_time_offset_sweep.csv`: RF point-measurement sweep on training flights.
- `radar_time_corrected.csv`: held-out radar rows after applying the learned offset.
- `rf_time_corrected.csv`: held-out RF rows after applying the learned offset.
- `selected_radar.csv`: selected tracklet-Viterbi radar path.
- `estimates.csv`: fused tracking estimates.
- `lofo_time_offset_metrics.json`: calibration and tracking summary.

A top-level `lofo_time_offset_summary.csv` aggregates all held-out flights.

Useful ablations:

```bash
# Apply only radar offset.
python scripts/run_lofo_time_offset_calibration.py data/raw/AADM2025Dryad --disable-rf-offset

# Calibrate offsets but do not run tracking.
python scripts/run_lofo_time_offset_calibration.py data/raw/AADM2025Dryad --skip-tracking
```
