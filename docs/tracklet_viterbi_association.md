# Tracklet-Viterbi radar association

`tracklet_viterbi.py` adds a deterministic radar association primitive for AERPAW UAV tracking. It selects one globally consistent Fortem radar row per radar frame before running the existing constant-velocity fusion baseline.

The cost combines:

- constant-velocity transition residuals,
- Fortem track-ID switch penalties and same-track rewards,
- UAV class probability,
- track persistence,
- overspeed penalties,
- optional RF proximity support.

Run the standalone evaluator with:

```bash
python scripts/run_tracklet_viterbi_association.py data/raw/AADM2025Dryad \
  --flight Opt1 \
  --radar-catprob-threshold 0.4 \
  --smoother fixed-lag \
  --smoother-lag-s 20
```

A useful first sweep is:

```bash
for sw in 6 9 12 18; do
  for ts in 40 60 90; do
    python scripts/run_tracklet_viterbi_association.py data/raw/AADM2025Dryad \
      --flight Opt1 \
      --output-dir outputs/tracklet_viterbi_sw${sw}_ts${ts} \
      --switch-penalty "$sw" \
      --transition-std-m "$ts"
  done
done
```

This is intended as the next association baseline to test against `prediction-nis`, `track-continuity`, `geometry-score`, `pda-mixture`, and `track-bank`.
