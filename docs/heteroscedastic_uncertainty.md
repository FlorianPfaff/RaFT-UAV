# Heteroscedastic RF/radar uncertainty

This adds a dependency-light first version of learned, row-wise sensor covariance for the RF/radar fusion baseline.

## Train

Use leave-one-flight-out training when evaluating a held-out flight:

```bash
python scripts/train_heteroscedastic_uncertainty.py data/raw/AADM2025Dryad \
  --exclude-flight Opt3 \
  --output outputs/uncertainty/no_opt3_uncertainty.json
```

The trainer aligns normalized RF and radar rows with the nearest truth sample and fits ridge-regularized log-linear variance heads for each measured axis.

## Evaluate

```bash
python scripts/run_heteroscedastic_baseline.py data/raw/AADM2025Dryad \
  --flight Opt3 \
  --uncertainty-model outputs/uncertainty/no_opt3_uncertainty.json \
  --radar-selection catprob \
  --smoother fixed-lag
```

The runner keeps the existing constant-velocity Kalman baseline and replaces static measurement covariance with per-row covariance columns predicted by the model.

## Model outputs

RF rows receive:

- `cov_ee`
- `cov_nn`
- `cov_en`
- `std_east_m`
- `std_north_m`

Radar rows receive:

- `cov_ee`
- `cov_nn`
- `cov_uu`
- `cov_en`
- `cov_eu`
- `cov_nu`
- `std_east_m`
- `std_north_m`
- `std_up_m`

`covariance_from_row(...)` also prefers `association_cov_*` columns over `cov_*` columns so later PDA/MHT association code can override the measurement covariance with a mixture covariance.

## Notes

This implementation intentionally avoids adding scikit-learn or a neural dependency. It is meant as a safe first step: fit calibrated covariance, compare against the fixed-covariance CV baseline, and then use the same row-wise covariance in PDA/MHT/IMM extensions.
