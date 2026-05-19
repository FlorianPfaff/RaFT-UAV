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

## Evaluate through the canonical baseline CLI

The regular `raft-uav run-baseline` command can consume a fitted model directly:

```bash
raft-uav run-baseline data/raw/AADM2025Dryad \
  --flight Opt3 \
  --uncertainty-model outputs/uncertainty/no_opt3_uncertainty.json \
  --radar-association geometry-score \
  --smoother fixed-lag
```

This applies the learned row-wise covariance before RF/radar measurement conversion.
The canonical CLI preserves the historical position-only radar measurement model.
Radar association also uses those covariances for candidate NIS scoring, PDA
mixture covariance, and track-bank assignment.

The standalone runner remains useful for compact ablation scripts:

```bash
python scripts/run_heteroscedastic_baseline.py data/raw/AADM2025Dryad \
  --flight Opt3 \
  --uncertainty-model outputs/uncertainty/no_opt3_uncertainty.json \
  --radar-selection catprob \
  --smoother fixed-lag
```

## Programmatic measurement conversion

For experiments that already normalize RF/radar rows and want to consume learned `cov_*` columns directly, use:

```python
from raft_uav.heteroscedastic_measurements import (
    radar_measurements_to_enu_with_uncertainty,
    rf_measurements_to_enu_with_uncertainty,
)

rf = model.apply_rf(rf)
radar = model.apply_radar(radar)
rf_measurements = rf_measurements_to_enu_with_uncertainty(rf)
radar_measurements = radar_measurements_to_enu_with_uncertainty(radar)
```

The radar converter keeps six-dimensional position-plus-velocity measurements when Fortem velocity components are present.  The learned covariance is used for the position block and the historical fixed velocity covariance is retained for the velocity block.

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

`covariance_from_row(...)` prefers `association_cov_*` columns over `cov_*` columns so PDA/MHT association code can override the measurement covariance with a mixture covariance.

## Notes

This implementation intentionally avoids adding scikit-learn or a neural dependency. It is meant as a safe first step: fit calibrated covariance, compare against the fixed-covariance CV baseline, and then use the same row-wise covariance in PDA/MHT/IMM extensions.
