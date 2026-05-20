# Results-improvement playbook

This note collects the additional infrastructure added for improving RaFT-UAV
results without using held-out truth for online decisions.

## Failure budget / oracle gap

Use `raft-uav-oracle-gap-report` after a run to classify radar frames into:

- no usable radar candidate,
- missed association,
- wrong association,
- filter rejection,
- large post-filter/smoother error,
- nominal.

This should be run before adding new filters.  If most frames fall into the
candidate-availability bucket, association/model tuning cannot recover the run.

## Nested LOFO tuning

Use `raft-uav-nested-lofo-tuning` for hand-tuned constants.  Every candidate is
run on training flights only; the best candidate is then evaluated once on the
held-out flight.  This is intended for Viterbi costs, PDA temperatures, IMM
mode-switch constants, stable-segment gates, and any future adaptive-process
noise settings.

## Experiment provenance

`raft_uav.experiment_config` provides JSON config loading and resolved-provenance
writing.  Store the resolved config next to `metrics.json` when building paper
runs so command-line arguments, `RAFT_UAV_*` environment variables, git SHA, and
calibration artifacts are visible.

## Confidence and calibration

`raft_uav.evaluation.confidence` derives a bounded confidence score from NIS,
residuals, association score, accepted/rejected status, and any available
covariance trace.  Use the reliability and selective-tracking curves to evaluate
whether low-confidence estimates can be detected instead of silently polluting
RMSE.

## Initialization

`raft_uav.baselines.delayed_initialization` builds multiple initial track
hypotheses from a short RF/radar window.  This is a safer replacement for
single-frame bootstrap when the first radar frame is ambiguous.

## Track-level learned features

`raft_uav.baselines.track_context_features` adds temporal Fortem track features:
age, hit streak, recent class-probability mean, range/speed history, velocity
smoothness, and frame gaps.  These are truth-free features for learned radar
association.

## Adaptive process noise

`raft_uav.baselines.adaptive_process_noise` converts rolling NIS consistency into
a recommended acceleration-noise schedule.  Start with diagnostic use; only feed
it back into a tracker after nested LOFO validation.

## Online time bias

`raft_uav.calibration.online_time_bias` contains a small per-source estimator for
slowly varying timestamp bias.  Initialize it from LOFO time-offset calibration
and keep it tightly bounded.

## Stress perturbations

`raft-uav-stress-perturbations` generates deterministic perturbation CSVs: frame
drops, false tracks, timestamp jitter, class-probability attenuation, velocity
noise, and position noise.  Use degradation curves rather than a single nominal
score when claiming robustness.

## Golden artifacts

`raft-uav-golden-artifact-checks` validates tiny smoke-test runs: required files,
parseability, monotonic times, basic metrics keys, and NaN fractions.

## Track-switch reporting

`raft_uav.evaluation.track_stability` reports track switches, unique selected
track IDs, dominant-track fraction, track entropy, and radar-update gaps.  Add
these to leaderboards so RMSE cannot hide unstable identity behavior.

## Soft hypothesis fusion

`raft_uav.baselines.soft_hypothesis_fusion` provides moment matching for state
hypotheses and candidate-position fusion with spread covariance.  It is a
building block for preserving top-K ambiguity rather than forcing a single hard
association whenever candidates are nearly tied.
