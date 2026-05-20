# Fifth-wave reliability and deployability diagnostics

This patch adds additive scaffolding for result improvements that focus on
trustworthiness, deployment safety, and hard-segment debugging rather than
changing the default tracker path.

## Included ideas

1. **Block-bootstrap confidence intervals** for leaderboard metrics, so
   autocorrelated trajectory samples are not treated as independent.
2. **Paired method comparison** at the same truth timestamps using
   `scripts/compare_tracking_methods.py`.
3. **Do-no-harm radar update policy** helpers that classify a radar update as
   `apply`, `soften`, `defer`, or `skip` from NIS, ambiguity, RF disagreement,
   and recovery-state signals.
4. **Counterfactual / shadow-tracking support** via explicit decision and
   ambiguity summaries; these are intended to be logged by future tracker hooks.
5. **Time-to-recovery metrics** for catastrophic error events.
6. **Safe fallback policy hooks** through the do-no-harm and ensemble-decision
   helpers.
7. **Calibration transfer diagnostics** comparing training-flight and held-out
   NIS distributions.
8. **Error attribution by source sequence**, e.g. first radar after a gap.
9. **Candidate ambiguity index** and per-frame ambiguity tables.
10. **Conservative leaderboard selection** with hard constraints.
11. **Runtime and memory instrumentation** via `RuntimeMonitor`.
12. **Determinism checks** with `scripts/run_determinism_check.py`.
13. **Bad-segment mining** with `scripts/mine_bad_segments.py`.
14. **Weak-label pseudo-labeling** for high-confidence candidate expansion.
15. **Adaptive smoothing lag** heuristic from ambiguity and recovery signals.
16. **Residual whiteness diagnostics** for NIS autocorrelation.
17. **Vertical-error-specific metrics**.
18. **Track-purity diagnostics** for selected radar rows.
19. **Oracle replay gap summaries** under realistic filtering/gating.
20. **Rule-based method-family ensemble decisions** using confidence and
    ambiguity diagnostics.

## Example commands

Paired method comparison:

```bash
python scripts/compare_tracking_methods.py \
  --truth-csv outputs/truth/Opt2.csv \
  --method-a-estimates outputs/method_a/Opt2/estimates.csv \
  --method-b-estimates outputs/method_b/Opt2/estimates.csv \
  --label-a learned_tracklet \
  --label-b tracklet_viterbi
```

Reliability report for one run:

```bash
python scripts/run_tracking_reliability_report.py \
  outputs/leave_flight_out_sota/heldout_Opt2/imm_tracklet_viterbi_fixed_lag/Opt2 \
  --truth-csv outputs/truth/Opt2.csv
```

Mine challenge slices:

```bash
python scripts/mine_bad_segments.py \
  --estimates-csv outputs/run/Opt2/estimates.csv \
  --truth-csv outputs/truth/Opt2.csv
```

Check determinism between two repeated runs:

```bash
python scripts/run_determinism_check.py outputs/repeat_a/Opt2 outputs/repeat_b/Opt2 \
  --fail-on-difference
```

Apply conservative leaderboard ranking:

```bash
python scripts/rank_conservative_leaderboard.py outputs/aggregate_summary.csv \
  --objective error_3d_p95_m \
  --constraint truth_coverage_rate:ge:0.95 \
  --constraint track_switch_count:le:5
```

## Integration notes

The new module is `raft_uav.evaluation.fifth_wave_diagnostics`.  Its functions
are pure utilities; they can be called from future tracker hooks, SOTA runners,
or paper-result scripts without changing the default algorithms.  This is
intentional: the goal is to make it easier to decide which algorithmic changes
actually improve hard intervals, improve calibration transfer, and avoid harmful
radar updates.
