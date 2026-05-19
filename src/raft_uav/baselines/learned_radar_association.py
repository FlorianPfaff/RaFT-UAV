"""Learned radar association runner for the asynchronous CV baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
)
from raft_uav.baselines.learned_radar_likelihood import (
    LearnedRadarAssociationModel,
    score_radar_candidates_with_learned_likelihood,
)
from raft_uav.baselines.radar_association import (
    _bootstrap_record,
    _catprob_candidate_pool,
    _empty_selected_radar,
    _events,
    _gate_threshold_for_measurement,
    _initial_observation,
    _inflation_alpha_for_measurement,
    _max_residual_norm_for_measurement,
    _nis_scored_candidates,
    _optional_float,
    _optional_track_id,
    _radar_row_to_measurement,
    _record,
    _robust_update_for_measurement,
    _selected_rows_frame,
)


def run_async_cv_baseline_with_learned_radar_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    model: LearnedRadarAssociationModel | str | Path,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    candidate_catprob_threshold: float | None = 0.5,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion with a learned radar-candidate likelihood.

    The learned model is used only for radar row association. The Kalman update,
    gating, and robust covariance inflation stay identical to the baseline path.
    """

    learned_model = (
        model if isinstance(model, LearnedRadarAssociationModel) else LearnedRadarAssociationModel.load(model)
    )
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    events = _events(list(rf_measurements), radar)
    if not events:
        return [], _empty_selected_radar(radar)

    initial_observation = _initial_observation(
        events[0],
        association="prediction-nis",
        covariance=covariance,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    if initial_observation is None:
        return [], _empty_selected_radar(radar)

    initial_measurement = initial_observation.measurement
    initial_selected_row = initial_observation.selected_radar_row
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []
    current_track_id: int | None = None
    if initial_selected_row is not None:
        current_track_id = _optional_track_id(initial_selected_row)
        selected_rows.append(initial_selected_row)
    records.append(
        _bootstrap_record(
            initial_measurement,
            tracker,
            selected_row=initial_selected_row,
            association_mode="learned-likelihood",
        )
    )

    for event in events[1:]:
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            diagnostics = tracker.update(
                measurement,
                gate_threshold=_gate_threshold_for_measurement(
                    measurement,
                    gate_probabilities_by_source=gate_probabilities_by_source,
                    gate_thresholds_by_source=gate_thresholds_by_source,
                ),
                safety_gate_threshold=_gate_threshold_for_measurement(
                    measurement,
                    gate_probabilities_by_source=safety_gate_probabilities_by_source,
                    gate_thresholds_by_source=safety_gate_thresholds_by_source,
                ),
                max_residual_norm=_max_residual_norm_for_measurement(
                    measurement,
                    max_residual_norms_by_source=max_residual_norms_by_source,
                ),
                robust_update=_robust_update_for_measurement(
                    measurement,
                    robust_update_by_source=robust_update_by_source,
                ),
                inflation_alpha=_inflation_alpha_for_measurement(
                    measurement,
                    inflation_alpha_by_source=inflation_alpha_by_source,
                ),
            )
            records.append(_record(measurement, tracker, diagnostics))
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        time_s = float(event["time_s"])
        tracker.predict_to(time_s)
        candidates = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
        if candidates.empty:
            continue
        scored = _nis_scored_candidates(candidates, tracker, covariance)
        if scored.empty:
            continue
        learned_scored = score_radar_candidates_with_learned_likelihood(
            scored,
            model=learned_model,
            tracker_state=tracker.state,
            current_track_id=current_track_id,
        )
        selected = learned_scored.loc[learned_scored["association_score"].idxmin()].copy()

        measurement = _radar_row_to_measurement(selected, covariance)
        diagnostics = tracker.update(
            measurement,
            gate_threshold=_gate_threshold_for_measurement(
                measurement,
                gate_probabilities_by_source=gate_probabilities_by_source,
                gate_thresholds_by_source=gate_thresholds_by_source,
            ),
            safety_gate_threshold=_gate_threshold_for_measurement(
                measurement,
                gate_probabilities_by_source=safety_gate_probabilities_by_source,
                gate_thresholds_by_source=safety_gate_thresholds_by_source,
            ),
            max_residual_norm=_max_residual_norm_for_measurement(
                measurement,
                max_residual_norms_by_source=max_residual_norms_by_source,
            ),
            robust_update=_robust_update_for_measurement(
                measurement,
                robust_update_by_source=robust_update_by_source,
            ),
            inflation_alpha=_inflation_alpha_for_measurement(
                measurement,
                inflation_alpha_by_source=inflation_alpha_by_source,
            ),
        )
        if diagnostics.accepted:
            current_track_id = _optional_track_id(selected)
            selected_rows.append(selected)
        records.append(
            _record(
                measurement,
                tracker,
                diagnostics,
                track_id=_optional_track_id(selected),
                association_nis=_optional_float(selected.get("association_nis")),
                association_score=_optional_float(selected.get("association_score")),
                association_mode="learned-likelihood",
            )
        )

    return records, _selected_rows_frame(radar, selected_rows)
