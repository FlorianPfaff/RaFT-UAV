"""Replay-preserving result API for tracklet-Viterbi association."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker, TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _build_rf_anchor_states,
    _first_rf_bootstrap_index,
    _nodes_for_radar_frame,
    _optional_float,
    _optional_track_id,
    _radar_event_key,
    _select_tracklet_viterbi_path,
    _selected_row_event_key,
)


@dataclass(frozen=True)
class TrackletViterbiResult:
    """Outputs from tracklet-Viterbi association and Kalman replay."""

    records: list[dict[str, object]]
    accepted_radar: pd.DataFrame
    viterbi_selected_radar: pd.DataFrame
    radar_candidate_ledger: pd.DataFrame


def run_async_cv_baseline_with_tracklet_viterbi_result(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
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
    candidate_catprob_threshold: float | None = 0.4,
    config: TrackletViterbiAssociationConfig | None = None,
) -> TrackletViterbiResult:
    """Run CV fusion and return accepted plus all non-miss Viterbi choices.

    ``accepted_radar`` contains only radar rows accepted by Kalman replay,
    ``viterbi_selected_radar`` contains every non-miss Viterbi choice annotated
    with replay acceptance, NIS, residual norm, and gating diagnostics, and
    ``radar_candidate_ledger`` contains the scored top candidate pool for each
    radar frame with the selected Viterbi row marked.
    """

    from raft_uav.baselines.radar_association import (
        _empty_selected_radar,
        _events,
        _initial_measurement,
        _selected_rows_frame,
    )

    cfg = config or TrackletViterbiAssociationConfig()
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    events = _events(list(rf_measurements), radar)
    if not events:
        empty = _empty_selected_radar(radar)
        return TrackletViterbiResult(
            [],
            empty,
            _empty_replayed_rows(empty),
            _empty_candidate_ledger(radar),
        )
    bootstrap_index = _first_rf_bootstrap_index(events)
    if bootstrap_index is None:
        empty = _empty_selected_radar(radar)
        return TrackletViterbiResult(
            [],
            empty,
            _empty_replayed_rows(empty),
            _empty_candidate_ledger(radar),
        )
    events = events[bootstrap_index:]

    initial = _initial_measurement(
        events[0],
        association="tracklet-viterbi",
        covariance=covariance,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    if initial is None:
        empty = _empty_selected_radar(radar)
        return TrackletViterbiResult(
            [],
            empty,
            _empty_replayed_rows(empty),
            _empty_candidate_ledger(radar),
        )

    anchors = _build_rf_anchor_states(
        events=events,
        acceleration_std_mps2=acceleration_std_mps2,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )
    selected = _select_tracklet_viterbi_path(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=cfg,
    )
    candidate_ledger = _tracklet_candidate_ledger(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=cfg,
        selected_rows=selected,
    )
    records, accepted, replayed = _replay_selected_tracklet_path_with_replay(
        events=events,
        selected_rows=selected,
        initial_measurement=initial,
        acceleration_std_mps2=acceleration_std_mps2,
        covariance=covariance,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )
    accepted_frame = _selected_rows_frame(radar, accepted)
    replayed_frame = _selected_rows_frame(radar, replayed)
    return TrackletViterbiResult(records, accepted_frame, replayed_frame, candidate_ledger)


def _tracklet_candidate_ledger(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, object],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    selected_rows: list[pd.Series],
) -> pd.DataFrame:
    """Return the scored top candidate pool used by tracklet Viterbi."""

    selected_by_event_key = {
        _event_key_token(_selected_row_event_key(row)): row for row in selected_rows
    }
    rows: list[pd.Series] = []
    template = pd.DataFrame()
    for event_index, event in enumerate(events):
        if event["kind"] != "radar":
            continue
        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        if template.empty:
            template = candidates.iloc[0:0].copy()
        nodes = [
            node
            for node in _nodes_for_radar_frame(
                event_index=event_index,
                candidates=candidates,
                anchor=anchors.get(event_index),
                covariance=covariance,
                candidate_catprob_threshold=candidate_catprob_threshold,
                config=config,
            )
            if not node.is_miss and node.row is not None
        ]
        selected = selected_by_event_key.get(_event_key_token(_radar_event_key(candidates)))
        for candidate_rank, node in enumerate(nodes):
            assert node.row is not None
            row = node.row.copy()
            selected_here = selected is not None and _same_radar_candidate(row, selected)
            row["association_mode"] = "tracklet-viterbi"
            row["association_action"] = "candidate_ledger"
            row["association_event_index"] = int(event_index)
            row["association_event_key_type"] = str(node.event_key[0])
            row["association_event_key_value"] = _event_key_value(node.event_key)
            row["association_candidate_rank"] = int(candidate_rank)
            row["association_candidate_source_index"] = _series_name_value(row)
            row["association_candidate_pool_rows"] = int(len(nodes))
            row["association_viterbi_selected"] = bool(selected_here)
            row["association_nis"] = float(node.anchor_nis)
            row["association_score"] = float(node.unary_cost)
            row["association_anchor_nis"] = float(node.anchor_nis)
            row["association_catprob_cost"] = float(node.catprob_cost)
            row["association_range_cost"] = float(node.range_cost)
            row["association_viterbi_path_cost"] = (
                _optional_float(selected.get("association_viterbi_path_cost"))
                if selected_here and selected is not None
                else None
            )
            rows.append(row)
    if not rows:
        return _empty_candidate_ledger(template)
    sort_columns = [
        column
        for column in (
            "time_s",
            "frame_index",
            "association_candidate_rank",
            "track_id",
            "track_index",
        )
        if column in rows[0].index
    ]
    return pd.DataFrame(rows).sort_values(sort_columns).reset_index(drop=True)


def _replay_selected_tracklet_path_with_replay(
    *,
    events: list[dict[str, object]],
    selected_rows: list[pd.Series],
    initial_measurement: TrackingMeasurement,
    acceleration_std_mps2: float,
    covariance: np.ndarray,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
) -> tuple[list[dict[str, object]], list[pd.Series], list[pd.Series]]:
    from raft_uav.baselines.radar_association import (
        _gate_threshold_for_measurement,
        _inflation_alpha_for_measurement,
        _max_residual_norm_for_measurement,
        _radar_row_to_measurement,
        _record,
        _robust_update_for_measurement,
    )

    selected_by_key = {_selected_row_event_key(row): row for row in selected_rows}
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    accepted_rows: list[pd.Series] = []
    replayed_rows: list[pd.Series] = []
    for event in events:
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
        selected = selected_by_key.get(_radar_event_key(candidates))
        if selected is None:
            continue
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
        replayed = selected.copy()
        replayed["association_replay_accepted"] = bool(diagnostics.accepted)
        replayed["association_replay_update_action"] = diagnostics.update_action
        replayed["association_replay_nis"] = float(diagnostics.nis)
        replayed["association_replay_residual_norm_m"] = float(diagnostics.residual_norm_m)
        replayed["association_replay_covariance_scale"] = float(diagnostics.covariance_scale)
        replayed["association_replay_gate_threshold"] = diagnostics.gate_threshold
        replayed["association_replay_safety_gate_threshold"] = diagnostics.safety_gate_threshold
        replayed_rows.append(replayed)
        if diagnostics.accepted:
            accepted_rows.append(replayed)
        records.append(
            _record(
                measurement,
                tracker,
                diagnostics,
                track_id=_optional_track_id(selected.get("track_id")),
                association_nis=_optional_float(selected.get("association_nis")),
                association_score=_optional_float(selected.get("association_score")),
                association_mode="tracklet-viterbi",
            )
        )
    return records, accepted_rows, replayed_rows


def _empty_replayed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    replayed = frame.copy()
    for column in (
        "association_replay_accepted",
        "association_replay_update_action",
        "association_replay_nis",
        "association_replay_residual_norm_m",
        "association_replay_covariance_scale",
        "association_replay_gate_threshold",
        "association_replay_safety_gate_threshold",
    ):
        if column not in replayed.columns:
            replayed[column] = []
    return replayed


def _empty_candidate_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    ledger = frame.iloc[0:0].copy()
    for column in (
        "association_mode",
        "association_action",
        "association_event_index",
        "association_event_key_type",
        "association_event_key_value",
        "association_candidate_rank",
        "association_candidate_source_index",
        "association_candidate_pool_rows",
        "association_viterbi_selected",
        "association_nis",
        "association_score",
        "association_anchor_nis",
        "association_catprob_cost",
        "association_range_cost",
        "association_viterbi_path_cost",
    ):
        if column not in ledger.columns:
            ledger[column] = []
    return ledger


def _same_radar_candidate(left: pd.Series, right: pd.Series) -> bool:
    left_name = _series_name_value(left)
    right_name = _series_name_value(right)
    if left_name is not None and right_name is not None and left_name == right_name:
        return True
    for column in ("track_index", "track_id"):
        left_value = _optional_float(left.get(column))
        right_value = _optional_float(right.get(column))
        if (
            left_value is not None
            and right_value is not None
            and int(left_value) == int(right_value)
        ):
            return True
    required = ("time_s", "east_m", "north_m", "up_m")
    if not all(column in left.index and column in right.index for column in required):
        return False
    left_values = [_optional_float(left.get(column)) for column in required]
    right_values = [_optional_float(right.get(column)) for column in required]
    if any(value is None for value in left_values + right_values):
        return False
    return bool(np.allclose(left_values, right_values, rtol=0.0, atol=1.0e-9))


def _series_name_value(row: pd.Series) -> int | float | str | None:
    name = getattr(row, "name", None)
    if name is None:
        return None
    number = _optional_float(name)
    if number is not None:
        return int(number) if float(number).is_integer() else float(number)
    return str(name)


def _event_key_token(key: tuple[str, int | float]) -> tuple[str, int | float | str]:
    kind, value = key
    if isinstance(value, float) and not np.isfinite(value):
        return str(kind), "nan"
    return str(kind), value


def _event_key_value(key: tuple[str, int | float]) -> int | float | str:
    _, value = key
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return value
