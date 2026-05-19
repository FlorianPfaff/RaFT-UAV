from __future__ import annotations

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "stable-radar-segment-ablation.yml"
)


def test_stable_radar_segment_workflow_runs_dataset_backed_ablation() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "uses: ./.github/actions/ensure-aadm2025dryad-dataset" in workflow
    assert "scripts/run_stable_radar_segment_ablation.py" in workflow
    assert "--flights ${FLIGHTS}" in workflow
    assert "--interpolation-max-gaps-s ${INTERPOLATION_MAX_GAPS_S}" in workflow
    assert "--interpolation-max-speeds-mps ${INTERPOLATION_MAX_SPEEDS_MPS}" in workflow
    assert "interpolation_dropped_frame_count" in workflow
    assert "interpolation_high_speed_dropped_count" in workflow
    assert "--ranking-output" in workflow
    assert "--recommendation-output" in workflow
    assert "stable_radar_segment_ablation_recommendation.json" in workflow
    assert "actions/upload-artifact@v7" in workflow
