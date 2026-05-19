import pandas as pd

from scripts.diagnose_rejected_viterbi_choices import (
    CLASS_BAD_ASSOCIATION_BAD_GATE,
    CLASS_BAD_ASSOCIATION_GOOD_GATE,
    CLASS_GOOD_ASSOCIATION_BAD_GATE,
    CLASS_GOOD_ASSOCIATION_GOOD_GATE,
    _summary_row,
    classify_viterbi_choices,
)


def test_classify_viterbi_choices_splits_association_and_gate_failures():
    replay = pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_replay_accepted": True,
            },
            {
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_replay_accepted": False,
            },
            {
                "time_s": 2.0,
                "east_m": 500.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_replay_accepted": True,
            },
            {
                "time_s": 3.0,
                "east_m": 500.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_replay_accepted": False,
            },
        ]
    )
    truth = pd.DataFrame(
        [
            {"time_s": 0.0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0},
            {"time_s": 1.0, "east_m": 10.0, "north_m": 0.0, "up_m": 0.0},
            {"time_s": 2.0, "east_m": 20.0, "north_m": 0.0, "up_m": 0.0},
            {"time_s": 3.0, "east_m": 30.0, "north_m": 0.0, "up_m": 0.0},
        ]
    )

    classified = classify_viterbi_choices(
        replay,
        truth,
        flight="OptX",
        max_time_delta_s=0.5,
        good_association_threshold_m=75.0,
    )

    assert classified["replay_truth_classification"].tolist() == [
        CLASS_GOOD_ASSOCIATION_GOOD_GATE,
        CLASS_GOOD_ASSOCIATION_BAD_GATE,
        CLASS_BAD_ASSOCIATION_GOOD_GATE,
        CLASS_BAD_ASSOCIATION_BAD_GATE,
    ]
    summary = _summary_row(classified, scope="fold", flight="OptX")
    assert summary[f"{CLASS_GOOD_ASSOCIATION_BAD_GATE}_count"] == 1
    assert summary[f"{CLASS_BAD_ASSOCIATION_GOOD_GATE}_count"] == 1
    assert summary["viterbi_selected_radar_rejected_rows"] == 2
    assert summary["good_association_bad_gate_rate_among_rejected"] == 0.5
