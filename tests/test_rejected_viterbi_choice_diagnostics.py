from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_rejected_viterbi_choices import (  # noqa: E402
    CLASS_BAD_ASSOCIATION_BAD_GATE,
    CLASS_BAD_ASSOCIATION_GOOD_GATE,
    CLASS_GOOD_ASSOCIATION_BAD_GATE,
    CLASS_GOOD_ASSOCIATION_GOOD_GATE,
    classify_viterbi_choices,
)


def test_classify_viterbi_choices_covers_all_association_gate_cases():
    truth = pd.DataFrame(
        [
            {"time_s": 0.0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0},
            {"time_s": 1.0, "east_m": 10.0, "north_m": 0.0, "up_m": 0.0},
            {"time_s": 2.0, "east_m": 20.0, "north_m": 0.0, "up_m": 0.0},
            {"time_s": 3.0, "east_m": 30.0, "north_m": 0.0, "up_m": 0.0},
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "east_m": 2.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_replay_accepted": True,
            },
            {
                "time_s": 1.0,
                "east_m": 12.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_replay_accepted": False,
            },
            {
                "time_s": 2.0,
                "east_m": 200.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_replay_accepted": True,
            },
            {
                "time_s": 3.0,
                "east_m": 300.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_replay_accepted": False,
            },
        ]
    )

    classified = classify_viterbi_choices(
        replay,
        truth,
        flight_name="Synthetic",
        max_time_delta_s=0.5,
        good_threshold_m=20.0,
    )

    assert classified["rejected_choice_classification"].tolist() == [
        CLASS_GOOD_ASSOCIATION_GOOD_GATE,
        CLASS_GOOD_ASSOCIATION_BAD_GATE,
        CLASS_BAD_ASSOCIATION_GOOD_GATE,
        CLASS_BAD_ASSOCIATION_BAD_GATE,
    ]
    assert classified["association_is_good"].tolist() == [True, True, False, False]
    assert classified["replay_accepted"].tolist() == [True, False, True, False]
    assert classified["truth_match_valid"].tolist() == [True, True, True, True]
