import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker
from raft_uav.baselines.learned_radar_association import (
    run_async_cv_baseline_with_learned_radar_association,
)
from raft_uav.baselines.learned_radar_likelihood import LearnedRadarAssociationModel
from raft_uav.baselines.radar_association import _nis_scored_candidates


def test_learned_radar_association_prefers_higher_likelihood_candidate():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 30.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.95,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 0.5,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.05,
            },
        ]
    )
    model = LearnedRadarAssociationModel(
        feature_names=("cat_prob_uav",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([10.0]),
        intercept=-5.0,
    )

    records, selected = run_async_cv_baseline_with_learned_radar_association(
        rf_measurements=[],
        radar=radar,
        model=model,
        candidate_catprob_threshold=None,
    )

    assert len(records) == 1
    assert selected["association_mode"].tolist() == ["learned-likelihood"]
    assert selected["track_id"].tolist() == [1]
    assert selected["association_learned_probability"].iloc[0] > 0.5


def test_model_scores_candidate_features_without_sklearn():
    tracker = AsyncConstantVelocityKalmanTracker(initial_position=np.zeros(3), initial_time_s=0.0)
    candidates = pd.DataFrame(
        [
            {"track_id": 1, "time_s": 0.0, "east_m": 1.0, "north_m": 0.0, "up_m": 0.0},
            {"track_id": 2, "time_s": 0.0, "east_m": 50.0, "north_m": 0.0, "up_m": 0.0},
        ]
    )
    scored = _nis_scored_candidates(candidates, tracker, np.eye(3))
    model = LearnedRadarAssociationModel(
        feature_names=("log1p_association_nis",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([-1.0]),
        intercept=0.0,
    )

    probabilities = model.predict_proba_features(
        pd.DataFrame({"log1p_association_nis": np.log1p(scored["association_nis"])})
    )

    assert probabilities[0] > probabilities[1]
