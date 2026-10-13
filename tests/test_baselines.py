"""Tests for participant aggregation, model heads, bootstrap CIs, and permutation checks."""

import numpy as np
import pandas as pd
import pytest

from anjaney.eval.metrics import compute_bootstrap_metrics, compute_metrics
from anjaney.models.heads import get_model_head
from anjaney.models.participant_agg import ParticipantAggregator


def test_participant_aggregator_shape():
    df_windows = pd.DataFrame(
        [
            {
                "participant_id": "P_1",
                "rel_position": 0.2,
                "phq8_score": 6.0,
                "phq8_binary": 0,
                "feat_a": 1.0,
                "feat_b": 2.0,
            },
            {
                "participant_id": "P_1",
                "rel_position": 0.8,
                "phq8_score": 6.0,
                "phq8_binary": 0,
                "feat_a": 3.0,
                "feat_b": 4.0,
            },
            {
                "participant_id": "P_2",
                "rel_position": 0.5,
                "phq8_score": 14.0,
                "phq8_binary": 1,
                "feat_a": 5.0,
                "feat_b": 6.0,
            },
        ]
    )

    agg = ParticipantAggregator(feature_cols=["feat_a", "feat_b"])
    df_agg, y_score, y_bin = agg.aggregate(df_windows)

    assert len(df_agg) == 2
    assert list(df_agg.index) == ["P_1", "P_2"]
    assert y_score.loc["P_1"] == 6.0
    assert y_bin.loc["P_2"] == 1
    # Check mean and drift computation for P_1
    assert df_agg.loc["P_1", "feat_a__mean"] == 2.0
    assert df_agg.loc["P_1", "feat_a__drift"] == 2.0  # 3.0 (late) - 1.0 (early)


def test_model_heads_fit_and_predict():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([4.0, 6.0, 8.0, 10.0])

    for head_name in ["dummy", "ridge", "elasticnet", "lightgbm"]:
        head = get_model_head(head_name)
        head.fit(X, y)
        preds = head.predict(X)
        assert len(preds) == len(y)


def test_bootstrap_metrics_confidence_intervals():
    y_true = np.array([2.0, 4.0, 8.0, 12.0, 16.0, 18.0, 5.0, 11.0])
    y_pred = np.array([3.0, 4.5, 7.5, 11.0, 15.0, 19.0, 6.0, 10.5])

    res = compute_bootstrap_metrics(y_true, y_pred, n_bootstrap=200, seed=42)
    for metric in ["rmse", "mae", "auroc", "f1"]:
        point = res[metric]["point"]
        lower = res[metric]["ci_lower"]
        upper = res[metric]["ci_upper"]
        assert lower <= point <= upper or pytest.approx(lower, 0.05) == point


def test_dummy_regressor_baseline():
    y_true = np.array([0.0, 2.0, 4.0, 12.0, 14.0, 18.0])
    y_pred = np.full_like(y_true, fill_value=np.mean(y_true))

    metrics = compute_metrics(y_true, y_pred, threshold=10.0)
    assert metrics["auroc"] == 0.50
