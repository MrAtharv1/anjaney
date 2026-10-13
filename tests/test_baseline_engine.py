"""Tests verifying both directions of RQ1 hypothesis and position conditioning isolation."""

import numpy as np
import pandas as pd

from anjaney.baseline.engine import PersonalBaselineEngine
from anjaney.baseline.position import PositionConditioner
from anjaney.data.synthetic import SyntheticCohortGenerator
from anjaney.features.registry import FeatureRegistry
from anjaney.preprocess.windows import WindowBuilder


def test_position_conditioner_fit_transform():
    df = pd.DataFrame(
        {
            "rel_position": [0.1, 0.3, 0.5, 0.7, 0.9],
            "combined_deviation": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    pc = PositionConditioner(n_bins=5)
    transformed = pc.fit_transform(df, "combined_deviation")
    assert np.allclose(transformed, 0.0)


def test_personal_baseline_engine_both_ways_rq1():
    # 1. Cohort WITHOUT extra high-risk drift (generic fatigue only)
    gen_falsified = SyntheticCohortGenerator(
        n_participants=20, seed=101, extra_high_risk_drift=False
    )
    sessions_f = gen_falsified.generate()
    builder = WindowBuilder(window_size=5)
    df_win_f = builder.build_dataset_dataframe(sessions_f)
    registry = FeatureRegistry()
    df_feat_f = registry.extract_features(df_win_f, groups=["timing"])

    engine = PersonalBaselineEngine(warmup_windows_k=2)
    timing_cols = [c for c in df_feat_f.columns if c.startswith("timing_")]

    priors = {
        c: (float(df_feat_f[c].mean()), float(df_feat_f[c].std()) + 1e-4) for c in timing_cols
    }

    dev_dfs = [
        engine.compute_session_deviations(
            grp, embed_cols=[], timing_cols=timing_cols, population_timing_priors=priors
        )
        for _, grp in df_feat_f.groupby("participant_id")
    ]
    df_all_f = pd.concat(dev_dfs, ignore_index=True)

    conditioner = PositionConditioner(n_bins=4)
    valid_f = df_all_f.dropna(subset=["combined_deviation"]).copy()
    valid_f["conditioned_deviation"] = conditioner.fit_transform(valid_f, "combined_deviation")

    late_f = valid_f[valid_f["rel_position"] >= 0.5]
    diff_falsified = abs(
        late_f[late_f["phq8_binary"] == 1]["conditioned_deviation"].mean()
        - late_f[late_f["phq8_binary"] == 0]["conditioned_deviation"].mean()
    )

    # 2. Cohort WITH injected high-risk drift
    gen_supported = SyntheticCohortGenerator(
        n_participants=20, seed=101, extra_high_risk_drift=True
    )
    sessions_s = gen_supported.generate()
    df_win_s = builder.build_dataset_dataframe(sessions_s)
    df_feat_s = registry.extract_features(df_win_s, groups=["timing"])

    dev_dfs_s = [
        engine.compute_session_deviations(
            grp, embed_cols=[], timing_cols=timing_cols, population_timing_priors=priors
        )
        for _, grp in df_feat_s.groupby("participant_id")
    ]
    df_all_s = pd.concat(dev_dfs_s, ignore_index=True)

    valid_s = df_all_s.dropna(subset=["combined_deviation"]).copy()
    valid_s["conditioned_deviation"] = conditioner.fit_transform(valid_s, "combined_deviation")

    late_s = valid_s[valid_s["rel_position"] >= 0.5]
    diff_supported = abs(
        late_s[late_s["phq8_binary"] == 1]["conditioned_deviation"].mean()
        - late_s[late_s["phq8_binary"] == 0]["conditioned_deviation"].mean()
    )

    # Injected drift yields higher group separation
    assert diff_supported > diff_falsified
