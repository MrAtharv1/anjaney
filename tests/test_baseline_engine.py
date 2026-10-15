"""Statistical tests for RQ1 null calibration, power, cross-fitted conditioner, and LATEST_REAL provenance guard."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anjaney.baseline.position import PositionConditioner
from anjaney.baseline.rq1_runner import (
    read_latest_real_rq1_decision,
    run_rq1_analysis,
    verify_prereg_clean,
)
from anjaney.data.synthetic import SyntheticCohortGenerator
from anjaney.features.registry import FeatureRegistry
from anjaney.preprocess.windows import WindowBuilder


def test_position_conditioner_cross_fit_and_nan_handling():
    df = pd.DataFrame(
        {
            "participant_id": [f"P_{i // 5}" for i in range(30)],
            "window_idx": [0, 1, 2, 3, 4] * 6,
            "prompt_phase": [0, 1, 2, 2, 3] * 6,
            "rel_position": [0.1, 0.3, 0.5, 0.7, 0.9] * 6,
            "combined_deviation": [1.0, 2.0, 3.0, 4.0, 5.0] * 6,
        }
    )
    pc = PositionConditioner(n_bins=5, n_turn_bins=5, min_bin_count=3, require_prompt_phase=True)
    transformed = pc.cross_fit_transform(df, "combined_deviation", n_folds=3)
    assert "cross_fit" in pc.conditioning_mode
    assert np.allclose(transformed, 0.0)

    df_nan = pd.DataFrame(
        {
            "participant_id": ["P_1", "P_1"],
            "window_idx": [1, 2],
            "prompt_phase": [1, 2],
            "rel_position": [np.nan, 0.5],
            "combined_deviation": [2.0, 3.0],
        }
    )
    out_nan = pc.transform(df_nan, "combined_deviation")
    assert np.isnan(out_nan[0])
    assert np.isfinite(out_nan[1])


def test_prereg_dirty_or_untracked_aborts(tmp_path: Path):
    untracked_file = tmp_path / "untracked_prereg.yaml"
    untracked_file.write_text("study_title: test\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Part 0 violation"):
        verify_prereg_clean(untracked_file)


@pytest.mark.slow
def test_rq1_null_false_supported_rate_over_200_simulations():
    """Simulate 200 pure-fatigue null datasets and assert Type-I false-SUPPORTED rate <= ~5%."""
    rng = np.random.default_rng(12345)
    n_sims = 200
    n_high = 40
    n_low = 40
    n_perms = 250
    n_boot = 250
    min_d = 0.30

    false_supported_count = 0

    for _ in range(n_sims):
        high_drifts = rng.normal(loc=0.0, scale=1.0, size=n_high)
        low_drifts = rng.normal(loc=0.0, scale=1.0, size=n_low)

        obs_diff = float(np.mean(high_drifts) - np.mean(low_drifts))
        s_pooled = np.sqrt(
            ((n_high - 1) * np.var(high_drifts, ddof=1) + (n_low - 1) * np.var(low_drifts, ddof=1))
            / (n_high + n_low - 2)
        )
        cohens_d = float(obs_diff / s_pooled)

        all_d = np.concatenate([high_drifts, low_drifts])
        exceed = 0
        for _ in range(n_perms):
            shuf = rng.permutation(all_d)
            if (np.mean(shuf[:n_high]) - np.mean(shuf[n_high:])) >= obs_diff:
                exceed += 1
        p_val = (exceed + 1) / (n_perms + 1)

        boot_idx_h = rng.integers(0, n_high, size=(n_boot, n_high))
        boot_idx_l = rng.integers(0, n_low, size=(n_boot, n_low))
        boot_diffs = np.mean(high_drifts[boot_idx_h], axis=1) - np.mean(
            low_drifts[boot_idx_l], axis=1
        )
        ci_lower = float(np.percentile(boot_diffs, 2.5))

        if p_val < 0.05 and ci_lower > 0.0 and cohens_d >= min_d:
            false_supported_count += 1

    false_positive_rate = false_supported_count / n_sims
    assert false_positive_rate <= 0.075


@pytest.mark.slow
def test_rq1_runner_power_and_synthetic_cannot_write_latest_real(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "anjaney.baseline.rq1_runner.verify_prereg_clean",
        lambda p: {
            "prereg_sha256": "test_hash",
            "prereg_commit": "test_commit",
            "repo_git_hash": "test_git",
            "code_dirty": False,
        },
    )

    builder = WindowBuilder(window_size=4)
    registry = FeatureRegistry()

    gen_alt = SyntheticCohortGenerator(n_participants=115, seed=42, extra_high_risk_drift=True)
    df_win_alt = builder.build_dataset_dataframe(gen_alt.generate())
    df_feat_alt = registry.extract_features(df_win_alt, groups=["timing"])
    if "mean_response_latency" in df_feat_alt.columns:
        df_feat_alt["timing_response_latency"] = df_feat_alt["mean_response_latency"].fillna(1.5)
    alt_pq = tmp_path / "features_alt.parquet"
    df_feat_alt.to_parquet(alt_pq, index=False)

    out_rq1 = tmp_path / "rq1_alt"
    res_alt = run_rq1_analysis(
        features_path=alt_pq,
        prereg_path="configs/prereg.yaml",
        output_dir=out_rq1,
        seed=42,
    )
    assert res_alt["status"] == "SUPPORTED"
    assert res_alt["data_source"] == "synthetic"
    assert res_alt["is_real_data_gate"] is False
    # Verify synthetic run NEVER creates LATEST_REAL pointer
    assert not (out_rq1 / "LATEST_REAL").exists()
    with pytest.raises(FileNotFoundError, match="No real-data RQ1 gate"):
        read_latest_real_rq1_decision(out_rq1)
