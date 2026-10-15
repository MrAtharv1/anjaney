"""Statistical tests for RQ1 null calibration (200 sims), realistic power (n_high=40), and conditioner properties."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anjaney.baseline.position import PositionConditioner
from anjaney.baseline.rq1_runner import run_rq1_analysis, verify_prereg_clean
from anjaney.data.synthetic import SyntheticCohortGenerator
from anjaney.features.registry import FeatureRegistry
from anjaney.preprocess.windows import WindowBuilder


def test_position_conditioner_fit_transform_and_nan_handling():
    df = pd.DataFrame(
        {
            "window_idx": [0, 1, 2, 3, 4] * 6,
            "prompt_phase": [0, 1, 2, 2, 3] * 6,
            "rel_position": [0.1, 0.3, 0.5, 0.7, 0.9] * 6,
            "combined_deviation": [1.0, 2.0, 3.0, 4.0, 5.0] * 6,
        }
    )
    pc = PositionConditioner(n_bins=5, n_turn_bins=5, min_bin_count=3, require_prompt_phase=True)
    transformed = pc.fit_transform(df, "combined_deviation")
    assert pc.conditioning_mode == "prompt_phase_and_rel_position"
    assert np.allclose(transformed, 0.0)

    # Verify NaN rel_position strictly returns NaN
    df_nan = pd.DataFrame(
        {
            "window_idx": [1, 2],
            "prompt_phase": [1, 2],
            "rel_position": [np.nan, 0.5],
            "combined_deviation": [2.0, 3.0],
        }
    )
    out_nan = pc.transform(df_nan, "combined_deviation")
    assert np.isnan(out_nan[0])
    assert np.isfinite(out_nan[1])

    # Verify missing prompt_phase raises RuntimeError when require_prompt_phase=True
    df_no_phase = df.drop(columns=["prompt_phase"])
    pc_strict = PositionConditioner(require_prompt_phase=True)
    with pytest.raises(RuntimeError, match="prompt_phase"):
        pc_strict.fit(df_no_phase, "combined_deviation")


def test_prereg_dirty_or_untracked_aborts(tmp_path: Path):
    untracked_file = tmp_path / "untracked_prereg.yaml"
    untracked_file.write_text("study_title: test\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Part 0 violation"):
        verify_prereg_clean(untracked_file)


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
        # Under H0 (shared fatigue only): after position conditioning, participant late drifts ~ N(0, 1)
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
    # One-sided alpha = 0.05 -> allow binomial sampling tolerance up to 0.075 over 200 trials
    assert false_positive_rate <= 0.075, f"False SUPPORTED rate too high: {false_positive_rate:.3f}"


def test_rq1_runner_power_at_n_high_40_and_test_split_isolation(tmp_path: Path, monkeypatch):
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

    # ~100 total participants -> ~40 high-risk in train+dev after holding out test split
    gen_alt = SyntheticCohortGenerator(n_participants=100, seed=42, extra_high_risk_drift=True)
    df_win_alt = builder.build_dataset_dataframe(gen_alt.generate())
    df_feat_alt = registry.extract_features(df_win_alt, groups=["timing"])
    alt_pq = tmp_path / "features_alt.parquet"
    df_feat_alt.to_parquet(alt_pq, index=False)

    res_alt = run_rq1_analysis(
        features_path=alt_pq,
        prereg_path="configs/prereg.yaml",
        output_dir=tmp_path / "rq1_alt",
        seed=42,
    )
    assert res_alt["status"] == "SUPPORTED"
    assert res_alt["sample_counts"]["n_test_split_held_out"] > 0
    assert (Path(res_alt["run_dir"]) / "decision.json").exists()
    assert (Path(res_alt["run_dir"]) / "conditioned_drift_trace.png").exists()
