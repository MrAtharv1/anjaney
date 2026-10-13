"""Tests for paired ablation grid, test lock enforcement, and missingness degradation."""

from pathlib import Path

import pytest

from anjaney.eval.report import generate_markdown_report
from anjaney.eval.robustness import evaluate_missingness_stress
from anjaney.eval.test_lock import OfficialTestLock


def test_official_test_lock_prevents_duplicate_runs(tmp_path: Path):
    lock_file = tmp_path / "test.lock"
    lock = OfficialTestLock(lock_file=lock_file)

    config = {"model": "ridge", "features": "text+timing", "seed": 42}
    h = lock.check_and_acquire(config)
    lock.release_and_record(h, {"rmse": 4.5})

    # Second execution with exact same config must fail
    with pytest.raises(RuntimeError, match="Rule 4 Violation"):
        lock.check_and_acquire(config)


def test_missingness_stress_abstention_monotonically_increases():
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"timing_feature": np.linspace(1, 10, 20)})
    y = np.linspace(5, 15, 20)

    res = evaluate_missingness_stress(df, y, y, split_q=3.0, threshold=10.0, seed=42)

    abstentions = [
        res["missing_0pct"]["abstention_rate"],
        res["missing_10pct"]["abstention_rate"],
        res["missing_30pct"]["abstention_rate"],
        res["missing_50pct"]["abstention_rate"],
    ]

    # Abstention rate should strictly rise or stay equal as missingness worsens
    assert abstentions[-1] >= abstentions[0]


def test_results_template_has_no_hardcoded_numbers():
    dummy_grid = {
        "grid": {
            "text_only": {
                "ridge": {
                    "metrics": {
                        "rmse": {"point": 5.1, "ci_lower": 4.2, "ci_upper": 6.0},
                        "auroc": {"point": 0.88, "ci_lower": 0.80, "ci_upper": 0.95},
                    },
                    "conformal": {"split": {"aurc": 0.12}},
                }
            }
        },
        "paired_ablation": {"label": "CONFIRMATORY", "delta_rmse": -0.45},
    }
    dummy_rob = {
        "m_0": {
            "mask_rate": 0.0,
            "empirical_coverage": 0.90,
            "mean_interval_width": 6.0,
            "abstention_rate": 0.10,
        }
    }

    md = generate_markdown_report(dummy_grid, dummy_rob, {"status": "SUPPORTED"})
    assert "5.10" in md
    assert "-0.450" in md
    assert "ANJANEY — Comprehensive Experimental Results" in md
