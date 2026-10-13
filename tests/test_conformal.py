"""Tests verifying finite-sample coverage, interval monotonicity, small-n guards, and triage logic."""

import numpy as np

from anjaney.conformal.policy import ConformalTriagePolicy, TriageDecision
from anjaney.conformal.split import SplitConformalRegressor


def test_split_conformal_finite_sample_coverage():
    rng = np.random.default_rng(1234)
    # 100 exchangeable calibration participants
    y_true = rng.normal(10.0, 3.0, size=100)
    y_pred = y_true + rng.normal(0.0, 1.0, size=100)

    # 1000 new test participants
    y_test_true = rng.normal(10.0, 3.0, size=1000)
    y_test_pred = y_test_true + rng.normal(0.0, 1.0, size=1000)

    conformal = SplitConformalRegressor(alpha=0.10)
    conformal.calibrate(y_true, y_pred)

    lowers, uppers = conformal.predict_intervals(y_test_pred)
    empirical_coverage = np.mean((y_test_true >= lowers) & (y_test_true <= uppers))

    # Empirical coverage near 90% (allowing binomial and calibration variance on n=100)
    assert 0.85 <= empirical_coverage <= 0.96


def test_split_conformal_intervals_nest_monotonically():
    y_true = np.array([2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 21.0, 23.0])
    y_pred = np.array([3.0, 4.0, 9.0, 11.0, 16.0, 17.0, 22.0, 22.0])

    c_90 = SplitConformalRegressor(alpha=0.10)
    c_80 = SplitConformalRegressor(alpha=0.20)

    q_90 = c_90.calibrate(y_true, y_pred)
    q_80 = c_80.calibrate(y_true, y_pred)

    # Stricter coverage requirement (alpha=0.10 vs 0.20) requires larger or equal quantile
    assert q_90 >= q_80


def test_small_n_guard():
    # Only 3 calibration points with alpha = 0.05
    # ceil((3 + 1) * 0.95) = ceil(3.8) = 4 > 3 -> impossible finite-sample guarantee
    y_true = np.array([5.0, 10.0, 15.0])
    y_pred = np.array([5.2, 9.8, 14.9])

    conformal = SplitConformalRegressor(alpha=0.05)
    q_hat = conformal.calibrate(y_true, y_pred)

    assert np.isinf(q_hat)
    low, up = conformal.predict_intervals(np.array([10.0]))
    assert np.isneginf(low[0])
    assert np.isposinf(up[0])


def test_conformal_triage_policy_mutually_exclusive():
    policy = ConformalTriagePolicy(threshold=10.0, max_interval_width=6.0, margin=0.5)

    # 1. Clear low-risk: interval safely below 10
    d1 = policy.evaluate_interval(3.0, 7.0)
    assert d1 == TriageDecision.LOW_RISK_TREND

    # 2. Clear elevated: interval safely above 10
    d2 = policy.evaluate_interval(12.0, 16.0)
    assert d2 == TriageDecision.ELEVATED_TREND

    # 3. Straddles threshold: interval contains 10 -> Escalate
    d3 = policy.evaluate_interval(8.0, 12.0)
    assert d3 == TriageDecision.ESCALATE_TO_HUMAN

    # 4. Too wide: width 8.0 > max_width 6.0 -> Escalate
    d4 = policy.evaluate_interval(1.0, 9.0)
    assert d4 == TriageDecision.ESCALATE_TO_HUMAN
