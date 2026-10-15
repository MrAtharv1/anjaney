"""Tests for Split Conformal, CQR, Difficulty-Normalized, CV+, MAPIE parity, and fail-closed guards."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from anjaney.conformal.cqr import ConformalQuantileRegressor
from anjaney.conformal.cross import CrossConformalCVPlus
from anjaney.conformal.normalized import DifficultyNormalizedConformal
from anjaney.conformal.policy import ConformalTriagePolicy, TriageDecision
from anjaney.conformal.split import SplitConformalRegressor


def test_predict_before_calibrate_raises_across_all_engines():
    X = np.random.randn(20, 6)
    y = np.random.randn(20)

    split_eng = SplitConformalRegressor(alpha=0.10)
    with pytest.raises(RuntimeError, match="calibrated"):
        split_eng.predict_intervals(y)

    cqr_eng = ConformalQuantileRegressor(alpha=0.10)
    cqr_eng.fit_heads(X, y)
    with pytest.raises(RuntimeError, match="calibrated"):
        cqr_eng.predict_intervals(X)

    norm_eng = DifficultyNormalizedConformal(alpha=0.10)
    norm_eng.fit_variance_model(X, y)
    with pytest.raises(RuntimeError, match="calibrated"):
        norm_eng.predict_intervals(X, y)

    cv_eng = CrossConformalCVPlus(alpha=0.10)
    with pytest.raises(RuntimeError, match="calibrated"):
        cv_eng.predict_intervals(X)


def test_repeated_splits_empirical_coverage_matches_nominal():
    """Verify empirical coverage over 100 repeated random participant splits matches 1 - alpha within binomial SE."""
    rng = np.random.default_rng(42)
    alpha = 0.10
    coverages = []

    for _ in range(100):
        y_true = rng.uniform(2.0, 20.0, size=120)
        y_pred = y_true + rng.normal(0.0, 2.5, size=120)

        perm = rng.permutation(120)
        cal_idx, test_idx = perm[:60], perm[60:]

        eng = SplitConformalRegressor(alpha=alpha, clip_bounds=None)
        eng.calibrate(y_true[cal_idx], y_pred[cal_idx])
        lo, hi = eng.predict_intervals(y_pred[test_idx])
        coverages.append(np.mean((y_true[test_idx] >= lo) & (y_true[test_idx] <= hi)))

    mean_cov = float(np.mean(coverages))
    # Expected coverage for split conformal is in [1 - alpha, 1 - alpha + 1/(n_cal + 1)] = [0.90, 0.916]
    assert 0.88 <= mean_cov <= 0.94, f"Empirical coverage {mean_cov:.3f} outside nominal bounds"


def test_mapie_cross_check_equivalence():
    """Cross-check SplitConformalRegressor nonconformity quantile against exact Lei et al. (2018) order statistic and MAPIE."""
    rng = np.random.default_rng(99)
    X_tr = rng.normal(size=(50, 4))
    X_cal = rng.normal(size=(50, 4))
    beta = np.array([1.5, -0.8, 0.5, 0.2])
    y_tr = X_tr @ beta + rng.normal(0, 1.5, size=50)
    y_cal = X_cal @ beta + rng.normal(0, 1.5, size=50)

    reg = Ridge(alpha=1.0).fit(X_tr, y_tr)
    y_pred_cal = reg.predict(X_cal)

    alpha = 0.10
    split_eng = SplitConformalRegressor(alpha=alpha, clip_bounds=None)
    q_ours = split_eng.calibrate(y_cal, y_pred_cal)

    # 1. Exact order statistic from Lei et al. (2018): k = ceil((n + 1) * (1 - alpha))
    residuals = np.abs(y_cal - y_pred_cal)
    n = len(y_cal)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    # In numpy.quantile(method='higher'), index = ceil(q * (n - 1)), so q = (k - 1) / (n - 1)
    q_expected = float(np.quantile(residuals, (k - 1) / (n - 1), method="higher"))
    assert np.isclose(q_ours, q_expected, atol=1e-6)

    # 2. If MAPIE is installed, cross-check interval width directly against MapieRegressor
    try:
        from mapie.regression import MapieRegressor

        mapie = MapieRegressor(estimator=reg, cv="prefit", method="base")
        mapie.fit(X_cal, y_cal)
        _, y_pis = mapie.predict(X_cal[:5], alpha=alpha)
        mapie_half_width = float((y_pis[0, 1, 0] - y_pis[0, 0, 0]) / 2.0)
        assert np.isclose(q_ours, mapie_half_width, atol=1e-5)
    except ImportError:
        pass


def test_cqr_no_quantile_crossing_and_valid_coverage():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(90, 12))
    y = 8.0 + 2.0 * X[:, 0] + rng.normal(0, 1.5, size=90)

    cqr = ConformalQuantileRegressor(alpha=0.10, n_pca_components=5, head_type="linear_qr")
    cqr.fit_heads(X[:45], y[:45])
    cqr.calibrate(X[45:70], y[45:70])
    lo, hi = cqr.predict_intervals(X[70:])

    assert np.all(hi >= lo)
    assert np.mean((y[70:] >= lo) & (y[70:] <= hi)) >= 0.80


def test_cv_plus_cross_conformal_intervals():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 8))
    y = 10.0 + 1.5 * X[:, 0] + rng.normal(0, 2.0, size=60)

    cv_plus = CrossConformalCVPlus(alpha=0.10, n_folds=5, seed=42)
    cv_plus.fit_calibrate(X[:45], y[:45])
    lo, hi = cv_plus.predict_intervals(X[45:])

    assert len(lo) == 15
    assert np.all(hi > lo)


def test_small_n_guard():
    eng = SplitConformalRegressor(alpha=0.10)
    q = eng.calibrate(np.array([5.0, 10.0, 15.0]), np.array([6.0, 9.0, 14.0]))
    assert np.isinf(q)


def test_conformal_policy_fails_closed_on_nan_and_inf():
    policy = ConformalTriagePolicy(threshold=10.0, max_interval_width=6.0, margin=0.5)
    assert policy.evaluate_interval(float("nan"), 5.0) == TriageDecision.ESCALATE_TO_HUMAN
    assert policy.evaluate_interval(2.0, float("nan")) == TriageDecision.ESCALATE_TO_HUMAN
    assert policy.evaluate_interval(-float("inf"), 5.0) == TriageDecision.ESCALATE_TO_HUMAN
    assert policy.evaluate_interval(8.0, 4.0) == TriageDecision.ESCALATE_TO_HUMAN


def test_cv_plus_small_n_off_by_one_and_mapie_plus_parity():
    """Verify exact 1-indexed floor(alpha*(n+1)) - 1 order statistic at n=19, alpha=0.10 and group uniqueness check."""
    rng = np.random.default_rng(123)
    n = 19
    alpha = 0.10
    X = rng.normal(size=(n, 3))
    y = 10.0 + 2.0 * X[:, 0] + rng.normal(0, 1.0, size=n)
    X_test = rng.normal(size=(5, 3))

    # 1. Duplicate participant IDs must raise ValueError (AGENTS.md Rule 4)
    dup_groups = [f"P_{i % 10}" for i in range(n)]
    cv_dup = CrossConformalCVPlus(alpha=alpha, n_folds=5, seed=42)
    with pytest.raises(ValueError, match="Duplicate participant IDs"):
        cv_dup.fit_calibrate(X, y, groups=dup_groups)

    # 2. Unique participant IDs succeed and match exact 0-indexed k_lo - 1 = 1 and k_up - 1 = 17
    unique_groups = [f"P_{i:03d}" for i in range(n)]
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    base_reg = Ridge(alpha=1.0, random_state=42)

    cv_plus = CrossConformalCVPlus(alpha=alpha, n_folds=5, seed=42, standardize=False)
    cv_plus.fit_calibrate(X, y, groups=unique_groups, base_estimator=base_reg, cv_splitter=kf)
    lo_ours, hi_ours = cv_plus.predict_intervals(X_test)

    # Verify k_lo = floor(0.10 * 20) = 2 (1-indexed) -> index 1 (0-indexed)
    # Verify k_up = ceil(0.90 * 20) = 18 (1-indexed) -> index 17 (0-indexed)
    fold_preds = np.vstack([m.predict(X_test) for _, m in cv_plus.fold_models])
    sample_preds = fold_preds[cv_plus.sample_fold_idx, :]
    expected_lo = np.sort(sample_preds - cv_plus.oof_residuals[:, None], axis=0)[1, :]
    expected_hi = np.sort(sample_preds + cv_plus.oof_residuals[:, None], axis=0)[17, :]

    assert np.allclose(lo_ours, expected_lo, atol=1e-6)
    assert np.allclose(hi_ours, expected_hi, atol=1e-6)

    # 3. Cross-check against MAPIE method='plus' if installed
    try:
        from mapie.regression import MapieRegressor

        mapie_plus = MapieRegressor(estimator=base_reg, method="plus", cv=kf)
        mapie_plus.fit(X, y)
        _, y_pis = mapie_plus.predict(X_test, alpha=alpha)
        assert np.allclose(lo_ours, y_pis[:, 0, 0], atol=1e-5)
        assert np.allclose(hi_ours, y_pis[:, 1, 0], atol=1e-5)
    except ImportError:
        pass
