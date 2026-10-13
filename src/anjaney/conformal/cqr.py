"""Conformalized Quantile Regression (Romano, Patterson, Candès, 2019)."""

import lightgbm as lgb
import numpy as np


class ConformalQuantileRegressor:
    """Heteroscedastic prediction intervals via pinball loss regression."""

    def __init__(self, alpha: float = 0.10, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.q_low_model = None
        self.q_high_model = None
        self.conformal_offset: float = 0.0
        self.n_calib: int = 0
        self.is_infinite: bool = False

    def fit_heads(self, X_train: np.ndarray, y_train: np.ndarray) -> "ConformalQuantileRegressor":
        """Fit lower (alpha/2) and upper (1 - alpha/2) quantile regression heads."""
        low_quantile = self.alpha / 2.0
        high_quantile = 1.0 - (self.alpha / 2.0)

        self.q_low_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=low_quantile,
            n_estimators=50,
            num_leaves=7,
            min_child_samples=4,
            random_state=self.seed,
            verbose=-1,
        )
        self.q_high_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=high_quantile,
            n_estimators=50,
            num_leaves=7,
            min_child_samples=4,
            random_state=self.seed,
            verbose=-1,
        )

        self.q_low_model.fit(X_train, y_train)
        self.q_high_model.fit(X_train, y_train)
        return self

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray) -> float:
        """Compute the CQR nonconformity score E_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))."""
        if self.q_low_model is None or self.q_high_model is None:
            raise RuntimeError("Quantile heads must be fit before calibration.")

        self.n_calib = len(y_calib)
        q_low_preds = self.q_low_model.predict(X_calib)
        q_high_preds = self.q_high_model.predict(X_calib)

        scores = np.maximum(q_low_preds - y_calib, y_calib - q_high_preds)

        k = int(np.ceil((self.n_calib + 1) * (1.0 - self.alpha)))
        if k > self.n_calib:
            self.is_infinite = True
            self.conformal_offset = float("inf")
            return self.conformal_offset

        self.is_infinite = False
        sorted_scores = np.sort(scores)
        self.conformal_offset = float(sorted_scores[k - 1])
        return self.conformal_offset

    def predict_intervals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construct calibrated asymmetric prediction intervals."""
        if self.q_low_model is None or self.q_high_model is None:
            raise RuntimeError("Model is uncalibrated.")

        q_low = self.q_low_model.predict(X)
        q_high = self.q_high_model.predict(X)

        if self.is_infinite:
            return np.full_like(q_low, -np.inf), np.full_like(q_high, np.inf)

        lower = q_low - self.conformal_offset
        upper = q_high + self.conformal_offset
        return lower, upper
