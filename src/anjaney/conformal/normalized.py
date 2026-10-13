"""Difficulty-normalized conformal prediction (Lei et al., 2018)."""

import numpy as np
from sklearn.linear_model import Ridge


class DifficultyNormalizedConformal:
    """Scales intervals by an auxiliary model estimating local difficulty/variance."""

    def __init__(self, alpha: float = 0.10, seed: int = 42, epsilon: float = 1e-2):
        self.alpha = alpha
        self.seed = seed
        self.epsilon = epsilon
        self.sigma_model = Ridge(alpha=5.0, random_state=seed)
        self.q_hat: float = 0.0
        self.is_infinite: bool = False

    def fit_variance_model(
        self, X_train: np.ndarray, y_train_true: np.ndarray, y_train_pred: np.ndarray
    ) -> "DifficultyNormalizedConformal":
        abs_residuals = np.abs(y_train_true - y_train_pred)
        self.sigma_model.fit(X_train, abs_residuals)
        return self

    def _predict_sigma(self, X: np.ndarray) -> np.ndarray:
        raw_sigma = self.sigma_model.predict(X)
        return np.maximum(self.epsilon, raw_sigma)

    def calibrate(
        self,
        X_calib: np.ndarray,
        y_calib_true: np.ndarray,
        y_calib_pred: np.ndarray,
    ) -> float:
        n_calib = len(y_calib_true)
        sigmas = self._predict_sigma(X_calib)
        normalized_scores = np.abs(y_calib_true - y_calib_pred) / sigmas

        k = int(np.ceil((n_calib + 1) * (1.0 - self.alpha)))
        if k > n_calib:
            self.is_infinite = True
            self.q_hat = float("inf")
            return self.q_hat

        self.is_infinite = False
        sorted_scores = np.sort(normalized_scores)
        self.q_hat = float(sorted_scores[k - 1])
        return self.q_hat

    def predict_intervals(self, X: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.is_infinite:
            return np.full_like(y_pred, -np.inf), np.full_like(y_pred, np.inf)

        sigmas = self._predict_sigma(X)
        margin = self.q_hat * sigmas
        return y_pred - margin, y_pred + margin
