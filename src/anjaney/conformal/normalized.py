"""Difficulty-normalized split conformal regression using out-of-fold residuals and summary features."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class DifficultyNormalizedConformal:
    """Adapts interval width using a low-dimensional difficulty estimator fit on OOF residuals."""

    def __init__(
        self,
        alpha: float = 0.10,
        seed: int = 42,
        sigma_floor_fraction: float = 0.20,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.seed = seed
        self.sigma_floor_fraction = sigma_floor_fraction
        self.sigma_model = Ridge(alpha=10.0, random_state=seed)
        self.sigma_floor: float | None = None
        self.q_hat: float | None = None
        self.n_calib: int = 0
        self.is_infinite: bool = False

    @staticmethod
    def extract_summary_difficulty_features(X: np.ndarray) -> np.ndarray:
        """Compress high-dimensional feature matrix into 4 low-variance summary statistics:
        [finite_completeness_ratio, l2_norm, feature_dispersion_std, max_abs_z].
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        finite_mask = np.isfinite(X_arr)
        completeness = np.mean(finite_mask, axis=1, keepdims=True)
        X_safe = np.where(finite_mask, X_arr, 0.0)

        l2_norm = np.linalg.norm(X_safe, axis=1, keepdims=True) / np.sqrt(max(1, X_arr.shape[1]))
        dispersion = np.std(X_safe, axis=1, keepdims=True)
        max_abs = np.max(np.abs(X_safe), axis=1, keepdims=True)

        return np.hstack([completeness, l2_norm, dispersion, max_abs])

    def fit_variance_model(
        self,
        X_train: np.ndarray,
        y_train_true: np.ndarray,
        y_train_oof_pred: np.ndarray | None = None,
    ) -> "DifficultyNormalizedConformal":
        """Fit variance estimator strictly on OUT-OF-FOLD residuals from the training partition."""
        X_tr = np.asarray(X_train, dtype=float)
        y_tr = np.asarray(y_train_true, dtype=float)

        if y_train_oof_pred is None:
            # Generate internal 5-fold OOF predictions if caller did not supply precomputed OOF preds
            oof_preds = np.zeros_like(y_tr)
            kf = KFold(n_splits=min(5, len(y_tr)), shuffle=True, random_state=self.seed)
            for tr_idx, val_idx in kf.split(X_tr):
                base_reg = Ridge(alpha=10.0, random_state=self.seed)
                base_reg.fit(X_tr[tr_idx], y_tr[tr_idx])
                oof_preds[val_idx] = base_reg.predict(X_tr[val_idx])
        else:
            oof_preds = np.asarray(y_train_oof_pred, dtype=float)

        abs_oof_residuals = np.abs(y_tr - oof_preds)
        med_resid = float(np.median(abs_oof_residuals))
        self.sigma_floor = max(self.sigma_floor_fraction * med_resid, 0.25)

        S_train = self.extract_summary_difficulty_features(X_tr)
        self.sigma_model.fit(S_train, abs_oof_residuals)
        return self

    def _predict_sigma(self, X: np.ndarray) -> np.ndarray:
        if self.sigma_floor is None:
            raise RuntimeError("Variance model must be fitted before predicting sigma.")
        S = self.extract_summary_difficulty_features(X)
        raw_sigma = self.sigma_model.predict(S)
        return np.maximum(self.sigma_floor, raw_sigma)

    def calibrate(
        self, X_calib: np.ndarray, y_calib_true: np.ndarray, y_calib_pred: np.ndarray
    ) -> float:
        if self.sigma_floor is None:
            raise RuntimeError("Call fit_variance_model before calibrate.")

        y_true = np.asarray(y_calib_true, dtype=float)
        y_pred = np.asarray(y_calib_pred, dtype=float)
        if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
            raise ValueError("Calibration inputs contain non-finite values.")

        self.n_calib = len(y_true)
        if self.n_calib == 0:
            raise ValueError("Calibration set cannot be empty.")

        sigmas = self._predict_sigma(X_calib)
        normalized_scores = np.abs(y_true - y_pred) / sigmas

        k = int(np.ceil((self.n_calib + 1) * (1.0 - self.alpha)))
        if k > self.n_calib:
            self.is_infinite = True
            self.q_hat = float("inf")
            return self.q_hat

        self.is_infinite = False
        self.q_hat = float(np.sort(normalized_scores)[k - 1])
        return self.q_hat

    def predict_intervals(self, X: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_hat is None:
            raise RuntimeError(
                "DifficultyNormalizedConformal must be calibrated before predicting."
            )

        y_p = np.asarray(y_pred, dtype=float)
        if self.is_infinite:
            return np.full_like(y_p, -np.inf), np.full_like(y_p, np.inf)

        margin = self.q_hat * self._predict_sigma(X)
        lower = y_p - margin
        upper = y_p + margin

        bad_mask = ~(np.isfinite(lower) & np.isfinite(upper))
        if np.any(bad_mask):
            lower = np.where(bad_mask, -np.inf, lower)
            upper = np.where(bad_mask, np.inf, upper)

        return lower, upper
