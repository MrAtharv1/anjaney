"""Conformalized Quantile Regression (CQR) with PCA-Linear Quantile heads and crossing guards."""

import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler


class ConformalQuantileRegressor:
    """Romano et al. (2019) CQR adapted for small-n clinical cohorts."""

    def __init__(
        self,
        alpha: float = 0.10,
        n_pca_components: int = 10,
        head_type: str = "linear_qr",
        seed: int = 42,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.n_pca = n_pca_components
        self.head_type = head_type
        self.seed = seed

        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.q_low_model = None
        self.q_high_model = None
        self.conformal_offset: float | None = None
        self.n_calib: int = 0
        self.is_infinite: bool = False

    def _fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.fit_transform(X)
        n_comp = min(self.n_pca, X_scaled.shape[0] - 1, X_scaled.shape[1])
        if n_comp >= 2:
            self.pca = PCA(n_components=n_comp, random_state=self.seed)
            return self.pca.fit_transform(X_scaled)
        self.pca = None
        return X_scaled

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled

    def _predict_sorted_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_low_model is None or self.q_high_model is None:
            raise RuntimeError("Quantile heads must be fitted first.")
        Z = self._transform_features(X)
        raw_low = self.q_low_model.predict(Z)
        raw_high = self.q_high_model.predict(Z)
        # Prevent quantile crossing (guarantee q_low <= q_high per row)
        q_low = np.minimum(raw_low, raw_high)
        q_high = np.maximum(raw_low, raw_high)
        return q_low, q_high

    def fit_heads(self, X_train: np.ndarray, y_train: np.ndarray) -> "ConformalQuantileRegressor":
        Z_train = self._fit_transform_features(np.asarray(X_train, dtype=float))
        y_tr = np.asarray(y_train, dtype=float)

        # Use 10th / 90th inner quantiles on small samples so pinball loss is well-supported
        low_q = max(0.10, self.alpha / 2.0)
        high_q = min(0.90, 1.0 - (self.alpha / 2.0))

        if self.head_type == "lightgbm":
            self.q_low_model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=low_q,
                n_estimators=25,
                num_leaves=5,
                min_child_samples=6,
                random_state=self.seed,
                verbose=-1,
            )
            self.q_high_model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=high_q,
                n_estimators=25,
                num_leaves=5,
                min_child_samples=6,
                random_state=self.seed,
                verbose=-1,
            )
        else:
            self.q_low_model = QuantileRegressor(quantile=low_q, alpha=0.05, solver="highs")
            self.q_high_model = QuantileRegressor(quantile=high_q, alpha=0.05, solver="highs")

        self.q_low_model.fit(Z_train, y_tr)
        self.q_high_model.fit(Z_train, y_tr)
        return self

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray) -> float:
        if self.q_low_model is None or self.q_high_model is None:
            raise RuntimeError("Quantile heads must be fit before calibration.")

        y_cal = np.asarray(y_calib, dtype=float)
        self.n_calib = len(y_cal)
        if self.n_calib == 0:
            raise ValueError("Calibration set cannot be empty.")

        q_low_preds, q_high_preds = self._predict_sorted_quantiles(np.asarray(X_calib, dtype=float))
        scores = np.maximum(q_low_preds - y_cal, y_cal - q_high_preds)

        k = int(np.ceil((self.n_calib + 1) * (1.0 - self.alpha)))
        if k > self.n_calib:
            self.is_infinite = True
            self.conformal_offset = float("inf")
            return self.conformal_offset

        self.is_infinite = False
        self.conformal_offset = float(np.sort(scores)[k - 1])
        return self.conformal_offset

    def predict_intervals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.conformal_offset is None:
            raise RuntimeError("CQR must be calibrated before calling predict_intervals.")

        q_low, q_high = self._predict_sorted_quantiles(np.asarray(X, dtype=float))
        if self.is_infinite:
            return np.full_like(q_low, -np.inf), np.full_like(q_high, np.inf)

        return q_low - self.conformal_offset, q_high + self.conformal_offset
