"""Cross-conformal (CV+) prediction intervals pooling all calibration scores across folds."""

import numpy as np


class CrossConformalRegressor:
    """Aggregates out-of-fold nonconformity scores across CV folds."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.pooled_residuals: list[float] = []
        self.q_hat: float = 0.0

    def add_fold_residuals(self, y_true_fold: np.ndarray, y_pred_fold: np.ndarray) -> None:
        residuals = np.abs(y_true_fold - y_pred_fold).tolist()
        self.pooled_residuals.extend(residuals)

    def finalize_calibration(self) -> float:
        n = len(self.pooled_residuals)
        if n == 0:
            raise ValueError("No residuals collected for CV+.")

        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        if k > n:
            self.q_hat = float("inf")
            return self.q_hat

        sorted_res = np.sort(np.array(self.pooled_residuals))
        self.q_hat = float(sorted_res[k - 1])
        return self.q_hat

    def predict_intervals(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_p = np.asarray(y_pred, dtype=float)
        return y_p - self.q_hat, y_p + self.q_hat
