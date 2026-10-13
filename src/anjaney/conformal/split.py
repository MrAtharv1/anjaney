"""Standard split conformal prediction for regression with finite-sample correction."""

import numpy as np


class SplitConformalRegressor:
    """Standard split-conformal regression engine.

    Statistical Assumption:
        Participant-level exchangeability: Each participant contributes exactly one
        calibration pair (x_i, y_i). The finite-sample 1-alpha guarantee applies to
        a newly arriving participant drawn from the same campus distribution.
    """

    def __init__(self, alpha: float = 0.10):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Alpha must be in (0, 1), received {alpha}")
        self.alpha = alpha
        self.q_hat: float | None = None
        self.n_calib: int = 0
        self.is_infinite: bool = False

    def calibrate(self, y_calib_true: np.ndarray, y_calib_pred: np.ndarray) -> float:
        """Compute the empirical nonconformity quantile over calibration participants."""
        y_true = np.asarray(y_calib_true, dtype=float)
        y_pred = np.asarray(y_calib_pred, dtype=float)
        self.n_calib = len(y_true)

        if self.n_calib == 0:
            raise ValueError("Calibration set cannot be empty.")

        residuals = np.abs(y_true - y_pred)

        # Finite-sample quantile index: ceil((n + 1) * (1 - alpha))
        k = int(np.ceil((self.n_calib + 1) * (1.0 - self.alpha)))

        # Small-n guard
        if k > self.n_calib:
            self.is_infinite = True
            self.q_hat = float("inf")
            return self.q_hat

        self.is_infinite = False
        sorted_residuals = np.sort(residuals)
        # 0-indexed lookup for rank k
        self.q_hat = float(sorted_residuals[k - 1])
        return self.q_hat

    def predict_intervals(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Produce [lower_bound, upper_bound] for point predictions."""
        if self.q_hat is None:
            raise RuntimeError("Model must be calibrated before predicting intervals.")

        y_p = np.asarray(y_pred, dtype=float)
        if self.is_infinite:
            lower = np.full_like(y_p, -np.inf)
            upper = np.full_like(y_p, np.inf)
            return lower, upper

        lower = y_p - self.q_hat
        upper = y_p + self.q_hat
        return lower, upper
