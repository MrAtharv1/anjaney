"""Standard split conformal prediction for regression with finite-sample correction and finite checks."""

import numpy as np


class SplitConformalRegressor:
    """Standard split-conformal regression engine.

    Statistical Assumption:
        Participant-level exchangeability: Each participant contributes ONE
        calibration residual score. The 1 - alpha guarantee applies to a newly
        arriving participant, not an individual turn.
    """

    def __init__(self, alpha: float = 0.10, clip_bounds: tuple[float, float] | None = (0.0, 24.0)):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Alpha must be in (0, 1), received {alpha}")
        self.alpha = alpha
        self.clip_bounds = clip_bounds
        self.q_hat: float | None = None
        self.n_calib: int = 0
        self.is_infinite: bool = False

    def calibrate(self, y_calib_true: np.ndarray, y_calib_pred: np.ndarray) -> float:
        y_true = np.asarray(y_calib_true, dtype=float)
        y_pred = np.asarray(y_calib_pred, dtype=float)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
        if len(y_true) == 0:
            raise ValueError("Calibration set cannot be empty.")
        if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
            raise ValueError("Calibration inputs contain non-finite (NaN or Inf) values.")

        self.n_calib = len(y_true)
        residuals = np.abs(y_true - y_pred)

        k = int(np.ceil((self.n_calib + 1) * (1.0 - self.alpha)))
        if k > self.n_calib:
            self.is_infinite = True
            self.q_hat = float("inf")
            return self.q_hat

        self.is_infinite = False
        sorted_residuals = np.sort(residuals)
        self.q_hat = float(sorted_residuals[k - 1])
        return self.q_hat

    def predict_intervals(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_hat is None:
            raise RuntimeError("Model must be calibrated before predicting intervals.")

        y_p = np.asarray(y_pred, dtype=float)
        if self.is_infinite:
            return np.full_like(y_p, -np.inf), np.full_like(y_p, np.inf)

        lower = y_p - self.q_hat
        upper = y_p + self.q_hat

        # Non-finite predictions produce infinite intervals so policy escalates
        non_finite_mask = ~np.isfinite(y_p)
        if np.any(non_finite_mask):
            lower = np.where(non_finite_mask, -np.inf, lower)
            upper = np.where(non_finite_mask, np.inf, upper)

        if self.clip_bounds is not None:
            lo_clip, hi_clip = self.clip_bounds
            finite_mask = np.isfinite(lower) & np.isfinite(upper)
            lower = np.where(finite_mask, np.clip(lower, lo_clip, hi_clip), lower)
            upper = np.where(finite_mask, np.clip(upper, lo_clip, hi_clip), upper)

        return lower, upper
