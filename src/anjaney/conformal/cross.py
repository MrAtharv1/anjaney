"""CV+ (Cross-Conformal / Jackknife+-after-K-Fold) regression engine (Barber et al., 2021)."""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class CrossConformalCVPlus:
    """Barber et al. (2021) CV+ interval estimator for participant-level data.

    Theoretical Guarantee:
        Barber et al. (2021) prove worst-case finite-sample coverage >= 1 - 2*alpha
        (and empirical coverage typically ~ 1 - alpha).
        Lower bound is the floor(alpha * (n + 1))-th smallest candidate (1-indexed),
        which is 0-indexed `floor(alpha * (n + 1)) - 1` (or -inf if < 0).
        Upper bound is the ceil((1 - alpha) * (n + 1))-th smallest candidate (1-indexed),
        which is 0-indexed `ceil((1 - alpha) * (n + 1)) - 1` (or +inf if >= n).
    """

    def __init__(
        self,
        alpha: float = 0.10,
        n_folds: int = 5,
        seed: int = 42,
        standardize: bool = True,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.n_folds = n_folds
        self.seed = seed
        self.standardize = standardize

        self.fold_models: list[tuple[StandardScaler | None, Ridge]] = []
        self.oof_residuals: np.ndarray | None = None
        self.sample_fold_idx: np.ndarray | None = None
        self.n_samples: int = 0
        self.is_infinite: bool = False

    def fit_calibrate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | list[str] | None = None,
        base_estimator=None,
        cv_splitter=None,
    ) -> "CrossConformalCVPlus":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if not (np.all(np.isfinite(X_arr)) and np.all(np.isfinite(y_arr))):
            raise ValueError("CV+ inputs must be finite.")

        self.n_samples = len(y_arr)
        if self.n_samples < self.n_folds:
            raise ValueError(f"Requires at least {self.n_folds} samples, got {self.n_samples}.")

        # AGENTS.md Rule 4: Enforce one row per unique participant
        if groups is not None:
            g_arr = np.asarray(groups)
            if len(g_arr) != self.n_samples:
                raise ValueError("Length of groups must match X and y.")
            if len(np.unique(g_arr)) != self.n_samples:
                raise ValueError(
                    "AGENTS.md Rule 4 violation: Duplicate participant IDs detected in CV+ calibration. "
                    "Aggregate to one row per participant before calibrating."
                )

        k_up_1idx = int(np.ceil((self.n_samples + 1) * (1.0 - self.alpha)))
        self.is_infinite = k_up_1idx > self.n_samples

        splitter = (
            cv_splitter
            if cv_splitter is not None
            else KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        )
        self.fold_models = []
        self.oof_residuals = np.zeros(self.n_samples, dtype=float)
        self.sample_fold_idx = np.zeros(self.n_samples, dtype=int)

        for fold_k, (tr_idx, val_idx) in enumerate(splitter.split(X_arr)):
            if self.standardize:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_arr[tr_idx])
                X_val_s = scaler.transform(X_arr[val_idx])
            else:
                scaler = None
                X_tr_s = X_arr[tr_idx]
                X_val_s = X_arr[val_idx]

            model = (
                clone(base_estimator)
                if base_estimator is not None
                else Ridge(alpha=10.0, random_state=self.seed)
            )
            model.fit(X_tr_s, y_arr[tr_idx])
            preds = model.predict(X_val_s)

            self.fold_models.append((scaler, model))
            self.oof_residuals[val_idx] = np.abs(y_arr[val_idx] - preds)
            self.sample_fold_idx[val_idx] = fold_k

        return self

    def predict_intervals(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.oof_residuals is None or self.sample_fold_idx is None or not self.fold_models:
            raise RuntimeError("CrossConformalCVPlus must be calibrated via fit_calibrate first.")

        X_te = np.asarray(X_test, dtype=float)
        if X_te.ndim == 1:
            X_te = X_te.reshape(1, -1)

        n_test = len(X_te)
        if self.is_infinite:
            return np.full(n_test, -np.inf), np.full(n_test, np.inf)

        fold_preds = np.vstack(
            [
                model.predict(scaler.transform(X_te) if scaler is not None else X_te)
                for scaler, model in self.fold_models
            ]
        )

        sample_preds = fold_preds[self.sample_fold_idx, :]
        res_col = self.oof_residuals[:, None]

        upper_candidates = np.sort(sample_preds + res_col, axis=0)
        lower_candidates = np.sort(sample_preds - res_col, axis=0)

        # 1-indexed order statistics converted to 0-indexed positions
        k_up_1idx = int(np.ceil((self.n_samples + 1) * (1.0 - self.alpha)))
        k_lo_1idx = int(np.floor((self.n_samples + 1) * self.alpha))

        if k_up_1idx > self.n_samples:
            uppers = np.full(n_test, np.inf)
        else:
            uppers = upper_candidates[k_up_1idx - 1, :]

        if k_lo_1idx < 1:
            lowers = np.full(n_test, -np.inf)
        else:
            lowers = lower_candidates[k_lo_1idx - 1, :]

        return lowers, uppers


CrossConformalRegressor = CrossConformalCVPlus
