"""Conformal classification for binary screening (PHQ-8 >= 10)."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class ConformalClassifier:
    """Produces prediction sets {0}, {1}, or {0, 1} with guaranteed coverage."""

    def __init__(self, alpha: float = 0.10, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.model = LogisticRegression(random_state=seed)
        self.tau_hat: float = 0.0

    def fit(self, X_train: np.ndarray, y_bin_train: np.ndarray) -> "ConformalClassifier":
        self.model.fit(X_train, y_bin_train)
        return self

    def calibrate(self, X_calib: np.ndarray, y_bin_calib: np.ndarray) -> float:
        """Calibrate using LAC (Least Ambiguous Classifier) nonconformity scores: 1 - P(Y_true)."""
        probs = self.model.predict_proba(X_calib)
        n = len(y_bin_calib)

        scores = [1.0 - probs[i, int(y_bin_calib[i])] for i in range(n)]

        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        if k > n:
            self.tau_hat = 1.0
            return self.tau_hat

        sorted_scores = np.sort(scores)
        self.tau_hat = float(sorted_scores[k - 1])
        return self.tau_hat

    def predict_sets(self, X: np.ndarray) -> list[set[int]]:
        """Return the prediction set for each participant."""
        probs = self.model.predict_proba(X)
        pred_sets = []

        for i in range(len(X)):
            s = set()
            for label in [0, 1]:
                score = 1.0 - probs[i, label]
                if score <= self.tau_hat:
                    s.add(label)
            if not s:
                s.add(int(np.argmax(probs[i])))
            pred_sets.append(s)

        return pred_sets
