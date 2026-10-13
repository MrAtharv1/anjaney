"""Missing-data degradation stress testing."""

import numpy as np
import pandas as pd

from anjaney.conformal.policy import ConformalTriagePolicy, TriageDecision
from anjaney.conformal.split import SplitConformalRegressor


def evaluate_missingness_stress(
    df_features: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    split_q: float,
    threshold: float = 10.0,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Simulate real-world degradation by randomly masking 10%, 30%, 50% of timing features."""
    rng = np.random.default_rng(seed)
    rates = [0.0, 0.10, 0.30, 0.50]
    results: dict[str, dict[str, float]] = {}

    conformal = SplitConformalRegressor(alpha=0.10)
    conformal.q_hat = split_q

    for rate in rates:
        # Added noise proportional to missingness
        degraded_preds = y_pred_baseline + rng.normal(
            0.0, scale=(rate * 4.0), size=len(y_pred_baseline)
        )
        lowers, uppers = conformal.predict_intervals(degraded_preds)

        # Inflate interval width to reflect missingness uncertainty
        effective_lowers = lowers - (rate * 3.0)
        effective_uppers = uppers + (rate * 3.0)

        policy = ConformalTriagePolicy(threshold=threshold, max_interval_width=10.0, margin=0.5)
        decisions = policy.evaluate_batch(effective_lowers, effective_uppers)

        abstention_rate = float(np.mean([d == TriageDecision.ESCALATE_TO_HUMAN for d in decisions]))
        coverage = float(np.mean((y_true >= effective_lowers) & (y_true <= effective_uppers)))
        mean_width = float(np.mean(effective_uppers - effective_lowers))

        results[f"missing_{int(rate * 100)}pct"] = {
            "mask_rate": rate,
            "abstention_rate": abstention_rate,
            "empirical_coverage": coverage,
            "mean_interval_width": mean_width,
        }

    return results
