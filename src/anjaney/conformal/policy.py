"""Three-way triage decision policy and operating point sweeps."""

from enum import StrEnum
from typing import Any

import numpy as np


class TriageDecision(StrEnum):
    LOW_RISK_TREND = "LOW_RISK_TREND"
    ELEVATED_TREND = "ELEVATED_TREND"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


class ConformalTriagePolicy:
    """Maps predictive intervals [lo, hi] to deterministic clinical triage actions."""

    def __init__(
        self, threshold: float = 10.0, max_interval_width: float = 8.0, margin: float = 0.5
    ):
        self.threshold = threshold
        self.w_max = max_interval_width
        self.margin = margin

    def evaluate_interval(self, lower: float, upper: float) -> TriageDecision:
        width = upper - lower

        # Escalation condition: Ambiguity across threshold OR interval is too wide
        crosses_threshold = (lower - self.margin) <= self.threshold <= (upper + self.margin)
        is_too_wide = width > self.w_max

        if crosses_threshold or is_too_wide or np.isinf(width):
            return TriageDecision.ESCALATE_TO_HUMAN

        if lower > self.threshold:
            return TriageDecision.ELEVATED_TREND

        return TriageDecision.LOW_RISK_TREND

    def evaluate_batch(self, lowers: np.ndarray, uppers: np.ndarray) -> list[TriageDecision]:
        return [
            self.evaluate_interval(float(low), float(high))
            for low, high in zip(lowers, uppers, strict=False)
        ]


def sweep_operating_points(
    lowers: np.ndarray,
    uppers: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 10.0,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    cost_abstain: float = 0.5,
) -> dict[str, Any]:
    """Sweep max width and threshold margins to construct Risk-Coverage curves and find optimal operating points."""
    w_max_candidates = np.linspace(4.0, 14.0, 11)
    margin_candidates = [0.0, 0.5, 1.0, 1.5]

    y_true_bin = (y_true >= threshold).astype(int)
    total_samples = len(y_true)

    records: list[dict] = []

    for w_max in w_max_candidates:
        for m in margin_candidates:
            policy = ConformalTriagePolicy(threshold=threshold, max_interval_width=w_max, margin=m)
            decisions = policy.evaluate_batch(lowers, uppers)

            abstained = [d == TriageDecision.ESCALATE_TO_HUMAN for d in decisions]
            abstention_rate = float(np.mean(abstained))
            coverage = 1.0 - abstention_rate

            # Selective risk: Error rate among non-abstained cases
            non_abstained_idx = [i for i, abs_flag in enumerate(abstained) if not abs_flag]
            if len(non_abstained_idx) > 0:
                errors = []
                for idx in non_abstained_idx:
                    pred_binary = 1 if decisions[idx] == TriageDecision.ELEVATED_TREND else 0
                    errors.append(pred_binary != y_true_bin[idx])
                selective_risk = float(np.mean(errors))
            else:
                selective_risk = 0.0

            # Expected cost computation
            total_cost = 0.0
            for i, d in enumerate(decisions):
                true_val = y_true_bin[i]
                if d == TriageDecision.ESCALATE_TO_HUMAN:
                    total_cost += cost_abstain
                elif d == TriageDecision.LOW_RISK_TREND and true_val == 1:
                    total_cost += cost_fn  # False Negative: Missed elevated distress
                elif d == TriageDecision.ELEVATED_TREND and true_val == 0:
                    total_cost += cost_fp  # False Positive

            expected_cost = total_cost / total_samples

            records.append(
                {
                    "max_width": float(w_max),
                    "margin": float(m),
                    "abstention_rate": abstention_rate,
                    "coverage": coverage,
                    "selective_risk": selective_risk,
                    "expected_cost": expected_cost,
                }
            )

    # Compute AURC (Area Under Risk-Coverage Curve)
    sorted_by_cov = sorted(records, key=lambda r: r["coverage"])
    covs = [r["coverage"] for r in sorted_by_cov]
    risks = [r["selective_risk"] for r in sorted_by_cov]
    aurc = float(np.trapezoid(risks, covs)) if len(covs) > 1 else 0.0

    # Best operating point minimizing cost
    best_point = min(records, key=lambda r: r["expected_cost"])

    return {
        "sweep_records": records,
        "aurc": aurc,
        "optimal_operating_point": best_point,
    }
