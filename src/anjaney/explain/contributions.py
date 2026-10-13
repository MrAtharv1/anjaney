"""Exact linear feature attribution grouped into plain-language semantic categories."""

import re
from typing import Any

import numpy as np

BANNED_WORDS = [
    r"\bdiagnos\w*",
    r"\byou have depression\b",
    r"\bclinical depression\b",
    r"\bdepressed\b",
    r"\bdisorder\b",
]


class LinearExplainer:
    def __init__(
        self,
        coefficients: np.ndarray,
        intercept: float,
        feature_names: list[str],
        train_means: np.ndarray,
    ):
        self.coef = np.asarray(coefficients, dtype=float)
        self.intercept = float(intercept)
        self.feature_names = feature_names
        self.train_means = np.asarray(train_means, dtype=float)

    def explain_instance(
        self,
        x_vector: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        triage_category: str,
        n_warmup_checkins: int = 3,
    ) -> dict[str, Any]:
        centered = np.asarray(x_vector, dtype=float) - self.train_means
        raw_contributions = self.coef * centered

        group_contributions = {
            "text_semantic": 0.0,
            "conversational_pacing": 0.0,
            "lexical_sentiment": 0.0,
            "baseline_deviation": 0.0,
        }

        for fname, val in zip(self.feature_names, raw_contributions, strict=False):
            if fname.startswith("embed_") or "pca" in fname:
                group_contributions["text_semantic"] += float(val)
            elif fname.startswith("timing_"):
                group_contributions["conversational_pacing"] += float(val)
            elif fname.startswith("lex_"):
                group_contributions["lexical_sentiment"] += float(val)
            elif "drift" in fname or "deviation" in fname:
                group_contributions["baseline_deviation"] += float(val)
            else:
                group_contributions["text_semantic"] += float(val)

        # Plain-language summary
        top_group = max(group_contributions.items(), key=lambda item: abs(item[1]))

        if triage_category == "LOW_RISK_TREND":
            plain_summary = (
                "Your check-in aligns comfortably with your established campus baseline."
            )
        elif triage_category == "ELEVATED_TREND":
            plain_summary = f"Your pattern shows notable variation from your usual routine, primarily reflected in {top_group[0].replace('_', ' ')}."
        else:
            plain_summary = "We noticed some pattern variation that would benefit from a quick, confidential chat with a campus counselor."

        # Scan against banned diagnostic vocabulary
        for pattern in BANNED_WORDS:
            if re.search(pattern, plain_summary, re.IGNORECASE):
                raise ValueError(f"Language safety violation: matched banned pattern '{pattern}'")

        return {
            "triage_category": triage_category,
            "interval_lower": float(lower_bound),
            "interval_upper": float(upper_bound),
            "group_contributions": group_contributions,
            "top_factor": top_group[0],
            "plain_language_summary": plain_summary,
            "data_quality": {
                "warmup_complete": n_warmup_checkins >= 3,
                "n_baseline_checkins": n_warmup_checkins,
            },
        }
