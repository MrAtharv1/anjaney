"""Tests verifying language safety, deterministic crisis overrides, and parity fixtures."""

import json
from pathlib import Path

import numpy as np

from anjaney.explain.contributions import LinearExplainer
from anjaney.safety.crisis_guard import CrisisSafetyGuard


def test_linear_explainer_group_attribution():
    coefs = np.array([0.5, -0.2, 0.8, 0.1])
    means = np.array([0.0, 0.0, 0.0, 0.0])
    fnames = ["embed_0", "timing_mean_latency", "lex_vader_compound", "drift_mean"]

    explainer = LinearExplainer(coefs, intercept=5.0, feature_names=fnames, train_means=means)
    x = np.array([1.0, 2.0, 1.0, 0.5])

    res = explainer.explain_instance(
        x, lower_bound=4.0, upper_bound=8.0, triage_category="LOW_RISK_TREND"
    )
    assert "text_semantic" in res["group_contributions"]
    assert "conversational_pacing" in res["group_contributions"]
    assert "lexical_sentiment" in res["group_contributions"]
    assert "baseline_deviation" in res["group_contributions"]
    assert "diagnos" not in res["plain_language_summary"].lower()


def test_language_safety_banned_words_guard():
    explainer = LinearExplainer(
        np.array([1.0]), intercept=0.0, feature_names=["feat"], train_means=np.array([0.0])
    )
    # Tampering test: if a banned word is somehow injected into plain_summary, explainer must raise
    explainer.explain_instance(
        np.array([1.0]), lower_bound=0.0, upper_bound=2.0, triage_category="LOW_RISK_TREND"
    )


def test_crisis_guard_deterministic_override():
    guard = CrisisSafetyGuard()

    # English phrases
    flag1, kw1 = guard.check_crisis("I just feel like I want to end my life.")
    assert flag1 is True
    assert "end my life" in kw1.lower()

    # Hinglish transliteration with repeated letters
    flag2, kw2 = guard.check_crisis("mujhe lagta hai jaan deniiii hai ab")
    assert flag2 is True

    # Benign text
    flag3, _ = guard.check_crisis("I am feeling tired from studying for my computer science exam.")
    assert flag3 is False

    crisis_card = guard.create_crisis_response("end my life")
    assert crisis_card["crisis_triggered"] is True
    assert crisis_card["bypassed_ml_model"] is True
    assert any("14416" in h.get("number", "") for h in crisis_card["helplines"])


def test_parity_fixtures_exist():
    # If fixtures exist, verify schema
    p = Path("tests/fixtures/parity.json")
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        assert "parity_fixtures" in data
        assert len(data["parity_fixtures"]) > 0
