# ANJANEY Model Card

## Model Overview
- **Name:** ANJANEY Triage Engine (v1.0.0)
- **Role:** Non-diagnostic, personal-baseline campus mental-health triage screening aid.
- **Model Architecture:** Frozen sentence encoder (`all-MiniLM-L6-v2`, INT8 quantized) + content-free conversation timing dynamics + in-fold position-conditioned baseline deviation + Ridge regression head + Split-Conformal prediction layer.

## Intended Use
- **Primary Population:** University campus student populations undergoing weekly or daily wellness check-ins.
- **Decision Space:**
  - `LOW_RISK_TREND`: Predictive interval upper bound $< 10.0$ PHQ-8.
  - `ELEVATED_TREND`: Predictive interval lower bound $\ge 10.0$ PHQ-8.
  - `ESCALATE_TO_HUMAN`: Ambiguous intervals crossing $10.0$ or exceeding width threshold ($w > w_{\max}$).

## Safety Boundaries
- **Never a Diagnostic System:** Output categories reflect screening priorities, never clinical diagnoses.
- **Deterministic Crisis Bypass:** Keyword matches for self-harm or suicide trigger immediate Tele-MANAS (14416) intervention cards, overriding all model inferences.