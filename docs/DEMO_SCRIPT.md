# ANJANEY: 5-Minute Examiner Defence Demo Script

## Minute 0–1: Introduction & Non-Diagnostic Stance
- **Talking Point:** "ANJANEY is a campus mental-health triage screening aid, not a diagnostic system. It solves three critical failure modes: conversational feature leakage, population-only confounding, and uncalibrated point estimates."

## Minute 1–2: In-Browser Offline Scoring (Airplane Mode)
- **Action:** Open `http://localhost:3000` in the browser. Open DevTools -> Network -> toggle **Offline (Airplane Mode)**.
- **Action:** Type a benign student check-in: *"Classes were busy today, but I met my project group and we completed our lab report."*
- **Observation:** Click **Complete Check-In**. Show that scoring completes in < 50ms while completely offline. Point estimate and conformal interval $[lo, hi]$ are rendered with a `Usual-Range Pattern` card.

## Minute 2–3: Conformal Abstention & Counselor Triage
- **Action:** Type an ambiguous, elevated check-in.
- **Observation:** Show the conformal interval crossing the clinical threshold ($\tau = 10$). The system resolves to `Support Check-In Recommended` (`ESCALATE_TO_HUMAN`).
- **Talking Point:** "Instead of making an overconfident point guess near the threshold, the split-conformal layer abstains and refers the student to a human counselor."

## Minute 3–4: Deterministic Crisis Override
- **Action:** Type: *"I feel completely broken and want to end my life."*
- **Observation:** The ML model is immediately bypassed. An intervention card appears displaying **Tele-MANAS (14416)** and campus emergency contacts.

## Minute 4–5: Wellness Cell Dashboard & Privacy Guarantees
- **Action:** Navigate to `http://localhost:3000` dashboard view (or query `/api/v1/dashboard/summary`).
- **Observation:** Show $k$-anonymity suppression in action: cells with $< 5$ students display `< 5 (Suppressed for Privacy)`. Show the counselor capacity load estimator adjusting dynamically based on empirical abstention rates.