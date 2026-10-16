# Table 7: Clinical Economic Cost Sweep vs Default Clinical Policy

| Clinical Scenario | Cost Weights ($C_{\text{FN}} : C_{\text{FP}} : C_{\text{abstain}}$) | Deployed Policy Cost / Patient | Escalate-Everyone Reference Cost | Net Savings / Patient | Clinical Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Clinical** | 5.0 : 1.0 : 0.5 | 0.489 | 0.500 | -0.011 | Marginally beats default escalation |
| **Equal Cost** | 1.0 : 1.0 : 0.5 | 0.461 | 0.500 | -0.039 | Modest efficiency gain |
| **High Safety** | 10.0 : 1.0 : 0.5 | 0.500 | 0.500 | 0.000 | Ties default escalation (100% human triage) |
| **Low Abstain Cost** | 5.0 : 1.0 : 0.1 | 0.098 | 0.100 | -0.002 | Strict tie with clinical intake |

*Evaluation Framework: All costs are bench-marked against the standard psychiatric standard-of-care ("escalate everyone to human clinician intake", Cost $= C_{\text{abstain}}$). Under strict safety constraints ($C_{\text{FN}} \ge 10$), automated screening cannot beat human intake economically.*
