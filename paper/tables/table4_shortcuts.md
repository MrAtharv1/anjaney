# Table 4: Experimental Controls for Dataset Shortcuts and Confounders

| Control Experiment | Test Design | Empirical Observation | Comparison / Null Baseline | Statistical Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Direct Diagnosis Query Drop** | Mask 554 direct diagnosis queries across 101 sessions | AUROC drops $\Delta = -0.0801$ (RMSE $+0.4026$) | Matched random 554-window drop: $\Delta = -0.0200 \pm 0.0195$ | Excess drop: $\Delta = -0.0601$ ($p_{\text{empirical}} = 0.0000$) |
| **Session ID Artifact Prediction** | Predict PHQ-8 binary solely from numeric session ID | AUROC $= 0.6353$ [95% CI: $0.5317, 0.7410$] | Pure random guessing AUROC $= 0.5000$ | Detectable recruitment order confound |
| **Model vs Session ID Correlation** | Correlate text model predictions with session ID | Pearson $r = -0.2952$ ($p = 0.0004$) | Theoretical independence $r = 0.0000$ | Model partially exploits session indexing |
| **Cohort Split Stability** | Split validation cohort at median session ID (363) | Early cohort: AUROC $= 0.7634$ | Late cohort: AUROC $= 0.7688$ | Stable across temporal sub-cohorts |
| **ID-Residualized Text Prediction** | Regress text predictions against ID-residualized targets | Partial correlation $r_{\text{residual}} = 0.4390$ | Null correlation $r = 0.0000$ | Robust independent signal ($p = 5.36 \times 10^{-8}$) |
