# Ethics, Clinical Stance & Technical Limitations

## Ethical Framework
1. **Screening Aid, Never Diagnostic Tool:**
   - ANJANEY produces risk trends to help students reflect and to assist wellness staff with queue prioritization. It does not diagnose clinical depression.
2. **Domain Shift Caveat:**
   - Models trained on semi-structured interview transcripts (DAIC-WOZ) may exhibit distribution shift when evaluated on informal campus check-ins.
3. **False Negative Mitigation:**
   - The conformal layer optimizes for conservative coverage ($\alpha = 0.10$). Ambiguous scores default to `ESCALATE_TO_HUMAN` rather than false reassurance.

## Structural & Methodological Limitations

1. **Clinical Interview vs. Campus Check-in Distribution Shift:**
   - Training and validation rely on DAIC-WOZ semi-structured clinical interviews conducted by an animated virtual interviewer (Ellie) with US English-speaking military veterans and psychiatric outpatients ($N=141$ validation pool, $N=188$ total on disk; session 440 zip archive on the USC server is truncated at source, excluding transcript/features). This setting differs substantially from informal, voluntary conversational text check-ins conducted by university students on campus.

2. **Inconclusive Primary Hypothesis on DAIC-WOZ (RQ1 & RQ2 Regression):**
   - **RQ1**: Within-session semantic drift conditioning remains `INCONCLUSIVE` (permutation $p = 0.046$, Cohen's $d = 0.314$, failing pre-registered threshold).
   - **RQ2**: In paired participant-bootstrap ablations ($N=141$, $B=2000$), fusing conversational timing with text embeddings does not produce a statistically significant reduction in primary regression error ($\Delta\text{RMSE} = -0.003$ [95% CI: $-0.036, +0.033$], $p_{\text{Holm}} = 1.000$) or discrimination ($\Delta\text{AUROC} = -0.0002$ [95% CI: $-0.014, +0.014$], $p_{\text{Holm}} = 1.000$). However, multimodal fusion achieves a statistically significant compression of conformal prediction intervals ($\Delta\text{Width} = -0.133$ [95% CI: $-0.169, -0.100$], $p_{\text{Holm}} < 0.001$).

3. **Recruitment Batch Shortcut (Session-ID Confound):**
   - The numeric participant session ID alone predicts depression above chance (Ridge OOF AUROC $0.635$, RMSE $5.455$, MAE $4.457$), reflecting recruitment cohort imbalance in DAIC-WOZ: ground truth PHQ-8 scores negatively correlate with session ID (Pearson $r = -0.281, p < 0.001$; Spearman $\rho = -0.314, p < 0.001$).
   - Out-of-fold predictions of `text_only` Ridge (which has zero access to session ID) exhibit an identical correlation with session ID ($r = -0.295, p < 0.001$; $\rho = -0.285, p < 0.001$).
   - Performance remains consistent across recruitment cohorts: for session ID $\le 363$ ($N=47$), `text_only` achieves RMSE $4.981$ and AUROC $0.763$; for session ID $> 363$ ($N=94$), `text_only` achieves RMSE $4.856$ and AUROC $0.769$ ($\Delta\text{AUROC} = 0.006$).
   - When evaluated on ID-residualised targets (removing linear session ID trends within training folds), `text_only` maintains an AUROC of $0.775$ (RMSE $4.904$), and the text residual predictions significantly predict unexplained variance ($r = 0.439, p = 5.36 \times 10^{-8}$).

4. **Interviewer Diagnostic Prompt Shortcut Sensitivity & Matched Random Control:**
   - 8.78% of feature windows (554 windows across 1,212 participant turns) directly follow explicit interviewer inquiries regarding prior depression diagnoses, PTSD, therapy, medication, or psychiatric visits.
   - Dropping all 554 diagnosis windows drops `text_only` Ridge performance from AUROC $0.781$ (RMSE $4.898$) to AUROC $0.701$ (RMSE $5.300$), yielding $\Delta\text{AUROC} = -0.0801$ and $\Delta\text{RMSE} = +0.4026$.
   - **Matched Random Control (100 draws)**: Removing an identical number of random windows matched per participant ($k_p$ random windows dropped for each participant who had $k_p$ diagnosis windows) yields an average drop of only $\Delta\text{AUROC} = -0.0200 \pm 0.0195$ [95% CI: $-0.0609, +0.0142$] and $\Delta\text{RMSE} = +0.0866 \pm 0.0949$.
   - **Excess Drop**: The excess drop uniquely attributable to diagnostic interview questions is $\Delta\text{AUROC}_{\text{excess}} = -0.0601$ and $\Delta\text{RMSE}_{\text{excess}} = +0.3160$ (empirical $p = 0.0000$ across 100 draws; 0/100 random draws produced a drop $\le -0.0801$).
   - Consequently, while ~2.0 AUROC points reflect general sample size reduction, ~6.0 AUROC points represent explicit clinical self-disclosure rather than latent conversational markers. Genuine conversational signal persists (AUROC $0.701 > 0.500$), but evaluations on raw DAIC-WOZ transcripts substantially benefit from explicit diagnostic prompting.

5. **Acoustic Extension is Exploratory and Offline-Only:**
   - Pre-extracted acoustic voice features (COVAREP glottal parameters and Formants) were evaluated strictly as an exploratory extension for benchmarking against AVEC2017 literature. Voice features are never extracted, exported, or processed by on-device student check-in bundles or web clients.

6. **Triage Coverage at High Confidence:**
   - Under single-session evaluations with 90% nominal confidence ($\alpha = 0.10$) and conservative uncertainty tolerance ($W_{\max} = 8.0$), 100% of participants are escalated to human clinicians due to wide predictive intervals ($\sim 14.6$ points). Meaningful autonomous triage throughput requires either lower confidence coverage ($\alpha \ge 0.30$) or wider uncertainty thresholds ($W_{\max} \ge 10.0$).