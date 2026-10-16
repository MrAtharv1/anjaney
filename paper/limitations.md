# Limitations of the ANJANEY Screening Architecture

This document provides a comprehensive accounting of methodological, statistical, dataset, and clinical limitations of the ANJANEY study. Any deployment, reproduction, or clinical translation must be interpreted in light of these constraints.

---

## 1. Dataset Scale and Statistical Power

- **Limited Sample Size**: The entire intact DAIC-WOZ corpus on disk comprises only 188 participant sessions across three official partitions:
  - Training: $N = 107$
  - Development (validation): $N = 34$ (following server-side truncation of session 440)
  - Held-out Test: $N = 47$
- **Under-Powered Sub-Analyses**: With $N=41$ high-risk participants in the pooled validation cohort, the minimum detectable effect size (MDE) at 80% statistical power is Cohen's $d = 0.463$. Consequently, small-to-moderate behavioral phenomena ($d \approx 0.30$) cannot be reliably confirmed or falsified, as demonstrated by the `INCONCLUSIVE` status of RQ1.
- **Wide Estimation Uncertainty**: While point metrics on the held-out test partition indicate screening capability (AUROC $= 0.797$), the 95% bootstrap confidence interval spans $[0.652, 0.917]$, reflecting substantial variance inherent to $N=47$ test cases.

---

## 2. Demographic and Population Mismatch (Ecological Validity)

- **Clinical Adult / Veteran Cohort vs Campus Students**:
  - DAIC-WOZ was collected at the University of Southern California Institute for Creative Technologies primarily involving adult community members, including a significant proportion of military veterans experiencing post-traumatic stress and major depression.
  - The target deployment setting for ANJANEY is an undergraduate and graduate university student population experiencing transient academic distress, sleep dysregulation, and adjustment challenges.
  - Clinical symptom manifestation, linguistic idioms, emotional expression, and willingness to disclose distress differ fundamentally between military veterans and modern college students.
- **Absence of In-Situ Student Data**:
  - No real university student check-in data was collected or evaluated in this study.
  - The student check-in client interface (`apps/web`) and edge-device ONNX runtime benchmarks are evaluated entirely on synthetic student interaction sessions and offline laboratory recordings.

---

## 3. Dataset Shortcuts and Confounding Artifacts

### A. The Direct Diagnosis Query Shortcut
- In the DAIC-WOZ semi-structured protocol, the virtual interviewer (Ellie) explicitly asks direct diagnostic questions (e.g., *"Have you been diagnosed with depression?"*, *"Are you currently taking medication?"*).
- Our controlled query-ablation experiment revealed:
  - Dropping these 554 diagnosis windows across 101 participants caused an immediate AUROC drop of $-0.0801$ (and RMSE increase of $+0.4026$).
  - Compared against 100 random window-drop controls (mean drop: $-0.0200 \pm 0.0195$), the **excess drop uniquely attributable to direct diagnosis inquiries was $-0.0601$ ($p < 0.0001$)**.
  - In a prospective, non-directive campus wellness check-in, an AI assistant would not repeatedly ask diagnostic questions. A substantial fraction of the reported screening accuracy may not generalize to naturalistic conversations.

### B. The Session-ID Recruitment Index Confound
- Numeric participant session IDs alone predict depression status with an AUROC of $0.6353$ [95% CI: $0.5317, 0.7410$], demonstrating non-random recruitment or temporal batching during DAIC-WOZ data collection.
- Unconstrained text model predictions correlate significantly with participant ID ($r = -0.2952, p = 0.0004$).
- While our partial-correlation control confirms that text embeddings retain independent predictive signal after residualizing out session ID ($r_{\text{residual}} = 0.4390, p = 5.36 \times 10^{-8}$), uncorrected deployment on sequential user identifiers carries severe risk of exploiting temporal artifacts.

---

## 4. Modality Transferability and Typing Dynamics

- **Spoken vs Typed Modalities**:
  - The timing features analyzed in this study (speech pauses, turn-taking latencies, response duration) were derived from spoken audio recordings of face-to-face virtual agent interviews.
  - The ANJANEY client application is a typed, asynchronous smartphone web application.
  - Keystroke dynamics and mobile typing pauses are driven by autocorrect, touch latency, multitasking, and screen keyboards, which have zero proven equivalence to acoustic conversational pauses.
- **Parsimony Protocol Enforced**:
  - Because RQ1 drift was inconclusive ($d = 0.314$) and RQ2 paired difference was non-significant ($\Delta\text{RMSE} = -0.003, p_{\text{Holm}} = 1.000$), timing features were formally excluded from the deployed model bundle (`final_config.yaml`).
  - Typed conversational timing remains an exploratory, unvalidated hypothesis.

---

## 5. Conformal Interval Inflation and Triage Economics

- **Wide Predictive Intervals**:
  - The split conformal regression interval width on the held-out test split at $\alpha = 0.10$ is $14.84 \pm 2.06$ points on the 24-point PHQ-8 scale.
  - Because the clinically relevant screening zone ($\text{PHQ-8} \in [8, 12]$) has a span of only 4 points, a 15-point predictive interval necessarily overlaps the clinical threshold for almost all borderline students.
  - Under rigorous triage criteria ($W_{\max} = 10.0$), the abstention rate reaches $97.9\%$ on test data.
- **Economic Value Limits**:
  - Automated screening cannot replace human clinical judgment under conservative clinical loss functions.
  - When the cost of a false negative is high ($C_{\text{FN}} = 10.0$), automated triage ties the default psychiatric standard-of-care ("escalate everyone to human intake"), providing zero economic advantage.

---

## 6. Single Test Evaluation Policy (Confirmatory Trade-Off)

- Under our pre-registered confirmatory protocol (`TEST_EVAL_SPEC.md`), the held-out test partition was evaluated exactly **once** behind a cryptographic lock harness (`experiments/test_eval.lock`).
- This design completely eliminates $p$-hacking, feature snooping, and post-hoc threshold selection.
- However, it means any post-test insight (e.g., suboptimal calibration of set-valued classification thresholds on the test split) cannot be re-tuned or adjusted. All results reported in this paper reflect true single-shot performance.
