# ANJANEY Claims Register

This document registers all scientific and empirical claims made in the ANJANEY manuscript. Each claim is formally classified as **SUPPORTED**, **INCONCLUSIVE**, or **EXPLORATORY**, accompanied by exact statistical evidence, source file citations, and pre-specified wording constraints.

---

## Summary Registry

| ID | Claim Description | Status | Evidence Summary | Source File | Wording Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | Late-session check-in drift differentiates high-risk from low-risk participants (RQ1) | **INCONCLUSIVE** | Cohen's $d = 0.314$ [95% CI: $-0.039, 0.697$], $p = 0.0461$; length-residualized $d = 0.282 < 0.30$; TOST 90% CI $[0.018, 0.630]$ | [`decision.json`](file:///c:/Users/intel/vscodews/anjaney/experiments/rq1/20261016_054735_rq1_gating/decision.json) | Do NOT claim drift is established. Must state "under pre-registered criteria on this dataset, RQ1 remains inconclusive." |
| **C2** | Multimodal conversational timing reduces screening prediction error over text embeddings (RQ2) | **FALSIFIED / NOT SUPPORTED** | $\Delta\text{RMSE} = -0.003$ [$-0.037, 0.031$], $p_{\text{Holm}} = 1.000$; $\Delta\text{AUROC} = -0.0004$ [$-0.0143, 0.0143$], $p_{\text{Holm}} = 1.000$ | [`table3_rq2_paired.md`](file:///c:/Users/intel/vscodews/anjaney/paper/tables/table3_rq2_paired.md) | Must state "fuses timing with text without statistically significant error reduction on this dataset." |
| **C3** | Multimodal timing significantly compresses predictive interval width | **NOT SUPPORTED** | $\Delta\text{Width} = -0.134 \pm 0.341$ points ($-1.03\%$), Nadeau-Bengio corrected resampled $t = -0.758, p = 0.452$ | [`table3_rq2_paired.md`](file:///c:/Users/intel/vscodews/anjaney/paper/tables/table3_rq2_paired.md) | Must state "width compression is clinically negligible (~1%) and not statistically significant under fold-dependency correction." |
| **C4** | Deployed `text_only` Ridge model predicts depression severity better than dummy baseline on held-out test data | **SUPPORTED** | RMSE: $5.15$ [$4.15, 6.08$] vs $6.42$ [$5.23, 7.58$]; MAE: $4.20$ [$3.35, 5.09$] vs $5.42$ [$4.47, 6.45$]; AUROC: $0.797$ [$0.652, 0.917$] vs $0.500$ | [`results.json`](file:///c:/Users/intel/vscodews/anjaney/experiments/test_eval/results.json) | Must specify "this model on this held-out test partition ($N=47$) improves over a mean prediction baseline." No ungrounded claims of clinical diagnostic capability. |
| **C5** | Split conformal regression maintains valid empirical coverage on held-out test participants | **SUPPORTED** | $\alpha=0.10$ nominal (90% target) achieves $91.5\%$ empirical coverage ($43/47$ participants) with mean width $14.84 \pm 2.06$ | [`results.json`](file:///c:/Users/intel/vscodews/anjaney/experiments/test_eval/results.json) | Must note that interval width spans $\approx 15$ PHQ-8 points, requiring human clinician triage for nearly all borderline cases. |
| **C6** | Set-valued conformal classification (LAC) provides autonomous screening throughput with bounded singleton error | **SUPPORTED** | LAC at $\alpha=0.20$ achieves $87.2\%$ throughput ($41/47$) with empirical coverage $74.5\%$ and singleton error $29.3\%$ [95% Wilson CI: $17.6\%, 44.5\%$] | [`results.json`](file:///c:/Users/intel/vscodews/anjaney/experiments/test_eval/results.json) | Must report singleton sample size ($N=41$) and Wilson CI. Must disclose that deterministic APS is degenerate ($N=0$). |
| **C7** | Model performance is partly driven by interviewer direct-diagnosis probes (shortcut artifact) | **SUPPORTED** | Dropping 554 diagnosis windows causes an excess AUROC drop of $-0.0601$ ($p < 0.0001$) beyond matched random control drops | [`stageC_diagnosis_shortcut.json`](file:///c:/Users/intel/vscodews/anjaney/experiments/stageC_diagnosis_shortcut.json) | Must explicitly disclose that ~6 AUROC points rely on explicit clinical inquiries ("have you been diagnosed with depression?"). |
| **C8** | Model predictions exploit participant session-ID recruitment indexing | **SUPPORTED** | Session ID alone predicts depression (AUROC $0.635$); text predictions correlate with ID ($r = -0.295, p = 0.0004$) | [`stageC_session_id_shortcut.json`](file:///c:/Users/intel/vscodews/anjaney/experiments/stageC_session_id_shortcut.json) | Must qualify that while partial correlation confirms independent signal ($r = 0.439, p = 5.36 \times 10^{-8}$), uncorrected deployment on sequential IDs carries risk. |
| **C9** | Acoustic voice features improve screening performance | **EXPLORATORY** | Acoustic features are strictly exploratory offline benchmarks, omitted from client deployment bundle | [`final_config.yaml`](file:///c:/Users/intel/vscodews/anjaney/configs/final_config.yaml) | Must state "acoustic voice parameters are evaluated offline as exploratory benchmarks and are NOT included in the deployed tool." |

---

## Detailed Claim Declarations

### Claim 1: Within-Session Drift Gating (RQ1)
- **Statement**: *Within-session conversational drift differs between high-risk and low-risk participants after position and script-phase conditioning.*
- **Classification**: `INCONCLUSIVE`
- **Primary Evidence**:
  - Sample sizes: $N_{\text{high}} = 41, N_{\text{low}} = 98$ (Train+Dev pooled, excluding 3 non-Ellie sessions).
  - Observed group difference: $+0.1235$ drift deviation units.
  - One-sided permutation test: $p = 0.0461$.
  - 95% bootstrap CI of Cohen's $d$: $[-0.0388, 0.6968]$ (crosses zero).
  - Equivalence test (TOST 90% CI): $[0.0181, 0.6299]$ (not contained within $[-0.30, 0.30]$).
  - Length-residualized sensitivity: Cohen's $d = 0.2819 < 0.30$.
  - Minimum Detectable Effect at 80% power: $\text{MDE} = 0.4625$.
- **Required Manuscript Phrasing**:
  > *"Under our pre-registered gating rules on the DAIC-WOZ dataset, within-session drift differences remain inconclusive. While an uncorrected permutation test reached nominal significance ($p = 0.046$), the 95% bootstrap confidence interval crosses zero, length-residualization attenuates the effect below our minimum threshold ($d = 0.282 < 0.30$), and equivalence cannot be established. Consequently, drift signals are not deployed for autonomous decision gating."*

---

### Claim 2: Multimodal Timing Ablation (RQ2)
- **Statement**: *Adding conversational timing features to text embeddings improves screening accuracy and reduces selective risk.*
- **Classification**: `NOT SUPPORTED`
- **Primary Evidence**:
  - 5-fold CV paired participant bootstrap differences ($B=2000$):
    - $\Delta\text{RMSE} = -0.003$ [95% CI: $-0.037, 0.031$], $p_{\text{Holm}} = 1.000$.
    - $\Delta\text{AUROC} = -0.0004$ [95% CI: $-0.0143, 0.0143$], $p_{\text{Holm}} = 1.000$.
    - $\Delta\text{Interval Width} = -0.134 \pm 0.341$ points ($-1.03\%$), $t = -0.758, p = 0.452$ (Nadeau-Bengio corrected resampled $t$-test).
- **Required Manuscript Phrasing**:
  > *"On this dataset, adding spoken conversational timing features to 384-dimensional MiniLM embeddings did not produce a statistically significant reduction in prediction error or interval width. All primary paired difference confidence intervals include zero after Holm-Bonferroni adjustment, and the 1% width compression is not statistically significant ($p = 0.452$). Following our pre-registered parsimony protocol, the deployed screening tool is strictly text-only."*

---

### Claim 3: Single Confirmatory Held-Out Test Evaluation
- **Statement**: *The deployed text-only Ridge model demonstrates significant predictive capability over baseline on the held-out test split.*
- **Classification**: `SUPPORTED`
- **Primary Evidence**:
  - Test set ($N=47$): RMSE $= 5.15$ [$4.15, 6.08$] vs Dummy $6.42$ [$5.23, 7.58$].
  - AUROC $= 0.797$ [$0.652, 0.917$] vs Dummy $0.500$.
  - Split conformal coverage at $\alpha=0.10$: $91.5\%$ ($43/47$ sessions covered).
- **Required Manuscript Phrasing**:
  > *"In a single confirmatory evaluation on the held-out test split ($N=47$) locked under pre-registered protocol `prereg-v2`, the text-only Ridge model achieved an RMSE of $5.15$ and an AUROC of $0.797$, outperforming the training-mean baseline. Conformal calibration achieved 91.5% empirical coverage at the nominal 90% level, though wide predictive intervals ($14.8$ points) mandate human clinical referral for nearly all autonomous triage cases."*
