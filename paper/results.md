# Results

> All numbers in this section trace to `paper/numbers.json` with source files listed per entry.
> Language follows 'this model on this dataset' conventions.
> No claim uses 'proves', 'confirms', 'beats', or 'outperforms' unless the paired 95 % CI excludes zero.
> Permutation *p*-values are reported one-sided. CIs are 95 % bootstrap (participant-level) unless otherwise stated.

---

## 4.1 Data and Splits

The DAIC-WOZ corpus contains 189 participants in the official AVEC 2017 release (sessions 342, 394, 398, 460 withheld per the exclusion list; session 440 truncated at the USC source server—missing transcript, COVAREP and formant files—and excluded per Amendment 1). After these exclusions, **188 intact sessions** are available on disk (Table 1; source: `data/daic_woz/MANIFEST.json`).

Official split counts: Train 107, Dev 35 (34 on disk after exclusion of 440), Test 47.
Window counts: Train 3,323 | Dev 1,251 | Test 1,739 | Total 6,313.

Three participants (451, 458, 480) have no interviewer turns in the transcript and are excluded from the RQ1 cohort but remain in the regression training pool; they contribute 47 windows combined.

For RQ1, the analysis cohort is **N = 139** train+dev participants (after excluding 440 and the three no-interviewer-turn sessions), with 41 classified high-risk and 98 low-risk (source: `experiments/rq1/20261016_054735_rq1_gating/decision.json`).

---

## 4.2 Held-Out Test Performance

The deployed model is a **text-only Ridge** (α = 10.0, 20 PCA components, 384-dimensional MiniLM-L6-v2 embeddings), refit on all 141 train+dev participants before test evaluation. Exactly one test evaluation was performed under `experiments/test_eval.lock` on git hash `a97469938c93a71091484fd8206265c130c80739` (source: `experiments/test_eval/results.json`).

**Table 8 — Held-out test performance (N = 47 participants, 1,739 windows)**

| Metric | Model | 95 % CI | Dummy (train+dev mean) | 95 % CI | Paired Δ | 95 % CI |
|--------|-------|---------|------------------------|---------|----------|---------|
| RMSE | 5.153 | [4.15, 6.08] | 6.415 | [5.23, 7.58] | −1.262 | — |
| MAE | 4.197 | [3.35, 5.09] | 5.424 | [4.47, 6.45] | −1.227 | — |
| Pearson *r* | 0.645 | [0.498, 0.770] | 0.000 | — | — | — |
| AUROC | 0.797 | [0.652, 0.917] | 0.500 | — | +0.297 | — |
| F₁ (in-fold τ = 7.1) | 0.632 | [0.429, 0.791] | 0.000 | — | — | — |
| Brier | 0.196 | [0.114, 0.287] | 0.280 | [0.160, 0.400] | −0.084 | — |

*Source: `experiments/test_eval/results.json` → `paper/numbers.json::table8_test_evaluation`.*
*All CIs are participant-level bootstrap (B = 2,000). Paired participant-bootstrap differences are not computed separately for test; directional comparisons are descriptive only. The dummy baseline predicts the train+dev mean (6.582) for every participant.*

For comparison, the cross-validated pooled estimate on train+dev was RMSE 4.92 (95 % CI [4.31, 5.55]) and AUROC 0.78 (95 % CI [0.68, 0.86]) (source: `paper/numbers.json::table2_ablation`). The test RMSE of 5.153 is 0.233 points above the CV-pooled estimate (a small positive generalisation gap consistent with the small N and optimistic CV bias); the test AUROC of 0.797 is within the CV-pooled CI. These directional comparisons are descriptive—no paired CI is available across CV and test partitions.

---

## 4.3 Conformal Prediction on the Held-Out Test Set

Conformal intervals were constructed using residuals from a 5-fold leave-one-out calibration on train+dev; q̂ values were computed before test was unsealed (source: `experiments/test_eval/results.json::conformal_regression`).

**Regression intervals (α = 0.10, W_max = 10):**
- Empirical coverage: **43/47 participants** (91.5 %, nominal 90.0 %; source: `table8_test_evaluation::conformal_alpha_10_cov`)
- Mean interval width: **14.84** PHQ-8 score points
- Abstention rate: **46/47** (97.9 %; only 1 participant falls within the W_max = 10 triage window)
- That 1 autonomous prediction: RMSE 0.747, MAE 0.747

Because all intervals exceed W_max = 8 on the test set, the practical operating point is **near-universal abstention** (escalate for counsellor review). Under a standard clinical cost ratio (C_FN=5, C_FP=1, C_abstain=0.5), the conformal triage policy achieves expected cost 0.489 vs escalate-everyone cost 0.500 on validation (source: `stageD_statistics_conformal.json::cost_ratio_sweep`; test-set cost analysis not performed separately as N=1 autonomous is anecdotally small).

**Classification track (LAC, α = 0.20, nominal 80 % coverage):**
- Empirical coverage on test: **35/47** (74.5 %, slightly below nominal due to small N)
- Singleton rate: **41/47** (87.2 %)
- Abstain rate: **6/47** (12.8 %; all as {both-class} sets)
- Error rate among singletons: **12/41** (29.3 %, Wilson 95 % CI [17.6 %, 44.5 %])

Singleton predictions are approximately **71 % accurate** on this dataset; this is not autonomous triage capability. The system is designed to flag uncertain cases for human review, not to replace clinical assessment.

---

## 4.4 Research Questions

### RQ1: Within-Session Linguistic Drift (Inconclusive)

RQ1 tests whether within-session linguistic drift (conditioned on interview phase and relative position) differs between participants classified high-risk vs low-risk for depression. The pre-registered gate is Cohen's *d* ≥ 0.50; the observed value does not reach this threshold (source: `experiments/rq1/20261016_054735_rq1_gating/decision.json`).

- N: 41 high-risk, 98 low-risk (train+dev; 47 test held out)
- Cohen's *d* = **0.314** (95 % bootstrap CI [−0.039, 0.697])
- One-sided permutation *p* = **0.046**
- TOST 90 % CI: [0.018, 0.630] — does not fall within equivalence bounds [−0.30, +0.30], so equivalence is also not established
- MDE (80 % power, N=139): **0.463** — the study is underpowered to detect the pre-registered threshold
- Length-residualised sensitivity *d*: **0.282** (slightly attenuated; session length is a meaningful covariate)
- Mixed-model: interaction coefficient 0.068 (95 % CI [−0.113, 0.249], *p* = 0.462) — consistent with null

**Conclusion: INCONCLUSIVE.** The point estimate lies in a practically interesting range but the CI is wide and crosses zero. The null is bounded (equivalence not established), but the study is underpowered for the pre-registered effect size. No decision gates on RQ1 change the deployed model.

*K-sensitivity (fixed cohort, N=139):* K=3 *d* = 0.314 (PRIMARY); K=5 *d* = 0.172; K=8 *d* = 0.036 — effect attenuates as the warm-up window grows, suggesting early-session effects dominate (exploratory).

*Conditioning-mode ablation:* The primary analysis uses `prompt_phase_and_rel_position_cross_fit_5fold`. Alternative modes (position-only, none) were not re-run on the final commit; this is a noted limitation.

### RQ2: Timing Improves Prediction (Not Supported)

RQ2 tests whether adding conversational timing features to the text-only ridge reduces RMSE or interval width. (Source: `experiments/stageD_statistics_conformal.json`, `experiments/stageB_rq1_rq3.json`.)

- ΔRMSE (text+timing − text-only): **−0.003** (95 % CI [−0.037, +0.031], Holm-corrected *p* = 1.000)
- ΔAUROC: **−0.0004** (95 % CI [−0.0143, +0.0143], Holm-corrected *p* = 1.000)
- Δ mean interval width: **−0.134** score points (−1.03 %), Nadeau–Bengio corrected resampled *t* = −0.758, *p* = 0.452

Neither metric shows a statistically detectable improvement. A ~1 % width compression is clinically negligible. Timing features are classified as **exploratory** and are not deployed.

---

## 4.5 Shortcut and Confound Controls

### Diagnosis-Prompt Sensitivity (Source: `experiments/stageC_diagnosis_shortcut.json`)

The corpus contains 554 windows (8.8 % of 6,313) whose interviewer turn includes the word "diagnosis". Dropping these windows reduces AUROC by **−0.080** (from 0.781 to 0.701). Dropping the same number of matched random windows reduces AUROC by **−0.020 ± 0.020** (mean ± SD, 100 draws). The excess drop attributable specifically to diagnosis-prompt windows is **−0.060** (empirical one-sided *p* < 0.01 against random draws). This confirms partial reliance on diagnosis-proximal text but does not explain all of the AUROC signal; the model retains AUROC 0.700 with those windows removed.

### Session-ID Analysis (Source: `experiments/stageC_session_id_shortcut.json`)

A ridge model using only session-ID one-hot encoding achieves AUROC **0.635** (95 % CI [0.532, 0.741]) on train+dev CV, substantially below the text-only model. Pearson correlation between text-model predictions and session ID is −0.295 (*p* < 0.001). The prediction residuals after regressing out session-ID remain correlated with session ID (*r* = 0.439, *p* < 0.001), indicating that session-level structure is partially captured by text embeddings but not explained away by it. Temporal-stability probes (early vs late sessions) yield AUROC 0.763 vs 0.769—a negligible difference suggesting no strong temporal drift in leakage.

---

## 4.6 Exploratory Modality Ablation

The following are descriptive benchmarks on the **official dev split** (N=34 on disk, train N=107). No statistical comparisons are claimed; all modalities beyond `text_only` are exploratory offline benchmarks only (source: `experiments/stageD_statistics_conformal.json::dev_split_descriptive`).

| Modality | RMSE | MAE | AUROC | Note |
|----------|------|-----|-------|------|
| Dummy (mean) | 6.310 | 5.289 | 0.500 | Baseline |
| text_only (Ridge) | 5.638 | 4.417 | 0.794 | Deployed |
| text_plus_timing (Ridge) | 5.669 | 4.473 | 0.779 | Exploratory |
| timing_only (Ridge) | 6.115 | 5.013 | 0.514 | Exploratory |
| lexical_only (Ridge) | 6.615 | 5.238 | 0.526 | Exploratory |
| acoustic_only (Ridge) | 6.727 | 5.053 | 0.585 | Exploratory, offline |
| text+timing+acoustic (Ridge) | 5.493 | 4.362 | 0.806 | Exploratory, offline |

*APS classification is DEGENERATE at all α levels tested (100 % abstention, N=0 singletons). These rows are footnoted DEGENERATE in the corresponding LaTeX table.*

---

## 4.7 On-Device Simulated Benchmark

ONNX int8-quantised model size: **22.8 MB** (vs fp32 89.2 MB). Peak RAM: 48.5 MB. Per-turn median latency: 18.4 ms (P95: 24.2 ms). Throughput: 54.3 turns/s. Battery drain per session: ~0.04 %. These figures are **SIMULATED** (source: `docs/ondevice_benchmark.md`; obtained from synthetic profiling on an emulated device, not from hardware measurement). Parity between int8 and fp32 predictions is verified by the npm test fixture suite (source: `paper/numbers.json::table9_ondevice`).
