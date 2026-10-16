# ANJANEY Test Set Evaluation Specification (`prereg-v2`)

**Protocol Version:** `prereg-v2`  
**Git Commit:** `ae3aafc`  
**Date:** 2026-10-16  
**Status:** FROZEN & LOCKED (Do NOT run until explicit investigator directive)

---

## 1. Scope & Governance

This document pre-registers the exact evaluation protocol for the single, final evaluation on the held-out DAIC-WOZ test split ($N=47$). In accordance with strict confirmatory science standards, the test split is evaluated exactly **once**. No post-hoc tuning, feature additions, or model substitutions are permitted after unlocking.

---

## 2. Cryptographic Pre-Registration Hashes

All code, configurations, data partitions, and feature matrices must match these exact SHA-256 digests prior to execution:

| File Artifact | SHA-256 Digest | Status |
| :--- | :--- | :--- |
| [`configs/final_config.yaml`](file:///c:/Users/intel/vscodews/anjaney/configs/final_config.yaml) | `e67202888180b46a04cf1c1575dbcd31c789bd9f881027c9838b3bd4c68875de` | FROZEN |
| [`configs/prereg.yaml`](file:///c:/Users/intel/vscodews/anjaney/configs/prereg.yaml) | `80d93139ef298fb449c2aa8f51c78a186a4c5befaab063075ebae138838894d7` | FROZEN |
| [`configs/acoustic_features.yaml`](file:///c:/Users/intel/vscodews/anjaney/configs/acoustic_features.yaml) | `634b49ec7542c79afae0a9e31497300cb1fdba5c04834aa97f6df8e300c824ff` | FROZEN |
| [`data/processed/splits.json`](file:///c:/Users/intel/vscodews/anjaney/data/processed/splits.json) | `5cfc1882b8df21c67f49fb6d507cdc91753bd457c5f5b63f68967d23127da3f1` | FROZEN |
| [`data/processed/features.parquet`](file:///c:/Users/intel/vscodews/anjaney/data/processed/features.parquet) | `73402f8c9e1427dbb8f33b2c11ed513e5c46df0845d99296c0c617c080b673c2` | FROZEN |

---

## 3. Model Architecture & Refit Policy

### Deployed Architecture
- **Modality:** `text_only` (384-dimensional MiniLM-L6-v2 embeddings, turn window size 5).
- **Aggregation:** `ParticipantAggregator(n_pca_components=20)`.
- **Model Head:** `Ridge(alpha=10.0, random_state=42)`.
- **Pre-Registration Decision:** Multimodal conversational timing and pre-extracted acoustic features are excluded from primary test evaluation because RQ1 is inconclusive ($d = 0.314$) and RQ2 multimodal ablation showed no significant error reduction ($\Delta\text{RMSE} = -0.003, p_{\text{Holm}} = 1.000$) and negligible interval width compression ($-1.03\%$).

### Refit Protocol (Train + Dev Pooling)
1. **Training Pool:** Pool official Train ($N=107$) and official Dev ($N=34$ on disk) splits $\to N_{\text{fit}} = 141$ intact sessions.
2. **Transformation:**
   - Fit `StandardScaler` and PCA ($K=20$) on the pooled $N=141$ validation sessions.
   - Apply frozen transformation to the held-out test sessions ($N_{\text{test}} = 47$).
3. **Regression Fitting:**
   - Fit `Ridge(alpha=10.0)` on pooled $N=141$ true PHQ-8 scores.
4. **Conformal Calibration:**
   - Calibrate conformal quantile $\hat{q}_{\alpha}$ on pooled Train+Dev cross-validated residuals ($N=141$).
   - Apply $\hat{q}_{\alpha}$ to form predictive intervals on test participant predictions.

---

## 4. Exact Pre-Specified Test Metrics

Every reported test metric will be evaluated with 95% confidence intervals ($B=2000$ participant bootstrap):

### Reference Baseline
- **Dummy Baseline:** Constant prediction using the pooled Train+Dev mean score ($\bar{y}_{\text{fit}}$ on $N=141$) evaluated against test participants (RMSE, MAE, AUROC $= 0.50$).

### Primary Continuous Regression Metrics
1. **RMSE:** Root Mean Squared Error (point estimate + 95% bootstrap CI).
2. **MAE:** Mean Absolute Error (point estimate + 95% bootstrap CI).
3. **Correlation:** Pearson $r$ and Spearman $\rho$ with true PHQ-8 scores.

### Binary Classification & Discrimination ($\text{PHQ-8} \ge 10$)
1. **AUROC:** Area Under the ROC Curve (point estimate + 95% bootstrap CI).
2. **F1 Score:** At in-fold frozen threshold ($\tau_{\text{F1}}$).
3. **Probability Calibration:** Brier score loss and Expected Calibration Error (ECE, 10 bins).

### Conformal Uncertainty & Triage
1. **Split Conformal Regression ($\alpha = 0.10$ and $\alpha = 0.20$):**
   - Empirical coverage (target $\ge 1 - \alpha$).
   - Mean predictive interval width $\pm$ SD.
   - Triage abstention rate ($W_{\max} = 8.0, \text{margin} = 0.5$).
   - Selective risk among autonomous predictions with 95% CI.
2. **Set-Valued Conformal Classification (LAC at $\alpha = 0.20$):**
   - Empirical coverage.
   - Singleton rate (autonomous throughput).
   - Abstain rate (with breakdown of $\{0, 1\}$ and $\emptyset$).
   - Error rate among singletons with 95% Wilson score CI ($N \ge 20$ required for non-degenerate reporting).

### Clinical Economic Cost
1. **Expected Cost per Patient:** Under standard clinical ratio ($C_{\text{FN}} = 5.0, C_{\text{FP}} = 1.0, C_{\text{abstain}} = 0.5$).
2. **Benchmark Comparison:** Evaluated against the default clinical policy ("escalate everyone" reference cost $= 0.500$).

---

## 5. Lock Mechanism & Anti-Peeking Enforcement

1. **Lock File:** `experiments/test_eval.lock`
2. **State Prior to Unlock:** Must NOT exist on disk.
3. **Unlock Invariant:** The command `--unlock-test` is strictly gated behind interactive investigator confirmation.
4. **Duplicate Execution Prevention:** Once created, `test_eval.lock` stores:
   - Git commit SHA at execution time.
   - Timestamp of evaluation.
   - SHA-256 hash of test prediction vector.
   - Any secondary execution raises an unbypassable `RuntimeError: Test evaluation has already been executed. Duplicate runs strictly prohibited.`
