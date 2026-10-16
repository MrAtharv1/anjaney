# Protocol Amendments & Superseded Results Log

This document records formal deviations, methodological amendments, and supersessions across research stages.

---

## Amendment 1: Pre-Stage A Results Superseded

**Date:** 2026-10-16  
**Status:** ACTIVE  
**Superseded Scope:** All experimental tables, metric estimates, and reports produced prior to Stage A are marked as **SUPERSEDED**. Old files are retained on disk for provenance and auditability.

### Rationales for Supersession:

1. **Strict In-Fold Acoustic Imputation**:
   - *Previous state*: Initial acoustic feature building applied dataset-wide median imputation for missing values (affecting 6 windows / 0.1%).
   - *Amended state*: Removed whole-dataset imputation from src/anjaney/features/acoustic.py. All missing feature values are imputed strictly within each cross-validation fold using training-fold medians only (src/anjaney/models/participant_agg.py:fit_transform_fold).

2. **Acoustic Feature List Freezing Protocol Disclosure**:
   - The 16 acoustic features (configs/acoustic_features.yaml) were frozen in code at commit d55c76e during feature implementation rather than pre-registered in a separate configuration file prior to extraction.
   - Modality classification: The acoustic voice modality is strictly classified as an **exploratory extension** for benchmarking against AVEC2017 literature. It is entirely offline and omitted from the client deployment bundle (undle.json).

3. **Session 440 Server-Side Archive Audit**:
   - Session 440 is listed in the official AVEC2017 dev_split_Depression_AVEC2017.csv (=35$).
   - Direct HTTP byte inspection of https://dcapswoz.ict.usc.edu/wwwdaicwoz/440_P.zip (619,905,024 bytes) confirmed that the archive terminates mid-stream at byte 78,273,750 inside 440_CLNF_hog.bin and lacks the central directory (PK\x05\x06), 440_TRANSCRIPT.csv, 440_COVAREP.csv, and 440_FORMANT.csv.
   - Per investigator directive, session 440 is documented as truncated at the USC server source and excluded from downstream analysis.
   - The dataset reconciliation on disk comprises 188 intact sessions (Train: 107, Dev: 34, Test: 47), cataloged in data/daic_woz/MANIFEST.json.

4. **Enhanced Data Privacy Pre-Commit Guard**:
   - .pre-commit-config.yaml was upgraded to include data/daic_woz/ alongside data/(raw|interim|processed)/.

5. **Acoustic Extraction Safety Compliance**:
   - `scripts/extract_daic_acoustics.py` was refactored to enforce max_workers=2, remove spoofed User-Agent headers, halt on HTTP 403 Forbidden, and apply exponential backoff on HTTP 429 Too Many Requests.

---

## Amendment 2: Pre-declared Deployed Model Configuration (`prereg-v2`)

**Date:** 2026-10-16  
**Git Tag:** `prereg-v2`  
**Configuration File:** `configs/final_config.yaml`  
**Status:** ACTIVE & FROZEN

### Formal Changes to Deployed Architecture:

1. **Selection of Parsimonious `text_only` Ridge Head**:
   - *Previous protocol assumption*: Anticipated multimodal deployment fusing conversational timing with text embeddings (`text_plus_timing`).
   - *Amended decision*: Because RQ1 is `INCONCLUSIVE` (Cohen's $d = 0.314$, failing the pre-registered $d \ge 0.50$ criterion) and the confirmatory RQ2 paired ablation showed zero statistically significant reduction in prediction error ($\Delta\text{RMSE} = -0.003, p_{\text{Holm}} = 1.000$; $\Delta\text{AUROC} = -0.0004, p_{\text{Holm}} = 1.000$) with clinically negligible interval width change ($-1.03\%$, Nadeau-Bengio corrected resampled $p = 0.452$), the final deployed screening model is formally pre-declared as the simpler, lower-complexity **`text_only` Ridge** ($\alpha = 10.0$, 20 PCA components over 384-dimensional MiniLM embeddings).

2. **Exploratory Reclassification of Timing and Acoustic Features**:
   - Conversational timing (speech pauses, turn latency) and acoustic voice parameters (COVAREP glottal features, formants) remain strictly **exploratory offline benchmarks** and are NEVER included in the primary risk estimation pipeline or student check-in client bundles.
   - Typed-timing is explicitly disclosed as unvalidated on real student typing dynamics.

3. **Operating Point Selection Frozen Exclusively on Validation Folds**:
   - Operating points for uncertainty triage and decision thresholds are frozen strictly on validation folds ($N=141$) prior to any test set evaluation:
     - Primary regression conformal threshold: $\tau = 10.0$, $\alpha = 0.10$ / $0.20$, $W_{\max} = 8.0$, margin $= 0.5$.
     - Set-valued conformal classification: LAC at $\alpha = 0.20$ (nominal 80% coverage).
     - Standard clinical economic operating point: $W_{\max} = 10.0$ ($90.8\%$ abstention, $7.7\%$ selective risk).

4. **Strict "Exactly One Test Run" Protocol**:
   - The test split ($N=47$) remains locked under `experiments/test_eval.lock` until the investigator explicitly invokes `--unlock-test`.
   - Exactly ONE evaluation on the held-out test split is permitted. No post-hoc tuning, model switching, or hyperparameter changes are permitted after unlock.

