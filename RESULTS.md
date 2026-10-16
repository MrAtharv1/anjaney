# ANJANEY — Comprehensive Experimental Results

## 1. Confirmatory Research Questions
- **RQ1 Status:** `INCONCLUSIVE`
  - Observed Group Difference: `0.1235`
  - Permutation $p$-value: `0.0461`
  - Cohen's $d$: `0.314`
- **RQ2 (Confirmatory Paired Ablation: `text_plus_timing` vs `text_only` on Ridge):** `CONFIRMATORY (RQ2)`
  - $\Delta$ RMSE: `-0.003` [95% CI: `-0.037`, `0.031`] ($p_{\text{Holm}} = 1.000$)
  - $\Delta$ AUROC: `-0.0004` [95% CI: `-0.0143`, `0.0143`] ($p_{\text{Holm}} = 1.000$)
  - $\Delta$ Interval Width: `-0.134` $\pm$ `0.341` points (`-1.03%` relative change; 50-fold paired Nadeau-Bengio corrected resampled $t = -0.758, p = 0.452$; naive participant bootstrap: [95% CI: `-0.168`, `-0.098`])
  - Confirmatory Assessment: Primary prediction error differences (RMSE and AUROC) include zero after Holm-Bonferroni correction ($p_{\text{Holm}} = 1.000$). While interval width exhibits a minor point reduction of 0.13 score points, a ~1% width change is clinically negligible and fails to reach statistical significance under the Nadeau-Bengio corrected resampled t-test ($p = 0.452$). Multimodal timing is therefore not justified for deployment.

## 2. Modality Ablation Table
| Modality | Head | RMSE [95% CI] | AUROC [95% CI] | AURC (Split) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dummy` | `mean` | 5.67 [5.05, 6.33] | 0.50 [0.50, 0.50] | -- | BASELINE |
| `text_only` | `ridge` | 4.92 [4.31, 5.55] | 0.78 [0.68, 0.86] | 0.004 | EXPLORATORY |
| `text_only` | `lightgbm` | 5.43 [4.77, 6.14] | 0.67 [0.56, 0.77] | 0.000 (DEGENERATE)* | EXPLORATORY |
| `timing_only` | `ridge` | 5.92 [5.27, 6.59] | 0.56 [0.45, 0.67] | 0.004 | EXPLORATORY |
| `timing_only` | `lightgbm` | 5.95 [5.34, 6.62] | 0.52 [0.42, 0.62] | 0.000 (DEGENERATE)* | EXPLORATORY |
| `lexical_only` | `ridge` | 5.24 [4.70, 5.81] | 0.69 [0.59, 0.79] | 0.004 | EXPLORATORY |
| `lexical_only` | `lightgbm` | 5.43 [4.82, 6.03] | 0.63 [0.52, 0.73] | 0.000 (DEGENERATE)* | EXPLORATORY |
| `acoustic_only` | `ridge` | 5.47 [4.80, 6.14] | 0.63 [0.52, 0.72] | 0.000 (DEGENERATE)* | EXPLORATORY |
| `acoustic_only` | `lightgbm` | 5.73 [5.03, 6.39] | 0.56 [0.46, 0.67] | 0.000 (DEGENERATE)* | EXPLORATORY |
| `text_plus_timing` | `ridge` | 4.91 [4.31, 5.52] | 0.78 [0.68, 0.86] | 0.004 | CONFIRMATORY |
| `text_plus_timing` | `lightgbm` | 5.20 [4.54, 5.89] | 0.71 [0.60, 0.79] | 0.000 (DEGENERATE)* | CONFIRMATORY |
| `text_plus_timing_plus_acoustic` | `ridge` | 4.87 [4.29, 5.47] | 0.78 [0.68, 0.86] | 0.000 (DEGENERATE)* | EXPLORATORY |
| `text_plus_timing_plus_acoustic` | `lightgbm` | 5.34 [4.66, 6.06] | 0.66 [0.54, 0.76] | 0.000 (DEGENERATE)* | EXPLORATORY |

\* *AURC = 0.000 denotes a degenerate triage artifact where 100% of participants are escalated to human clinicians ($W_{\max} = 8.0$ < interval width), rendering selective risk identically zero.*

## 3. Missing-Data Stress Testing
| Masking Rate | Empirical Coverage | Mean Interval Width | Abstention Rate |
| :--- | :--- | :--- | :--- |
| 0% | 100.0% | 7.83 | 40.4% |
| 10% | 100.0% | 8.46 | 43.6% |
| 30% | 100.0% | 9.55 | 61.2% |
| 50% | 100.0% | 10.50 | 71.8% |
