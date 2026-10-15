# ANJANEY — Comprehensive Experimental Results

## 1. Confirmatory Research Questions
- **RQ1 Status:** `INCONCLUSIVE`
  - Observed Group Difference: `0.1209`
  - Permutation $p$-value: `0.0535`
  - Cohen's $d$: `0.305`
- **RQ2 (Multimodal Fusion vs Text-Only):** `CONFIRMATORY (RQ2)`
  - $\Delta$ RMSE: `-0.004`

## 2. Modality Ablation Table
| Modality | Head | RMSE [95% CI] | AUROC [95% CI] | AURC (Split) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `text_only` | `ridge` | 4.93 [4.36, 5.52] | 0.78 [0.68, 0.86] | 0.004 | EXPLORATORY |
| `text_only` | `lightgbm` | 4.90 [4.36, 5.48] | 0.75 [0.65, 0.84] | 0.000 | EXPLORATORY |
| `timing_only` | `ridge` | 5.99 [5.26, 6.80] | 0.48 [0.37, 0.59] | 0.000 | EXPLORATORY |
| `timing_only` | `lightgbm` | 5.83 [5.11, 6.55] | 0.53 [0.42, 0.62] | 0.000 | EXPLORATORY |
| `lexical_only` | `ridge` | 5.61 [5.05, 6.19] | 0.62 [0.53, 0.72] | 0.000 | EXPLORATORY |
| `lexical_only` | `lightgbm` | 5.71 [5.11, 6.29] | 0.63 [0.53, 0.72] | 0.000 | EXPLORATORY |
| `text_plus_timing` | `ridge` | 4.93 [4.35, 5.51] | 0.78 [0.68, 0.86] | 0.004 | CONFIRMATORY |
| `text_plus_timing` | `lightgbm` | 5.40 [4.74, 6.07] | 0.67 [0.56, 0.77] | 0.000 | CONFIRMATORY |

## 3. Missing-Data Stress Testing
| Masking Rate | Empirical Coverage | Mean Interval Width | Abstention Rate |
| :--- | :--- | :--- | :--- |
| 0% | 100.0% | 7.83 | 40.4% |
| 10% | 100.0% | 8.46 | 43.6% |
| 30% | 100.0% | 9.55 | 61.2% |
| 50% | 100.0% | 10.50 | 71.8% |
