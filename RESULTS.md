# ANJANEY — Comprehensive Experimental Results

## 1. Confirmatory Research Questions
- **RQ1 Status:** `SUPPORTED`
  - Observed Group Difference: `0.1528`
  - Permutation $p$-value: `0.0474`
  - Cohen's $d$: `0.613`
- **RQ2 (Multimodal Fusion vs Text-Only):** `CONFIRMATORY (RQ2)`
  - $\Delta$ RMSE: `-0.107`

## 2. Modality Ablation Table
| Modality | Head | RMSE [95% CI] | AUROC [95% CI] | AURC (Split) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `text_only` | `ridge` | 4.69 [3.73, 5.57] | 0.99 [0.97, 1.00] | 0.000 | EXPLORATORY |
| `text_only` | `lightgbm` | 3.81 [3.13, 4.42] | 0.98 [0.92, 1.00] | 0.009 | EXPLORATORY |
| `timing_only` | `ridge` | 4.59 [3.82, 5.30] | 1.00 [1.00, 1.00] | 0.000 | EXPLORATORY |
| `timing_only` | `lightgbm` | 4.14 [3.04, 5.45] | 0.93 [0.77, 1.00] | 0.009 | EXPLORATORY |
| `lexical_only` | `ridge` | 3.61 [2.78, 4.52] | 0.99 [0.97, 1.00] | 0.000 | EXPLORATORY |
| `lexical_only` | `lightgbm` | 4.42 [3.10, 5.83] | 0.99 [0.94, 1.00] | 0.000 | EXPLORATORY |
| `text_plus_timing` | `ridge` | 4.59 [3.64, 5.47] | 1.00 [1.00, 1.00] | 0.000 | CONFIRMATORY |
| `text_plus_timing` | `lightgbm` | 3.72 [3.14, 4.26] | 0.98 [0.93, 1.00] | 0.016 | CONFIRMATORY |

## 3. Missing-Data Stress Testing
| Masking Rate | Empirical Coverage | Mean Interval Width | Abstention Rate |
| :--- | :--- | :--- | :--- |
| 0% | 100.0% | 9.00 | 46.0% |
| 10% | 100.0% | 9.60 | 46.0% |
| 30% | 100.0% | 10.80 | 100.0% |
| 50% | 100.0% | 12.00 | 100.0% |
