# Table 8: Single Confirmatory Evaluation on Held-Out Test Split ($N=47$)

| Metric Category | Performance Metric | Deployed Model (`text_only` Ridge) [95% CI] | Dummy Baseline (Train+Dev Mean) [95% CI] |
| :--- | :--- | :--- | :--- |
| **Continuous Error** | RMSE | **5.15** [4.15, 6.08] | 6.42 [5.23, 7.58] |
| | MAE | **4.20** [3.35, 5.09] | 5.42 [4.47, 6.45] |
| | Pearson $r$ | **0.65** [0.50, 0.77] | 0.00 [0.00, 0.00] |
| **Screening Discrimination** | AUROC (PHQ-8 $\ge 10$) | **0.797** [0.652, 0.917] | 0.500 [0.500, 0.500] |
| | $F_1$ Score (in-fold frozen $\tau = 7.1$) | **0.632** [0.429, 0.791] | 0.000 [0.000, 0.000] |
| | $F_1$ Score (standard screening $\tau = 10.0$) | **0.400** [0.111, 0.640] | 0.000 [0.000, 0.000] |
| | Brier Calibration Loss | **0.196** [0.114, 0.287] | 0.280 [0.160, 0.400] |
| **Conformal Regression** | Empirical Coverage ($\alpha = 0.10$, target 90%) | **91.5%** (43/47) | -- |
| | Mean Interval Width ($\alpha = 0.10$) | **14.84** $\pm$ 2.06 score points | -- |
| | Triage Abstention Rate ($W_{\max} = 10.0$) | **97.9%** (46/47) | 100.0% |
| **Classification Conformal** | Empirical Coverage (LAC at $\alpha = 0.20$, target 80%) | **74.5%** (35/47) | -- |
| | Autonomous Throughput (Singletons) | **87.2%** (41/47) | 0.0% |
| | Error Among Singletons ($N=41$) | **29.3%** (12/41) [95% Wilson: 17.6%, 44.5%] | -- |

*Confirmatory Provenance: Evaluated exactly ONCE at commit `a974699`. Predictions vector SHA-256: `f2fdf653f1af7975...` locked in `experiments/test_eval.lock`.*
