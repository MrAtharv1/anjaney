# Table 3: RQ2 Confirmatory Paired Differences (`text_plus_timing` vs `text_only` on Ridge)

| Evaluation Metric | Point Difference | 95% Bootstrap CI | Statistical Test | Adjusted $p$-value | Confirmatory Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\Delta$ RMSE | -0.003 | [-0.037, 0.031] | Participant bootstrap | $p_{\text{Holm}} = 1.000$ | Not Significant |
| $\Delta$ AUROC | -0.0004 | [-0.0143, 0.0143] | Participant bootstrap | $p_{\text{Holm}} = 1.000$ | Not Significant |
| $\Delta$ Interval Width | -0.134 $\pm$ 0.341 | [-1.03% rel.] | Nadeau-Bengio resampled $t$ | $p = 0.452$ | Clinically Negligible / Not Sig. |

*Protocol Assessment: Confirmatory prediction error differences cross zero after Holm-Bonferroni correction. Interval width compression (-1.03%) is not statistically significant under the Nadeau-Bengio test accounting for fold dependency ($t = -0.758, p = 0.452$). Parsimonious `text_only` Ridge selected for deployment.*
