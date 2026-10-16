# Table 6: Set-Valued Conformal Classification (Validation Folds, $N=141$)

| Engine | Nominal $\alpha$ | Target Coverage | Empirical Coverage | Autonomous Throughput (Singletons) | Abstention Rate (Total) | Abstention: Both $\{0, 1\}$ | Abstention: Empty $\emptyset$ | Singleton Error Rate [95% Wilson CI] | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LAC** | 0.20 | 80.0% | 89.4% | 63.1% (89/141) | 36.9% (52/141) | 52 | 0 | 16.9% [10.5%, 26.0%] ($N=89$) | **VALID** |
| **APS (Det.)** | 0.20 | 80.0% | 100.0% | 0.0% (0/141) | 100.0% (141/141) | 141 | 0 | Undefined ($N=0$) | **DEGENERATE**$^*$ |

\* *APS Degeneracy Note: Deterministic APS calibrates to $\tau = 1.0$ whenever base classifier training accuracy is imperfect, forcing all prediction sets to $\{0, 1\}$ ($100\%$ abstention, $0$ singletons). It is marked DEGENERATE/anecdotal per protocol ($N < 20$ evaluates to zero statistical power).*
