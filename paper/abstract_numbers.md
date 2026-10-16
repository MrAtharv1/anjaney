# Abstract Headline Numbers

All numbers below trace to `paper/numbers.json` with source files. Intended for abstract and summary statements only.

| # | Claim | Value | Source key in `numbers.json` |
|---|-------|-------|------------------------------|
| 1 | Dataset size (intact sessions on disk) | N = 188 | `table1_dataset::total_disk_n` |
| 2 | Test-set AUROC (text-only Ridge, N=47) | 0.797 (95 % CI [0.652, 0.917]) | `table8_test_evaluation::model_auroc` |
| 3 | Test-set RMSE vs dummy baseline | 5.153 vs 6.415 | `table8_test_evaluation::model_rmse`, `::dummy_rmse` |
| 4 | Conformal coverage at α=0.10 (test) | 43/47 (91.5 %) | `table8_test_evaluation::conformal_alpha_10_cov` |
| 5 | Conformal abstention rate at α=0.10, W_max=10 (test) | 46/47 (97.9 %) | `table8_test_evaluation::conformal_alpha_10_abstain_wmax10` |
| 6 | LAC singleton accuracy on test (1 − error rate) | ~70.7 % (12/41 errors; Wilson CI [55.5 %, 82.4 %]) | `table8_test_evaluation::lac_alpha_20_singleton_error` |
| 7 | RQ1 — Cohen's *d* (inconclusive; gate not reached) | 0.314 (95 % CI [−0.039, 0.697]) | `rq1_gating::cohens_d` |
| 8 | RQ2 — ΔRMSE from adding timing (not significant) | −0.003 (95 % CI [−0.037, +0.031]) | `table3_rq2_paired::delta_rmse` |
| 9 | Diagnosis-prompt excess AUROC drop (empirical *p* < 0.01) | −0.060 vs matched random −0.020 | `table4_shortcuts::excess_drop_diagnosis` |
| 10 | ONNX int8 model size (SIMULATED benchmark) | 22.8 MB (vs fp32 89.2 MB) | `table9_ondevice::onnx_int8_size_mb` |

---

## Wording Rules
- Do not write "proves", "confirms", "beats", or "outperforms" unless the paired 95 % CI excludes zero.
- Write "p < 0.01" (never "p = 0.0").
- RQ1 conclusion: "inconclusive — the point estimate lies in a practically interesting range but the 95 % CI crosses zero and the study is underpowered for the pre-registered effect size."
- Conformal classification: "approximately 71 % accurate singleton predictions; not autonomous triage."
