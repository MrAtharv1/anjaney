# ANJANEY Red-Team Security & Integrity Audit

## Audit Findings & Verification Checklist
- [x] **Zero Raw Text in Logs/Reports:** Verified by `test_dataset_report_contains_zero_raw_utterance_text`. Inspection reports contain exclusively aggregate statistics.
- [x] **No Protected Data in Git History:** Verified by `.pre-commit-config.yaml` privacy guard blocking `data/raw`, `data/interim`, and `data/processed`.
- [x] **Zero Participant Overlap Across CV Partitions:** Verified by `assert_no_participant_overlap()` called in all preprocessing and evaluation modules.
- [x] **In-Fold Fitting Only:** All scalers, PCA projections, position conditioners, and conformal quantiles are fit strictly on training partitions inside each cross-validation fold.
- [x] **Single Official Test Set Evaluation:** Enforced cryptographically by `OfficialTestLock` (`experiments/test_eval.lock`).
- [x] **Non-Diagnostic Language Compliance:** Verified by `test_language_safety_banned_words_guard`. All user-facing strings are free of clinical diagnostic terms.
- [x] **Deterministic Crisis Override:** Verified by `test_crisis_guard_deterministic_override`. Matches bypass ML scoring and display Tele-MANAS (14416).