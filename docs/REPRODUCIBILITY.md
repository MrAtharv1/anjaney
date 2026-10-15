# ANJANEY Reproduction Guide

To reproduce every table, metric, and figure from scratch:

```bash
# 1. Generate clean synthetic cohort
anjaney data generate-synthetic --n-participants 50 --output-dir data/synthetic

# 2. Build check-in windows and leak-proof splits
anjaney data build --source synthetic --output-dir data/processed --window-size 5

# 3. Extract sentence embeddings and conversational dynamics
anjaney features build --windows-file data/processed/windows.parquet --output-file data/processed/features.parquet --groups text_embed,timing,lexical

# 4. Train population baselines and run permutation checks
anjaney train baseline --features-file data/processed/features.parquet --splits-file data/processed/splits.json

# 5. Execute pre-registered RQ1 gating analysis
anjaney rq1 run --features-file data/processed/features.parquet --prereg-file configs/prereg.yaml

# 6. Calibrate split-conformal prediction intervals
anjaney conformal run --features-file data/processed/features.parquet --splits-file data/processed/splits.json --alpha 0.10

# 7. Execute paired evaluation grid and compile RESULTS.md
anjaney eval run --features-file data/processed/features.parquet --splits-file data/processed/splits.json

# 8. Export INT8 model bundle and on-device latency benchmark
anjaney export bundle --features-file data/processed/features.parquet --output-dir data/processed/bundle
anjaney export benchmark --output-dir docs