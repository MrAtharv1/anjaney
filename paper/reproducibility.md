# Reproducibility Appendix & Protocol Audit

This document provides the exact technical specifications, cryptographic digests, random seed sequences, and execution steps necessary to reproduce every result, table, and figure in the ANJANEY manuscript.

---

## 1. Cryptographic File Integrity Digests (SHA-256)

All experiments, cross-validation folds, feature tables, and evaluation locks are pinned to these exact SHA-256 digests:

| File Artifact | SHA-256 Checksum | Protocol Role |
| :--- | :--- | :--- |
| [`configs/final_config.yaml`](file:///c:/Users/intel/vscodews/anjaney/configs/final_config.yaml) | `e67202888180b46a04cf1c1575dbcd31c789bd9f881027c9838b3bd4c68875de` | Frozen Deployed Model Architecture |
| [`configs/prereg.yaml`](file:///c:/Users/intel/vscodews/anjaney/configs/prereg.yaml) | `80d93139ef298fb449c2aa8f51c78a186a4c5befaab063075ebae138838894d7` | Pre-registered Statistical Hypotheses |
| [`configs/acoustic_features.yaml`](file:///c:/Users/intel/vscodews/anjaney/configs/acoustic_features.yaml) | `634b49ec7542c79afae0a9e31497300cb1fdba5c04834aa97f6df8e300c824ff` | Exploratory Acoustic Feature Spec |
| [`data/processed/splits.json`](file:///c:/Users/intel/vscodews/anjaney/data/processed/splits.json) | `5cfc1882b8df21c67f49fb6d507cdc91753bd457c5f5b63f68967d23127da3f1` | 50-Fold Repeated CV & Official Splits |
| [`data/processed/features.parquet`](file:///c:/Users/intel/vscodews/anjaney/data/processed/features.parquet) | `73402f8c9e1427dbb8f33b2c11ed513e5c46df0845d99296c0c617c080b673c2` | Master Extracted Feature Matrix |
| [`experiments/test_eval.lock`](file:///c:/Users/intel/vscodews/anjaney/experiments/test_eval.lock) | `3309a47d251d54d9095c5bc39e2ee1cb53b0e1e612eb318f773663be826fcb5b` | Single Test Evaluation Lock Harness |

---

## 2. Version Control Provenance

- **Pre-Registration Tag**: `prereg-v2` (Git Commit: `ae3aafc`)
  - Confirmed ancestor of evaluation HEAD via: `git merge-base --is-ancestor prereg-v2 HEAD`
- **Single Test Evaluation Commit**: `a974699`
- **Test Set Prediction Vector SHA-256**:
  `f2fdf653f1af797516f9784dba3e7d089ae648ff4e07177dbd01480f55168e2e`
- **Lock Invariant**: The lock harness (`experiments/test_eval.lock`) forbids duplicate runs on the test partition. Any subsequent execution aborts immediately with a `RuntimeError`.

---

## 3. Computational Environment & Dependencies

- **Python Version**: `3.11.x` (CPython 64-bit on Windows / Linux / macOS)
- **Primary Libraries**:
  - `numpy >= 1.26.0`
  - `pandas >= 2.1.0`
  - `scipy >= 1.11.0`
  - `scikit-learn >= 1.3.0`
  - `statsmodels >= 0.14.0`
  - `pyarrow >= 14.0.0`
  - `matplotlib >= 3.8.0`
  - `fastparquet >= 2023.10.0`
- **Node.js Environment** (Client parity verification):
  - `Node.js >= 18.0.0`
  - `npm >= 9.0.0`

---

## 4. End-to-End Step-by-Step Reproduction Instructions

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/anjaney.git
cd anjaney

# Checkout the pre-registered release tag
git checkout prereg-v2

# Install Python dependencies in virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Step 2: Reproduce Preflight Audit & Dataset Manifest
```bash
# Verify integrity of raw data downloads and exclusion of session 440
python -c "
import json
with open('data/daic_woz/MANIFEST.json') as f:
    m = json.load(f)
assert m['splits_on_disk']['dev'] == 34
assert m['session_440_status']['status'] == 'TRUNCATED_AT_SOURCE'
print('Dataset manifest verified successfully.')
"
```

### Step 3: Reproduce RQ1 Gating Analysis
```bash
# Executes within-session drift gating across 10,000 permutations and 2,000 bootstraps
anjaney baseline rq1 --features data/processed/features.parquet --prereg configs/prereg.yaml
```

### Step 4: Reproduce RQ2 Ablation Grid & Conformal Statistics
```bash
# Runs 50-fold repeated CV across modalities and Nadeau-Bengio interval width test
python -c "
from anjaney.eval.grid import EvaluationGridRunner
runner = EvaluationGridRunner('data/processed/features.parquet', 'data/processed/splits.json', seed=42)
results = runner.run_grid()
print('Ablation grid completed.')
"
```

### Step 5: Reproduce Shortcut Controls (Diagnosis Drop & Session ID)
```bash
# Run diagnosis query ablation and session ID permutation tests
python scripts/build_paper_pack.py
```

### Step 6: Verify Client Parity Fixtures (TypeScript ONNX Web)
```bash
# Run Jest parity test suite in web application
cd apps/web
npm test
cd ../..
```

### Step 7: Linting and Unit Test Suite
```bash
# Code quality check
ruff check .

# Fast unit and statistical test suite
pytest -m "not slow"
```

---

## 5. Random Seed Registry

| Component | Random Seed | Target Operation |
| :--- | :--- | :--- |
| Participant CV Folds | `42` | Stratified 5-fold CV splits generation |
| Participant Bootstrap | `42` | Resampling indices for 95% CIs ($B=2000$) |
| Permutation Tests | `42` | Label shuffling for null drift distribution ($M=10000$) |
| Model Heads (Ridge) | `42` | In-fold regression solver initialization |
| Conformal Split | `42` | Residual ranking and quantile computation |
| Client Fixture Generation | `42` | ONNX web parity clamp fixtures |
