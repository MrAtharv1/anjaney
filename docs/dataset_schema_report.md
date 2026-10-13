# ANJANEY — Dataset Schema & Integrity Report

> ⚠️ **SYNTHETIC DATA BANNER**: This inspection report was generated using synthetic data fixtures.

## 1. Split Distribution & Label Statistics
- **Total Sessions Analyzed:** 50
- **Sessions Per Split:** {'train': 30, 'test': 10, 'dev': 10}
- **PHQ-8 Score Distribution:**
  - Mean: 8.54 ± 6.65
  - Median: 7.0 (IQR: 4.0 - 13.5)
  - Min / Max: 0 / 22
- **Binary High-Risk Prevalence (PHQ-8 ≥ 10):**
  - `train`: 33.3% positive (10 / 30)
  - `test`: 40.0% positive (4 / 10)
  - `dev`: 20.0% positive (2 / 10)

## 2. Demographic Breakdown
- **Reported Gender:** {'F': 24, 'M': 26}

## 3. Session Volume & Timing Aggregates
- **Participant Turns per Session:**
  - Min: 18
  - Median: 23.0
  - Max: 28
- **Session Duration (seconds):**
  - Min: 172.6s
  - Median: 236.6s
  - Max: 338.2s

## 4. Anomaly & Integrity Audit
- **Non-monotonic Timestamps:** 0
- **Overlapping Turns:** 0
- **Zero or Negative Turn Durations:** 0
- **Sessions Under Threshold (< 5 participant turns):** 0
- **Missing Clinical Labels:** 0
- **Cross-Split Participant Leaks:** 0

## 5. Which Proposed Signals Are Actually Usable
- **Frozen Sentence Embeddings (Participant-Only):** True
- **Content-Free Latency & Turn Pacing:** True
- **Session-Relative Drift Tracking:** True
- **PHQ-8 Regression Calibration:** True
