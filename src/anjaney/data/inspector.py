"""Dataset inspection reporting aggregate statistics with strict zero-text privacy."""

from collections import Counter
from pathlib import Path

import numpy as np

from anjaney.data.daic_loader import DAICLoader
from anjaney.data.schemas import DatasetAnomalyReport
from anjaney.data.synthetic import SyntheticCohortGenerator


def run_inspection(
    source: str = "synthetic",
    daic_root: Path | str | None = None,
    synthetic_count: int = 50,
) -> tuple[DatasetAnomalyReport, str]:
    if source == "daic":
        loader = DAICLoader(daic_root)
        sessions = list(loader.iter_sessions())
        if not sessions:
            raise RuntimeError(f"No DAIC-WOZ sessions could be loaded from root: {loader.root}")
        is_synthetic = False
    elif source == "synthetic":
        gen = SyntheticCohortGenerator(n_participants=synthetic_count)
        sessions = gen.generate()
        is_synthetic = True
    else:
        raise ValueError(f"Unknown data source: {source}. Choose 'synthetic' or 'daic'.")

    report = DatasetAnomalyReport()
    report.total_sessions = len(sessions)

    # Participant splits & cross-split duplicate checks
    split_memberships: dict[str, set[str]] = {}
    splits_counter: Counter[str] = Counter()
    genders: Counter[str] = Counter()
    phq_scores: list[float] = []
    binary_positives_by_split: dict[str, list[int]] = {}
    participant_turn_counts: list[int] = []
    durations: list[float] = []

    for s in sessions:
        splits_counter[s.official_split] += 1
        split_memberships.setdefault(s.official_split, set()).add(s.participant_id)

        if s.gender:
            genders[s.gender] += 1
        if s.phq8_score is not None:
            phq_scores.append(s.phq8_score)
        else:
            report.missing_labels += 1

        if s.phq8_binary is not None:
            binary_positives_by_split.setdefault(s.official_split, []).append(s.phq8_binary)

        # Anomaly checks on turns
        p_turns = s.participant_turns
        participant_turn_counts.append(len(p_turns))
        if len(p_turns) < 5:
            report.sessions_under_min_turns += 1

        durations.append(s.total_duration_s)

        for i, turn in enumerate(s.turns):
            if turn.end_s <= turn.start_s:
                report.zero_or_negative_durations += 1
            if i > 0 and turn.start_s < s.turns[i - 1].start_s:
                report.non_monotonic_timestamps += 1
            if i > 0 and turn.start_s < s.turns[i - 1].end_s:
                # Flag overlapping turns
                report.overlapping_turns += 1

    report.sessions_per_split = dict(splits_counter)

    # Check cross-split participant leakage
    all_splits = list(split_memberships.keys())
    for i in range(len(all_splits)):
        for j in range(i + 1, len(all_splits)):
            overlap = split_memberships[all_splits[i]].intersection(
                split_memberships[all_splits[j]]
            )
            report.duplicate_participant_across_splits += len(overlap)

    # Usable signals check
    report.usable_signals = {
        "content_free_latency": report.overlapping_turns < (0.2 * sum(participant_turn_counts)),
        "within_session_position_tracking": report.non_monotonic_timestamps == 0,
        "participant_only_embeddings": True,
        "phq8_regression_target": report.missing_labels == 0,
    }

    # Generate Markdown documentation containing strictly aggregate statistics
    banner = (
        "> ⚠️ **SYNTHETIC DATA BANNER**: This inspection report was generated using synthetic data fixtures.\n\n"
        if is_synthetic
        else ""
    )

    phq_arr = np.array(phq_scores) if phq_scores else np.array([0.0])
    dur_arr = np.array(durations) if durations else np.array([0.0])
    turns_arr = np.array(participant_turn_counts) if participant_turn_counts else np.array([0])

    md = f"""# ANJANEY — Dataset Schema & Integrity Report

{banner}## 1. Split Distribution & Label Statistics
- **Total Sessions Analyzed:** {report.total_sessions}
- **Sessions Per Split:** {dict(report.sessions_per_split)}
- **PHQ-8 Score Distribution:**
  - Mean: {np.mean(phq_arr):.2f} ± {np.std(phq_arr):.2f}
  - Median: {np.median(phq_arr):.1f} (IQR: {np.percentile(phq_arr, 25):.1f} - {np.percentile(phq_arr, 75):.1f})
  - Min / Max: {np.min(phq_arr):.0f} / {np.max(phq_arr):.0f}
- **Binary High-Risk Prevalence (PHQ-8 ≥ 10):**
"""
    for sp, bin_vals in binary_positives_by_split.items():
        rate = np.mean(bin_vals) * 100 if bin_vals else 0.0
        md += f"  - `{sp}`: {rate:.1f}% positive ({sum(bin_vals)} / {len(bin_vals)})\n"

    md += f"""
## 2. Demographic Breakdown
- **Reported Gender:** {dict(genders)}

## 3. Session Volume & Timing Aggregates
- **Participant Turns per Session:**
  - Min: {np.min(turns_arr)}
  - Median: {np.median(turns_arr):.1f}
  - Max: {np.max(turns_arr)}
- **Session Duration (seconds):**
  - Min: {np.min(dur_arr):.1f}s
  - Median: {np.median(dur_arr):.1f}s
  - Max: {np.max(dur_arr):.1f}s

## 4. Anomaly & Integrity Audit
- **Non-monotonic Timestamps:** {report.non_monotonic_timestamps}
- **Overlapping Turns:** {report.overlapping_turns}
- **Zero or Negative Turn Durations:** {report.zero_or_negative_durations}
- **Sessions Under Threshold (< 5 participant turns):** {report.sessions_under_min_turns}
- **Missing Clinical Labels:** {report.missing_labels}
- **Cross-Split Participant Leaks:** {report.duplicate_participant_across_splits}

## 5. Which Proposed Signals Are Actually Usable
- **Frozen Sentence Embeddings (Participant-Only):** {report.usable_signals.get("participant_only_embeddings")}
- **Content-Free Latency & Turn Pacing:** {report.usable_signals.get("content_free_latency")}
- **Session-Relative Drift Tracking:** {report.usable_signals.get("within_session_position_tracking")}
- **PHQ-8 Regression Calibration:** {report.usable_signals.get("phq8_regression_target")}
"""
    return report, md
