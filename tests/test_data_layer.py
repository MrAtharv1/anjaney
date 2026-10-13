"""Tests for schemas, loaders, anomaly detection, and privacy guarantees."""

from pathlib import Path

import pytest

from anjaney.data.daic_loader import DAICLoader
from anjaney.data.inspector import run_inspection
from anjaney.data.schemas import SpeakerRole, Turn
from anjaney.data.synthetic import HIGH_RISK_TEMPLATES, LOW_RISK_TEMPLATES, SyntheticCohortGenerator


def test_synthetic_cohort_generation_and_schema():
    gen = SyntheticCohortGenerator(n_participants=10, seed=123)
    sessions = gen.generate()
    assert len(sessions) == 10
    for s in sessions:
        assert s.is_synthetic is True
        assert s.phq8_score is not None
        assert 0.0 <= s.phq8_score <= 24.0
        assert s.phq8_binary in [0, 1]
        assert len(s.turns) > 0
        assert len(s.participant_turns) > 0


def test_split_independence_no_participant_overlap():
    gen = SyntheticCohortGenerator(n_participants=30, seed=42)
    sessions = gen.generate()
    splits: dict[str, set[str]] = {}
    for s in sessions:
        splits.setdefault(s.official_split, set()).add(s.participant_id)

    assert "train" in splits and "dev" in splits and "test" in splits
    # No participant may appear in more than one partition
    assert len(splits["train"].intersection(splits["dev"])) == 0
    assert len(splits["train"].intersection(splits["test"])) == 0
    assert len(splits["dev"].intersection(splits["test"])) == 0


def test_daic_loader_unconfigured_error():
    loader = DAICLoader(daic_root=None)
    with pytest.raises(ValueError, match="DAIC_ROOT is not configured"):
        loader.validate_root()


def test_tag_cleaning_in_transcript():
    raw_text = (
        "I felt <laughter> really happy </laughter> about my project [laughter] scrubbed_entry_123"
    )
    cleaned, stripped_count = DAICLoader._clean_text(raw_text)
    assert "<laughter>" not in cleaned
    assert "[laughter]" not in cleaned
    assert "scrubbed_entry" not in cleaned
    assert "I felt really happy about my project" == cleaned
    assert stripped_count >= 3


def test_dataset_report_contains_zero_raw_utterance_text(tmp_path: Path):
    out_file = tmp_path / "schema_report.md"
    report, report_md = run_inspection(source="synthetic", synthetic_count=20)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    content = out_file.read_text(encoding="utf-8")
    assert report.total_sessions == 20
    assert "ANJANEY — Dataset Schema & Integrity Report" in content

    # STRICT PRIVACY AUDIT: Assert that no template utterances leak into reports
    for template in LOW_RISK_TEMPLATES + HIGH_RISK_TEMPLATES:
        assert template not in content, f"Privacy violation: raw text template leaked: {template}"


def test_turn_duration_property():
    turn = Turn(speaker=SpeakerRole.PARTICIPANT, start_s=10.0, end_s=14.5, text="Sample text")
    assert pytest.approx(turn.duration_s, 0.01) == 4.5
