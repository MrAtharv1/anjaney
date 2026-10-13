"""Tests for cleaning, windowing, and leak-free StratifiedGroupKFold partitions."""

import pytest

from anjaney.data.schemas import Session, SpeakerRole, Turn
from anjaney.data.synthetic import SyntheticCohortGenerator
from anjaney.preprocess.clean import clean_session
from anjaney.preprocess.splits import assert_no_participant_overlap, build_splits
from anjaney.preprocess.windows import WindowBuilder


def test_clean_session_timestamps_and_text():
    turns = [
        Turn(speaker=SpeakerRole.INTERVIEWER, start_s=10.0, end_s=14.0, text="  Hi   there! "),
        Turn(
            speaker=SpeakerRole.PARTICIPANT,
            start_s=16.0,
            end_s=20.0,
            text="I'm feeling    okay. ",
        ),
    ]
    raw_session = Session(participant_id="P_001", turns=turns)
    cleaned = clean_session(raw_session)

    # First turn starts at 0.0s
    assert cleaned.turns[0].start_s == 0.0
    assert cleaned.turns[0].end_s == 4.0
    assert cleaned.turns[0].text == "Hi there!"

    # Second turn rebased
    assert cleaned.turns[1].start_s == 6.0
    assert cleaned.turns[1].end_s == 10.0
    assert cleaned.turns[1].text == "I'm feeling okay."


def test_window_builder_never_mixes_participants():
    turns1 = [
        Turn(speaker=SpeakerRole.PARTICIPANT, start_s=float(i), end_s=float(i + 1), text=f"T{i}")
        for i in range(10)
    ]
    s1 = Session(participant_id="P_A", turns=turns1, phq8_score=5.0)

    turns2 = [
        Turn(speaker=SpeakerRole.PARTICIPANT, start_s=float(i), end_s=float(i + 1), text=f"T{i}")
        for i in range(10)
    ]
    s2 = Session(participant_id="P_B", turns=turns2, phq8_score=15.0)

    builder = WindowBuilder(window_size=5)
    df = builder.build_dataset_dataframe([s1, s2])

    assert len(df) == 4  # 2 windows each
    assert set(df["participant_id"].unique()) == {"P_A", "P_B"}
    assert all(df[df["participant_id"] == "P_A"]["phq8_score"] == 5.0)
    assert all(df[df["participant_id"] == "P_B"]["phq8_score"] == 15.0)


def test_short_windows_are_flagged_not_dropped():
    short_turns = [
        Turn(speaker=SpeakerRole.PARTICIPANT, start_s=1.0, end_s=2.0, text="Yes."),
        Turn(speaker=SpeakerRole.PARTICIPANT, start_s=3.0, end_s=4.0, text="No."),
    ]
    s = Session(participant_id="P_SHORT", turns=short_turns)
    builder = WindowBuilder(window_size=2, min_tokens=10)
    df = builder.build_dataset_dataframe([s])

    assert len(df) == 1
    assert bool(df.iloc[0]["is_short"]) is True


def test_assert_no_participant_overlap_catches_leak():
    leaked_splits = {
        "official": {"train": ["P_1", "P_2"], "dev": ["P_2"], "test": ["P_3"]},
        "cv_folds": [],
    }
    with pytest.raises(AssertionError, match="Leakage detected"):
        assert_no_participant_overlap(leaked_splits)

    leaked_cv = {
        "official": {"train": ["P_1"], "dev": [], "test": []},
        "cv_folds": [
            {
                "repeat": 0,
                "fold": 0,
                "train_pids": ["P_A", "P_B"],
                "val_pids": ["P_B"],
            }
        ],
    }
    with pytest.raises(AssertionError, match="CV Leakage"):
        assert_no_participant_overlap(leaked_cv)


def test_build_splits_guarantees_independence():
    gen = SyntheticCohortGenerator(n_participants=40, seed=99)
    sessions = gen.generate()
    splits = build_splits(sessions, n_splits=5, n_repeats=5, seed=42)

    # Validate all 25 folds
    assert len(splits["cv_folds"]) == 25
    assert_no_participant_overlap(splits)


def test_preprocessing_determinism():
    gen = SyntheticCohortGenerator(n_participants=10, seed=777)
    sessions = gen.generate()
    builder = WindowBuilder(window_size=5)

    df1 = builder.build_dataset_dataframe(sessions)
    df2 = builder.build_dataset_dataframe(sessions)

    assert df1.equals(df2)
