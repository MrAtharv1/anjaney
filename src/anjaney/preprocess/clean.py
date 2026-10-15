"""Session text and timestamp cleaning with explicit missingness tracking."""

import re

from anjaney.data.schemas import Session, Turn

"""Backward-compatibility re-export shim for cleaner.py."""

from anjaney.preprocess.cleaner import TranscriptCleaner, clean_session

__all__ = ["TranscriptCleaner", "clean_session"]


def clean_session(session: Session, lowercase: bool = False) -> Session:
    """Normalize session timestamps to start from 0.0s, order turns, and collapse text whitespace."""
    if not session.turns:
        return session

    # Sort turns chronologically
    sorted_turns = sorted(session.turns, key=lambda t: t.start_s)
    session_start = sorted_turns[0].start_s

    cleaned_turns: list[Turn] = []
    for turn in sorted_turns:
        text = turn.text
        if lowercase:
            text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()

        # Rebase timestamps relative to session start
        rebased_start = max(0.0, round(turn.start_s - session_start, 3))
        rebased_end = max(rebased_start, round(turn.end_s - session_start, 3))

        cleaned_turns.append(
            Turn(speaker=turn.speaker, start_s=rebased_start, end_s=rebased_end, text=text)
        )

    return Session(
        participant_id=session.participant_id,
        turns=cleaned_turns,
        phq8_score=session.phq8_score,
        phq8_binary=session.phq8_binary,
        gender=session.gender,
        official_split=session.official_split,
        is_synthetic=session.is_synthetic,
    )
