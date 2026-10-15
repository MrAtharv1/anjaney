"""Transcript timestamp normalization and acoustic/PII tag cleaning."""

import re

from anjaney.data.schemas import Session, Turn

TAG_PATTERN = re.compile(r"<[^>]+>|\[[^\]]+\]|scrubbed_entry_\w*", re.IGNORECASE)


def clean_session(session: Session) -> Session:
    """Clean acoustic tags, normalize timestamps, and drop empty turns."""
    cleaned_turns: list[Turn] = []
    for t in sorted(session.turns, key=lambda x: x.start_s):
        if t.end_s < t.start_s:
            continue
        clean_txt = TAG_PATTERN.sub("", t.text)
        clean_txt = " ".join(clean_txt.split()).strip()
        if not clean_txt:
            continue
        cleaned_turns.append(
            Turn(
                speaker=t.speaker,
                start_s=float(t.start_s),
                end_s=float(t.end_s),
                text=clean_txt,
            )
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


class TranscriptCleaner:
    @staticmethod
    def clean_session(session: Session) -> Session:
        return clean_session(session)
