"""Transcript timestamp normalization and acoustic/PII tag cleaning."""

import re

from anjaney.data.schemas import Session, Turn

TAG_PATTERN = re.compile(r"<[^>]+>|\[[^\]]+\]|scrubbed_entry_\w*", re.IGNORECASE)


def clean_session(session: Session, lowercase: bool = False) -> Session:
    """Clean acoustic tags, normalize timestamps to start at 0.0s, and drop empty turns."""
    if not session.turns:
        return session

    sorted_turns = sorted(session.turns, key=lambda x: x.start_s)
    valid_raw: list[Turn] = []

    for t in sorted_turns:
        if t.end_s < t.start_s:
            continue
        clean_txt = TAG_PATTERN.sub("", t.text)
        clean_txt = " ".join(clean_txt.split()).strip()
        if lowercase:
            clean_txt = clean_txt.lower()
        if not clean_txt:
            continue
        valid_raw.append(
            Turn(
                speaker=t.speaker,
                start_s=float(t.start_s),
                end_s=float(t.end_s),
                text=clean_txt,
            )
        )

    if not valid_raw:
        return Session(
            participant_id=session.participant_id,
            turns=[],
            phq8_score=session.phq8_score,
            phq8_binary=session.phq8_binary,
            gender=session.gender,
            official_split=session.official_split,
            is_synthetic=session.is_synthetic,
        )

    offset = valid_raw[0].start_s
    cleaned_turns = [
        Turn(
            speaker=t.speaker,
            start_s=round(max(0.0, t.start_s - offset), 4),
            end_s=round(max(0.0, t.end_s - offset), 4),
            text=t.text,
        )
        for t in valid_raw
    ]

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
