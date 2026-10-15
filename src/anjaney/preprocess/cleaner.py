"""Transcript cleaning: DAIC-WOZ token rules, Ellie utterance_id parsing, and interruption masking."""

import re

# Define regex patterns first so other modules can import them without circular dependencies
SCRUBBED_RE = re.compile(r"\bscrubbed_entry\w*\b", re.IGNORECASE)
XXX_TOKEN_RE = re.compile(r"\bxxx+\b", re.IGNORECASE)
ANGLE_OR_BRACKET_TAG_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")
ELLIE_ID_PAREN_RE = re.compile(r"^([a-zA-Z0-9_\-]+)\s*[\(\[](.+)[\)\]]\s*$")

TAG_PATTERN = re.compile(r"<[^>]+>|\[[^\]]+\]|\bscrubbed_entry\w*\b|\bxxx+\b", re.IGNORECASE)

# Known interruption intervals per participant id evaluated on raw timestamps before offset normalization
# Session 373: 352.0s-420.0s (5:52-7:00), Session 444: 286.0s-387.0s (4:46-6:27)
KNOWN_INTERRUPTION_SPANS: dict[str, list[tuple[float, float]]] = {
    "373": [(352.0, 420.0)],
    "444": [(286.0, 387.0)],
}


def parse_ellie_utterance(text: str) -> tuple[str | None, str]:
    """Parse Ellie utterance ID and inner prompt text if formatted like '<id> (<prompt>)'.

    Returns (utterance_id, inner_prompt_text). If no utterance ID prefix is found,
    returns (None, text).
    """
    match = ELLIE_ID_PAREN_RE.match(text.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, text.strip()


def is_in_interruption_span(pid: str, start_s: float, end_s: float) -> bool:
    """Check if turn timestamp overlaps with any known interruption span for the participant."""
    spans = KNOWN_INTERRUPTION_SPANS.get(str(pid).strip(), [])
    for span_start, span_end in spans:
        if max(start_s, span_start) < min(end_s, span_end):
            return True
    return False


# Import schemas after defining module-level constants and functions
from anjaney.data.schemas import Session, Turn  # noqa: E402


def clean_session(session: Session, lowercase: bool = False) -> Session:
    """Clean acoustic tags, normalize timestamps to start at 0.0s, and drop empty turns."""
    if not session.turns:
        return session

    sorted_turns = sorted(session.turns, key=lambda x: x.start_s)
    valid_raw: list[Turn] = []

    for t in sorted_turns:
        if t.end_s < t.start_s:
            continue

        has_scrub = bool(t.has_scrubbed) or bool(SCRUBBED_RE.search(t.text))
        in_interruption = bool(t.in_interruption_span) or is_in_interruption_span(
            session.participant_id, t.start_s, t.end_s
        )

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
                utterance_id=t.utterance_id,
                has_scrubbed=has_scrub,
                in_interruption_span=in_interruption,
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
            numeric_id=session.numeric_id,
            has_interviewer_turns=session.has_interviewer_turns,
        )

    offset = valid_raw[0].start_s
    cleaned_turns = [
        Turn(
            speaker=t.speaker,
            start_s=round(max(0.0, t.start_s - offset), 4),
            end_s=round(max(0.0, t.end_s - offset), 4),
            text=t.text,
            utterance_id=t.utterance_id,
            has_scrubbed=t.has_scrubbed,
            in_interruption_span=t.in_interruption_span,
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
        numeric_id=session.numeric_id,
        has_interviewer_turns=session.has_interviewer_turns,
    )


class TranscriptCleaner:
    @staticmethod
    def clean_session(session: Session) -> Session:
        return clean_session(session)
