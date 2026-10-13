"""Data contracts for transcripts, participant sessions, and dataset diagnostics."""

from enum import StrEnum
from pydantic import BaseModel, Field, model_validator


class SpeakerRole(StrEnum):
    PARTICIPANT = "Participant"
    INTERVIEWER = "Interviewer"


class Turn(BaseModel):
    speaker: SpeakerRole
    start_s: float = Field(ge=0.0, description="Turn start in seconds from session start")
    end_s: float = Field(ge=0.0, description="Turn end in seconds from session start")
    text: str = Field(description="Utterance text with annotation tags stripped")

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class Session(BaseModel):
    participant_id: str
    turns: list[Turn]
    phq8_score: float | None = Field(default=None, ge=0.0, le=24.0)
    phq8_binary: int | None = Field(default=None, ge=0, le=1)
    gender: str | None = None
    official_split: str = Field(default="train", description="train, dev, or test")
    is_synthetic: bool = False

    @model_validator(mode="after")
    def derive_binary_if_missing(self) -> "Session":
        if self.phq8_binary is None and self.phq8_score is not None:
            self.phq8_binary = 1 if self.phq8_score >= 10.0 else 0
        return self

    @property
    def participant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker == SpeakerRole.PARTICIPANT]

    @property
    def total_duration_s(self) -> float:
        if not self.turns:
            return 0.0
        return max(t.end_s for t in self.turns) - min(t.start_s for t in self.turns)


class DatasetAnomalyReport(BaseModel):
    total_sessions: int = 0
    sessions_per_split: dict[str, int] = Field(default_factory=dict)
    non_monotonic_timestamps: int = 0
    overlapping_turns: int = 0
    zero_or_negative_durations: int = 0
    sessions_under_min_turns: int = 0
    missing_labels: int = 0
    duplicate_participant_across_splits: int = 0
    usable_signals: dict[str, bool] = Field(default_factory=dict)