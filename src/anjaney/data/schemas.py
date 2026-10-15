"""Pydantic schemas for sessions, turns, and aggregate dataset reports with zero raw-text leakage."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SpeakerRole(StrEnum):
    INTERVIEWER = "Interviewer"
    PARTICIPANT = "Participant"


class Turn(BaseModel):
    speaker: SpeakerRole
    start_s: float
    end_s: float
    text: str
    utterance_id: str | None = None
    has_scrubbed: bool = False
    in_interruption_span: bool = False

    @model_validator(mode="after")
    def validate_times(self) -> "Turn":
        if self.end_s < self.start_s:
            raise ValueError(f"end_s ({self.end_s}) cannot be less than start_s ({self.start_s})")
        return self

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def duration(self) -> float:
        return self.duration_s


class Session(BaseModel):
    participant_id: str
    turns: list[Turn] = Field(default_factory=list)
    phq8_score: float = 0.0
    phq8_binary: int = 0
    gender: str = "U"
    official_split: str = "train"
    is_synthetic: bool = False
    numeric_id: int | None = None
    has_interviewer_turns: bool = True

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


class DatasetReport(BaseModel):
    """Aggregate-only dataset inspection report (AGENTS.md Rule 1: zero raw text)."""

    source: str = "daic"
    total_sessions_loaded: int = 0
    excluded_official_ids_skipped: list[int] = Field(default_factory=list)
    split_counts: dict[str, int] = Field(default_factory=dict)
    high_risk_count: int = 0
    low_risk_count: int = 0
    sessions_missing_interviewer_turns: list[str] = Field(default_factory=list)
    total_turns: int = 0
    overlapping_turn_transitions_count: int = 0
    scrubbed_turns_count: int = 0
    interruption_masked_turns_count: int = 0
    ellie_format_counts: dict[str, int] = Field(default_factory=dict)
    phase_distribution_le_363: dict[str, int] = Field(default_factory=dict)
    phase_distribution_gt_363: dict[str, int] = Field(default_factory=dict)
    top_ellie_utterance_id_counts: dict[str, int] = Field(default_factory=dict)
    top_ellie_prompt_hash_counts: dict[str, int] = Field(default_factory=dict)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)
