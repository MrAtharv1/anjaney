"""Sliding check-in window segmentation with interviewer script-phase extraction and zero leakage."""

import numpy as np
import pandas as pd

from anjaney.data.schemas import Session, SpeakerRole
from anjaney.preprocess.cleaner import clean_session


class WindowBuilder:
    """Segments participant turns into non-overlapping check-in windows.

    Per AGENTS.md Rule 5 and Burdisso et al. (2024):
    Interviewer prompts are NEVER included in text features, but the preceding
    interviewer prompt identity is used to derive `prompt_phase` for script conditioning.
    """

    def __init__(
        self,
        window_size: int = 5,
        turns_per_window: int | None = None,
        min_turns_for_valid: int = 3,
        min_tokens: int = 5,
    ):
        self.window_size = turns_per_window if turns_per_window is not None else window_size
        self.min_turns = min_turns_for_valid
        self.min_tokens = min_tokens

    @staticmethod
    def _classify_interviewer_prompt_phase(prompt_text: str, rel_pos: float) -> int:
        """Map preceding interviewer prompt to canonical DAIC script phase (0..3)."""
        txt = prompt_text.lower()
        if any(
            w in txt
            for w in (
                "rapport_intro",
                "where are you from",
                "originally",
                "about yourself",
                "feeling over the past",
            )
        ):
            return 0
        if any(
            w in txt
            for w in (
                "behavior_routine",
                "academic routine",
                "daily",
                "sleep",
                "relax",
                "weekend",
            )
        ):
            return 1
        if any(
            w in txt
            for w in (
                "distress_probe",
                "depressed",
                "hopeless",
                "down",
                "overwhelming",
                "stressful",
                "stressed",
                "past",
            )
        ):
            return 2
        if any(
            w in txt
            for w in (
                "closing_reflection",
                "plans",
                "goals",
                "thank you",
                "anything else",
                "friends and social",
            )
        ):
            return 3
        return min(3, int(np.clip(rel_pos, 0.0, 0.999) * 4))

    def build_session_windows(self, session: Session) -> list[dict]:
        cleaned = clean_session(session)

        part_turns = []
        part_latencies = []
        part_prompt_texts = []

        last_interviewer_end: float | None = None
        last_interviewer_text: str = ""

        for t in cleaned.turns:
            if t.speaker == SpeakerRole.INTERVIEWER:
                last_interviewer_end = t.end_s
                last_interviewer_text = t.text
            elif t.speaker == SpeakerRole.PARTICIPANT:
                part_turns.append(t)
                if last_interviewer_end is not None:
                    lat = max(0.0, t.start_s - last_interviewer_end)
                else:
                    lat = np.nan
                part_latencies.append(lat)
                part_prompt_texts.append(last_interviewer_text)

        n_total = len(part_turns)
        if n_total == 0:
            return []

        windows = []
        n_windows = int(np.ceil(n_total / self.window_size))

        for w_idx in range(n_windows):
            start_i = w_idx * self.window_size
            end_i = min(start_i + self.window_size, n_total)
            chunk = part_turns[start_i:end_i]
            chunk_lats = part_latencies[start_i:end_i]
            chunk_prompts = part_prompt_texts[start_i:end_i]

            combined_text = " ".join(t.text for t in chunk).strip()
            tokens = combined_text.split()
            durations = [max(0.0, t.end_s - t.start_s) for t in chunk]
            valid_lats = [val for val in chunk_lats if not np.isnan(val)]

            rel_pos = float(w_idx / max(1, n_windows - 1)) if n_windows > 1 else 0.0
            lead_prompt = chunk_prompts[0] if chunk_prompts else ""
            prompt_phase = self._classify_interviewer_prompt_phase(lead_prompt, rel_pos)

            is_short = len(chunk) < self.min_turns or len(tokens) < self.min_tokens

            windows.append(
                {
                    "participant_id": cleaned.participant_id,
                    "official_split": cleaned.official_split,
                    "phq8_score": cleaned.phq8_score,
                    "phq8_binary": cleaned.phq8_binary,
                    "gender": cleaned.gender,
                    "window_idx": w_idx,
                    "rel_position": rel_pos,
                    "prompt_phase": int(prompt_phase),
                    "n_turns": len(chunk),
                    "n_turns_in_window": len(chunk),
                    "is_short": is_short,
                    "is_short_window": is_short,
                    "text": combined_text,
                    "n_tokens": len(tokens),
                    "start_s": chunk[0].start_s,
                    "end_s": chunk[-1].end_s,
                    "window_start_time": chunk[0].start_s,
                    "window_stop_time": chunk[-1].end_s,
                    "mean_turn_duration": float(np.mean(durations)) if durations else 0.0,
                    "total_speech_time": float(np.sum(durations)) if durations else 0.0,
                    "mean_response_latency": float(np.mean(valid_lats)) if valid_lats else np.nan,
                    "max_response_latency": float(np.max(valid_lats)) if valid_lats else np.nan,
                }
            )

        return windows

    def build_dataset_dataframe(self, sessions: list[Session]) -> pd.DataFrame:
        all_rows = []
        for s in sessions:
            all_rows.extend(self.build_session_windows(s))
        return pd.DataFrame(all_rows)
