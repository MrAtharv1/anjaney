"""Check-in window builder aggregating participant turns and preserving content-free timing arrays."""

from typing import Any

import pandas as pd

from anjaney.data.schemas import Session, SpeakerRole


class WindowBuilder:
    """Chunks participant turns into fixed-count check-in windows."""

    def __init__(self, window_size: int = 5, min_tokens: int = 8):
        self.window_size = window_size
        self.min_tokens = min_tokens

    def build_session_windows(self, session: Session) -> list[dict[str, Any]]:
        """Extract participant check-in windows along with prior interviewer stops."""
        turns = session.turns
        if not turns:
            return []

        # Find preceding interviewer stop time for every participant turn
        participant_turn_data: list[dict[str, Any]] = []
        last_interviewer_end: float | None = None

        for turn in turns:
            if turn.speaker == SpeakerRole.INTERVIEWER:
                last_interviewer_end = turn.end_s
            elif turn.speaker == SpeakerRole.PARTICIPANT:
                participant_turn_data.append(
                    {
                        "start_s": turn.start_s,
                        "end_s": turn.end_s,
                        "duration_s": turn.duration_s,
                        "text": turn.text,
                        "prev_interviewer_end": last_interviewer_end,
                    }
                )

        if not participant_turn_data:
            return []

        # Chunk into windows of window_size participant turns
        chunks: list[list[dict[str, Any]]] = []
        for i in range(0, len(participant_turn_data), self.window_size):
            chunk = participant_turn_data[i : i + self.window_size]
            chunks.append(chunk)

        n_windows = len(chunks)
        window_records: list[dict[str, Any]] = []

        for w_idx, chunk in enumerate(chunks):
            concatenated_text = " ".join(item["text"] for item in chunk)
            tokens = concatenated_text.split()
            n_tokens = len(tokens)

            start_s = chunk[0]["start_s"]
            end_s = chunk[-1]["end_s"]
            rel_position = round(w_idx / max(1, n_windows - 1), 4) if n_windows > 1 else 0.0

            turn_starts = [item["start_s"] for item in chunk]
            turn_ends = [item["end_s"] for item in chunk]
            prev_interviewer_ends = [
                -1.0 if item["prev_interviewer_end"] is None else item["prev_interviewer_end"]
                for item in chunk
            ]
            latency_missing = [item["prev_interviewer_end"] is None for item in chunk]

            window_records.append(
                {
                    "participant_id": session.participant_id,
                    "window_idx": w_idx,
                    "n_windows_for_participant": n_windows,
                    "rel_position": rel_position,
                    "start_s": start_s,
                    "end_s": end_s,
                    "text": concatenated_text,
                    "n_tokens": n_tokens,
                    "is_short": n_tokens < self.min_tokens,
                    "turn_starts": turn_starts,
                    "turn_ends": turn_ends,
                    "prev_interviewer_ends": prev_interviewer_ends,
                    "latency_missing": latency_missing,
                    "phq8_score": session.phq8_score,
                    "phq8_binary": session.phq8_binary,
                    "gender": session.gender,
                    "official_split": session.official_split,
                    "is_synthetic": session.is_synthetic,
                }
            )

        return window_records

    def build_dataset_dataframe(self, sessions: list[Session]) -> pd.DataFrame:
        """Process multiple sessions into a unified DataFrame."""
        all_windows: list[dict[str, Any]] = []
        for s in sessions:
            all_windows.extend(self.build_session_windows(s))

        if not all_windows:
            return pd.DataFrame()

        df = pd.DataFrame(all_windows)
        # Enforce deterministic row sorting
        df = df.sort_values(by=["participant_id", "window_idx"]).reset_index(drop=True)
        return df
