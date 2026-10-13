"""DAIC-WOZ and E-DAIC transcript loader with flexible column matching."""

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from anjaney.data.schemas import Session, SpeakerRole, Turn

# Strips DAIC acoustic/scrub annotations while tracking cleaned instances
ANNOTATION_TAG_REGEX = re.compile(
    r"<[^>]+>|\[[^\]]+\]|scrubbed_entry_\w+|scrubbed_entry", re.IGNORECASE
)


class DAICLoader:
    def __init__(self, daic_root: Path | str | None = None):
        root = daic_root or os.environ.get("DAIC_ROOT")
        if not root:
            self.root: Path | None = None
        else:
            self.root = Path(root)

    def validate_root(self) -> Path:
        if self.root is None:
            raise ValueError(
                "DAIC_ROOT is not configured. Provide path or set DAIC_ROOT environment variable."
            )
        if not self.root.exists():
            raise FileNotFoundError(f"DAIC_ROOT directory does not exist at '{self.root}'.")
        return self.root

    @staticmethod
    def _clean_text(raw_text: str) -> tuple[str, int]:
        if not isinstance(raw_text, str):
            return "", 0
        matches = len(ANNOTATION_TAG_REGEX.findall(raw_text))
        cleaned = ANNOTATION_TAG_REGEX.sub(" ", raw_text)
        return " ".join(cleaned.split()), matches

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        normalized_cols = {re.sub(r"[\s_\-#]+", "", col.lower()): col for col in df.columns}
        for cand in candidates:
            norm_cand = re.sub(r"[\s_\-#]+", "", cand.lower())
            if norm_cand in normalized_cols:
                return normalized_cols[norm_cand]
        return None

    def load_labels(self) -> pd.DataFrame:
        root = self.validate_root()
        label_files = list(root.glob("*split*.csv")) + list(root.glob("*labels*.csv"))
        if not label_files:
            label_files = list(root.glob("*.csv"))

        records: list[pd.DataFrame] = []
        for file in label_files:
            try:
                df = pd.read_csv(file)
                pid_col = self._find_column(df, ["participant_id", "participantid", "pid", "id"])
                score_col = self._find_column(
                    df, ["phq8_score", "phq_score", "phq8score", "phqscore"]
                )
                if pid_col and score_col:
                    split_name = "train"
                    if "dev" in file.name.lower() or "val" in file.name.lower():
                        split_name = "dev"
                    elif "test" in file.name.lower():
                        split_name = "test"
                    df["official_split"] = split_name
                    records.append(df)
            except Exception:
                continue

        if not records:
            return pd.DataFrame(
                columns=["participant_id", "phq8_score", "phq8_binary", "gender", "official_split"]
            )

        merged = pd.concat(records, ignore_index=True)
        return merged

    def load_session(self, participant_id: str, label_row: dict[str, Any] | None = None) -> Session:
        root = self.validate_root()
        pattern = f"*{participant_id}*TRANSCRIPT*.csv"
        matching_files = list(root.rglob(pattern))
        if not matching_files:
            # Fallback search for tab-separated text files
            matching_files = list(root.rglob(f"*{participant_id}*.tsv"))

        if not matching_files:
            raise FileNotFoundError(f"No transcript file found for participant {participant_id}")

        filepath = matching_files[0]
        sep = "\t" if filepath.suffix == ".tsv" or "\t" in open(filepath).readline() else ","
        df = pd.read_csv(filepath, sep=sep)

        start_col = self._find_column(df, ["start_time", "starttime", "start_s", "start"])
        end_col = self._find_column(df, ["stop_time", "stoptime", "end_time", "end_s", "end"])
        speaker_col = self._find_column(df, ["speaker", "speaker_role", "role"])
        text_col = self._find_column(df, ["value", "text", "utterance", "transcript"])

        if not all([start_col, end_col, speaker_col, text_col]):
            raise ValueError(
                f"Transcript {filepath.name} missing necessary columns. Available: {list(df.columns)}"
            )

        turns: list[Turn] = []
        for _, row in df.iterrows():
            clean_str, _ = self._clean_text(str(row[text_col]))
            spk_raw = str(row[speaker_col]).strip().lower()
            role = (
                SpeakerRole.PARTICIPANT
                if "participant" in spk_raw or spk_raw == "1"
                else SpeakerRole.INTERVIEWER
            )

            try:
                s_time = float(row[start_col])
                e_time = float(row[end_col])
            except (ValueError, TypeError):
                continue

            turns.append(Turn(speaker=role, start_s=s_time, end_s=e_time, text=clean_str))

        labels = label_row or {}
        return Session(
            participant_id=str(participant_id),
            turns=turns,
            phq8_score=labels.get("phq8_score"),
            phq8_binary=labels.get("phq8_binary"),
            gender=labels.get("gender"),
            official_split=labels.get("official_split", "train"),
            is_synthetic=False,
        )

    def iter_sessions(self):
        labels_df = self.load_labels()
        for _, row in labels_df.iterrows():
            pid_col = self._find_column(labels_df, ["participant_id", "participantid", "pid", "id"])
            if not pid_col:
                continue
            pid = str(row[pid_col])
            try:
                yield self.load_session(pid, row.to_dict())
            except FileNotFoundError:
                continue
