"""DAIC-WOZ and E-DAIC dataset loader supporting real AVEC2017 layout and aggregate-only inspection."""

import csv
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from anjaney.data.schemas import DatasetReport, Session, SpeakerRole, Turn
from anjaney.preprocess.cleaner import (
    SCRUBBED_RE,
    is_in_interruption_span,
    parse_ellie_utterance,
)

EXCLUDED_DAIC_IDS: set[int] = {342, 394, 398, 460}
NO_INTERVIEWER_DAIC_IDS: set[int] = {451, 458, 480}


def _discover_col(headers: list[str], candidates: tuple[str, ...]) -> str | None:
    norm_map = {h.strip().lower(): h for h in headers}
    for cand in candidates:
        if cand.lower() in norm_map:
            return norm_map[cand.lower()]
    for h in headers:
        h_low = h.strip().lower()
        if any(cand.lower() in h_low for cand in candidates):
            return h
    return None


ANNOTATION_TAG_REGEX = re.compile(
    r"<[^>]+>|\[[^\]]+\]|scrubbed_entry_\w+|scrubbed_entry", re.IGNORECASE
)


class DAICLoader:
    """Loads DAIC-WOZ / E-DAIC sessions from local disk without ever logging raw transcript text."""

    def __init__(
        self,
        data_root: Path | str | None = None,
        root_dir: Path | str | None = None,
        daic_root: Path | str | None = None,
    ):
        env_root = os.environ.get("DAIC_WOZ_ROOT") or os.environ.get("DAIC_ROOT")
        resolved = data_root or root_dir or daic_root or env_root
        if resolved:
            self.root: Path | None = Path(resolved)
        else:
            self.root = None

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

    def iter_sessions(self):
        yield from self.load_sessions()

    def _load_split_metadata(self) -> dict[str, dict]:
        """Discover and parse official AVEC2017 train/dev/test split CSVs or synthetic_labels.csv."""
        meta: dict[str, dict] = {}

        split_sources = [
            ("train", self.root / "train_split_Depression_AVEC2017.csv"),
            ("dev", self.root / "dev_split_Depression_AVEC2017.csv"),
            ("test", self.root / "full_test_split.csv"),
            ("test", self.root / "test_split_Depression_AVEC2017.csv"),
            ("synthetic", self.root / "synthetic_labels.csv"),
        ]

        for default_split, csv_path in split_sources:
            if not csv_path.exists():
                continue
            with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
                sample = f.read(2048)
                f.seek(0)
                delim = "\t" if "\t" in sample and sample.count("\t") > sample.count(",") else ","
                reader = csv.DictReader(f, delimiter=delim)
                if not reader.fieldnames:
                    continue
                headers = list(reader.fieldnames)

                id_col = _discover_col(headers, ("participant_id", "id"))
                score_col = _discover_col(headers, ("phq8_score", "phq_score", "phq8", "score"))
                bin_col = _discover_col(
                    headers, ("phq8_binary", "phq_binary", "binary", "depressed")
                )
                gender_col = _discover_col(headers, ("gender", "sex"))
                split_col = _discover_col(headers, ("official_split", "split"))

                if not id_col:
                    continue

                for row in reader:
                    raw_id = str(row.get(id_col, "")).strip()
                    if not raw_id:
                        continue
                    if raw_id in meta and meta[raw_id].get("has_labels") and not score_col:
                        continue

                    score_val = (
                        float(row[score_col])
                        if score_col and row.get(score_col, "").strip() != ""
                        else 0.0
                    )
                    if bin_col and row.get(bin_col, "").strip() != "":
                        bin_val = int(float(row[bin_col]))
                    else:
                        bin_val = 1 if score_val >= 10.0 else 0

                    gender_val = str(row.get(gender_col, "U")).strip() if gender_col else "U"
                    split_val = (
                        str(row.get(split_col, default_split)).strip()
                        if split_col and row.get(split_col)
                        else ("train" if default_split == "synthetic" else default_split)
                    )

                    meta[raw_id] = {
                        "phq8_score": score_val,
                        "phq8_binary": bin_val,
                        "gender": gender_val,
                        "official_split": split_val,
                        "is_synthetic": default_split == "synthetic" or raw_id.startswith("SYNTH_"),
                        "has_labels": bool(score_col and row.get(score_col, "").strip() != ""),
                    }
        return meta

    def _discover_transcript_files(
        self,
    ) -> tuple[list[tuple[str, int | None, Path]], list[int]]:
        """Locate both nested (<id>_P/<id>_TRANSCRIPT.csv) and flat (*_TRANSCRIPT.csv) transcripts."""
        found: dict[str, tuple[str, int | None, Path]] = {}
        skipped_excluded: set[int] = set()

        for p_dir in sorted(self.root.glob("*_P")):
            if not p_dir.is_dir() or p_dir.name.startswith((".", "__")):
                continue
            prefix = p_dir.name[:-2]
            num_id = int(prefix) if prefix.isdigit() else None
            if num_id in EXCLUDED_DAIC_IDS:
                skipped_excluded.add(num_id)
                continue
            t_file = p_dir / f"{prefix}_TRANSCRIPT.csv"
            if t_file.exists() and not t_file.name.startswith((".", "__")):
                found[prefix] = (prefix, num_id, t_file)

        for t_file in sorted(self.root.glob("*_TRANSCRIPT.csv")):
            if t_file.name.startswith((".", "__")):
                continue
            prefix = t_file.name.replace("_TRANSCRIPT.csv", "")
            num_id = int(prefix) if prefix.isdigit() else None
            if num_id in EXCLUDED_DAIC_IDS:
                skipped_excluded.add(num_id)
                continue
            if prefix not in found:
                found[prefix] = (prefix, num_id, t_file)

        return list(found.values()), sorted(skipped_excluded)

    def _parse_transcript_file(self, pid: str, num_id: int | None, path: Path) -> list[Turn]:
        with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            delim = "\t" if "\t" in sample else ","
            reader = csv.DictReader(f, delimiter=delim)
            if not reader.fieldnames:
                return []
            headers = list(reader.fieldnames)

            start_col = _discover_col(headers, ("start_time", "start_s", "start"))
            stop_col = _discover_col(headers, ("stop_time", "end_time", "end_s", "stop", "end"))
            speaker_col = _discover_col(headers, ("speaker", "role", "who"))
            val_col = _discover_col(headers, ("value", "text", "utterance", "transcript"))

            if not (start_col and stop_col and speaker_col and val_col):
                raise ValueError(
                    f"Could not discover required transcript columns in {path.name}: {headers}"
                )

            turns: list[Turn] = []
            for row in reader:
                raw_speaker = str(row.get(speaker_col) or "").strip().lower()
                if raw_speaker in ("ellie", "interviewer"):
                    role = SpeakerRole.INTERVIEWER
                elif raw_speaker in ("participant", "user", "subject"):
                    role = SpeakerRole.PARTICIPANT
                else:
                    continue

                try:
                    start_s = float(row.get(start_col) or 0.0)
                    end_s = float(row.get(stop_col) or start_s)
                except ValueError:
                    continue

                if end_s < start_s:
                    continue

                raw_val = str(row.get(val_col) or "").strip()
                uid = None
                if role == SpeakerRole.INTERVIEWER:
                    uid, _ = parse_ellie_utterance(raw_val)

                has_scrub = bool(SCRUBBED_RE.search(raw_val))
                in_interruption = is_in_interruption_span(pid, start_s, end_s)

                turns.append(
                    Turn(
                        speaker=role,
                        start_s=start_s,
                        end_s=end_s,
                        text=raw_val,
                        utterance_id=uid,
                        has_scrubbed=has_scrub,
                        in_interruption_span=in_interruption,
                    )
                )
            return turns

    def load_sessions(self) -> list[Session]:
        meta_map = self._load_split_metadata()
        transcript_items, _ = self._discover_transcript_files()

        sessions: list[Session] = []
        for pid, num_id, t_path in transcript_items:
            turns = self._parse_transcript_file(pid, num_id, t_path)
            has_ellie = any(t.speaker == SpeakerRole.INTERVIEWER for t in turns)
            if num_id in NO_INTERVIEWER_DAIC_IDS:
                has_ellie = False

            m = meta_map.get(pid, {})
            sessions.append(
                Session(
                    participant_id=pid,
                    numeric_id=num_id,
                    turns=turns,
                    phq8_score=float(m.get("phq8_score", 0.0)),
                    phq8_binary=int(m.get("phq8_binary", 0)),
                    gender=str(m.get("gender", "U")),
                    official_split=str(m.get("official_split", "train")),
                    is_synthetic=bool(m.get("is_synthetic", pid.startswith("SYNTH_"))),
                    has_interviewer_turns=has_ellie,
                )
            )
        return sessions

    def inspect_dataset(
        self, phase_map_path: Path | str = "configs/prompt_phase_map.json"
    ) -> DatasetReport:
        """Produce an aggregate-only DatasetReport (ZERO raw utterance text ever output)."""
        from anjaney.preprocess.windows import (
            WindowBuilder,
            normalize_and_hash_prompt,
        )

        _, skipped_excluded = self._discover_transcript_files()
        sessions = self.load_sessions()
        wb = WindowBuilder(window_size=5, phase_map_path=phase_map_path)

        split_counts: Counter[str] = Counter()
        high_risk = 0
        low_risk = 0
        missing_ellie_pids: list[str] = []
        total_turns = 0
        overlap_transitions = 0
        scrubbed_turns = 0
        interruption_turns = 0

        ellie_formats: Counter[str] = Counter()
        uid_counter: Counter[str] = Counter()
        hash_counter: Counter[str] = Counter()
        phase_le_363: Counter[str] = Counter()
        phase_gt_363: Counter[str] = Counter()

        for s in sessions:
            split_counts[s.official_split] += 1
            if s.phq8_binary == 1:
                high_risk += 1
            else:
                low_risk += 1

            if not s.has_interviewer_turns or not any(
                t.speaker == SpeakerRole.INTERVIEWER for t in s.turns
            ):
                missing_ellie_pids.append(s.participant_id)

            last_end: float | None = None
            for t in s.turns:
                total_turns += 1
                if t.has_scrubbed:
                    scrubbed_turns += 1
                if t.in_interruption_span:
                    interruption_turns += 1

                if last_end is not None and t.start_s < last_end:
                    overlap_transitions += 1
                last_end = max(last_end or 0.0, t.end_s)

                if t.speaker == SpeakerRole.INTERVIEWER:
                    uid, inner = parse_ellie_utterance(t.text)
                    if uid is not None:
                        ellie_formats["id_prefixed_gt_363_style"] += 1
                        uid_counter[uid] += 1
                    else:
                        ellie_formats["plain_text_le_363_style"] += 1
                    if inner:
                        hash_counter[normalize_and_hash_prompt(inner)] += 1

            win_rows = wb.build_session_windows(s)
            num_id = s.numeric_id
            if num_id is None:
                digits = re.findall(r"\d+", s.participant_id)
                num_id = int(digits[0]) if digits else 0

            for w in win_rows:
                p_val = w["prompt_phase"]
                p_key = "NaN" if (p_val is None or np.isnan(p_val)) else f"phase_{int(p_val)}"
                if num_id <= 363:
                    phase_le_363[p_key] += 1
                else:
                    phase_gt_363[p_key] += 1

        is_synth = all(s.is_synthetic for s in sessions) if sessions else False
        return DatasetReport(
            source="synthetic" if is_synth else "daic",
            total_sessions_loaded=len(sessions),
            excluded_official_ids_skipped=skipped_excluded,
            split_counts=dict(split_counts),
            high_risk_count=high_risk,
            low_risk_count=low_risk,
            sessions_missing_interviewer_turns=missing_ellie_pids,
            total_turns=total_turns,
            overlapping_turn_transitions_count=overlap_transitions,
            scrubbed_turns_count=scrubbed_turns,
            interruption_masked_turns_count=interruption_turns,
            ellie_format_counts=dict(ellie_formats),
            phase_distribution_le_363=dict(phase_le_363),
            phase_distribution_gt_363=dict(phase_gt_363),
            top_ellie_utterance_id_counts=dict(uid_counter.most_common(25)),
            top_ellie_prompt_hash_counts=dict(hash_counter.most_common(25)),
        )
