"""Sliding check-in window segmentation with hash-based interviewer script-phase lookup and protocol inheritance."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from anjaney.data.schemas import Session, SpeakerRole
from anjaney.preprocess.cleaner import clean_session, parse_ellie_utterance

SYNTHETIC_PHASE_PREFIXES: dict[str, int] = {
    "rapport_intro:": 0,
    "behavior_routine:": 1,
    "distress_probe:": 2,
    "closing_reflection:": 3,
}

BACKCHANNELS = {
    "yeah",
    "uh huh",
    "okay",
    "right",
    "mhm",
    "i see",
    "really",
    "yes",
    "cool",
    "sure",
    "mm",
    "mm-hmm",
    "uh-huh",
    "ok",
    "got it",
    "i understand",
    "yep",
    "nod",
    "m-hm",
    "oh",
    "wow",
    "thats good",
    "good",
    "alright",
    "that sounds good",
    "thats great",
    "nice",
    "awesome",
    "see what you mean",
    "aw",
    "thank you",
    "thanks",
    "i see what you mean",
    "great",
    "that sucks",
    "im sorry",
    "sorry to hear that",
    "sorry hear",
    "makes sense",
    "sounds interesting",
    "oh no",
}

PROBES_FOLLOWUPS = {
    "why",
    "why2",
    "tell me more about that",
    "tell_about_that",
    "like what",
    "can you elaborate",
    "how come",
    "in what way",
    "could you tell me more",
    "how so",
    "can you give me an example",
    "what do you mean",
    "give_example",
    "how_know",
}


def normalize_and_hash_prompt(prompt_text: str) -> str:
    """Normalize whitespace/punctuation and return 16-char SHA-256 hex digest (zero raw text stored)."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", prompt_text.lower())
    collapsed = " ".join(cleaned.split()).strip()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()[:16]


def is_backchannel_or_probe(text: str, uid: str | None = None) -> bool:
    """Identify backchannels and short follow-up probes that inherit preceding topic phase."""
    u = (uid or "").lower()
    if any(
        b in u
        for b in [
            "backchannel",
            "continuer",
            "acknowledg",
            "laughter",
            "awesome",
            "thats_great",
            "see_mean",
            "nice",
            "that_sucks",
            "im_sorry",
            "sorry_hear",
            "makes_sense",
            "sounds_interesting",
            "oh_no",
        ]
    ):
        return True
    if any(p in u for p in ["why", "tell_about_that", "like_what", "give_example", "how_know"]):
        return True
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()
    return cleaned in BACKCHANNELS or cleaned in PROBES_FOLLOWUPS


class WindowBuilder:
    """Segments participant turns into non-overlapping check-in windows.

    Per AGENTS.md Rule 5 and Burdisso et al. (2024):
    Interviewer prompts are NEVER included in text features. For real data, the
    preceding interviewer prompt is normalized, hashed (SHA-256), and looked up
    in configs/prompt_phase_map.json (built strictly from train split). Short
    backchannels inherit preceding substantive question phase so they do not reset script phase.
    """

    def __init__(
        self,
        window_size: int = 5,
        turns_per_window: int | None = None,
        min_turns_for_valid: int = 3,
        min_tokens: int = 5,
        phase_map_path: Path | str = "configs/prompt_phase_map.json",
    ):
        self.window_size = turns_per_window if turns_per_window is not None else window_size
        self.min_turns = min_turns_for_valid
        self.min_tokens = min_tokens
        self.phase_map_path = Path(phase_map_path)
        self.hash_to_phase: dict[str, int] = {}
        self.uid_to_phase: dict[str, int] = {}

        if self.phase_map_path.exists():
            with open(self.phase_map_path, encoding="utf-8") as f:
                data = json.load(f)
                self.hash_to_phase = {
                    str(k): int(v) for k, v in data.get("prompt_hashes", {}).items()
                }
                self.uid_to_phase = {
                    str(k): int(v) for k, v in data.get("utterance_ids", {}).items()
                }

    def _resolve_prompt_phase(
        self, prompt_text: str, uid: str | None = None, is_synthetic: bool = False
    ) -> int | float:
        """Resolve script phase (0..3) from hash map (real data) or synthetic tag prefix."""
        txt_lower = prompt_text.strip().lower()
        if not txt_lower and not uid:
            return np.nan

        # 1. Synthetic prefix tag
        if is_synthetic:
            for prefix, phase_id in SYNTHETIC_PHASE_PREFIXES.items():
                if txt_lower.startswith(prefix):
                    return int(phase_id)
            return 0

        # 2. Check utterance_id (e.g. for DAIC sessions > 363)
        if uid and uid in self.uid_to_phase:
            return int(self.uid_to_phase[uid])

        # 3. Check inner text if id-prefixed (e.g. 'where_originally (where are you from)')
        parsed_uid, inner = parse_ellie_utterance(txt_lower)
        if parsed_uid and parsed_uid in self.uid_to_phase:
            return int(self.uid_to_phase[parsed_uid])

        if inner:
            inner_hash = normalize_and_hash_prompt(inner)
            if inner_hash in self.hash_to_phase:
                return int(self.hash_to_phase[inner_hash])

        # 4. Check full text hash
        prompt_hash = normalize_and_hash_prompt(txt_lower)
        if prompt_hash in self.hash_to_phase:
            return int(self.hash_to_phase[prompt_hash])

        return np.nan

    def build_session_windows(self, session: Session) -> list[dict]:
        cleaned = clean_session(session)

        # Missing-Ellie sessions (e.g. 451, 458, 480): prompt_phase is np.nan
        has_ellie = cleaned.has_interviewer_turns and any(
            t.speaker == SpeakerRole.INTERVIEWER for t in cleaned.turns
        )

        # Find scrubbed turns and mark adjacent turns in session
        n_turns = len(cleaned.turns)
        scrubbed_indices = {i for i, t in enumerate(cleaned.turns) if t.has_scrubbed}
        adjacent_to_scrub = {
            j for i in scrubbed_indices for j in (i - 1, i, i + 1) if 0 <= j < n_turns
        }

        part_turns = []
        part_latencies = []
        part_durations = []
        part_phases = []
        prev_interviewer_ends_list = []

        last_interviewer_end: float | None = None
        last_interviewer_is_interrupted: bool = False
        last_interviewer_is_scrubbed: bool = False

        current_substantive_phase: int | float = 0 if has_ellie else np.nan

        for idx, t in enumerate(cleaned.turns):
            if t.speaker == SpeakerRole.INTERVIEWER:
                last_interviewer_end = t.end_s
                last_interviewer_is_interrupted = t.in_interruption_span
                last_interviewer_is_scrubbed = idx in adjacent_to_scrub

                if has_ellie:
                    # Check if backchannel or follow-up probe
                    if not is_backchannel_or_probe(t.text, t.utterance_id):
                        resolved = self._resolve_prompt_phase(
                            t.text, uid=t.utterance_id, is_synthetic=bool(cleaned.is_synthetic)
                        )
                        if not np.isnan(resolved):
                            current_substantive_phase = int(resolved)

            elif t.speaker == SpeakerRole.PARTICIPANT:
                part_turns.append(t)

                # Duration: mask with np.nan if in interruption span
                if t.in_interruption_span:
                    part_durations.append(np.nan)
                else:
                    part_durations.append(max(0.0, float(t.end_s - t.start_s)))

                # Response Latency: signed (overlaps valid, DO NOT clip negative to 0.0)
                # Mask with np.nan if in interruption span or adjacent to scrubbed turn
                if (
                    last_interviewer_end is None
                    or t.in_interruption_span
                    or last_interviewer_is_interrupted
                    or (idx in adjacent_to_scrub)
                    or last_interviewer_is_scrubbed
                ):
                    lat = np.nan
                    prev_end_val = -1.0
                else:
                    lat = float(t.start_s - last_interviewer_end)
                    prev_end_val = float(last_interviewer_end)

                part_latencies.append(lat)
                prev_interviewer_ends_list.append(prev_end_val)
                part_phases.append(current_substantive_phase if has_ellie else np.nan)

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
            chunk_durs = part_durations[start_i:end_i]
            chunk_phases = part_phases[start_i:end_i]
            chunk_prev_ends = prev_interviewer_ends_list[start_i:end_i]

            combined_text = " ".join(t.text for t in chunk).strip()
            tokens = combined_text.split()

            valid_durs = [d for d in chunk_durs if not np.isnan(d)]
            valid_lats = [val for val in chunk_lats if not np.isnan(val)]

            rel_pos = float(w_idx / max(1, n_windows - 1)) if n_windows > 1 else 0.0

            if not has_ellie:
                prompt_phase = np.nan
            else:
                valid_win_phases = [p for p in chunk_phases if not np.isnan(p)]
                prompt_phase = int(valid_win_phases[0]) if valid_win_phases else np.nan

            is_short = len(chunk) < self.min_turns or len(tokens) < self.min_tokens

            # Overlap turn ratio: proportion of valid turns where participant started before interviewer finished
            overlap_turn_ratio = (
                float(sum(1 for lat in valid_lats if lat < 0.0) / len(valid_lats))
                if valid_lats
                else 0.0
            )

            windows.append(
                {
                    "participant_id": cleaned.participant_id,
                    "official_split": cleaned.official_split,
                    "is_synthetic": bool(cleaned.is_synthetic),
                    "has_interviewer_turns": bool(has_ellie),
                    "phq8_score": cleaned.phq8_score,
                    "phq8_binary": cleaned.phq8_binary,
                    "gender": cleaned.gender,
                    "window_idx": w_idx,
                    "rel_position": rel_pos,
                    "prompt_phase": prompt_phase,
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
                    "mean_turn_duration": float(np.mean(valid_durs)) if valid_durs else 0.0,
                    "total_speech_time": float(np.sum(valid_durs)) if valid_durs else 0.0,
                    "mean_response_latency": float(np.mean(valid_lats)) if valid_lats else np.nan,
                    "max_response_latency": float(np.max(valid_lats)) if valid_lats else np.nan,
                    "overlap_turn_ratio": overlap_turn_ratio,
                    "turn_starts": [
                        t.start_s if not t.in_interruption_span else np.nan for t in chunk
                    ],
                    "turn_ends": [t.end_s if not t.in_interruption_span else np.nan for t in chunk],
                    "prev_interviewer_ends": chunk_prev_ends,
                }
            )

        return windows

    def build_dataset_dataframe(self, sessions: list[Session]) -> pd.DataFrame:
        all_rows = []
        for s in sessions:
            all_rows.extend(self.build_session_windows(s))
        return pd.DataFrame(all_rows)
