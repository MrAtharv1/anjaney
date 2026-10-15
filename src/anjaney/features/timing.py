"""Content-free conversational timing and pacing dynamics."""

from typing import Any

import numpy as np


class TimingFeatureExtractor:
    """Extracts non-content timing patterns from turn start/stop intervals."""

    def __init__(self, max_latency_cap: float = 30.0):
        self.max_latency_cap = max_latency_cap

    def extract_window_features(self, window_row: dict[str, Any]) -> dict[str, float]:
        starts = np.array(window_row.get("turn_starts", []), dtype=float)
        ends = np.array(window_row.get("turn_ends", []), dtype=float)
        prev_interviewer_ends = np.array(window_row.get("prev_interviewer_ends", []), dtype=float)

        n_turns = len(starts)
        if n_turns == 0:
            if "mean_turn_duration" in window_row:
                mean_dur = float(window_row.get("mean_turn_duration", 0.0) or 0.0)
                mean_lat = float(window_row.get("mean_response_latency", 0.0) or 0.0)
                if np.isnan(mean_lat):
                    mean_lat = 0.0
                n_tokens = float(window_row.get("n_tokens", 0) or 0.0)
                total_speech_time = float(window_row.get("total_speech_time", 1.0) or 1.0)
                speech_rate = n_tokens / max(1.0, total_speech_time)
                w_dur = max(
                    1.0,
                    float(
                        (window_row.get("end_s", 0) or 0.0) - (window_row.get("start_s", 0) or 0.0)
                    ),
                )
                n_t = float(window_row.get("n_turns", 1) or 1.0)
                rel_pos = float(window_row.get("rel_position", 0.0) or 0.0)
                return {
                    "timing_mean_turn_duration": mean_dur,
                    "timing_std_turn_duration": 0.0,
                    "timing_mean_latency": mean_lat,
                    "timing_median_latency": mean_lat,
                    "timing_std_latency": 0.0,
                    "timing_iqr_latency": 0.0,
                    "timing_latency_cv": 0.0,
                    "timing_mean_pause": 0.0,
                    "timing_speech_rate_wps": speech_rate,
                    "timing_turns_per_minute": (n_t / w_dur) * 60.0,
                    "timing_engagement_decay": (n_tokens / max(1.0, n_t)) * (1.0 - rel_pos),
                    "timing_overlap_turn_ratio": float(
                        window_row.get("overlap_turn_ratio", 0.0) or 0.0
                    ),
                }
            return self._empty_features()

        # 1. Turn Durations (filter out masked NaNs)
        durations = np.maximum(0.0, ends - starts)
        durations = durations[~np.isnan(durations)]

        # 2. Response Latencies (signed, unclipped lower bound to preserve valid overlaps)
        valid_latency_mask = (
            (prev_interviewer_ends >= 0.0)
            & (~np.isnan(prev_interviewer_ends))
            & (~np.isnan(starts))
        )
        latencies = np.where(
            valid_latency_mask,
            starts - prev_interviewer_ends,
            np.nan,
        )
        # Cap outliers symmetrically
        valid_latencies = latencies[~np.isnan(latencies)]
        valid_latencies = np.clip(valid_latencies, -self.max_latency_cap, self.max_latency_cap)

        # 3. Inter-turn pauses between participant's consecutive turns (if any)
        if n_turns > 1:
            pauses = np.maximum(0.0, starts[1:] - ends[:-1])
            pauses = pauses[~np.isnan(pauses)]
        else:
            pauses = np.array([0.0])

        # 4. Words per second (speech rate)
        n_tokens = float(window_row.get("n_tokens", 0))
        total_speaking_time = float(np.sum(durations))
        speech_rate = n_tokens / max(1.0, total_speaking_time)

        # 5. Window-level aggregations
        mean_dur = float(np.mean(durations)) if len(durations) > 0 else 0.0
        std_dur = float(np.std(durations)) if len(durations) > 1 else 0.0

        if len(valid_latencies) > 0:
            mean_lat = float(np.mean(valid_latencies))
            median_lat = float(np.median(valid_latencies))
            std_lat = float(np.std(valid_latencies)) if len(valid_latencies) > 1 else 0.0
            iqr_lat = (
                float(np.percentile(valid_latencies, 75) - np.percentile(valid_latencies, 25))
                if len(valid_latencies) > 1
                else 0.0
            )
            lat_cv = std_lat / (mean_lat + 1e-4)
        else:
            mean_lat = 0.0
            median_lat = 0.0
            std_lat = 0.0
            iqr_lat = 0.0
            lat_cv = 0.0

        # Rhythm: turns per minute across window span
        window_duration = max(1.0, float(window_row.get("end_s", 0) - window_row.get("start_s", 0)))
        turns_per_minute = (n_turns / window_duration) * 60.0

        # Engagement decay proxy (utterance length normalized by relative position)
        rel_pos = float(window_row.get("rel_position", 0.0))
        engagement_decay = (n_tokens / max(1, n_turns)) * (1.0 - rel_pos)

        # Overlap turn ratio
        overlap_turn_ratio = (
            float(np.mean(valid_latencies < 0.0))
            if len(valid_latencies) > 0
            else float(window_row.get("overlap_turn_ratio", 0.0) or 0.0)
        )

        return {
            "timing_mean_turn_duration": mean_dur,
            "timing_std_turn_duration": std_dur,
            "timing_mean_latency": mean_lat,
            "timing_median_latency": median_lat,
            "timing_std_latency": std_lat,
            "timing_iqr_latency": iqr_lat,
            "timing_mean_pause": float(np.mean(pauses)) if len(pauses) > 0 else 0.0,
            "timing_speech_rate_wps": speech_rate,
            "timing_turns_per_minute": turns_per_minute,
            "timing_engagement_decay": engagement_decay,
            "timing_overlap_turn_ratio": overlap_turn_ratio,
        }

    def _empty_features(self) -> dict[str, float]:
        return {
            "timing_mean_turn_duration": 0.0,
            "timing_std_turn_duration": 0.0,
            "timing_mean_latency": 0.0,
            "timing_median_latency": 0.0,
            "timing_std_latency": 0.0,
            "timing_iqr_latency": 0.0,
            "timing_latency_cv": 0.0,
            "timing_mean_pause": 0.0,
            "timing_speech_rate_wps": 0.0,
            "timing_turns_per_minute": 0.0,
            "timing_engagement_decay": 0.0,
            "timing_overlap_turn_ratio": 0.0,
        }
