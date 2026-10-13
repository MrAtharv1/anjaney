"""Within-session personal-baseline deviation engine with shrinkage and persistence tracking."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


class PersonalBaselineEngine:
    """Calculates multidimensional deviations against individual warm-up baselines."""

    def __init__(
        self,
        warmup_windows_k: int = 3,
        empirical_bayes_k: float = 2.0,
        ewma_alpha: float = 0.3,
        cusum_threshold: float = 1.0,
        cusum_slack: float = 0.5,
    ):
        self.warmup_k = warmup_windows_k
        self.eb_k = empirical_bayes_k
        self.ewma_alpha = ewma_alpha
        self.cusum_threshold = cusum_threshold
        self.cusum_slack = cusum_slack

    def compute_session_deviations(
        self,
        session_windows: pd.DataFrame,
        embed_cols: list[str],
        timing_cols: list[str],
        population_timing_priors: dict[str, tuple[float, float]],
    ) -> pd.DataFrame:
        """Compute window deviations for one participant session."""
        df = session_windows.sort_values("window_idx").copy()
        n_windows = len(df)

        text_devs = np.full(n_windows, np.nan)
        timing_devs = np.full(n_windows, np.nan)
        combined_devs = np.full(n_windows, np.nan)

        if n_windows <= self.warmup_k:
            df["text_deviation"] = text_devs
            df["timing_deviation"] = timing_devs
            df["combined_deviation"] = combined_devs
            df["ewma_deviation"] = np.nan
            df["cusum_high"] = 0.0
            return df

        # 1. Warm-up baseline representations (windows 0 .. K-1)
        warmup_df = df.iloc[: self.warmup_k]
        warmup_embeddings = warmup_df[embed_cols].values
        baseline_text_centroid = np.mean(warmup_embeddings, axis=0)

        # Baseline timing statistics with empirical-Bayes shrinkage
        warmup_timing_stats: dict[str, tuple[float, float]] = {}
        for col in timing_cols:
            vals = warmup_df[col].values
            sample_mean = float(np.mean(vals))
            sample_std = float(np.std(vals)) + 1e-4

            pop_mean, pop_std = population_timing_priors.get(col, (sample_mean, sample_std))
            # Shrinkage towards population prior: mu_hat = (n * mu_i + k * mu_pop) / (n + k)
            shrunk_mean = (self.warmup_k * sample_mean + self.eb_k * pop_mean) / (
                self.warmup_k + self.eb_k
            )
            shrunk_std = (self.warmup_k * sample_std + self.eb_k * pop_std) / (
                self.warmup_k + self.eb_k
            )
            warmup_timing_stats[col] = (shrunk_mean, shrunk_std)

        # 2. Window deviations for post-warmup check-ins (t >= K)
        for t in range(self.warmup_k, n_windows):
            row = df.iloc[t]
            curr_emb = row[embed_cols].values

            # Text cosine distance to centroid
            norm_c = np.linalg.norm(baseline_text_centroid)
            norm_e = np.linalg.norm(curr_emb)
            if norm_c > 1e-6 and norm_e > 1e-6:
                d_text = float(cosine(curr_emb, baseline_text_centroid))
            else:
                d_text = 0.0
            text_devs[t] = d_text

            # Timing deviation: sum of absolute robust z-scores across metrics
            z_scores = []
            for col in timing_cols:
                mu, sigma = warmup_timing_stats[col]
                z_scores.append(abs((float(row[col]) - mu) / sigma))
            d_timing = float(np.mean(z_scores)) if z_scores else 0.0
            timing_devs[t] = d_timing

            # Combined deviation metric
            combined_devs[t] = 0.5 * d_text + 0.5 * d_timing

        df["text_deviation"] = text_devs
        df["timing_deviation"] = timing_devs
        df["combined_deviation"] = combined_devs

        # 3. Persistence: EWMA and one-sided CUSUM
        ewma_vals = np.full(n_windows, np.nan)
        cusum_vals = np.zeros(n_windows)

        s_prev = 0.0
        cusum_prev = 0.0
        initialized = False

        for t in range(self.warmup_k, n_windows):
            val = combined_devs[t]
            if not initialized:
                s_prev = val
                initialized = True
            else:
                s_prev = self.ewma_alpha * val + (1.0 - self.ewma_alpha) * s_prev
            ewma_vals[t] = s_prev

            # CUSUM accumulation
            cusum_prev = max(0.0, cusum_prev + (val - self.cusum_slack))
            cusum_vals[t] = cusum_prev

        df["ewma_deviation"] = ewma_vals
        df["cusum_high"] = cusum_vals
        return df

    @staticmethod
    def extract_participant_drift_features(df_participant_deviations: pd.DataFrame) -> dict:
        """Summarize post-warmup deviations into participant-level drift features."""
        valid = df_participant_deviations.dropna(subset=["conditioned_deviation"])
        if len(valid) == 0:
            return {
                "drift_mean": 0.0,
                "drift_max": 0.0,
                "drift_slope": 0.0,
                "drift_late_minus_early": 0.0,
                "cusum_max": 0.0,
            }

        devs = valid["conditioned_deviation"].values
        rel_pos = valid["rel_position"].values

        drift_mean = float(np.mean(devs))
        drift_max = float(np.max(devs))

        if len(devs) > 1 and (np.max(rel_pos) - np.min(rel_pos)) > 1e-4:
            slope, _ = np.polyfit(rel_pos, devs, 1)
        else:
            slope = 0.0

        half = len(devs) // 2
        late_minus_early = float(np.mean(devs[half:]) - np.mean(devs[:half])) if half > 0 else 0.0

        cusum_max = float(valid["cusum_high"].max()) if "cusum_high" in valid.columns else 0.0

        return {
            "drift_mean": drift_mean,
            "drift_max": drift_max,
            "drift_slope": float(slope),
            "drift_late_minus_early": late_minus_early,
            "cusum_max": cusum_max,
        }
