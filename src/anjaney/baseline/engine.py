"""Within-session personal-baseline deviation engine with vectorized PCA-Mahalanobis and persistence."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


class PersonalBaselineEngine:
    """Calculates multidimensional deviations against individual warm-up baselines."""

    def __init__(
        self,
        warmup_windows_k: int = 3,
        empirical_bayes_k: float = 2.0,
        pca_components: int = 16,
        fusion_weight_text: float = 0.50,
        fusion_weight_timing: float = 0.50,
        ewma_alpha: float = 0.3,
        cusum_slack: float = 0.5,
        winsorize_cap: float = 5.0,
    ):
        self.warmup_k = warmup_windows_k
        self.eb_k = empirical_bayes_k
        self.pca_components = pca_components
        self.w_text = fusion_weight_text
        self.w_timing = fusion_weight_timing
        self.ewma_alpha = ewma_alpha
        self.cusum_slack = cusum_slack
        self.winsorize_cap = winsorize_cap

        self.pca: PCA | None = None
        self.lw_cov: LedoitWolf | None = None
        self.timing_priors: dict[str, tuple[float, float]] = {}
        self.dev_scaler_stats: dict[str, tuple[float, float]] = {}
        self.missing_modality_window_drops: int = 0

    def fit_training_priors(
        self,
        df_train_windows: pd.DataFrame,
        embed_cols: list[str],
        timing_cols: list[str],
    ) -> "PersonalBaselineEngine":
        if df_train_windows.empty:
            raise ValueError("Cannot fit baseline priors on empty training DataFrame.")

        # 1. Pooled within-person timing priors
        for col in timing_cols:
            vals = df_train_windows[col].dropna().values
            pop_mean = float(np.mean(vals)) if len(vals) > 0 else 0.0

            within_person_vars = []
            for _, grp in df_train_windows.groupby("participant_id"):
                g_vals = grp[col].dropna().values
                if len(g_vals) > 1:
                    within_person_vars.append(np.var(g_vals, ddof=1))

            pooled_within_std = (
                float(np.sqrt(np.mean(within_person_vars)))
                if within_person_vars
                else float(np.std(vals) + 1e-3)
            )
            self.timing_priors[col] = (pop_mean, max(pooled_within_std, 1e-3))

        # 2. Fit PCA and Ledoit-Wolf on within-person centered embedding residuals
        if embed_cols:
            centered_residuals = []
            for _, grp in df_train_windows.groupby("participant_id"):
                sorted_g = grp.sort_values("window_idx")
                if len(sorted_g) <= self.warmup_k:
                    continue
                emb_vals = sorted_g[embed_cols].to_numpy(dtype=float)
                base_mean = emb_vals[: self.warmup_k].mean(axis=0)
                post_res = emb_vals[self.warmup_k :] - base_mean
                centered_residuals.append(post_res)

            if not centered_residuals:
                raise ValueError(
                    f"No training participants have > {self.warmup_k} windows to fit within-person covariance."
                )

            X_res = np.vstack(centered_residuals)
            n_comp = min(self.pca_components, X_res.shape[0] - 1, X_res.shape[1])
            if n_comp < 2:
                raise ValueError(f"Insufficient residual degrees of freedom for PCA: {X_res.shape}")

            self.pca = PCA(n_components=n_comp, random_state=42)
            X_pca = self.pca.fit_transform(X_res)

            self.lw_cov = LedoitWolf()
            self.lw_cov.fit(X_pca)

        return self

    def compute_raw_session_deviations(
        self,
        session_windows: pd.DataFrame,
        embed_cols: list[str],
        timing_cols: list[str],
    ) -> pd.DataFrame:
        df = session_windows.sort_values("window_idx").copy()
        n_windows = len(df)

        text_cos_devs = np.full(n_windows, np.nan)
        text_maha_devs = np.full(n_windows, np.nan)
        timing_devs = np.full(n_windows, np.nan)

        if n_windows <= self.warmup_k:
            df["text_deviation"] = text_cos_devs
            df["maha_deviation"] = text_maha_devs
            df["timing_deviation"] = timing_devs
            return df

        warmup_df = df.iloc[: self.warmup_k]
        post_df = df.iloc[self.warmup_k :]

        # 1. Vectorized Text Cosine & PCA-Mahalanobis Deviation
        if embed_cols:
            warmup_embs = warmup_df[embed_cols].to_numpy(dtype=float)
            post_embs = post_df[embed_cols].to_numpy(dtype=float)
            base_centroid = np.mean(warmup_embs, axis=0, keepdims=True)

            if np.linalg.norm(base_centroid) > 1e-6:
                cos_dists = cdist(post_embs, base_centroid, metric="cosine").ravel()
                cos_dists = np.where(np.isfinite(cos_dists), cos_dists, 0.0)
            else:
                cos_dists = np.zeros(len(post_df))
            text_cos_devs[self.warmup_k :] = cos_dists

            if self.pca is not None and self.lw_cov is not None:
                warmup_pca = self.pca.transform(warmup_embs)
                base_pca_centroid = np.mean(warmup_pca, axis=0)
                post_pca = self.pca.transform(post_embs)
                diffs = post_pca - base_pca_centroid
                maha_sq = np.einsum("ij,jk,ik->i", diffs, self.lw_cov.precision_, diffs)
                text_maha_devs[self.warmup_k :] = np.sqrt(np.maximum(0.0, maha_sq))
            else:
                text_maha_devs[self.warmup_k :] = cos_dists
        else:
            text_cos_devs[self.warmup_k :] = 0.0
            text_maha_devs[self.warmup_k :] = 0.0

        # 2. Timing warm-up baseline with Empirical-Bayes shrinkage
        if timing_cols:
            warmup_timing_stats: dict[str, tuple[float, float]] = {}
            for col in timing_cols:
                vals = warmup_df[col].dropna().to_numpy(dtype=float)
                pop_mean, pop_within_std = self.timing_priors.get(col, (0.0, 1.0))

                if len(vals) == 0:
                    warmup_timing_stats[col] = (pop_mean, pop_within_std)
                    continue

                n_i = len(vals)
                med_i = float(np.median(vals))
                mad_i = float(np.median(np.abs(vals - med_i))) * 1.4826

                shrunk_loc = (n_i * med_i + self.eb_k * pop_mean) / (n_i + self.eb_k)
                shrunk_scale = (n_i * mad_i + self.eb_k * pop_within_std) / (n_i + self.eb_k)
                safe_scale = max(shrunk_scale, 0.20 * pop_within_std, 1e-4)
                warmup_timing_stats[col] = (shrunk_loc, safe_scale)

            post_timing_z = []
            for col in timing_cols:
                mu, sigma = warmup_timing_stats[col]
                col_vals = post_df[col].to_numpy(dtype=float)
                z = np.minimum(np.abs((col_vals - mu) / sigma), self.winsorize_cap)
                post_timing_z.append(z)

            z_stack = np.vstack(post_timing_z)  # shape (n_timing_cols, n_post_windows)
            with np.errstate(invalid="ignore"):
                timing_devs[self.warmup_k :] = np.nanmean(z_stack, axis=0)
        else:
            timing_devs[self.warmup_k :] = 0.0

        df["text_deviation"] = text_cos_devs
        df["maha_deviation"] = text_maha_devs
        df["timing_deviation"] = timing_devs
        return df

    def fit_deviation_standardizer(self, df_train_devs: pd.DataFrame) -> "PersonalBaselineEngine":
        for col in ["text_deviation", "maha_deviation", "timing_deviation"]:
            if col in df_train_devs.columns:
                vals = df_train_devs[col].dropna().to_numpy(dtype=float)
                if len(vals) > 1:
                    self.dev_scaler_stats[col] = (
                        float(np.mean(vals)),
                        max(float(np.std(vals, ddof=1)), 1e-4),
                    )
                else:
                    self.dev_scaler_stats[col] = (0.0, 1.0)
        return self

    def apply_standardized_fusion(
        self, df_devs: pd.DataFrame, has_text_cols: bool = True, has_timing_cols: bool = True
    ) -> pd.DataFrame:
        df = df_devs.copy()

        def _std_col(col_name: str) -> np.ndarray:
            if col_name not in df.columns:
                return np.full(len(df), np.nan)
            mu, sd = self.dev_scaler_stats.get(col_name, (0.0, 1.0))
            return (df[col_name].to_numpy(dtype=float) - mu) / sd

        z_text = _std_col("text_deviation")
        z_maha = _std_col("maha_deviation")
        z_time = _std_col("timing_deviation")

        df["text_dev_z"] = z_text
        df["maha_dev_z"] = z_maha
        df["timing_dev_z"] = z_time

        post_mask = df["window_idx"].to_numpy() >= self.warmup_k

        if has_text_cols and has_timing_cols:
            combined = self.w_text * z_maha + self.w_timing * z_time
            # Count post-warmup windows dropped because one modality was NaN
            dropped_mask = post_mask & (np.isnan(z_maha) | np.isnan(z_time))
            self.missing_modality_window_drops = int(np.sum(dropped_mask))
            df["combined_deviation"] = combined
        elif has_text_cols:
            self.missing_modality_window_drops = int(np.sum(post_mask & np.isnan(z_maha)))
            df["combined_deviation"] = z_maha
        else:
            self.missing_modality_window_drops = int(np.sum(post_mask & np.isnan(z_time)))
            df["combined_deviation"] = z_time

        return df

    def compute_conditioned_persistence(
        self, df_conditioned: pd.DataFrame, cond_col: str = "conditioned_deviation"
    ) -> pd.DataFrame:
        df = df_conditioned.copy()
        out_frames = []

        for _, grp in df.groupby("participant_id", sort=False):
            g = grp.sort_values("window_idx").copy()
            vals = g[cond_col].to_numpy(dtype=float)
            ewma_arr = np.full(len(g), np.nan)
            cusum_arr = np.zeros(len(g))

            s_prev = 0.0
            c_prev = 0.0
            init = False

            for i, v in enumerate(vals):
                if not np.isfinite(v):
                    continue
                if not init:
                    s_prev = float(v)
                    init = True
                else:
                    s_prev = self.ewma_alpha * float(v) + (1.0 - self.ewma_alpha) * s_prev
                ewma_arr[i] = s_prev

                c_prev = max(0.0, c_prev + (float(v) - self.cusum_slack))
                cusum_arr[i] = c_prev

            g["ewma_conditioned"] = ewma_arr
            g["cusum_conditioned"] = cusum_arr
            out_frames.append(g)

        return pd.concat(out_frames, ignore_index=True) if out_frames else df
