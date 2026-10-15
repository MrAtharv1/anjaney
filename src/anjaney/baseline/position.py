"""Hierarchical, vectorized script-phase and position conditioner with fail-closed phase verification."""

import numpy as np
import pandas as pd


class PositionConditioner:
    """Removes expected interviewer script-phase and progression drift fitted strictly on training data.

    Conditioning Hierarchy:
        1. Joint cell: (prompt_phase, rel_position_bin) if cell count >= min_bin_count
        2. Script phase fallback: prompt_phase mean if count >= min_bin_count
        3. Coarse position fallback: rel_position_bin if count >= min_bin_count
        4. Global training mean fallback
    """

    def __init__(
        self,
        n_bins: int = 5,
        n_turn_bins: int = 5,
        min_bin_count: int = 5,
        require_prompt_phase: bool = True,
    ):
        self.n_bins = n_bins
        self.n_turn_bins = n_turn_bins
        self.min_bin_count = min_bin_count
        self.require_prompt_phase = require_prompt_phase

        self.pos_edges: np.ndarray = np.linspace(0.0, 1.0, n_bins + 1)
        self.turn_edges: np.ndarray | None = None

        self.joint_means: dict[tuple[int, int], float] = {}
        self.phase_means: dict[int, float] = {}
        self.coarse_pos_means: dict[int, float] = {}
        self.global_mean: float = 0.0
        self.conditioning_mode: str = "UNFITTED"
        self.is_fitted: bool = False

    def _compute_bins(
        self, df: pd.DataFrame, fit_turn_edges: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rel_pos = df["rel_position"].to_numpy(dtype=float)
        valid_pos_mask = np.isfinite(rel_pos)

        clipped_pos = np.where(valid_pos_mask, np.clip(rel_pos, 0.0, 1.0 - 1e-9), 0.0)
        pos_bins = np.digitize(clipped_pos, self.pos_edges[1:-1], right=False)

        if "prompt_phase" in df.columns:
            phase_vals = pd.to_numeric(df["prompt_phase"], errors="coerce").to_numpy(dtype=float)
            valid_phase = np.isfinite(phase_vals)
            valid_pos_mask = valid_pos_mask & valid_phase
            turn_bins = np.where(valid_phase, phase_vals, -1).astype(int)
            if fit_turn_edges:
                self.conditioning_mode = "prompt_phase_and_rel_position"
        else:
            if self.require_prompt_phase:
                raise RuntimeError(
                    "Confound guard: 'prompt_phase' column missing from features DataFrame. "
                    "Set require_prompt_phase=False only for explicitly labelled UNCONDITIONED runs."
                )
            if fit_turn_edges:
                self.conditioning_mode = "UNCONDITIONED_FALLBACK_WINDOW_IDX"
            if "window_idx" in df.columns:
                w_idx = df["window_idx"].to_numpy(dtype=float)
                valid_w = np.isfinite(w_idx)
                valid_pos_mask = valid_pos_mask & valid_w
                if fit_turn_edges or self.turn_edges is None:
                    max_w = float(np.nanmax(w_idx)) if np.any(valid_w) else 10.0
                    self.turn_edges = np.linspace(0.0, max(max_w + 1e-6, 1.0), self.n_turn_bins + 1)
                safe_w = np.where(valid_w, w_idx, 0.0)
                turn_bins = np.digitize(safe_w, self.turn_edges[1:-1], right=False)
            else:
                turn_bins = np.zeros(len(df), dtype=int)

        return pos_bins, turn_bins, valid_pos_mask

    def fit(self, df_windows: pd.DataFrame, deviation_col: str) -> "PositionConditioner":
        if deviation_col not in df_windows.columns or "rel_position" not in df_windows.columns:
            raise ValueError(f"Missing required columns '{deviation_col}' or 'rel_position'.")

        dev_vals = df_windows[deviation_col].to_numpy(dtype=float)
        pos_bins, turn_bins, valid_pos_mask = self._compute_bins(df_windows, fit_turn_edges=True)
        valid_mask = valid_pos_mask & np.isfinite(dev_vals)

        if not np.any(valid_mask):
            raise ValueError(
                f"Cannot fit PositionConditioner: zero valid finite rows in '{deviation_col}'"
            )

        valid_devs = dev_vals[valid_mask]
        self.global_mean = float(np.mean(valid_devs))

        df_work = pd.DataFrame(
            {"dev": valid_devs, "pos_b": pos_bins[valid_mask], "turn_b": turn_bins[valid_mask]}
        )

        joint_stats = df_work.groupby(["turn_b", "pos_b"])["dev"].agg(["count", "mean"])
        self.joint_means = {
            (int(idx[0]), int(idx[1])): float(row["mean"])
            for idx, row in joint_stats.iterrows()
            if row["count"] >= self.min_bin_count
        }

        phase_stats = df_work.groupby("turn_b")["dev"].agg(["count", "mean"])
        self.phase_means = {
            int(idx): float(row["mean"])
            for idx, row in phase_stats.iterrows()
            if row["count"] >= self.min_bin_count
        }

        coarse_stats = df_work.groupby("pos_b")["dev"].agg(["count", "mean"])
        self.coarse_pos_means = {
            int(idx): float(row["mean"])
            for idx, row in coarse_stats.iterrows()
            if row["count"] >= self.min_bin_count
        }

        self.is_fitted = True
        return self

    def transform(self, df_windows: pd.DataFrame, deviation_col: str) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PositionConditioner must be fitted before calling transform.")

        dev_vals = df_windows[deviation_col].to_numpy(dtype=float)
        pos_bins, turn_bins, valid_pos_mask = self._compute_bins(df_windows, fit_turn_edges=False)
        valid_mask = valid_pos_mask & np.isfinite(dev_vals)

        expected = np.full(len(df_windows), self.global_mean, dtype=float)
        for i in np.flatnonzero(valid_mask):
            t_b, p_b = int(turn_bins[i]), int(pos_bins[i])
            if (t_b, p_b) in self.joint_means:
                expected[i] = self.joint_means[(t_b, p_b)]
            elif t_b in self.phase_means:
                expected[i] = self.phase_means[t_b]
            elif p_b in self.coarse_pos_means:
                expected[i] = self.coarse_pos_means[p_b]

        return np.where(valid_mask, dev_vals - expected, np.nan)

    def fit_transform(self, df_windows: pd.DataFrame, deviation_col: str) -> np.ndarray:
        return self.fit(df_windows, deviation_col).transform(df_windows, deviation_col)
