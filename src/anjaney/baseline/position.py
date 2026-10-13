"""Confound removal: in-fold position conditioning across relative session progress."""

import numpy as np
import pandas as pd


class PositionConditioner:
    """Estimates population mean trajectory per position bin on training folds only.

    Statistical Assumption:
        Interview progression induces generic fatigue or engagement drift.
        Subtracting position bin expectations removes shared progression confounds.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.bin_means: dict[int, float] = {}
        self.global_mean: float = 0.0

    def _get_bin(self, rel_position: float) -> int:
        b = int(np.clip(rel_position, 0.0, 0.999) * self.n_bins)
        return b

    def fit(self, df_windows: pd.DataFrame, deviation_col: str) -> "PositionConditioner":
        """Compute mean deviation for each relative position bin on training participants only."""
        bins = [self._get_bin(pos) for pos in df_windows["rel_position"]]
        df_temp = df_windows.copy()
        df_temp["pos_bin"] = bins

        grouped = df_temp.groupby("pos_bin")[deviation_col].mean()
        self.bin_means = grouped.to_dict()
        self.global_mean = float(df_temp[deviation_col].mean())
        return self

    def transform(self, df_windows: pd.DataFrame, deviation_col: str) -> np.ndarray:
        """Subtract training-bin mean deviation from each window."""
        conditioned = []
        for _, row in df_windows.iterrows():
            pos_bin = self._get_bin(float(row["rel_position"]))
            expected_mean = self.bin_means.get(pos_bin, self.global_mean)
            val = float(row[deviation_col])
            conditioned.append(val - expected_mean)
        return np.array(conditioned, dtype=float)

    def fit_transform(self, df_windows: pd.DataFrame, deviation_col: str) -> np.ndarray:
        return self.fit(df_windows, deviation_col).transform(df_windows, deviation_col)
