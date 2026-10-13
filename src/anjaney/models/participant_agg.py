"""Window-to-participant feature aggregation with in-fold dimensionality reduction."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class ParticipantAggregator:
    """Aggregates window-level features to participant-level vectors with in-fold PCA."""

    def __init__(self, feature_cols: list[str], n_pca_components: int | None = None):
        self.feature_cols = feature_cols
        self.n_pca_components = n_pca_components
        self.pca: PCA | None = None

    def aggregate(self, df_windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Aggregate window features by participant into (X_df, y_score, y_binary).

        Statistical Assumption:
            Window sequence order is summarized via first-moment (mean),
            second-moment (std), and temporal gradient (late - early).
        """
        records: list[dict] = []
        pids: list[str] = []
        scores: list[float] = []
        binaries: list[int] = []

        grouped = df_windows.groupby("participant_id", sort=False)
        for pid, group in grouped:
            pids.append(str(pid))
            scores.append(float(group["phq8_score"].iloc[0]))
            binaries.append(int(group["phq8_binary"].iloc[0]))

            feat_vals = group[self.feature_cols].values
            mean_feats = np.mean(feat_vals, axis=0)
            std_feats = np.std(feat_vals, axis=0) if len(group) > 1 else np.zeros_like(mean_feats)

            # Late-session vs early-session drift
            early_mask = group["rel_position"] < 0.5
            late_mask = group["rel_position"] >= 0.5

            if early_mask.any() and late_mask.any():
                early_mean = np.mean(feat_vals[early_mask], axis=0)
                late_mean = np.mean(feat_vals[late_mask], axis=0)
                drift_feats = late_mean - early_mean
            else:
                drift_feats = np.zeros_like(mean_feats)

            row_dict = {"participant_id": str(pid)}
            for i, col in enumerate(self.feature_cols):
                row_dict[f"{col}__mean"] = float(mean_feats[i])
                row_dict[f"{col}__std"] = float(std_feats[i])
                row_dict[f"{col}__drift"] = float(drift_feats[i])

            records.append(row_dict)

        df_agg = pd.DataFrame(records).set_index("participant_id")
        y_score = pd.Series(scores, index=df_agg.index, name="phq8_score")
        y_binary = pd.Series(binaries, index=df_agg.index, name="phq8_binary")

        return df_agg, y_score, y_binary

    def fit_transform_fold(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit PCA strictly inside the training fold and transform validation fold."""
        if self.n_pca_components is not None and self.n_pca_components < X_train.shape[1]:
            self.pca = PCA(n_components=self.n_pca_components, random_state=42)
            X_tr_trans = self.pca.fit_transform(X_train.values)
            X_val_trans = self.pca.transform(X_val.values)
            return X_tr_trans, X_val_trans
        return X_train.values, X_val.values
