"""Window-to-participant feature aggregation with in-fold scaled dimensionality reduction."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ParticipantAggregator:
    def __init__(self, feature_cols: list[str], n_pca_components: int | None = None):
        self.feature_cols = feature_cols
        self.n_pca_components = n_pca_components
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None

    def aggregate(self, df_windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
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
            with np.errstate(all="ignore"):
                mean_feats = np.nanmean(feat_vals, axis=0)
                mean_feats = np.nan_to_num(mean_feats, nan=0.0)
                std_feats = (
                    np.nanstd(feat_vals, axis=0) if len(group) > 1 else np.zeros_like(mean_feats)
                )
                std_feats = np.nan_to_num(std_feats, nan=0.0)

                early_mask = group["rel_position"] < 0.5
                late_mask = group["rel_position"] >= 0.5

                if early_mask.any() and late_mask.any():
                    early_mean = np.nan_to_num(np.nanmean(feat_vals[early_mask], axis=0), nan=0.0)
                    late_mean = np.nan_to_num(np.nanmean(feat_vals[late_mask], axis=0), nan=0.0)
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
        """Fit scaler and PCA inside the training fold and transform validation fold."""
        X_tr_clean = np.nan_to_num(X_train.values, nan=0.0)
        X_val_clean = np.nan_to_num(X_val.values, nan=0.0)

        self.scaler = StandardScaler()
        X_tr_scaled = self.scaler.fit_transform(X_tr_clean)
        X_val_scaled = self.scaler.transform(X_val_clean)
        X_tr_scaled = np.nan_to_num(X_tr_scaled, nan=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)

        if self.n_pca_components is not None:
            max_components = min(
                self.n_pca_components,
                X_tr_scaled.shape[0] - 1,
                X_tr_scaled.shape[1],
            )
            if max_components >= 2:
                self.pca = PCA(n_components=max_components, random_state=42)
                X_tr_trans = self.pca.fit_transform(X_tr_scaled)
                X_val_trans = self.pca.transform(X_val_scaled)
                return X_tr_trans, X_val_trans

        return X_tr_scaled, X_val_scaled
