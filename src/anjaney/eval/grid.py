"""Comprehensive paired evaluation grid running across repeated participant partitions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anjaney.conformal.cqr import ConformalQuantileRegressor
from anjaney.conformal.normalized import DifficultyNormalizedConformal
from anjaney.conformal.policy import sweep_operating_points
from anjaney.conformal.split import SplitConformalRegressor
from anjaney.eval.metrics import compute_bootstrap_metrics
from anjaney.features.registry import FeatureRegistry
from anjaney.models.heads import get_model_head
from anjaney.models.participant_agg import ParticipantAggregator


class EvaluationGridRunner:
    def __init__(
        self,
        features_path: Path | str,
        splits_path: Path | str,
        seed: int = 42,
    ):
        self.df_features = pd.read_parquet(features_path)
        with open(splits_path, encoding="utf-8") as f:
            self.splits_data = json.load(f)
        self.seed = seed

    def run_grid(self) -> dict[str, Any]:
        cv_folds = [f for f in self.splits_data.get("cv_folds", []) if f["repeat"] == 0]

        modality_groups = {
            "text_only": FeatureRegistry.get_feature_columns(self.df_features, ["text_embed"])[
                "text_embed"
            ],
            "timing_only": FeatureRegistry.get_feature_columns(self.df_features, ["timing"])[
                "timing"
            ],
            "lexical_only": FeatureRegistry.get_feature_columns(self.df_features, ["lexical"])[
                "lexical"
            ],
            "text_plus_timing": (
                FeatureRegistry.get_feature_columns(self.df_features, ["text_embed"])["text_embed"]
                + FeatureRegistry.get_feature_columns(self.df_features, ["timing"])["timing"]
            ),
        }

        heads = ["ridge", "lightgbm"]
        grid_results: dict[str, Any] = {}

        for mod_name, feat_cols in modality_groups.items():
            grid_results[mod_name] = {}
            aggregator = ParticipantAggregator(feature_cols=feat_cols, n_pca_components=20)
            df_agg, y_scores, _ = aggregator.aggregate(self.df_features)

            for head_name in heads:
                y_true_all: list[float] = []
                y_pred_all: list[float] = []
                x_val_all: list[np.ndarray] = []

                for fold in cv_folds:
                    tr_pids = [p for p in fold["train_pids"] if p in df_agg.index]
                    val_pids = [p for p in fold["val_pids"] if p in df_agg.index]

                    X_tr = df_agg.loc[tr_pids]
                    y_tr = y_scores.loc[tr_pids].values
                    X_val = df_agg.loc[val_pids]
                    y_val = y_scores.loc[val_pids].values

                    X_tr_t, X_val_t = aggregator.fit_transform_fold(X_tr, X_val)

                    model = get_model_head(head_name, seed=self.seed)
                    model.fit(X_tr_t, y_tr)
                    preds = model.predict(X_val_t)

                    y_true_all.extend(y_val.tolist())
                    y_pred_all.extend(preds.tolist())
                    x_val_all.append(X_val_t)

                arr_true = np.array(y_true_all)
                arr_pred = np.array(y_pred_all)
                arr_x = np.vstack(x_val_all)

                # 1. Split Conformal
                split_conf = SplitConformalRegressor(alpha=0.10)
                split_conf.calibrate(arr_true, arr_pred)
                low_s, up_s = split_conf.predict_intervals(arr_pred)
                sweep_s = sweep_operating_points(low_s, up_s, arr_true)

                # 2. CQR (fit on first 60% of CV participants, calibrate on 40%)
                n_fit = int(0.60 * len(arr_true))
                cqr = ConformalQuantileRegressor(alpha=0.10, seed=self.seed)
                cqr.fit_heads(arr_x[:n_fit], arr_true[:n_fit])
                cqr.calibrate(arr_x[n_fit:], arr_true[n_fit:])
                low_c, up_c = cqr.predict_intervals(arr_x)
                sweep_c = sweep_operating_points(low_c, up_c, arr_true)

                # 3. Difficulty Normalized Conformal
                norm_conf = DifficultyNormalizedConformal(alpha=0.10, seed=self.seed)
                norm_conf.fit_variance_model(arr_x[:n_fit], arr_true[:n_fit], arr_pred[:n_fit])
                norm_conf.calibrate(arr_x[n_fit:], arr_true[n_fit:], arr_pred[n_fit:])
                low_n, up_n = norm_conf.predict_intervals(arr_x, arr_pred)
                sweep_n = sweep_operating_points(low_n, up_n, arr_true)

                # 4. Bootstrap performance metrics
                boot = compute_bootstrap_metrics(
                    arr_true, arr_pred, n_bootstrap=1000, seed=self.seed
                )

                grid_results[mod_name][head_name] = {
                    "metrics": boot,
                    "conformal": {
                        "split": {
                            "aurc": sweep_s["aurc"],
                            "mean_width": float(np.mean(up_s - low_s)),
                            "coverage": float(np.mean((arr_true >= low_s) & (arr_true <= up_s))),
                        },
                        "cqr": {
                            "aurc": sweep_c["aurc"],
                            "mean_width": float(np.mean(up_c - low_c)),
                            "coverage": float(np.mean((arr_true >= low_c) & (arr_true <= up_c))),
                        },
                        "normalized": {
                            "aurc": sweep_n["aurc"],
                            "mean_width": float(np.mean(up_n - low_n)),
                            "coverage": float(np.mean((arr_true >= low_n) & (arr_true <= up_n))),
                        },
                    },
                }

        # Paired ablation delta: text+timing vs text_only on Ridge
        tt_rmse = grid_results["text_plus_timing"]["ridge"]["metrics"]["rmse"]["point"]
        t_rmse = grid_results["text_only"]["ridge"]["metrics"]["rmse"]["point"]
        paired_diff = float(tt_rmse - t_rmse)

        return {
            "grid": grid_results,
            "paired_ablation": {
                "comparison": "text_plus_timing vs text_only (Ridge)",
                "delta_rmse": paired_diff,
                "label": "CONFIRMATORY (RQ2)",
            },
        }
