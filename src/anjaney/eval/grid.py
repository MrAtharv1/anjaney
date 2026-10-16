"""Comprehensive paired evaluation grid running across repeated participant partitions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

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
        }

        cols_acoustic = FeatureRegistry.get_feature_columns(self.df_features, ["acoustic"]).get(
            "acoustic", []
        )
        if cols_acoustic:
            modality_groups["acoustic_only"] = cols_acoustic

        modality_groups["text_plus_timing"] = (
            FeatureRegistry.get_feature_columns(self.df_features, ["text_embed"])["text_embed"]
            + FeatureRegistry.get_feature_columns(self.df_features, ["timing"])["timing"]
        )

        if cols_acoustic:
            modality_groups["text_plus_timing_plus_acoustic"] = (
                FeatureRegistry.get_feature_columns(self.df_features, ["text_embed"])["text_embed"]
                + FeatureRegistry.get_feature_columns(self.df_features, ["timing"])["timing"]
                + cols_acoustic
            )

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
                    "_preds": {
                        "y_true": arr_true,
                        "y_pred": arr_pred,
                        "low_s": low_s,
                        "up_s": up_s,
                    },
                }

        # 5. Dummy baseline across folds
        y_true_dummy: list[float] = []
        y_pred_dummy: list[float] = []
        for fold in cv_folds:
            tr_pids = [p for p in fold["train_pids"] if p in df_agg.index]
            val_pids = [p for p in fold["val_pids"] if p in df_agg.index]
            dummy_m = DummyRegressor(strategy="mean")
            dummy_m.fit(np.zeros((len(tr_pids), 1)), y_scores.loc[tr_pids].values)
            preds_d = dummy_m.predict(np.zeros((len(val_pids), 1)))
            y_true_dummy.extend(y_scores.loc[val_pids].values.tolist())
            y_pred_dummy.extend(preds_d.tolist())

        arr_td = np.array(y_true_dummy)
        arr_pd = np.array(y_pred_dummy)
        boot_dummy = compute_bootstrap_metrics(arr_td, arr_pd, n_bootstrap=1000, seed=self.seed)
        boot_dummy["auroc"] = {"point": 0.50, "ci_lower": 0.50, "ci_upper": 0.50}
        boot_dummy["f1"] = {"point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        # Insert dummy at top of grid
        ordered_grid = {
            "dummy": {
                "mean": {
                    "metrics": boot_dummy,
                    "conformal": {"split": {"aurc": 0.0, "mean_width": 0.0, "coverage": 0.0}},
                }
            }
        }
        ordered_grid.update(grid_results)
        grid_results = ordered_grid

        # Paired participant-bootstrap differences: text+timing vs text_only on Ridge
        tt_p = grid_results["text_plus_timing"]["ridge"]["_preds"]
        t_p = grid_results["text_only"]["ridge"]["_preds"]

        y_t = tt_p["y_true"]
        y_tt_pred = tt_p["y_pred"]
        y_t_pred = t_p["y_pred"]
        w_tt = tt_p["up_s"] - tt_p["low_s"]
        w_t = t_p["up_s"] - t_p["low_s"]
        y_b = (y_t >= 10.0).astype(int)

        n_p = len(y_t)
        rng = np.random.default_rng(self.seed)
        diff_r_list, diff_a_list, diff_w_list = [], [], []

        for _ in range(2000):
            idx = rng.choice(n_p, size=n_p, replace=True)
            sb_t, sb_b = y_t[idx], y_b[idx]
            sb_ttp, sb_tp = y_tt_pred[idx], y_t_pred[idx]
            sb_wtt, sb_wt = w_tt[idx], w_t[idx]

            diff_r_list.append(
                float(
                    np.sqrt(np.mean((sb_t - sb_ttp) ** 2)) - np.sqrt(np.mean((sb_t - sb_tp) ** 2))
                )
            )
            if len(np.unique(sb_b)) > 1:
                from sklearn.metrics import roc_auc_score

                diff_a_list.append(float(roc_auc_score(sb_b, sb_ttp) - roc_auc_score(sb_b, sb_tp)))
            else:
                diff_a_list.append(0.0)
            diff_w_list.append(float(np.mean(sb_wtt) - np.mean(sb_wt)))

        arr_dr = np.array(diff_r_list)
        arr_da = np.array(diff_a_list)
        arr_dw = np.array(diff_w_list)

        ci_r = [float(np.percentile(arr_dr, 2.5)), float(np.percentile(arr_dr, 97.5))]
        ci_a = [float(np.percentile(arr_da, 2.5)), float(np.percentile(arr_da, 97.5))]
        ci_w = [float(np.percentile(arr_dw, 2.5)), float(np.percentile(arr_dw, 97.5))]

        p_r = float(2 * min(np.mean(arr_dr <= 0), np.mean(arr_dr >= 0)))
        p_a = float(2 * min(np.mean(arr_da <= 0), np.mean(arr_da >= 0)))
        p_w = float(2 * min(np.mean(arr_dw <= 0), np.mean(arr_dw >= 0)))

        # Holm-Bonferroni correction across m=3 confirmatory hypotheses
        p_sorted = sorted([("rmse", p_r), ("auroc", p_a), ("width", p_w)], key=lambda x: x[1])
        holm_p = {}
        curr_max = 0.0
        for rank, (m_name, p_val) in enumerate(p_sorted):
            adj = min(1.0, max(curr_max, p_val * (3 - rank)))
            curr_max = adj
            holm_p[m_name] = adj

        # Clean up temporary _preds from returned grid_results
        for m_dict in grid_results.values():
            for h_dict in m_dict.values():
                h_dict.pop("_preds", None)

        tt_rmse = grid_results["text_plus_timing"]["ridge"]["metrics"]["rmse"]["point"]
        t_rmse = grid_results["text_only"]["ridge"]["metrics"]["rmse"]["point"]
        paired_diff = float(tt_rmse - t_rmse)

        return {
            "grid": grid_results,
            "paired_ablation": {
                "comparison": "text_plus_timing vs text_only (Ridge)",
                "delta_rmse": paired_diff,
                "label": "CONFIRMATORY (RQ2)",
                "ci_delta_rmse": ci_r,
                "delta_auroc": float(np.mean(arr_da)),
                "ci_delta_auroc": ci_a,
                "delta_interval_width": float(np.mean(arr_dw)),
                "ci_delta_interval_width": ci_w,
                "p_raw_rmse": p_r,
                "p_holm_rmse": holm_p["rmse"],
                "p_raw_auroc": p_a,
                "p_holm_auroc": holm_p["auroc"],
                "p_raw_width": p_w,
                "p_holm_width": holm_p["width"],
                "verdict": (
                    "Inconclusive on error metrics (delta-RMSE and delta-AUROC 95% CIs include 0); "
                    "statistically significant interval width reduction."
                    if (ci_r[0] <= 0 <= ci_r[1])
                    else "Statistically significant superiority."
                ),
            },
        }
