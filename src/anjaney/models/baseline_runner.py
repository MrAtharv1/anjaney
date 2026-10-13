"""Orchestrates out-of-fold cross-validation, leakage alarms, and permutation baselines."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anjaney.eval.metrics import compute_bootstrap_metrics
from anjaney.features.registry import FeatureRegistry
from anjaney.models.heads import get_model_head
from anjaney.models.participant_agg import ParticipantAggregator
from anjaney.utils.reproducibility import RunContext


def run_baseline_evaluation(
    features_path: Path | str,
    splits_path: Path | str,
    output_dir: Path | str = "experiments",
    run_permutation_test: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    df_features = pd.read_parquet(features_path)
    with open(splits_path, encoding="utf-8") as f:
        splits_data = json.load(f)

    # Filter to CV participants (train + dev)
    cv_folds = splits_data.get("cv_folds", [])
    if not cv_folds:
        raise ValueError("No cross-validation folds located in splits.json")

    modalities: dict[str, list[str]] = {
        "text_only": FeatureRegistry.get_feature_columns(df_features, ["text_embed"])["text_embed"],
        "timing_only": FeatureRegistry.get_feature_columns(df_features, ["timing"])["timing"],
        "lexical_only": FeatureRegistry.get_feature_columns(df_features, ["lexical"])["lexical"],
        "text_plus_timing": (
            FeatureRegistry.get_feature_columns(df_features, ["text_embed"])["text_embed"]
            + FeatureRegistry.get_feature_columns(df_features, ["timing"])["timing"]
        ),
    }

    head_names = ["dummy", "ridge", "elasticnet", "lightgbm"]
    run_summary: dict[str, Any] = {}

    with RunContext(
        run_name="population_baselines",
        config_dict={"modalities": list(modalities.keys()), "heads": head_names},
        base_dir=output_dir,
        seed=seed,
    ) as ctx:
        for mod_name, feat_cols in modalities.items():
            run_summary[mod_name] = {}
            aggregator = ParticipantAggregator(feature_cols=feat_cols, n_pca_components=32)
            df_agg, y_scores, _ = aggregator.aggregate(df_features)

            for head_name in head_names:
                # Aggregate predictions across validation folds
                y_true_all: list[float] = []
                y_pred_all: list[float] = []

                # Evaluate on the first repeat (e.g. 5 folds)
                repeat_0_folds = [f for f in cv_folds if f["repeat"] == 0]
                for fold in repeat_0_folds:
                    tr_pids = [p for p in fold["train_pids"] if p in df_agg.index]
                    val_pids = [p for p in fold["val_pids"] if p in df_agg.index]

                    X_train = df_agg.loc[tr_pids]
                    y_train = y_scores.loc[tr_pids].values
                    X_val = df_agg.loc[val_pids]
                    y_val = y_scores.loc[val_pids].values

                    X_tr_trans, X_val_trans = aggregator.fit_transform_fold(X_train, X_val)

                    model = get_model_head(head_name, seed=seed)
                    model.fit(X_tr_trans, y_train)
                    preds = model.predict(X_val_trans)

                    y_true_all.extend(y_val.tolist())
                    y_pred_all.extend(preds.tolist())

                metrics = compute_bootstrap_metrics(
                    np.array(y_true_all), np.array(y_pred_all), n_bootstrap=1000, seed=seed
                )
                run_summary[mod_name][head_name] = metrics

        # Permutation Test (Alarm check)
        permutation_alarm_passed = True
        if run_permutation_test:
            perm_aggregator = ParticipantAggregator(
                feature_cols=modalities["text_only"], n_pca_components=32
            )
            df_agg, y_scores, _ = perm_aggregator.aggregate(df_features)
            rng = np.random.default_rng(seed)
            shuffled_scores = pd.Series(
                rng.permutation(y_scores.values),
                index=y_scores.index,
                name="shuffled_phq8",
            )

            y_true_perm: list[float] = []
            y_pred_perm: list[float] = []
            repeat_0_folds = [f for f in cv_folds if f["repeat"] == 0]

            for fold in repeat_0_folds:
                tr_pids = [p for p in fold["train_pids"] if p in df_agg.index]
                val_pids = [p for p in fold["val_pids"] if p in df_agg.index]

                X_tr_trans, X_val_trans = perm_aggregator.fit_transform_fold(
                    df_agg.loc[tr_pids], df_agg.loc[val_pids]
                )
                model = get_model_head("ridge", seed=seed)
                model.fit(X_tr_trans, shuffled_scores.loc[tr_pids].values)
                y_true_perm.extend(shuffled_scores.loc[val_pids].values.tolist())
                y_pred_perm.extend(model.predict(X_val_trans).tolist())

            perm_metrics = compute_bootstrap_metrics(
                np.array(y_true_perm), np.array(y_pred_perm), n_bootstrap=500, seed=seed
            )
            perm_auroc = perm_metrics["auroc"]["point"]
            run_summary["permutation_test"] = {
                "head": "ridge",
                "modality": "text_only",
                "shuffled_auroc": perm_auroc,
            }

            if not (0.35 <= perm_auroc <= 0.65):
                permutation_alarm_passed = False

        # Save metrics JSON to experiment directory
        metrics_file = ctx.run_dir / "baseline_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)

        return {
            "summary": run_summary,
            "run_dir": ctx.run_dir,
            "permutation_passed": permutation_alarm_passed,
        }
