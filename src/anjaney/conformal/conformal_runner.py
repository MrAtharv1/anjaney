"""Orchestrates split-conformal calibration, operating point sweeps, and validity audits."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anjaney.conformal.classification import ConformalClassifier
from anjaney.conformal.cqr import ConformalQuantileRegressor
from anjaney.conformal.mapie_check import verify_against_mapie
from anjaney.conformal.normalized import DifficultyNormalizedConformal
from anjaney.conformal.policy import sweep_operating_points
from anjaney.conformal.split import SplitConformalRegressor
from anjaney.features.registry import FeatureRegistry
from anjaney.models.heads import get_model_head
from anjaney.models.participant_agg import ParticipantAggregator
from anjaney.utils.reproducibility import RunContext


def run_conformal_pipeline(
    features_path: Path | str = "data/processed/features.parquet",
    splits_path: Path | str = "data/processed/splits.json",
    output_dir: Path | str = "experiments/conformal",
    alpha: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    df_features = pd.read_parquet(features_path)
    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)

    # Participant-level feature aggregation
    feature_cols = FeatureRegistry.get_feature_columns(df_features, ["text_embed", "timing"])
    combined_cols = feature_cols["text_embed"] + feature_cols["timing"]

    aggregator = ParticipantAggregator(feature_cols=combined_cols, n_pca_components=24)
    df_agg, y_scores, y_bins = aggregator.aggregate(df_features)

    # Split train vs dev pool into training (60%) and calibration (40%)
    train_pids = splits["official"]["train"]
    dev_pids = splits["official"]["dev"]
    all_calib_pool = [p for p in train_pids + dev_pids if p in df_agg.index]

    rng = np.random.default_rng(seed)
    shuffled_pids = rng.permutation(all_calib_pool)
    n_train = int(0.65 * len(shuffled_pids))

    fit_pids = shuffled_pids[:n_train]
    cal_pids = shuffled_pids[n_train:]

    X_fit = df_agg.loc[fit_pids]
    y_fit = y_scores.loc[fit_pids].values
    X_cal = df_agg.loc[cal_pids]
    y_cal = y_scores.loc[cal_pids].values
    y_cal_bin = y_bins.loc[cal_pids].values

    # In-fold transformation
    X_fit_trans, X_cal_trans = aggregator.fit_transform_fold(X_fit, X_cal)

    # 1. Fit Point Model
    point_model = get_model_head("ridge", seed=seed)
    point_model.fit(X_fit_trans, y_fit)
    cal_preds = point_model.predict(X_cal_trans)

    # 2. Standard Split Conformal
    split_conf = SplitConformalRegressor(alpha=alpha)
    split_q = split_conf.calibrate(y_cal, cal_preds)
    low_split, high_split = split_conf.predict_intervals(cal_preds)

    # 3. CQR
    cqr = ConformalQuantileRegressor(alpha=alpha, seed=seed)
    cqr.fit_heads(X_fit_trans, y_fit)
    cqr.calibrate(X_cal_trans, y_cal)
    low_cqr, high_cqr = cqr.predict_intervals(X_cal_trans)

    # 4. Difficulty Normalized
    norm_conf = DifficultyNormalizedConformal(alpha=alpha, seed=seed)
    fit_preds = point_model.predict(X_fit_trans)
    norm_conf.fit_variance_model(X_fit_trans, y_fit, fit_preds)
    norm_conf.calibrate(X_cal_trans, y_cal, cal_preds)
    low_norm, high_norm = norm_conf.predict_intervals(X_cal_trans, cal_preds)

    # 5. Conformal Classification (Secondary track)
    clf_conf = ConformalClassifier(alpha=alpha, seed=seed)
    clf_conf.fit(X_fit_trans, y_bins.loc[fit_pids].values)
    clf_conf.calibrate(X_cal_trans, y_cal_bin)
    clf_sets = clf_conf.predict_sets(X_cal_trans)

    # 6. Operating Point Sweep on Split Intervals
    sweep_results = sweep_operating_points(low_split, high_split, y_cal, threshold=10.0)

    # 7. MAPIE Check
    mapie_diff = verify_against_mapie(X_fit_trans, y_fit, X_cal_trans, y_cal, alpha=alpha)

    # 8. Empirical Validity across nominal alphas
    validity_records = {}
    for a_val in [0.05, 0.10, 0.20, 0.30]:
        sc = SplitConformalRegressor(alpha=a_val)
        sc.calibrate(y_cal, cal_preds)
        low_b, up_b = sc.predict_intervals(cal_preds)
        covered = (y_cal >= low_b) & (y_cal <= up_b)
        validity_records[f"nominal_{1.0 - a_val:.2f}"] = {
            "empirical_coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(up_b - low_b)),
        }

    output_payload = {
        "alpha": alpha,
        "n_fit_participants": len(fit_pids),
        "n_calib_participants": len(cal_pids),
        "split_conformal": {
            "q_hat": split_q,
            "mean_interval_width": float(np.mean(high_split - low_split)),
            "coverage": float(np.mean((y_cal >= low_split) & (y_cal <= high_split))),
        },
        "cqr": {
            "mean_width": float(np.mean(high_cqr - low_cqr)),
            "coverage": float(np.mean((y_cal >= low_cqr) & (y_cal <= high_cqr))),
        },
        "difficulty_normalized": {
            "mean_width": float(np.mean(high_norm - low_norm)),
            "coverage": float(np.mean((y_cal >= low_norm) & (y_cal <= high_norm))),
        },
        "conformal_classification": {
            "abstention_rate_both_labels": float(np.mean([len(s) == 2 for s in clf_sets])),
            "coverage": float(np.mean([y_cal_bin[i] in clf_sets[i] for i in range(len(clf_sets))])),
        },
        "mapie_parity": mapie_diff,
        "operating_point_sweep": {
            "aurc": sweep_results["aurc"],
            "optimal_operating_point": sweep_results["optimal_operating_point"],
        },
        "validity_study": validity_records,
    }

    # Save artifacts and plot Risk-Coverage Curve
    with RunContext(
        run_name="conformal_calibration",
        config_dict={"alpha": alpha, "seed": seed},
        base_dir=output_dir,
        seed=seed,
    ) as ctx:
        res_file = ctx.run_dir / "conformal_results.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        # Plot Risk-Coverage Curve
        sweep_data = sweep_results["sweep_records"]
        covs = [r["coverage"] for r in sweep_data]
        risks = [r["selective_risk"] for r in sweep_data]

        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(covs, risks, color="#2ca02c", alpha=0.7, edgecolors="none")
        plt.title(f"Selective Risk vs. Coverage (AURC = {sweep_results['aurc']:.3f})")
        plt.xlabel("Coverage (1 - Abstention Rate)")
        plt.ylabel("Selective Risk (Screening Error on Covered)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(ctx.run_dir / "risk_coverage_curve.png", dpi=300)
        plt.close()

        output_payload["run_dir"] = ctx.run_dir

    return output_payload
