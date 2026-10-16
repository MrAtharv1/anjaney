"""Single Confirmatory Test Set Evaluation for ANJANEY Screening Model.

Protocol Version: prereg-v2
Refit on Train+Dev (N=141), evaluate on official held-out Test (N=47).
Locking enforcement: experiments/test_eval.lock written before evaluation.
Zero post-test modifications permitted.
"""

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold

from anjaney.features.registry import FeatureRegistry
from anjaney.models.participant_agg import ParticipantAggregator
from anjaney.utils.reproducibility import get_git_hash


def compute_wilson_score_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score interval for binomial proportions."""
    if total == 0:
        return 0.0, 1.0
    p_hat = successes / total
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    half_width = (z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total)) / denom
    lower = max(0.0, float(center - half_width))
    upper = min(1.0, float(center + half_width))
    return lower, upper


def evaluate_participant_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
    threshold: float = 10.0,
    f1_threshold: float = 10.0,
) -> dict[str, dict[str, float]]:
    """Compute metrics with 95% participant-bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def calc_metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
        rmse = float(root_mean_squared_error(yt, yp))
        mae = float(mean_absolute_error(yt, yp))
        corr = float(np.corrcoef(yt, yp)[0, 1]) if len(np.unique(yp)) > 1 else 0.0
        y_bin = (yt >= threshold).astype(int)
        if len(np.unique(y_bin)) > 1 and len(np.unique(yp)) > 1:
            auroc = float(roc_auc_score(y_bin, yp))
        else:
            auroc = 0.50
        y_pred_bin = (yp >= f1_threshold).astype(int)
        f1 = float(f1_score(y_bin, y_pred_bin, zero_division=0))
        # Logistic sigmoid probability estimate for Brier loss
        # Use margin from threshold
        margin = yp - threshold
        prob = 1.0 / (1.0 + np.exp(-margin))
        brier = float(brier_score_loss(y_bin, prob))
        return {
            "rmse": rmse,
            "mae": mae,
            "pearson_r": corr,
            "auroc": auroc,
            "f1": f1,
            "brier": brier,
        }

    point = calc_metrics(y_true, y_pred)
    boot: dict[str, list[float]] = {k: [] for k in point}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m = calc_metrics(y_true[idx], y_pred[idx])
        for k in m:
            boot[k].append(m[k])

    out: dict[str, dict[str, float]] = {}
    for k in point:
        arr = np.array(boot[k])
        out[k] = {
            "point": point[k],
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }
    return out


def run_single_test_evaluation(
    features_path: str = "data/processed/features.parquet",
    splits_path: str = "data/processed/splits.json",
    config_path: str = "configs/final_config.yaml",
    lock_path: str = "experiments/test_eval.lock",
    output_path: str = "experiments/test_eval/results.json",
    seed: int = 42,
) -> dict[str, Any]:
    lock_file = Path(lock_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Enforcement of single test evaluation rule
    if lock_file.exists():
        raise RuntimeError(
            f"PREFLIGHT ABORT: {lock_file} already exists. Single test evaluation policy strictly forbids duplicate execution."
        )

    git_hash = get_git_hash()
    timestamp_start = datetime.datetime.now(datetime.UTC).isoformat()

    # Write initial lock entry FIRST
    initial_lock = {
        "status": "RUNNING",
        "timestamp_start": timestamp_start,
        "git_hash": git_hash,
        "config_path": config_path,
        "model": "text_only Ridge (alpha=10.0, 20 PCA components)",
    }
    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(initial_lock, f, indent=2)

    try:
        # Load data
        df_feat = pd.read_parquet(features_path)
        with open(splits_path, encoding="utf-8") as f:
            splits_data = json.load(f)

        train_pids = sorted(list(splits_data["official"]["train"]))
        dev_pids = sorted(list(splits_data["official"]["dev"]))
        test_pids = sorted(list(splits_data["official"]["test"]))

        pool_val_pids = sorted(train_pids + dev_pids)
        assert len(pool_val_pids) == 141, f"Expected 141 train+dev, got {len(pool_val_pids)}"
        assert len(test_pids) == 47, f"Expected 47 test, got {len(test_pids)}"

        text_cols = FeatureRegistry.get_feature_columns(df_feat, ["text_embed"])["text_embed"]
        aggregator = ParticipantAggregator(feature_cols=text_cols, n_pca_components=20)
        df_agg, y_scores, _ = aggregator.aggregate(df_feat)

        # -------------------------------------------------------------
        # In-Fold Validation Calibration on Train+Dev (N=141)
        # -------------------------------------------------------------
        y_val_pool = y_scores.loc[pool_val_pids].values
        X_val_pool = df_agg.loc[pool_val_pids]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        y_val_bin = (y_val_pool >= 10.0).astype(int)

        oof_preds = np.zeros(len(y_val_pool))
        oof_probs = np.zeros((len(y_val_pool), 2))

        for tr_idx, cal_idx in skf.split(X_val_pool, y_val_bin):
            tr_pids_f = [pool_val_pids[i] for i in tr_idx]
            cal_pids_f = [pool_val_pids[i] for i in cal_idx]

            X_tr_f, X_cal_f = aggregator.fit_transform_fold(df_agg.loc[tr_pids_f], df_agg.loc[cal_pids_f])
            y_tr_f = y_scores.loc[tr_pids_f].values

            m_ridge = Ridge(alpha=10.0, random_state=seed)
            m_ridge.fit(X_tr_f, y_tr_f)
            p_cal = m_ridge.predict(X_cal_f)
            oof_preds[cal_idx] = np.clip(p_cal, 0.0, 24.0)

            # Classification model for LAC
            m_clf = LogisticRegression(C=1.0, random_state=seed)
            m_clf.fit(X_tr_f, y_val_bin[tr_idx])
            oof_probs[cal_idx] = m_clf.predict_proba(X_cal_f)

        # Find in-fold optimal F1 threshold
        threshold_grid = np.linspace(5.0, 15.0, 101)
        best_f1 = -1.0
        best_tau = 10.0
        for tau in threshold_grid:
            f1_cand = f1_score(y_val_bin, (oof_preds >= tau).astype(int), zero_division=0)
            if f1_cand > best_f1:
                best_f1 = f1_cand
                best_tau = float(tau)

        # Conformal regression calibration on pooled residuals
        cal_residuals = np.abs(y_val_pool - oof_preds)
        n_cal = len(cal_residuals)

        # Alpha = 0.10
        k_10 = int(np.ceil((n_cal + 1) * 0.90))
        q_hat_10 = float(np.sort(cal_residuals)[k_10 - 1])

        # Alpha = 0.20
        k_20 = int(np.ceil((n_cal + 1) * 0.80))
        q_hat_20 = float(np.sort(cal_residuals)[k_20 - 1])

        # Alpha = 0.05
        k_05 = int(np.ceil((n_cal + 1) * 0.95))
        q_hat_05 = float(np.sort(cal_residuals)[k_05 - 1])

        # LAC calibration threshold at alpha = 0.20
        cal_scores_lac = np.array([1.0 - oof_probs[i, y_val_bin[i]] for i in range(n_cal)])
        k_lac_20 = int(np.ceil((n_cal + 1) * 0.80))
        tau_lac_20 = float(np.sort(cal_scores_lac)[k_lac_20 - 1])

        # -------------------------------------------------------------
        # Refit on FULL Train+Dev (N=141) and Predict on Test (N=47)
        # -------------------------------------------------------------
        X_pool = df_agg.loc[pool_val_pids]
        y_pool = y_scores.loc[pool_val_pids].values

        X_test = df_agg.loc[test_pids]
        y_test = y_scores.loc[test_pids].values
        y_test_bin = (y_test >= 10.0).astype(int)

        X_pool_trans, X_test_trans = aggregator.fit_transform_fold(X_pool, X_test)

        final_ridge = Ridge(alpha=10.0, random_state=seed)
        final_ridge.fit(X_pool_trans, y_pool)
        test_preds_raw = final_ridge.predict(X_test_trans)
        test_preds = np.clip(test_preds_raw, 0.0, 24.0)

        # Prediction vector hash for tamper evidence
        pred_hash = hashlib.sha256(test_preds.tobytes()).hexdigest()

        # Classification refit for LAC
        final_clf = LogisticRegression(C=1.0, random_state=seed)
        final_clf.fit(X_pool_trans, (y_pool >= 10.0).astype(int))
        test_probs = final_clf.predict_proba(X_test_trans)

        # -------------------------------------------------------------
        # Dummy Baseline Refit & Evaluation
        # -------------------------------------------------------------
        dummy_mean = float(np.mean(y_pool))
        dummy_preds = np.full_like(y_test, dummy_mean)
        dummy_metrics = evaluate_participant_bootstrap(y_test, dummy_preds, n_bootstrap=2000, seed=seed)
        dummy_metrics["auroc"] = {"point": 0.50, "ci_lower": 0.50, "ci_upper": 0.50}
        dummy_metrics["f1"] = {"point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        # -------------------------------------------------------------
        # Primary Model Performance Metrics on Test
        # -------------------------------------------------------------
        model_metrics = evaluate_participant_bootstrap(
            y_test, test_preds, n_bootstrap=2000, seed=seed, threshold=10.0, f1_threshold=best_tau
        )
        std_f1_metrics = evaluate_participant_bootstrap(
            y_test, test_preds, n_bootstrap=2000, seed=seed, threshold=10.0, f1_threshold=10.0
        )["f1"]

        # -------------------------------------------------------------
        # Conformal Regression Intervals on Test
        # -------------------------------------------------------------
        conformal_eval: dict[str, Any] = {}
        for alpha_val, q_hat in [(0.05, q_hat_05), (0.10, q_hat_10), (0.20, q_hat_20)]:
            low = np.clip(test_preds - q_hat, 0.0, 24.0)
            up = np.clip(test_preds + q_hat, 0.0, 24.0)
            cov = float(np.mean((y_test >= low) & (y_test <= up)))
            widths = up - low
            mean_w = float(np.mean(widths))
            sd_w = float(np.std(widths, ddof=1))

            triage_policies: dict[str, Any] = {}
            for w_max in [6.0, 8.0, 10.0]:
                spans_margin = (low <= 10.0 + 0.5) & (up >= 10.0 - 0.5)
                exceeds_w = widths > w_max
                abstain_mask = spans_margin | exceeds_w

                n_abstain = int(np.sum(abstain_mask))
                abstain_rate = float(n_abstain / len(y_test))

                auto_mask = ~abstain_mask
                if np.sum(auto_mask) > 0:
                    sel_rmse = float(root_mean_squared_error(y_test[auto_mask], test_preds[auto_mask]))
                    sel_mae = float(mean_absolute_error(y_test[auto_mask], test_preds[auto_mask]))
                else:
                    sel_rmse = 0.0
                    sel_mae = 0.0

                triage_policies[f"w_max_{int(w_max)}"] = {
                    "abstain_count": n_abstain,
                    "abstain_rate": abstain_rate,
                    "autonomous_count": int(np.sum(auto_mask)),
                    "autonomous_rmse": sel_rmse,
                    "autonomous_mae": sel_mae,
                }

            conformal_eval[f"alpha_{alpha_val:.2f}"] = {
                "nominal_coverage": 1.0 - alpha_val,
                "q_hat": q_hat,
                "empirical_coverage": cov,
                "mean_width": mean_w,
                "sd_width": sd_w,
                "triage": triage_policies,
            }

        # -------------------------------------------------------------
        # Set-Valued Conformal Classification (LAC at alpha=0.20)
        # -------------------------------------------------------------
        test_sets = []
        for i in range(len(test_probs)):
            s = set()
            for c in [0, 1]:
                if 1.0 - test_probs[i, c] <= tau_lac_20:
                    s.add(c)
            test_sets.append(s)

        lac_cov = float(np.mean([y_test_bin[i] in test_sets[i] for i in range(len(y_test_bin))]))
        singletons = [i for i, s in enumerate(test_sets) if len(s) == 1]
        both_abstain = [i for i, s in enumerate(test_sets) if len(s) == 2]
        empty_abstain = [i for i, s in enumerate(test_sets) if len(s) == 0]

        n_singletons = len(singletons)
        singleton_rate = float(n_singletons / len(y_test_bin))
        abstain_rate = float((len(both_abstain) + len(empty_abstain)) / len(y_test_bin))

        singleton_errors = sum(1 for i in singletons if list(test_sets[i])[0] != y_test_bin[i])
        singleton_err_rate = float(singleton_errors / n_singletons) if n_singletons > 0 else 0.0
        w_low, w_high = compute_wilson_score_ci(singleton_errors, n_singletons)

        lac_results = {
            "nominal_coverage": 0.80,
            "tau_lac": tau_lac_20,
            "empirical_coverage": lac_cov,
            "singleton_count": n_singletons,
            "singleton_rate": singleton_rate,
            "abstain_count": len(both_abstain) + len(empty_abstain),
            "abstain_rate": abstain_rate,
            "abstain_breakdown": {
                "both_count": len(both_abstain),
                "empty_count": len(empty_abstain),
            },
            "error_among_singletons": {
                "n_singletons": n_singletons,
                "n_errors": singleton_errors,
                "point": singleton_err_rate,
                "wilson_ci_95": [w_low, w_high],
            },
        }

        # -------------------------------------------------------------
        # Clinical Cost Evaluation on Test
        # -------------------------------------------------------------
        cost_sweep_results: dict[str, Any] = {}
        ratios = [
            ("standard", 5.0, 1.0, 0.5),
            ("equal", 1.0, 1.0, 0.5),
            ("high_safety", 10.0, 1.0, 0.5),
            ("low_abstain_cost", 5.0, 1.0, 0.1),
        ]

        q10 = q_hat_10
        low10 = np.clip(test_preds - q10, 0.0, 24.0)
        up10 = np.clip(test_preds + q10, 0.0, 24.0)
        abstain10 = ((low10 <= 10.5) & (up10 >= 9.5)) | ((up10 - low10) > 8.0)

        for name, c_fn, c_fp, c_abs in ratios:
            cost_escalate_all = c_abs
            costs = []
            for i in range(len(y_test_bin)):
                if abstain10[i]:
                    costs.append(c_abs)
                else:
                    pred_bin = int(test_preds[i] >= 10.0)
                    true_bin = y_test_bin[i]
                    if true_bin == 1 and pred_bin == 0:
                        costs.append(c_fn)
                    elif true_bin == 0 and pred_bin == 1:
                        costs.append(c_fp)
                    else:
                        costs.append(0.0)
            mean_policy_cost = float(np.mean(costs))
            cost_sweep_results[name] = {
                "c_fn": c_fn,
                "c_fp": c_fp,
                "c_abstain": c_abs,
                "policy_cost_per_patient": mean_policy_cost,
                "escalate_all_cost_per_patient": cost_escalate_all,
                "delta_vs_escalate_all": float(mean_policy_cost - cost_escalate_all),
            }

        # -------------------------------------------------------------
        # Compile Master Results JSON
        # -------------------------------------------------------------
        timestamp_end = datetime.datetime.now(datetime.UTC).isoformat()
        results = {
            "status": "COMPLETED",
            "protocol": "prereg-v2",
            "model": "text_only Ridge (alpha=10.0, 20 PCA components)",
            "git_hash": git_hash,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "prediction_vector_sha256": pred_hash,
            "sample_counts": {
                "n_train_dev_pooled": len(pool_val_pids),
                "n_test": len(test_pids),
                "n_test_windows": len(df_feat[df_feat["participant_id"].isin(test_pids)]),
            },
            "operating_points": {
                "in_fold_optimal_f1_threshold": best_tau,
                "in_fold_optimal_f1_score": best_f1,
                "standard_screening_threshold": 10.0,
                "conformal_q_hat_alpha_0_10": q_hat_10,
                "conformal_q_hat_alpha_0_20": q_hat_20,
                "lac_tau_alpha_0_20": tau_lac_20,
            },
            "performance_metrics": {
                "deployed_model": model_metrics,
                "deployed_model_f1_at_std_threshold": std_f1_metrics,
                "dummy_baseline": dummy_metrics,
                "dummy_mean_prediction": dummy_mean,
            },
            "conformal_regression": conformal_eval,
            "conformal_classification_lac": lac_results,
            "clinical_cost_evaluation": cost_sweep_results,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        final_lock = {
            "status": "LOCKED",
            "git_hash": git_hash,
            "timestamp": timestamp_end,
            "prediction_vector_sha256": pred_hash,
            "results_summary": {
                "n_test": len(test_pids),
                "rmse": model_metrics["rmse"]["point"],
                "mae": model_metrics["mae"]["point"],
                "auroc": model_metrics["auroc"]["point"],
                "f1_in_fold_tau": model_metrics["f1"]["point"],
                "coverage_alpha_0_10": conformal_eval["alpha_0.10"]["empirical_coverage"],
                "width_alpha_0_10": conformal_eval["alpha_0.10"]["mean_width"],
                "lac_coverage_alpha_0_20": lac_results["empirical_coverage"],
                "lac_singleton_rate": lac_results["singleton_rate"],
            },
        }
        with open(lock_file, "w", encoding="utf-8") as f:
            json.dump(final_lock, f, indent=2)

        return results

    except Exception as e:
        error_lock = {
            "status": "FAILED",
            "git_hash": git_hash,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "error": str(e),
        }
        with open(lock_file, "w", encoding="utf-8") as f:
            json.dump(error_lock, f, indent=2)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single confirmatory test set evaluation")
    parser.add_argument("--unlock-test", action="store_true", help="Explicit investigator flag to unlock test split")
    args = parser.parse_args()

    if not args.unlock_test:
        raise RuntimeError("Refusing to run without explicit --unlock-test flag.")

    res = run_single_test_evaluation()
    print("Test set evaluation completed successfully.")
    print("Locked status recorded in experiments/test_eval.lock")
