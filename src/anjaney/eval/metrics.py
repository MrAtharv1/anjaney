"""Participant-level bootstrap confidence interval calculation for regression and derived screening triage."""

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score, root_mean_squared_error


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 10.0
) -> dict[str, float]:
    """Compute point metrics for PHQ-8 regression and derived binary triage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    # Derived binary discrimination
    if len(np.unique(y_true_bin)) > 1:
        auroc = float(roc_auc_score(y_true_bin, y_pred))
    else:
        auroc = 0.50

    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    return {"rmse": rmse, "mae": mae, "auroc": auroc, "f1": f1}


def compute_bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
    threshold: float = 10.0,
) -> dict[str, dict[str, float]]:
    """Compute point estimates and 95% bootstrap confidence intervals resampling participants.

    Statistical Assumption:
        Participants are independent and identically distributed draws from the campus population.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    point_estimates = compute_metrics(y_true, y_pred, threshold=threshold)

    boot_results: dict[str, list[float]] = {"rmse": [], "mae": [], "auroc": [], "f1": []}

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        sample_true = y_true[boot_idx]
        sample_pred = y_pred[boot_idx]

        metrics = compute_metrics(sample_true, sample_pred, threshold=threshold)
        for k in boot_results:
            boot_results[k].append(metrics[k])

    results: dict[str, dict[str, float]] = {}
    for metric_name, point_val in point_estimates.items():
        arr = np.array(boot_results[metric_name])
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        results[metric_name] = {
            "point": float(point_val),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return results
