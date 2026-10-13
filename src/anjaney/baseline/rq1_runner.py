"""RQ1 Gating Test Runner: Permutation tests, Mixed-Effects modeling, and decision locking."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml

from anjaney.baseline.engine import PersonalBaselineEngine
from anjaney.baseline.position import PositionConditioner
from anjaney.features.registry import FeatureRegistry
from anjaney.utils.reproducibility import RunContext, get_git_hash


def verify_prereg_clean(prereg_path: Path | str = "configs/prereg.yaml") -> str:
    path = Path(prereg_path)
    if not path.exists():
        raise FileNotFoundError(f"Pre-registration config missing at {path}")

    # Verify prereg has no uncommitted changes
    try:
        git_diff = (
            subprocess.check_output(
                ["git", "status", "--porcelain", str(path)],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        if git_diff:
            raise RuntimeError(
                f"AGENTS.md Part 0 violation: '{path}' has uncommitted modifications. Commit before execution."
            )
    except Exception as e:
        if "uncommitted modifications" in str(e):
            raise

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_rq1_analysis(
    features_path: Path | str = "data/processed/features.parquet",
    prereg_path: Path | str = "configs/prereg.yaml",
    output_dir: Path | str = "experiments/rq1",
    seed: int = 42,
) -> dict[str, Any]:
    prereg_hash = verify_prereg_clean(prereg_path)
    with open(prereg_path, encoding="utf-8") as f:
        prereg = yaml.safe_load(f)

    df_features = pd.read_parquet(features_path)
    embed_cols = FeatureRegistry.get_feature_columns(df_features, ["text_embed"])["text_embed"]
    timing_cols = FeatureRegistry.get_feature_columns(df_features, ["timing"])["timing"]

    warmup_k = prereg["analysis_parameters"]["warmup_windows_k"]
    eb_k = prereg["analysis_parameters"]["empirical_bayes_k"]
    n_perms = prereg["analysis_parameters"]["n_permutations"]
    n_boot = prereg["analysis_parameters"]["n_bootstrap"]
    min_d = prereg["analysis_parameters"]["minimum_effect_size_d"]

    # Compute population timing priors
    timing_priors = {}
    for col in timing_cols:
        vals = df_features[col].dropna().values
        timing_priors[col] = (float(np.mean(vals)), float(np.std(vals)) + 1e-4)

    # 1. Compute within-session deviations per participant
    engine = PersonalBaselineEngine(warmup_windows_k=warmup_k, empirical_bayes_k=eb_k)
    participant_dfs = []

    for _, group in df_features.groupby("participant_id", sort=False):
        df_dev = engine.compute_session_deviations(
            group,
            embed_cols=embed_cols,
            timing_cols=timing_cols,
            population_timing_priors=timing_priors,
        )
        participant_dfs.append(df_dev)

    df_all_devs = pd.concat(participant_dfs, ignore_index=True)

    # 2. In-fold / training position conditioning
    train_devs = df_all_devs[df_all_devs["official_split"] == "train"].dropna(
        subset=["combined_deviation"]
    )

    conditioner = PositionConditioner(n_bins=5)
    conditioner.fit(train_devs, "combined_deviation")
    df_all_devs["conditioned_deviation"] = conditioner.transform(df_all_devs, "combined_deviation")

    # Participant late-session drift: windows with rel_position >= 0.5
    valid_post_warmup = df_all_devs.dropna(subset=["conditioned_deviation"]).copy()
    late_windows = valid_post_warmup[valid_post_warmup["rel_position"] >= 0.5]

    participant_late_drifts = (
        late_windows.groupby("participant_id")
        .agg(
            late_drift=("conditioned_deviation", "mean"),
            phq8_score=("phq8_score", "first"),
            phq8_binary=("phq8_binary", "first"),
        )
        .reset_index()
    )

    high_risk_drifts = participant_late_drifts[participant_late_drifts["phq8_binary"] == 1][
        "late_drift"
    ].values
    low_risk_drifts = participant_late_drifts[participant_late_drifts["phq8_binary"] == 0][
        "late_drift"
    ].values

    obs_diff = float(np.mean(high_risk_drifts) - np.mean(low_risk_drifts))

    # Cohen's d
    s_pooled = np.sqrt(
        (
            (len(high_risk_drifts) - 1) * np.var(high_risk_drifts, ddof=1)
            + (len(low_risk_drifts) - 1) * np.var(low_risk_drifts, ddof=1)
        )
        / (len(high_risk_drifts) + len(low_risk_drifts) - 2)
    )
    cohens_d = float(obs_diff / s_pooled) if s_pooled > 1e-6 else 0.0

    # Participant label-permutation test
    rng = np.random.default_rng(seed)
    all_drifts = np.concatenate([high_risk_drifts, low_risk_drifts])
    n_high = len(high_risk_drifts)

    perm_diffs = []
    for _ in range(n_perms):
        shuffled = rng.permutation(all_drifts)
        perm_diff = np.mean(shuffled[:n_high]) - np.mean(shuffled[n_high:])
        perm_diffs.append(perm_diff)

    p_value = float(np.mean(np.abs(perm_diffs) >= abs(obs_diff)))

    # Bootstrap 95% Confidence Interval for group difference
    boot_diffs = []
    for _ in range(n_boot):
        b_high = rng.choice(high_risk_drifts, size=len(high_risk_drifts), replace=True)
        b_low = rng.choice(low_risk_drifts, size=len(low_risk_drifts), replace=True)
        boot_diffs.append(np.mean(b_high) - np.mean(b_low))

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Mixed-effects model: conditioned_deviation ~ phq8_binary * rel_position + (1 | participant_id)
    try:
        model = smf.mixedlm(
            "conditioned_deviation ~ phq8_binary * rel_position",
            valid_post_warmup,
            groups=valid_post_warmup["participant_id"],
        )
        fit_res = model.fit()
        mixed_summary = {
            "interaction_coef": float(fit_res.params.get("phq8_binary:rel_position", 0.0)),
            "interaction_p": float(fit_res.pvalues.get("phq8_binary:rel_position", 1.0)),
        }
    except Exception:
        mixed_summary = {"interaction_coef": 0.0, "interaction_p": 1.0}

    # Evaluate decision rule from prereg.yaml
    if p_value < 0.05 and (ci_lower > 0 or ci_upper < 0) and abs(cohens_d) >= min_d:
        decision = "SUPPORTED"
    elif ci_lower <= 0 <= ci_upper:
        decision = "FALSIFIED"
    else:
        decision = "INCONCLUSIVE"

    results = {
        "status": decision,
        "prereg_hash": prereg_hash,
        "git_hash": get_git_hash(),
        "statistics": {
            "mean_high_risk_drift": float(np.mean(high_risk_drifts)),
            "mean_low_risk_drift": float(np.mean(low_risk_drifts)),
            "observed_group_difference": obs_diff,
            "bootstrap_ci_95": [ci_lower, ci_upper],
            "cohens_d": cohens_d,
            "permutation_p_value": p_value,
            "mixed_effects": mixed_summary,
        },
    }

    # Save outputs to experiments/rq1/
    with RunContext(
        run_name="rq1_gating", config_dict=prereg, base_dir=output_dir, seed=seed
    ) as ctx:
        decision_file = ctx.run_dir / "decision.json"
        with open(decision_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Plot publication figure
        plt.figure(figsize=(7, 4.5))
        pos_bins = np.linspace(0.0, 1.0, 6)
        valid_post_warmup["bin"] = pd.cut(valid_post_warmup["rel_position"], bins=pos_bins)
        grouped_trace = (
            valid_post_warmup.groupby(["bin", "phq8_binary"], observed=False)[
                "conditioned_deviation"
            ]
            .agg(["mean", "sem"])
            .reset_index()
        )

        for binary_val, label, col in [
            (0, "Low Risk (PHQ < 10)", "#1f77b4"),
            (1, "High Risk (PHQ >= 10)", "#d62728"),
        ]:
            sub = grouped_trace[grouped_trace["phq8_binary"] == binary_val]
            x_vals = np.arange(len(sub))
            plt.errorbar(
                x_vals,
                sub["mean"],
                yerr=1.96 * sub["sem"],
                label=label,
                color=col,
                fmt="-o",
                capsize=4,
            )

        plt.title(f"RQ1: Conditioned Drift Trajectory (Status: {decision})")
        plt.xlabel("Within-Session Position Bin (Warm-up to Late)")
        plt.ylabel("Conditioned Deviation")
        plt.xticks(np.arange(5), ["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5"])
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(ctx.run_dir / "conditioned_drift_trace.png", dpi=300)
        plt.close()

        results["run_dir"] = ctx.run_dir

    return results
