"""RQ1 Gating Test Runner: Provenance locking, participant cross-fitting, mixed-model fallback, and length-confound downgrade rule."""

import hashlib
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy import stats as sp_stats

from anjaney.baseline.engine import PersonalBaselineEngine
from anjaney.baseline.position import PositionConditioner
from anjaney.features.registry import FeatureRegistry
from anjaney.utils.reproducibility import RunContext, get_git_hash


def verify_prereg_clean(prereg_path: Path | str = "configs/prereg.yaml") -> dict[str, Any]:
    """Fail-closed check: raises if git fails, if prereg.yaml is untracked, or if uncommitted modifications exist."""
    path = Path(prereg_path)
    if not path.exists():
        raise FileNotFoundError(f"Pre-registration config missing at {path}")

    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            check=True,
            capture_output=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"AGENTS.md Part 0 violation: '{path}' is untracked or git failed. Commit prereg.yaml before running."
        ) from e

    try:
        status_proc = subprocess.run(
            ["git", "status", "--porcelain", str(path)],
            check=True,
            capture_output=True,
        )
        git_diff = status_proc.stdout.decode("utf-8").strip()
    except Exception as e:
        raise RuntimeError(f"Git status check failed on '{path}': {e}") from e

    if git_diff:
        raise RuntimeError(
            f"AGENTS.md Part 0 violation: '{path}' has uncommitted modifications:\n{git_diff}"
        )

    try:
        commit_proc = subprocess.run(
            ["git", "log", "-n", "1", "--pretty=format:%H", "--", str(path)],
            check=True,
            capture_output=True,
        )
        prereg_commit = commit_proc.stdout.decode("utf-8").strip()
    except Exception as e:
        raise RuntimeError(f"Could not read git commit for '{path}': {e}") from e

    if not prereg_commit:
        raise RuntimeError(f"AGENTS.md Part 0 violation: '{path}' has no commit history.")

    full_git_hash = get_git_hash()
    with open(path, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()

    return {
        "prereg_sha256": file_sha256,
        "prereg_commit": prereg_commit,
        "repo_git_hash": full_git_hash,
        "code_dirty": full_git_hash.endswith("-dirty"),
    }


def read_latest_real_rq1_decision(base_dir: Path | str = "experiments/rq1") -> dict[str, Any]:
    """Read ONLY the explicit LATEST_REAL pointer (never synthetic runs) for downstream RQ3 gating."""
    pointer_file = Path(base_dir) / "LATEST_REAL"
    if not pointer_file.exists():
        raise FileNotFoundError(
            f"No real-data RQ1 gate found at '{pointer_file}'. "
            "Synthetic runs cannot unlock real-data RQ3 claims."
        )
    decision_path = Path(pointer_file.read_text(encoding="utf-8").strip())
    if not decision_path.exists():
        raise FileNotFoundError(f"LATEST_REAL pointer target missing: {decision_path}")
    with open(decision_path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("data_source") != "daic":
        raise RuntimeError("Invalid LATEST_REAL target: data_source is not 'daic'.")
    return data


def _compute_cohens_d(high_vals: np.ndarray, low_vals: np.ndarray) -> float:
    n_h, n_l = len(high_vals), len(low_vals)
    if n_h < 2 or n_l < 2:
        return 0.0
    var_h = np.var(high_vals, ddof=1)
    var_l = np.var(low_vals, ddof=1)
    s_pool = np.sqrt(((n_h - 1) * var_h + (n_l - 1) * var_l) / (n_h + n_l - 2))
    if s_pool <= 1e-8:
        return 0.0
    return float((np.mean(high_vals) - np.mean(low_vals)) / s_pool)


def _compute_mde_cohens_d(
    n_high: int, n_low: int, alpha_one_sided: float = 0.05, power: float = 0.80
) -> float:
    z_alpha = float(sp_stats.norm.ppf(1.0 - alpha_one_sided))
    z_beta = float(sp_stats.norm.ppf(power))
    se_factor = np.sqrt((1.0 / max(1, n_high)) + (1.0 / max(1, n_low)))
    return float((z_alpha + z_beta) * se_factor)


def _fit_mixed_effects_with_fallback(valid_post_warmup: pd.DataFrame) -> dict[str, Any]:
    """Try random intercept + slope first; fallback to random intercept + participant cluster-robust OLS on boundary/non-convergence."""
    formula = "conditioned_deviation ~ phq8_binary * rel_position + n_tokens"
    inter_key = "phq8_binary:rel_position"

    # Attempt 1: Random intercept + random slope on rel_position
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                formula,
                valid_post_warmup,
                groups=valid_post_warmup["participant_id"],
                re_formula="~rel_position",
            )
            fit_res = model.fit(disp=False)
            had_boundary_warn = any(
                "boundary" in str(w.message).lower() or "failed" in str(w.message).lower()
                for w in caught
            )

        converged = bool(getattr(fit_res, "converged", False)) and not had_boundary_warn
        if converged and np.all(np.isfinite(fit_res.bse)):
            conf_int = fit_res.conf_int()
            return {
                "converged": True,
                "model_used": "mixedlm_random_intercept_and_slope",
                "interaction_coef": float(fit_res.params.get(inter_key, np.nan)),
                "interaction_p": float(fit_res.pvalues.get(inter_key, np.nan)),
                "interaction_ci_95": [
                    float(conf_int.loc[inter_key, 0]),
                    float(conf_int.loc[inter_key, 1]),
                ],
                "n_tokens_coef": float(fit_res.params.get("n_tokens", np.nan)),
                "n_tokens_p": float(fit_res.pvalues.get("n_tokens", np.nan)),
            }
    except Exception:
        pass

    # Fallback: Random-intercept MixedLM or Cluster-Robust SEs by participant_id
    try:
        ols_mod = smf.ols(formula, data=valid_post_warmup).fit(
            cov_type="cluster",
            cov_kwds={"groups": valid_post_warmup["participant_id"]},
        )
        conf_int = ols_mod.conf_int()
        return {
            "converged": True,
            "model_used": "fallback_cluster_robust_se_by_participant",
            "interaction_coef": float(ols_mod.params.get(inter_key, np.nan)),
            "interaction_p": float(ols_mod.pvalues.get(inter_key, np.nan)),
            "interaction_ci_95": [
                float(conf_int.loc[inter_key, 0]),
                float(conf_int.loc[inter_key, 1]),
            ],
            "n_tokens_coef": float(ols_mod.params.get("n_tokens", np.nan)),
            "n_tokens_p": float(ols_mod.pvalues.get("n_tokens", np.nan)),
        }
    except Exception as exc:
        return {
            "converged": False,
            "model_used": "failed",
            "error": str(exc),
            "interaction_coef": None,
            "interaction_p": None,
        }


def _evaluate_for_warmup_k(
    df_features: pd.DataFrame,
    embed_cols: list[str],
    timing_cols: list[str],
    params: dict[str, Any],
    warmup_k: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    eb_k = float(params["empirical_bayes_k"])
    pca_comp = int(params.get("pca_components_maha", 16))
    w_text = float(params.get("fusion_weight_text", 0.50))
    w_timing = float(params.get("fusion_weight_timing", 0.50))
    n_bins = int(params.get("position_bins", 5))
    n_turn_bins = int(params.get("turn_index_bins", 5))
    min_bin_count = int(params.get("min_bin_count_n", 5))
    cf_folds = int(params.get("conditioner_cross_fit_folds", 5))
    require_phase = bool(params.get("require_prompt_phase", True))
    late_cutoff = float(params.get("late_session_cutoff", 0.5))
    ewma_alpha = float(params.get("ewma_alpha", 0.3))
    cusum_slack = float(params.get("cusum_slack_k", 0.5))
    winsorize_cap = float(params.get("z_score_winsorize_cap", 5.0))

    df_train_only = df_features[df_features["official_split"] == "train"]
    engine = PersonalBaselineEngine(
        warmup_windows_k=warmup_k,
        empirical_bayes_k=eb_k,
        pca_components=pca_comp,
        fusion_weight_text=w_text,
        fusion_weight_timing=w_timing,
        ewma_alpha=ewma_alpha,
        cusum_slack=cusum_slack,
        winsorize_cap=winsorize_cap,
    )
    engine.fit_training_priors(df_train_only, embed_cols=embed_cols, timing_cols=timing_cols)

    participant_dfs = []
    excluded_high_pids: list[str] = []
    excluded_low_pids: list[str] = []

    for pid, group in df_features.groupby("participant_id", sort=False):
        is_high = int(group["phq8_binary"].iloc[0]) == 1
        if len(group) <= warmup_k:
            if is_high:
                excluded_high_pids.append(str(pid))
            else:
                excluded_low_pids.append(str(pid))
        df_dev = engine.compute_raw_session_deviations(
            group, embed_cols=embed_cols, timing_cols=timing_cols
        )
        participant_dfs.append(df_dev)

    df_all_devs = pd.concat(participant_dfs, ignore_index=True)

    train_raw_devs = df_all_devs[df_all_devs["official_split"] == "train"]
    engine.fit_deviation_standardizer(train_raw_devs)
    df_all_devs = engine.apply_standardized_fusion(
        df_all_devs,
        has_text_cols=len(embed_cols) > 0,
        has_timing_cols=len(timing_cols) > 0,
    )

    # Cross-fit PositionConditioner by participant across train+dev so all participants are out-of-fold
    conditioner = PositionConditioner(
        n_bins=n_bins,
        n_turn_bins=n_turn_bins,
        min_bin_count=min_bin_count,
        require_prompt_phase=require_phase,
    )
    post_mask = df_all_devs["combined_deviation"].notna()
    df_post = df_all_devs[post_mask].copy()
    df_post["conditioned_deviation"] = conditioner.cross_fit_transform(
        df_post, "combined_deviation", n_folds=cf_folds
    )
    df_all_devs["conditioned_deviation"] = np.nan
    df_all_devs.loc[post_mask, "conditioned_deviation"] = df_post[
        "conditioned_deviation"
    ].to_numpy()

    df_all_devs = engine.compute_conditioned_persistence(
        df_all_devs, cond_col="conditioned_deviation"
    )

    valid_post_warmup = df_all_devs.dropna(subset=["conditioned_deviation"]).copy()
    late_windows = valid_post_warmup[valid_post_warmup["rel_position"] >= late_cutoff]

    participant_late = (
        late_windows.groupby("participant_id")
        .agg(
            late_drift=("conditioned_deviation", "mean"),
            mean_tokens=("n_tokens", "mean"),
            phq8_score=("phq8_score", "first"),
            phq8_binary=("phq8_binary", "first"),
        )
        .reset_index()
    )

    meta = {
        "conditioning_mode": conditioner.conditioning_mode,
        "excluded_high_pids": excluded_high_pids,
        "excluded_low_pids": excluded_low_pids,
        "missing_modality_window_drops": engine.missing_modality_window_drops,
        "participant_late": participant_late,
    }
    return meta, valid_post_warmup


def run_rq1_analysis(
    features_path: Path | str = "data/processed/features.parquet",
    prereg_path: Path | str = "configs/prereg.yaml",
    output_dir: Path | str = "experiments/rq1",
    seed: int = 42,
) -> dict[str, Any]:
    prereg_meta = verify_prereg_clean(prereg_path)
    with open(prereg_path, encoding="utf-8") as f:
        prereg = yaml.safe_load(f)

    params = prereg["analysis_parameters"]
    warmup_k = int(params["warmup_windows_k"])
    n_bins = int(params.get("position_bins", 5))
    n_perms = int(params["n_permutations"])
    n_boot = int(params["n_bootstrap"])
    fig_boot_n = int(params.get("figure_bootstrap_n", 500))
    min_d = float(params["minimum_effect_size_d"])
    equiv_d = float(params.get("equivalence_bound_d", 0.30))

    df_raw = pd.read_parquet(features_path)

    # Detect data_source ("synthetic" vs "daic") from features dataframe or participant IDs
    if "is_synthetic" in df_raw.columns and bool(df_raw["is_synthetic"].any()):
        data_source = "synthetic"
    elif df_raw["participant_id"].astype(str).str.startswith("SYNTH_").any():
        data_source = "synthetic"
    else:
        data_source = "daic"

    # STRICT ISOLATION: Restrict RQ1 strictly to train + dev
    test_count_excluded = int(
        df_raw[df_raw["official_split"] == "test"]["participant_id"].nunique()
    )
    df_features = df_raw[df_raw["official_split"].isin(["train", "dev"])].copy()
    if df_features.empty:
        raise ValueError("No train/dev participants found in features DataFrame.")

    # Missing-Ellie sessions (451, 458, 480): exclude cleanly from conditioned RQ1 run
    if "has_interviewer_turns" in df_raw.columns:
        no_ellie_pids = sorted(
            df_raw[~df_raw["has_interviewer_turns"]]["participant_id"].astype(str).unique().tolist()
        )
    else:
        no_ellie_pids = [
            str(p)
            for p in ["451", "458", "480"]
            if str(p) in df_raw["participant_id"].astype(str).values
        ]
    for daic_id in ["451", "458", "480"]:
        if daic_id in df_raw["participant_id"].astype(str).values and daic_id not in no_ellie_pids:
            no_ellie_pids.append(daic_id)
    no_ellie_pids = sorted(list(set(no_ellie_pids)))

    df_features = df_features[~df_features["participant_id"].astype(str).isin(no_ellie_pids)].copy()

    embed_cols = FeatureRegistry.get_feature_columns(df_features, ["text_embed"]).get(
        "text_embed", []
    )
    timing_cols = FeatureRegistry.get_feature_columns(df_features, ["timing"]).get("timing", [])

    primary_meta, valid_post_warmup = _evaluate_for_warmup_k(
        df_features, embed_cols, timing_cols, params, warmup_k=warmup_k, seed=seed
    )
    participant_late = primary_meta["participant_late"]

    high_risk_drifts = participant_late[participant_late["phq8_binary"] == 1][
        "late_drift"
    ].to_numpy(dtype=float)
    low_risk_drifts = participant_late[participant_late["phq8_binary"] == 0]["late_drift"].to_numpy(
        dtype=float
    )

    n_high = len(high_risk_drifts)
    n_low = len(low_risk_drifts)
    if n_high < 2 or n_low < 2:
        raise ValueError("Insufficient participants per risk group to run RQ1 comparison.")

    obs_diff = float(np.mean(high_risk_drifts) - np.mean(low_risk_drifts))
    cohens_d = _compute_cohens_d(high_risk_drifts, low_risk_drifts)
    mde_d_80 = _compute_mde_cohens_d(n_high, n_low, alpha_one_sided=0.05, power=0.80)

    # 1. Permutation test (one-sided with Phipson-Smyth (count+1)/(M+1))
    rng = np.random.default_rng(seed)
    all_drifts = np.concatenate([high_risk_drifts, low_risk_drifts])
    exceed_count = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(all_drifts)
        if (np.mean(shuffled[:n_high]) - np.mean(shuffled[n_high:])) >= obs_diff:
            exceed_count += 1
    p_value = float((exceed_count + 1) / (n_perms + 1))

    # 2. Bootstrap CIs in both raw units and Cohen's d units
    boot_raw_diffs = np.empty(n_boot, dtype=float)
    boot_d_vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        b_h = rng.choice(high_risk_drifts, size=n_high, replace=True)
        b_l = rng.choice(low_risk_drifts, size=n_low, replace=True)
        boot_raw_diffs[b] = np.mean(b_h) - np.mean(b_l)
        boot_d_vals[b] = _compute_cohens_d(b_h, b_l)

    raw_ci_95 = [
        float(np.percentile(boot_raw_diffs, 2.5)),
        float(np.percentile(boot_raw_diffs, 97.5)),
    ]
    d_ci_95 = [
        float(np.percentile(boot_d_vals, 2.5)),
        float(np.percentile(boot_d_vals, 97.5)),
    ]
    d_ci_90_tost = [
        float(np.percentile(boot_d_vals, 5.0)),
        float(np.percentile(boot_d_vals, 95.0)),
    ]

    # 3. Participant-level length-residualized sensitivity check
    tokens_arr = participant_late["mean_tokens"].to_numpy(dtype=float)
    drifts_arr = participant_late["late_drift"].to_numpy(dtype=float)
    if np.std(tokens_arr) > 1e-6:
        slope, intercept = np.polyfit(tokens_arr, drifts_arr, 1)
        resid_drifts = drifts_arr - (slope * tokens_arr + intercept)
    else:
        resid_drifts = drifts_arr
    high_resid = resid_drifts[participant_late["phq8_binary"].to_numpy() == 1]
    low_resid = resid_drifts[participant_late["phq8_binary"].to_numpy() == 0]
    length_adj_d = _compute_cohens_d(high_resid, low_resid)

    # 4. Pre-registered Decision Rule (with length-confound downgrade guard)
    downgraded_by_length_confound = False
    if primary_meta["conditioning_mode"].startswith("UNCONDITIONED"):
        decision = "UNCONDITIONED"
    elif p_value < 0.05 and d_ci_95[0] > 0.0 and cohens_d >= min_d:
        if length_adj_d < min_d:
            decision = "INCONCLUSIVE"
            downgraded_by_length_confound = True
        else:
            decision = "SUPPORTED"
    elif d_ci_90_tost[0] >= -equiv_d and d_ci_90_tost[1] <= equiv_d:
        decision = "FALSIFIED"
    else:
        decision = "INCONCLUSIVE"

    # 5. Mixed-Effects Model with automatic fallback to participant cluster-robust SEs
    mixed_summary = _fit_mixed_effects_with_fallback(valid_post_warmup)

    # 6. Exploratory Warm-up K Sensitivity (K in [3, 5, 8])
    k_grid = params.get("warmup_k_sensitivity_grid", [3, 5, 8])
    k_sensitivity: dict[str, Any] = {}
    for k_cand in k_grid:
        eligible_train = sum(
            len(g) > int(k_cand)
            for _, g in df_features[df_features["official_split"] == "train"].groupby(
                "participant_id"
            )
        )
        if eligible_train < 2:
            k_sensitivity[f"K_{k_cand}"] = {
                "status": "INSUFFICIENT_DATA",
                "eligible_train_participants": int(eligible_train),
                "label": "PRIMARY" if int(k_cand) == warmup_k else "EXPLORATORY",
            }
            continue
        try:
            k_meta, _ = _evaluate_for_warmup_k(
                df_features,
                embed_cols,
                timing_cols,
                params,
                warmup_k=int(k_cand),
                seed=seed,
            )
            p_df = k_meta["participant_late"]
            h_vals = p_df[p_df["phq8_binary"] == 1]["late_drift"].to_numpy(dtype=float)
            l_vals = p_df[p_df["phq8_binary"] == 0]["late_drift"].to_numpy(dtype=float)
            k_sensitivity[f"K_{k_cand}"] = {
                "status": "OK",
                "cohens_d": _compute_cohens_d(h_vals, l_vals)
                if len(h_vals) >= 2 and len(l_vals) >= 2
                else None,
                "n_high_retained": int(len(h_vals)),
                "n_low_retained": int(len(l_vals)),
                "label": "PRIMARY" if int(k_cand) == warmup_k else "EXPLORATORY",
            }
        except Exception as k_exc:
            k_sensitivity[f"K_{k_cand}"] = {
                "status": "INSUFFICIENT_DATA",
                "detail": str(k_exc),
                "label": "EXPLORATORY",
            }

    results: dict[str, Any] = {
        "status": decision,
        "data_source": data_source,
        "is_real_data_gate": data_source == "daic",
        "downgraded_by_length_confound": downgraded_by_length_confound,
        "conditioning_mode": primary_meta["conditioning_mode"],
        "prereg_hash": prereg_meta["prereg_sha256"],
        "prereg_commit": prereg_meta["prereg_commit"],
        "git_hash": prereg_meta["repo_git_hash"],
        "code_dirty": prereg_meta["code_dirty"],
        "sample_counts": {
            "n_high_risk_analyzed": n_high,
            "n_low_risk_analyzed": n_low,
            "n_excluded_no_interviewer_turns": len(no_ellie_pids),
            "excluded_no_interviewer_pids": no_ellie_pids,
            "n_excluded_short_high_risk": len(primary_meta["excluded_high_pids"]),
            "n_excluded_short_low_risk": len(primary_meta["excluded_low_pids"]),
            "excluded_high_risk_pids": primary_meta["excluded_high_pids"],
            "excluded_low_risk_pids": primary_meta["excluded_low_pids"],
            "n_missing_modality_windows_dropped": primary_meta["missing_modality_window_drops"],
            "n_test_split_held_out": test_count_excluded,
        },
        "statistics": {
            "mean_high_risk_drift": float(np.mean(high_risk_drifts)),
            "mean_low_risk_drift": float(np.mean(low_risk_drifts)),
            "observed_group_difference": obs_diff,
            "bootstrap_ci_95": raw_ci_95,
            "raw_diff_bootstrap_ci_95": raw_ci_95,
            "cohens_d": cohens_d,
            "cohens_d_bootstrap_ci_95": d_ci_95,
            "cohens_d_tost_ci_90": d_ci_90_tost,
            "equivalence_bounds_d": [-equiv_d, equiv_d],
            "minimum_detectable_effect_d_80pct_power": mde_d_80,
            "permutation_p_value": p_value,
            "length_residualized_sensitivity_cohens_d": length_adj_d,
            "mixed_effects": mixed_summary,
            "exploratory_warmup_k_sensitivity": k_sensitivity,
        },
    }

    with RunContext(
        run_name="rq1_gating", config_dict=prereg, base_dir=output_dir, seed=seed
    ) as ctx:
        decision_file = ctx.run_dir / "decision.json"
        with open(decision_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Write LATEST_REAL pointer ONLY when data_source == "daic"
        if data_source == "daic":
            latest_real_ptr = Path(output_dir) / "LATEST_REAL"
            latest_real_ptr.write_text(str(decision_file.resolve()), encoding="utf-8")

        # Participant-level cluster bootstrap for figure error bars
        plt.figure(figsize=(7, 4.5))
        pos_bins = np.linspace(0.0, 1.0, n_bins + 1)
        valid_post_warmup["bin"] = pd.cut(
            valid_post_warmup["rel_position"],
            bins=pos_bins,
            labels=False,
            include_lowest=True,
        )

        part_bin_df = (
            valid_post_warmup.groupby(["participant_id", "phq8_binary", "bin"], observed=False)[
                "conditioned_deviation"
            ]
            .mean()
            .reset_index()
        )

        for binary_val, label, col in [
            (0, "Low Risk (PHQ < 10)", "#1f77b4"),
            (1, "High Risk (PHQ >= 10)", "#d62728"),
        ]:
            sub_group = part_bin_df[part_bin_df["phq8_binary"] == binary_val]
            pids = sub_group["participant_id"].unique()
            bin_means = []
            bin_ci_half = []

            for b_idx in range(n_bins):
                b_vals = (
                    sub_group[sub_group["bin"] == b_idx]["conditioned_deviation"]
                    .dropna()
                    .to_numpy()
                )
                if len(b_vals) == 0:
                    bin_means.append(0.0)
                    bin_ci_half.append(0.0)
                    continue
                bin_means.append(float(np.mean(b_vals)))
                if len(b_vals) > 1 and len(pids) > 1:
                    boot_means = [
                        float(np.mean(rng.choice(b_vals, size=len(b_vals), replace=True)))
                        for _ in range(fig_boot_n)
                    ]
                    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
                    bin_ci_half.append(float((ci_hi - ci_lo) / 2.0))
                else:
                    bin_ci_half.append(0.0)

            plt.errorbar(
                np.arange(n_bins),
                bin_means,
                yerr=bin_ci_half,
                label=label,
                color=col,
                fmt="-o",
                capsize=4,
            )

        plt.axhline(0.0, color="gray", linestyle=":", alpha=0.8)
        plt.title(f"RQ1 Conditioned Deviation ({data_source.upper()}: {decision})")
        plt.xlabel("Within-Session Relative Position Bin")
        plt.ylabel("Conditioned Standardized Deviation")
        plt.xticks(np.arange(n_bins), [f"Bin {i + 1}" for i in range(n_bins)])
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(ctx.run_dir / "conditioned_drift_trace.png", dpi=300)
        plt.close()

        results["run_dir"] = str(ctx.run_dir)

    return results
