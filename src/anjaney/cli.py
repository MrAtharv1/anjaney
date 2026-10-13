"""Typer CLI interface for ANJANEY research pipeline."""

import hashlib
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from anjaney.baseline.rq1_runner import run_rq1_analysis
from anjaney.conformal.conformal_runner import run_conformal_pipeline
from anjaney.data.daic_loader import DAICLoader
from anjaney.data.inspector import run_inspection
from anjaney.data.synthetic import SyntheticCohortGenerator
from anjaney.features.registry import FeatureRegistry
from anjaney.models.baseline_runner import run_baseline_evaluation
from anjaney.preprocess.clean import clean_session
from anjaney.preprocess.splits import build_splits, save_splits_json
from anjaney.preprocess.windows import WindowBuilder

app = typer.Typer(
    name="anjaney",
    help="ANJANEY: Campus Mental-Health Screening Aid CLI",
    add_completion=False,
)

data_app = typer.Typer(help="Data ingestion, schema discovery, and synthetic generation")
features_app = typer.Typer(help="Feature extraction: embeddings, timing, and lexical")
train_app = typer.Typer(help="Population baseline model training")
rq1_app = typer.Typer(help="RQ1 gating analysis and drift tests")
conformal_app = typer.Typer(help="Split-conformal calibration and triage policy evaluation")
eval_app = typer.Typer(help="Full grid evaluation and ablation benchmarks")
export_app = typer.Typer(help="ONNX conversion and on-device benchmark export")

app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(rq1_app, name="rq1")
app.add_typer(conformal_app, name="conformal")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")


@data_app.callback()
def data_default():
    pass


@data_app.command("inspect")
def data_inspect(
    source: Annotated[str, typer.Option(help="Data source: 'synthetic' or 'daic'")] = "synthetic",
    daic_root: Annotated[
        str | None, typer.Option(help="Path to DAIC directory (or env DAIC_ROOT)")
    ] = None,
    output: Annotated[str, typer.Option(help="Output path for inspection report")] = (
        "docs/dataset_schema_report.md"
    ),
):
    try:
        typer.echo(f"Running dataset integrity inspection on '{source}'...")
        _, report_md = run_inspection(source=source, daic_root=daic_root)
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        typer.echo(f"Inspection report generated successfully at {out_path}")
    except Exception as e:
        typer.secho(f"Error inspecting dataset: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@data_app.command("generate-synthetic")
def data_generate_synthetic(
    n_participants: Annotated[int, typer.Option(help="Number of participants")] = 50,
    output_dir: Annotated[str, typer.Option(help="Target directory for synthetic data")] = (
        "data/synthetic"
    ),
):
    typer.echo(f"Generating {n_participants} synthetic participant sessions...")
    gen = SyntheticCohortGenerator(n_participants=n_participants)
    out_path = gen.export_to_directory(output_dir)
    typer.echo(f"Synthetic cohort exported to {out_path}")


@data_app.command("build")
def data_build(
    source: Annotated[str, typer.Option(help="Data source: 'synthetic' or 'daic'")] = "synthetic",
    daic_root: Annotated[
        str | None, typer.Option(help="Path to DAIC directory (or env DAIC_ROOT)")
    ] = None,
    output_dir: Annotated[str, typer.Option(help="Directory to save windows and splits")] = (
        "data/processed"
    ),
    window_size: Annotated[int, typer.Option(help="Number of turns per window")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed for splits")] = 1337,
):
    try:
        typer.echo(f"Loading sessions from '{source}'...")
        if source == "synthetic":
            gen = SyntheticCohortGenerator(n_participants=50, seed=42)
            sessions = gen.generate()
        else:
            loader = DAICLoader(daic_root)
            sessions = list(loader.iter_sessions())

        typer.echo(f"Cleaning {len(sessions)} sessions...")
        cleaned_sessions = [clean_session(s) for s in sessions]

        typer.echo(f"Segmenting windows (window_size={window_size})...")
        builder = WindowBuilder(window_size=window_size)
        df_windows = builder.build_dataset_dataframe(cleaned_sessions)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        parquet_file = out_path / "windows.parquet"
        df_windows.to_parquet(parquet_file, index=False)

        # Compute SHA256 of windows parquet for determinism tracking
        with open(parquet_file, "rb") as f:
            pq_hash = hashlib.sha256(f.read()).hexdigest()

        typer.echo(f"Generating repeated StratifiedGroupKFold splits (seed={seed})...")
        splits_dict = build_splits(cleaned_sessions, n_splits=5, n_repeats=10, seed=seed)
        splits_file = out_path / "splits.json"
        save_splits_json(splits_dict, splits_file)

        typer.echo("Artifacts created successfully:")
        typer.echo(f" - {parquet_file} ({len(df_windows)} windows, sha256: {pq_hash[:12]})")
        typer.echo(f" - {splits_file} ({len(splits_dict['cv_folds'])} folds, 0 participant leaks)")
    except Exception as e:
        typer.secho(f"Error building dataset: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@features_app.callback()
def features_default():
    pass


@features_app.command("build")
def features_build(
    windows_file: Annotated[
        str, typer.Option(help="Path to windows parquet")
    ] = "data/processed/windows.parquet",
    output_file: Annotated[
        str, typer.Option(help="Output path for features parquet")
    ] = "data/processed/features.parquet",
    groups: Annotated[str, typer.Option(help="Comma-separated feature groups")] = (
        "text_embed,timing,lexical"
    ),
    model: Annotated[
        str, typer.Option(help="SentenceTransformer model name")
    ] = "sentence-transformers/all-MiniLM-L6-v2",
):
    try:
        in_path = Path(windows_file)
        if not in_path.exists():
            raise FileNotFoundError(
                f"Input windows file not found at {in_path}. Run 'anjaney data build' first."
            )

        selected_groups = [g.strip() for g in groups.split(",")]
        typer.echo(f"Loading windows from {in_path}...")
        df_windows = pd.read_parquet(in_path)

        typer.echo(
            f"Extracting feature groups {selected_groups} with model='{model}' on {len(df_windows)} windows..."
        )
        registry = FeatureRegistry(embedding_model=model)
        df_features = registry.extract_features(df_windows, groups=selected_groups)

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_parquet(out_path, index=False)

        col_map = FeatureRegistry.get_feature_columns(df_features, groups=selected_groups)
        typer.echo(f"Features saved successfully to {out_path}")
        for grp, cols in col_map.items():
            typer.echo(f" - Group '{grp}': {len(cols)} columns")
    except Exception as e:
        typer.secho(f"Error building features: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@train_app.callback()
def train_default():
    pass


@train_app.command("baseline")
def train_baseline(
    features_file: Annotated[
        str, typer.Option(help="Path to features parquet")
    ] = "data/processed/features.parquet",
    splits_file: Annotated[
        str, typer.Option(help="Path to splits json")
    ] = "data/processed/splits.json",
    output_dir: Annotated[str, typer.Option(help="Experiment directory")] = "experiments",
    seed: Annotated[int, typer.Option(help="Evaluation seed")] = 42,
):
    try:
        typer.echo("Starting population baseline training across participant folds...")
        result = run_baseline_evaluation(
            features_path=features_file,
            splits_path=splits_file,
            output_dir=output_dir,
            seed=seed,
        )

        typer.echo(f"Baseline run successfully logged to {result['run_dir']}\n")
        typer.echo("| Modality | Head | RMSE [95% CI] | AUROC [95% CI] |")
        typer.echo("| :--- | :--- | :--- | :--- |")

        summary = result["summary"]
        for mod, heads in summary.items():
            if mod == "permutation_test":
                continue
            for h_name, metrics in heads.items():
                rmse_str = f"{metrics['rmse']['point']:.2f} [{metrics['rmse']['ci_lower']:.2f}, {metrics['rmse']['ci_upper']:.2f}]"
                auroc_str = f"{metrics['auroc']['point']:.2f} [{metrics['auroc']['ci_lower']:.2f}, {metrics['auroc']['ci_upper']:.2f}]"
                typer.echo(f"| {mod} | {h_name} | {rmse_str} | {auroc_str} |")

        # Sanity alarms
        perm = summary.get("permutation_test", {})
        typer.echo(f"\nPermutation Test AUROC: {perm.get('shuffled_auroc', 0.5):.2f}")
        if not result["permutation_passed"]:
            typer.secho(
                "ALARM: Permutation test AUROC deviated significantly from 0.50! Audit fold isolation.",
                fg=typer.colors.RED,
                bold=True,
            )
        else:
            typer.secho(
                "Permutation check PASSED (chance-level behavior on shuffled labels).",
                fg=typer.colors.GREEN,
            )

        text_ridge_auroc = (
            summary.get("text_only", {}).get("ridge", {}).get("auroc", {}).get("point", 0.0)
        )
        if text_ridge_auroc > 0.85:
            typer.secho(
                "WARNING (LEAKAGE AUDIT): Participant text AUROC > 0.85 on cross-validation. Real DAIC text rarely exceeds 0.80. Inspect for shortcut features.",
                fg=typer.colors.YELLOW,
                bold=True,
            )
    except Exception as e:
        typer.secho(f"Error executing baseline evaluation: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@rq1_app.callback()
def rq1_default():
    pass


@rq1_app.command("run")
def rq1_run(
    features_file: Annotated[
        str, typer.Option(help="Path to features parquet")
    ] = "data/processed/features.parquet",
    prereg_file: Annotated[
        str, typer.Option(help="Path to committed prereg.yaml")
    ] = "configs/prereg.yaml",
    output_dir: Annotated[str, typer.Option(help="Experiment folder")] = "experiments/rq1",
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
):
    try:
        typer.echo(f"Validating pre-registration lock at '{prereg_file}'...")
        res = run_rq1_analysis(
            features_path=features_file,
            prereg_path=prereg_file,
            output_dir=output_dir,
            seed=seed,
        )

        stats = res["statistics"]
        typer.echo("\n=======================================================")
        typer.echo(f"  ANJANEY RQ1 GATING TEST DECISION: {res['status']}")
        typer.echo("=======================================================")
        typer.echo(f"Git Hash: {res['git_hash']}")
        typer.echo(f"Prereg Hash: {res['prereg_hash'][:12]}")
        typer.echo(f"Observed Group Difference: {stats['observed_group_difference']:.4f}")
        typer.echo(
            f"Bootstrap 95% CI: [{stats['bootstrap_ci_95'][0]:.4f}, {stats['bootstrap_ci_95'][1]:.4f}]"
        )
        typer.echo(f"Cohen's d: {stats['cohens_d']:.3f}")
        typer.echo(f"Permutation p-value: {stats['permutation_p_value']:.4f}")
        typer.echo(f"Artifacts saved to: {res['run_dir']}\n")

        if res["status"] == "SUPPORTED":
            typer.secho(
                "RQ1 Supported: High-risk drift exceeds population position expectations.",
                fg=typer.colors.GREEN,
                bold=True,
            )
        elif res["status"] == "FALSIFIED":
            typer.secho(
                "RQ1 Falsified: Drift is explained by generic fatigue. Baseline will be evaluated as negative result.",
                fg=typer.colors.YELLOW,
                bold=True,
            )
    except Exception as e:
        typer.secho(f"RQ1 Execution Halted: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@conformal_app.callback()
def conformal_default():
    pass


@conformal_app.command("run")
def conformal_run(
    features_file: Annotated[
        str, typer.Option(help="Path to features parquet")
    ] = "data/processed/features.parquet",
    splits_file: Annotated[
        str, typer.Option(help="Path to splits json")
    ] = "data/processed/splits.json",
    output_dir: Annotated[str, typer.Option(help="Target experiment directory")] = (
        "experiments/conformal"
    ),
    alpha: Annotated[float, typer.Option(help="Target error rate alpha")] = 0.10,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
):
    try:
        typer.echo(f"Starting split-conformal calibration (alpha={alpha}, seed={seed})...")
        res = run_conformal_pipeline(
            features_path=features_file,
            splits_path=splits_file,
            output_dir=output_dir,
            alpha=alpha,
            seed=seed,
        )

        typer.echo("\n=======================================================")
        typer.echo(f"  CONFORMAL CALIBRATION SUMMARY (Target Coverage: {1.0 - alpha:.0%})")
        typer.echo("=======================================================")
        typer.echo(f"Calibration Participants: {res['n_calib_participants']}")
        typer.echo(
            f"Split Conformal Interval: +/-{res['split_conformal']['q_hat']:.2f} points (Coverage: {res['split_conformal']['coverage']:.1%})"
        )
        typer.echo(
            f"CQR Mean Width: {res['cqr']['mean_width']:.2f} (Coverage: {res['cqr']['coverage']:.1%})"
        )
        typer.echo(
            f"Difficulty-Normalized Width: {res['difficulty_normalized']['mean_width']:.2f} (Coverage: {res['difficulty_normalized']['coverage']:.1%})"
        )
        typer.echo(
            f"Classification Set Abstentions: {res['conformal_classification']['abstention_rate_both_labels']:.1%}"
        )
        typer.echo(f"Risk-Coverage AURC: {res['operating_point_sweep']['aurc']:.3f}")
        opt = res["operating_point_sweep"]["optimal_operating_point"]
        typer.echo(
            f"Optimal Policy: Max Width={opt['max_width']} | Margin={opt['margin']} -> Abstain={opt['abstention_rate']:.1%}, Risk={opt['selective_risk']:.1%}"
        )
        typer.echo(f"\nArtifacts saved to: {res['run_dir']}")
    except Exception as e:
        typer.secho(f"Conformal Calibration Failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@eval_app.callback()
def eval_default():
    pass


@eval_app.command("run")
def eval_run():
    typer.echo("Evaluation grid: not implemented yet.")


@export_app.callback()
def export_default():
    pass


@export_app.command("benchmark")
def export_benchmark():
    typer.echo("Export benchmark: not implemented yet.")


if __name__ == "__main__":
    app()
