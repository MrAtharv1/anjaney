"""Typer CLI interface for ANJANEY research pipeline."""

from pathlib import Path
from typing import Annotated

import typer

from anjaney.data.inspector import run_inspection
from anjaney.data.synthetic import SyntheticCohortGenerator

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


@features_app.callback()
def features_default():
    pass


@features_app.command("build")
def features_build():
    typer.echo("Features build command: not implemented yet.")


@train_app.callback()
def train_default():
    pass


@train_app.command("baseline")
def train_baseline():
    typer.echo("Train baseline command: not implemented yet.")


@rq1_app.callback()
def rq1_default():
    pass


@rq1_app.command("run")
def rq1_run():
    typer.echo("RQ1 gating test: not implemented yet.")


@conformal_app.callback()
def conformal_default():
    pass


@conformal_app.command("run")
def conformal_run():
    typer.echo("Conformal evaluation: not implemented yet.")


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
