"""Typer CLI interface for ANJANEY research pipeline."""

import typer

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
def data_inspect():
    typer.echo("Data inspect command: not implemented yet.")


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
