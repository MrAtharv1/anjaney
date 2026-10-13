"""Tests verifying CLI registration and help menus."""

from typer.testing import CliRunner

from anjaney.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ["data", "features", "train", "rq1", "conformal", "eval", "export"]:
        assert group in result.stdout
