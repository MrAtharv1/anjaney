"""Tests for RunContext artifact creation and reproducibility."""

from pathlib import Path

from anjaney.utils.reproducibility import RunContext, set_seed


def test_run_context_creates_manifest_files(tmp_path: Path):
    dummy_config = {"alpha": 0.1, "seed": 42}
    with RunContext(
        run_name="unit_test_run", config_dict=dummy_config, base_dir=tmp_path, seed=42
    ) as ctx:
        assert ctx.run_dir.exists()
        assert (ctx.run_dir / "config.json").exists()
        assert (ctx.run_dir / "git_hash.txt").exists()
        assert (ctx.run_dir / "seed.txt").exists()
        assert (ctx.run_dir / "env.txt").exists()

        with open(ctx.run_dir / "seed.txt", encoding="utf-8") as f:
            assert f.read().strip() == "42"


def test_set_seed_deterministic():
    import numpy as np

    set_seed(1234)
    val1 = np.random.rand()
    set_seed(1234)
    val2 = np.random.rand()
    assert val1 == val2
