"""Reproducibility utilities: deterministic seed locking, run logging, and environment capture."""

import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python standard library, NumPy, and PyTorch (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_hash() -> str:
    """Return the current commit hash or UNCOMMITTED/UNKNOWN."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        if status:
            return f"{commit}-dirty"
        return commit
    except Exception:
        return "UNKNOWN_GIT_HASH"


class RunContext:
    """Context manager and experiment manifest creator."""

    def __init__(
        self, run_name: str, config_dict: dict, base_dir: Path | str = "experiments", seed: int = 42
    ):
        self.run_name = run_name
        self.config_dict = config_dict
        self.base_dir = Path(base_dir)
        self.seed = seed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"{timestamp}_{run_name}"

    def __enter__(self) -> "RunContext":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        set_seed(self.seed)

        # 1. Save config.json
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config_dict, f, indent=2, default=str)

        # 2. Save git_hash.txt
        with open(self.run_dir / "git_hash.txt", "w", encoding="utf-8") as f:
            f.write(get_git_hash() + "\n")

        # 3. Save seed.txt
        with open(self.run_dir / "seed.txt", "w", encoding="utf-8") as f:
            f.write(str(self.seed) + "\n")

        # 4. Save env.txt (installed packages / python details)
        try:
            env_freeze = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
            ).decode("utf-8")
        except Exception:
            env_freeze = f"Python {sys.version}\n"
        with open(self.run_dir / "env.txt", "w", encoding="utf-8") as f:
            f.write(env_freeze)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
