"""Single-execution lock harness for official held-out test partitions."""

import hashlib
import json
from pathlib import Path
from typing import Any

from anjaney.utils.reproducibility import get_git_hash


class OfficialTestLock:
    def __init__(self, lock_file: Path | str = "experiments/test_eval.lock"):
        self.lock_path = Path(lock_file)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def compute_config_hash(self, config_dict: dict[str, Any]) -> str:
        serialized = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def check_and_acquire(self, config_dict: dict[str, Any]) -> str:
        c_hash = self.compute_config_hash(config_dict)

        if self.lock_path.exists():
            with open(self.lock_path, encoding="utf-8") as f:
                locked_records = json.load(f)
            if c_hash in locked_records:
                raise RuntimeError(
                    f"AGENTS.md Rule 4 Violation: Official test evaluation has already been run for config hash {c_hash[:12]}. Repeated runs on test set are strictly disallowed."
                )
        return c_hash

    def release_and_record(self, config_hash: str, results_summary: dict[str, Any]) -> None:
        records: dict[str, Any] = {}
        if self.lock_path.exists():
            with open(self.lock_path, encoding="utf-8") as f:
                records = json.load(f)

        records[config_hash] = {
            "git_hash": get_git_hash(),
            "results": results_summary,
        }

        with open(self.lock_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
