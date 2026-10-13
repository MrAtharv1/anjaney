"""Stratified Group K-Fold split generator with zero-leakage assertions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from anjaney.data.schemas import Session


def assert_no_participant_overlap(splits_data: dict[str, Any]) -> None:
    """Audit partition definitions to guarantee zero participant leakage across sets."""
    # 1. Audit official partitions
    official = splits_data.get("official", {})
    train_set = set(official.get("train", []))
    dev_set = set(official.get("dev", []))
    test_set = set(official.get("test", []))

    if train_set.intersection(dev_set):
        raise AssertionError(
            f"Leakage detected between official train and dev: {train_set.intersection(dev_set)}"
        )
    if train_set.intersection(test_set):
        raise AssertionError(
            f"Leakage detected between official train and test: {train_set.intersection(test_set)}"
        )
    if dev_set.intersection(test_set):
        raise AssertionError(
            f"Leakage detected between official dev and test: {dev_set.intersection(test_set)}"
        )

    # 2. Audit cross-validation folds
    cv_folds = splits_data.get("cv_folds", [])
    for fold_meta in cv_folds:
        f_train = set(fold_meta["train_pids"])
        f_val = set(fold_meta["val_pids"])
        intersection = f_train.intersection(f_val)
        if intersection:
            raise AssertionError(
                f"CV Leakage: Repeat {fold_meta['repeat']} Fold {fold_meta['fold']} shares participants: {intersection}"
            )


def build_splits(
    sessions: list[Session],
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 1337,
) -> dict[str, Any]:
    """Generate repeated StratifiedGroupKFold splits over train+dev partitions."""
    official_splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for s in sessions:
        if s.official_split in official_splits:
            official_splits[s.official_split].append(s.participant_id)

    # Cross-validation operates on official train + dev pool
    cv_sessions = [s for s in sessions if s.official_split in ["train", "dev"]]
    pids = np.array([s.participant_id for s in cv_sessions])
    labels = np.array([s.phq8_binary if s.phq8_binary is not None else 0 for s in cv_sessions])
    groups = pids.copy()

    cv_folds_list: list[dict[str, Any]] = []

    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f_idx, (train_idx, val_idx) in enumerate(sgkf.split(pids, labels, groups)):
            cv_folds_list.append(
                {
                    "repeat": r,
                    "fold": f_idx,
                    "train_pids": sorted(pids[train_idx].tolist()),
                    "val_pids": sorted(pids[val_idx].tolist()),
                }
            )

    result = {
        "metadata": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "seed": seed,
            "total_participants": len(sessions),
            "cv_participants": len(cv_sessions),
        },
        "official": {
            "train": sorted(official_splits["train"]),
            "dev": sorted(official_splits["dev"]),
            "test": sorted(official_splits["test"]),
        },
        "cv_folds": cv_folds_list,
    }

    assert_no_participant_overlap(result)
    return result


def save_splits_json(splits_data: dict[str, Any], filepath: Path | str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
