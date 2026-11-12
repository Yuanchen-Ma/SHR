from __future__ import annotations

import hashlib
import math

import numpy as np

from shr.schemas import UnitSplit


def make_unit_split(
    units: np.ndarray | list[int],
    outer_test_fraction: float,
    calibration_fraction_of_remaining: float,
    seed: int,
    split_id: str,
) -> UnitSplit:
    unique = np.array(sorted({int(unit) for unit in units}), dtype=int)
    if len(unique) < 8:
        raise ValueError("At least 8 units are required for dev/calibration/test splitting")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test = max(1, round(len(unique) * outer_test_fraction))
    remaining = shuffled[n_test:]
    n_calibration = max(1, round(len(remaining) * calibration_fraction_of_remaining))
    split = UnitSplit(
        development_units=tuple(sorted(int(value) for value in remaining[n_calibration:])),
        threshold_calibration_units=tuple(
            sorted(int(value) for value in remaining[:n_calibration])
        ),
        outer_test_units=tuple(sorted(int(value) for value in shuffled[:n_test])),
        split_id=split_id,
        seed=seed,
    )
    validate_disjoint(split)
    return split


def validate_disjoint(split: UnitSplit) -> None:
    groups = [
        set(split.development_units),
        set(split.threshold_calibration_units),
        set(split.outer_test_units),
    ]
    if any(not group for group in groups):
        raise ValueError("Development, calibration, and test unit sets must all be non-empty")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Unit split roles overlap")


def grouped_kfold(units: np.ndarray | list[int], n_splits: int, seed: int):
    unique = np.array(sorted({int(unit) for unit in units}), dtype=int)
    if n_splits < 2 or n_splits > len(unique):
        raise ValueError(f"inner_folds must be in [2, {len(unique)}]")
    shuffled = np.random.default_rng(seed).permutation(unique)
    folds = np.array_split(shuffled, n_splits)
    for fold in folds:
        validation = np.array(sorted(fold), dtype=int)
        training = np.array(sorted(np.setdiff1d(unique, validation)), dtype=int)
        yield training, validation


def group_train_validation_split(
    units: np.ndarray | list[int], validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.array(sorted({int(unit) for unit in units}), dtype=int)
    if len(unique) < 3:
        raise ValueError("At least 3 units are needed for grouped early stopping")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_validation = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    return np.sort(shuffled[n_validation:]), np.sort(shuffled[:n_validation])


def split_calibration_units(frame, job_key):
    """Split reserved engines into validation and online calibration roles."""
    units = np.array(sorted(frame.unit_id.unique()))
    if len(units) < 3:
        raise ValueError("Validation and policy calibration need at least three reserved engines")
    # The suffix is part of the published split seed. Changing it changes the experiment.
    digest = hashlib.sha256((job_key + "/validation_policy_v1").encode()).digest()
    seed = int.from_bytes(digest[:4], "little")
    units = np.random.default_rng(seed).permutation(units)
    n_validation = min(len(units) - 1, max(2, math.ceil(2 * len(units) / 3)))
    return sorted(units[:n_validation].tolist()), sorted(units[n_validation:].tolist())
