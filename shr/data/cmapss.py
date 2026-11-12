from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CMAPSS_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{index}" for index in range(1, 4)]
    + [f"sensor_{index}" for index in range(1, 22)]
)
SETTING_COLUMNS = [f"setting_{index}" for index in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]


@dataclass(frozen=True)
class CmapssDatasetPaths:
    train: Path
    test: Path
    rul: Path

    def as_list(self) -> list[Path]:
        return [self.train, self.test, self.rul]


def resolve_cmapss_paths(raw_data_dir: str | Path, dataset_id: str) -> CmapssDatasetPaths:
    """Resolve nested, flat, and dataset-specific C-MAPSS roots."""

    root = Path(raw_data_dir).expanduser()
    normalized_id = dataset_id.upper()
    searched: list[Path] = []
    for directory in (root / normalized_id, root):
        paths = CmapssDatasetPaths(
            train=directory / f"train_{normalized_id}.txt",
            test=directory / f"test_{normalized_id}.txt",
            rul=directory / f"RUL_{normalized_id}.txt",
        )
        searched.extend(paths.as_list())
        if all(path.is_file() for path in paths.as_list()):
            return paths
    locations = "\n".join(f"- {path}" for path in searched)
    raise FileNotFoundError(
        f"Could not resolve all files for {normalized_id}. Searched:\n{locations}"
    )


def read_cmapss_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing C-MAPSS file: {table_path}. Place NASA files under the configured "
            "raw_data_dir before running the scientific pilot."
        )
    frame = pd.read_csv(table_path, sep=r"\s+", header=None)
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(f"Expected {len(CMAPSS_COLUMNS)} columns, got {frame.shape[1]}")
    frame.columns = CMAPSS_COLUMNS
    frame["unit_id"] = frame["unit_id"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)
    frame = frame.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    validate_trajectory_table(frame)
    return frame


def validate_trajectory_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("C-MAPSS table is empty")
    if frame[["unit_id", "cycle"]].duplicated().any():
        raise ValueError("Duplicate (unit_id, cycle) rows found")
    cycle_differences = frame.groupby("unit_id", sort=False)["cycle"].diff().dropna()
    if (cycle_differences <= 0).any():
        raise ValueError("Cycles must be strictly increasing within each unit")
    if frame[CMAPSS_COLUMNS].isna().any().any():
        raise ValueError("Unexpected missing values in the raw C-MAPSS table")


def add_train_rul(frame: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    result = frame.copy()
    max_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul_raw"] = max_cycle - result["cycle"]
    result["rul"] = result["rul_raw"] if cap is None else np.minimum(result["rul_raw"], cap)
    return result


def read_test_rul(path: str | Path) -> np.ndarray:
    table = pd.read_csv(path, sep=r"\s+", header=None)
    if table.empty or table.shape[1] != 1:
        raise ValueError(f"Expected one nonempty RUL column in {path}")
    values = table.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Expected finite, nonnegative test RUL values in {path}")
    return values


def add_test_rul(
    frame: pd.DataFrame, terminal_rul: np.ndarray, cap: int | None = 125
) -> pd.DataFrame:
    units = np.sort(frame["unit_id"].unique())
    if len(units) != len(terminal_rul):
        raise ValueError(f"Test units ({len(units)}) and RUL values ({len(terminal_rul)}) differ")
    terminal_by_unit = dict(zip(units, terminal_rul, strict=True))
    result = frame.copy()
    max_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul_raw"] = result["unit_id"].map(terminal_by_unit) + max_cycle - result["cycle"]
    result["rul"] = result["rul_raw"] if cap is None else np.minimum(result["rul_raw"], cap)
    return result


def load_training_set(raw_data_dir: str | Path, dataset_id: str, cap: int | None) -> pd.DataFrame:
    paths = resolve_cmapss_paths(raw_data_dir, dataset_id)
    return add_train_rul(read_cmapss_table(paths.train), cap)


def load_official_test_set(
    raw_data_dir: str | Path, dataset_id: str, cap: int | None
) -> pd.DataFrame:
    paths = resolve_cmapss_paths(raw_data_dir, dataset_id)
    frame = read_cmapss_table(paths.test)
    terminal_rul = read_test_rul(paths.rul)
    return add_test_rul(frame, terminal_rul, cap)
