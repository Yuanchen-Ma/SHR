from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UnitSplit:
    development_units: tuple[int, ...]
    threshold_calibration_units: tuple[int, ...]
    outer_test_units: tuple[int, ...]
    split_id: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorruptionManifest:
    unit_id: int
    corruption_type: str
    severity: str
    sensor_indices: tuple[int, ...]
    onset_cycle: int | None
    duration: int | None
    amplitude_std: float | None
    seed: int
    corruption_id: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sensor_indices"] = list(self.sensor_indices)
        return result


@dataclass
class PreparedTrajectory:
    unit_id: int
    cycles: np.ndarray
    rul: np.ndarray
    sensors: np.ndarray
    settings: np.ndarray
    observed_mask: np.ndarray
    delta_t: np.ndarray
    condition_ids: np.ndarray
    manifest: CorruptionManifest


@dataclass
class WindowRecord:
    x: np.ndarray
    y: float
    unit_id: int
    cycle: int
    condition_id: int
    corruption_id: str
    corruption_type: str
    corruption_severity: str
    corruption_seed: int
    corruption_onset_cycle: int | None
    corruption_amplitude_std: float | None
    sample_weight: float
    quality: dict[str, float]
