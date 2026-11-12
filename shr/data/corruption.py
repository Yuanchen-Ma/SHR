from __future__ import annotations

import numpy as np

from shr.schemas import CorruptionManifest

MISSING_SPECS = {
    "low": (0.10, 0.10),
    "medium": (0.20, 0.30),
    "high": (0.30, 0.50),
}
DROPOUT_CHANNELS = {"low": 1, "medium": 3, "high": 5}
BIAS_AMPLITUDES = {"low": 0.5, "medium": 1.0, "high": 2.0}


def apply_corruption(
    sensor_values: np.ndarray,
    observed_mask: np.ndarray,
    corruption_type: str,
    severity: str,
    unit_id: int,
    cycles: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, CorruptionManifest]:
    if sensor_values.shape != observed_mask.shape:
        raise ValueError("sensor_values and observed_mask must have identical shapes")
    values = np.array(sensor_values, dtype=float, copy=True)
    mask = np.array(observed_mask, dtype=bool, copy=True)
    rng = np.random.default_rng(seed)
    n_time, n_sensors = values.shape
    sensors: np.ndarray = np.array([], dtype=int)
    onset_index: int | None = None
    duration: int | None = None
    amplitude: float | None = None

    if corruption_type == "clean":
        severity = "none"
    elif corruption_type == "burst_missing":
        fraction, duration_fraction = MISSING_SPECS[severity]
        n_bad = max(1, round(fraction * n_sensors))
        sensors = np.sort(rng.choice(n_sensors, n_bad, replace=False))
        duration = max(1, round(duration_fraction * n_time))
        onset_index = int(rng.integers(0, max(1, n_time - duration + 1)))
        mask[onset_index : onset_index + duration, sensors] = False
    elif corruption_type == "channel_dropout":
        n_bad = min(n_sensors, DROPOUT_CHANNELS[severity])
        sensors = np.sort(rng.choice(n_sensors, n_bad, replace=False))
        low = int(np.floor(0.30 * max(n_time - 1, 1)))
        high = max(low + 1, int(np.ceil(0.70 * max(n_time - 1, 1))) + 1)
        onset_index = min(n_time - 1, int(rng.integers(low, high)))
        duration = n_time - onset_index
        mask[onset_index:, sensors] = False
    elif corruption_type == "gradual_bias":
        amplitude = BIAS_AMPLITUDES[severity]
        n_bad = max(1, round(MISSING_SPECS[severity][0] * n_sensors))
        sensors = np.sort(rng.choice(n_sensors, n_bad, replace=False))
        low = int(np.floor(0.30 * max(n_time - 1, 1)))
        high = max(low + 1, int(np.ceil(0.70 * max(n_time - 1, 1))) + 1)
        onset_index = min(n_time - 1, int(rng.integers(low, high)))
        duration = n_time - onset_index
        ramp = np.linspace(0.0, amplitude, duration, dtype=float)[:, None]
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(sensors))[None, :]
        values[onset_index:, sensors] += ramp * signs
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    onset_cycle = None if onset_index is None else int(cycles[onset_index])
    corruption_id = f"u{unit_id}_{corruption_type}_{severity}_s{seed}"
    manifest = CorruptionManifest(
        unit_id=unit_id,
        corruption_type=corruption_type,
        severity=severity,
        sensor_indices=tuple(int(value) for value in sensors),
        onset_cycle=onset_cycle,
        duration=duration,
        amplitude_std=amplitude,
        seed=seed,
        corruption_id=corruption_id,
    )
    return values, mask, manifest


def causal_fill(values: np.ndarray, observed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.shape != observed_mask.shape:
        raise ValueError("values and observed_mask must have identical shapes")
    filled = np.zeros_like(values, dtype=float)
    delta_t = np.zeros_like(values, dtype=float)
    last = np.zeros(values.shape[1], dtype=float)
    gaps = np.zeros(values.shape[1], dtype=float)
    seen = np.zeros(values.shape[1], dtype=bool)
    for time_index in range(values.shape[0]):
        observed = observed_mask[time_index]
        last[observed] = values[time_index, observed]
        seen[observed] = True
        gaps[observed] = 0.0
        gaps[~observed] += 1.0
        filled[time_index] = np.where(seen, last, 0.0)
        delta_t[time_index] = gaps
    return filled, delta_t
