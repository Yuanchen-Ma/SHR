from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from shr.data.cmapss import SETTING_COLUMNS
from shr.data.corruption import apply_corruption, causal_fill
from shr.schemas import PreparedTrajectory, WindowRecord
from shr.randomness import derive_seed

QUALITY_FEATURES = [
    "missing_fraction_all",
    "missing_fraction_recent",
    "n_fully_missing_channels",
    "n_active_channels_last",
    "max_gap",
    "mean_gap",
    "mean_time_since_observed",
    "max_time_since_observed",
    "mask_transition_count",
    "flat_channel_count",
    "drift_score_mean",
    "drift_score_max",
]


def prepare_trajectories(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    corruption_type: str,
    severity: str,
    seed: int,
) -> tuple[list[PreparedTrajectory], list[dict[str, object]]]:
    trajectories: list[PreparedTrajectory] = []
    manifests: list[dict[str, object]] = []
    for unit_id, unit in frame.groupby("unit_id", sort=True):
        values = unit[sensor_columns].to_numpy(dtype=float)
        mask = np.ones_like(values, dtype=bool)
        cycles = unit["cycle"].to_numpy(dtype=int)
        unit_seed = derive_seed(seed, int(unit_id), corruption_type, severity)
        corrupted, corrupted_mask, manifest = apply_corruption(
            values, mask, corruption_type, severity, int(unit_id), cycles, unit_seed
        )
        filled, delta_t = causal_fill(corrupted, corrupted_mask)
        trajectories.append(
            PreparedTrajectory(
                unit_id=int(unit_id),
                cycles=cycles,
                rul=unit["rul"].to_numpy(dtype=float),
                sensors=filled,
                settings=unit[SETTING_COLUMNS].to_numpy(dtype=float),
                observed_mask=corrupted_mask,
                delta_t=delta_t,
                condition_ids=unit["condition_id"].to_numpy(dtype=int),
                manifest=manifest,
            )
        )
        manifests.append(manifest.as_dict())
    return trajectories, manifests


def build_windows(
    trajectories: list[PreparedTrajectory],
    window_length: int,
    stride: int,
    include_delta_t: bool,
    slope_mean: np.ndarray,
    slope_std: np.ndarray,
    max_windows_per_unit: int | None,
    seed: int,
) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    scenario_counts = Counter(trajectory.unit_id for trajectory in trajectories)
    for trajectory in trajectories:
        indices = list(range(0, len(trajectory.cycles), stride))
        if indices[-1] != len(trajectory.cycles) - 1:
            indices.append(len(trajectory.cycles) - 1)
        if max_windows_per_unit is not None and len(indices) > max_windows_per_unit:
            rng = np.random.default_rng(
                derive_seed(seed, trajectory.unit_id, trajectory.manifest.corruption_id)
            )
            selected = rng.choice(indices[:-1], max_windows_per_unit - 1, replace=False)
            indices = sorted([int(value) for value in selected] + [indices[-1]])
        for end_index in indices:
            start = max(0, end_index - window_length + 1)
            sensor_segment = trajectory.sensors[start : end_index + 1]
            setting_segment = trajectory.settings[start : end_index + 1]
            mask_segment = trajectory.observed_mask[start : end_index + 1]
            delta_segment = trajectory.delta_t[start : end_index + 1]
            pad = window_length - len(sensor_segment)
            sensors = _left_pad(sensor_segment, pad, 0.0)
            settings = _left_pad(setting_segment, pad, 0.0)
            mask = _left_pad(mask_segment.astype(float), pad, 0.0)
            delta = _left_pad(delta_segment, pad, float(pad))
            pieces = [sensors, settings, mask]
            if include_delta_t:
                pieces.append(delta)
            quality = compute_quality_features(sensors, mask, delta, slope_mean, slope_std)
            records.append(
                WindowRecord(
                    x=np.concatenate(pieces, axis=1).astype(np.float32),
                    y=float(trajectory.rul[end_index]),
                    unit_id=trajectory.unit_id,
                    cycle=int(trajectory.cycles[end_index]),
                    condition_id=int(trajectory.condition_ids[end_index]),
                    corruption_id=trajectory.manifest.corruption_id,
                    corruption_type=trajectory.manifest.corruption_type,
                    corruption_severity=trajectory.manifest.severity,
                    corruption_seed=trajectory.manifest.seed,
                    corruption_onset_cycle=trajectory.manifest.onset_cycle,
                    corruption_amplitude_std=trajectory.manifest.amplitude_std,
                    sample_weight=1.0 / scenario_counts[trajectory.unit_id],
                    quality=quality,
                )
            )
    _assign_engine_equal_weights(records)
    return records


def compute_quality_features(
    sensors: np.ndarray,
    observed_mask: np.ndarray,
    delta_t: np.ndarray,
    slope_mean: np.ndarray,
    slope_std: np.ndarray,
) -> dict[str, float]:
    recent = min(5, len(observed_mask))
    missing = observed_mask < 0.5
    gaps = _maximum_gaps(missing)
    recent_sensors = sensors[-min(10, len(sensors)) :]
    if len(recent_sensors) >= 2:
        time = np.arange(len(recent_sensors), dtype=float)
        centered = time - time.mean()
        slopes = centered @ recent_sensors / max(float(np.square(centered).sum()), 1.0)
    else:
        slopes = np.zeros(sensors.shape[1])
    drift = np.abs(slopes - slope_mean) / np.maximum(slope_std, 1e-6)
    transitions = np.abs(np.diff(observed_mask, axis=0)).sum() if len(observed_mask) > 1 else 0.0
    return {
        "missing_fraction_all": float(missing.mean()),
        "missing_fraction_recent": float(missing[-recent:].mean()),
        "n_fully_missing_channels": float(np.all(missing, axis=0).sum()),
        "n_active_channels_last": float(observed_mask[-1].sum()),
        "max_gap": float(gaps.max(initial=0)),
        "mean_gap": float(gaps.mean()),
        "mean_time_since_observed": float(delta_t[-1].mean()),
        "max_time_since_observed": float(delta_t[-1].max(initial=0)),
        "mask_transition_count": float(transitions),
        "flat_channel_count": float((recent_sensors.std(axis=0) < 1e-5).sum()),
        "drift_score_mean": float(np.mean(drift)),
        "drift_score_max": float(np.max(drift, initial=0)),
    }


def _maximum_gaps(missing: np.ndarray) -> np.ndarray:
    result = np.zeros(missing.shape[1], dtype=int)
    current = np.zeros(missing.shape[1], dtype=int)
    for row in missing:
        current = np.where(row, current + 1, 0)
        result = np.maximum(result, current)
    return result


def _left_pad(values: np.ndarray, pad: int, value: float) -> np.ndarray:
    if pad == 0:
        return values
    padding = np.full((pad, values.shape[1]), value, dtype=values.dtype)
    return np.concatenate([padding, values], axis=0)


def _assign_engine_equal_weights(records: list[WindowRecord]) -> None:
    counts = Counter((record.unit_id, record.corruption_id) for record in records)
    scenarios = Counter(record.unit_id for record in {r.corruption_id: r for r in records}.values())
    for record in records:
        record.sample_weight = 1.0 / (
            counts[(record.unit_id, record.corruption_id)] * scenarios[record.unit_id]
        )


def records_to_arrays(records: list[WindowRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("No windows were constructed")
    x = np.stack([record.x for record in records]).astype(np.float32)
    y = np.array([record.y for record in records], dtype=np.float32)
    weights = np.array([record.sample_weight for record in records], dtype=np.float32)
    return x, y, weights
