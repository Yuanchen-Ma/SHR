from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from shr.data.windows import QUALITY_FEATURES
from shr.schemas import WindowRecord
from shr.randomness import stable_hash

FEATURE_GROUPS = {
    "fusion_lite": ["rul_pred", "qr_width", *QUALITY_FEATURES, "latent_mahalanobis_conditioned"],
}

FORBIDDEN_RISK_COLUMNS = {
    "rul_true", "error", "late_loss", "severe_over_10", "severe_over_20",
    "critical_over_10", "rul_raw", "max_cycle", "life_fraction_true",
    "corruption_amplitude_true", "corruption_onset_true", "corruption_type_true",
}


@dataclass
class DistanceReference:
    mean: np.ndarray
    precision: np.ndarray
    training_distances: np.ndarray


class ConditionedLatentDistance:
    def fit(
        self, embeddings: np.ndarray, condition_ids: np.ndarray, min_samples: int = 20
    ) -> ConditionedLatentDistance:
        if len(embeddings) != len(condition_ids):
            raise ValueError("Embeddings and condition_ids have different lengths")
        self.global_reference = _fit_reference(embeddings)
        self.references: dict[int, DistanceReference] = {}
        for condition in np.unique(condition_ids):
            subset = embeddings[condition_ids == condition]
            self.references[int(condition)] = (
                _fit_reference(subset) if len(subset) >= min_samples else self.global_reference
            )
        return self

    def score(
        self, embeddings: np.ndarray, condition_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.empty(len(embeddings), dtype=float)
        percentiles = np.empty(len(embeddings), dtype=float)
        for index, (embedding, condition) in enumerate(zip(embeddings, condition_ids, strict=True)):
            reference = self.references.get(int(condition), self.global_reference)
            delta = embedding - reference.mean
            distance = float(np.sqrt(max(delta @ reference.precision @ delta, 0.0)))
            distances[index] = distance
            percentiles[index] = np.searchsorted(
                reference.training_distances, distance, side="right"
            ) / len(reference.training_distances)
        return distances, percentiles

    def score_global(self, embeddings: np.ndarray) -> np.ndarray:
        delta = embeddings - self.global_reference.mean
        squared = np.einsum("ni,ij,nj->n", delta, self.global_reference.precision, delta)
        return np.sqrt(np.maximum(squared, 0.0))


def build_risk_table(
    records: list[WindowRecord],
    predictions: dict[str, np.ndarray],
    distance_model: ConditionedLatentDistance,
    split_id: str,
    inner_fold_id: int,
    base_train_units: tuple[int, ...] | list[int],
    severe_delta: float,
    critical_horizon: float,
    dataset_id: str,
) -> pd.DataFrame:
    if len(records) != len(predictions["rul_pred"]):
        raise ValueError("Record and prediction counts differ")
    conditions = np.array([record.condition_id for record in records], dtype=int)
    distances, percentiles = distance_model.score(predictions["embedding"], conditions)
    global_distances = distance_model.score_global(predictions["embedding"])
    rows: list[dict[str, object]] = []
    train_hash = stable_hash(tuple(int(unit) for unit in base_train_units))
    for index, record in enumerate(records):
        prediction = float(predictions["rul_pred"][index])
        error = prediction - record.y
        row: dict[str, object] = {
            "unit_id": record.unit_id,
            "cycle": record.cycle,
            "dataset_id": dataset_id,
            "outer_split_id": split_id,
            "inner_fold_id": inner_fold_id,
            "base_train_unit_hash": train_hash,
            "corruption_id": record.corruption_id,
            "corruption_seed": record.corruption_seed,
            "corruption_onset_cycle": record.corruption_onset_cycle,
            "corruption_amplitude_std": record.corruption_amplitude_std,
            "corruption_type": record.corruption_type,
            "corruption_severity": record.corruption_severity,
            "rul_true": record.y,
            "rul_pred": prediction,
            "q_low": float(predictions["q_low"][index]),
            "q_high": float(predictions["q_high"][index]),
            "qr_width": float(predictions["q_high"][index] - predictions["q_low"][index]),
            "ensemble_std": float(predictions["ensemble_std"][index]),
            "prediction_disagreement": abs(
                prediction - 0.5 * float(predictions["q_low"][index] + predictions["q_high"][index])
            ),
            "latent_mahalanobis_conditioned": float(distances[index]),
            "latent_mahalanobis_global": float(global_distances[index]),
            "latent_distance_percentile": float(percentiles[index]),
            "condition_id": record.condition_id,
            "error": error,
            "late_loss": max(error, 0.0),
            "severe_over_10": int(error > severe_delta),
            "severe_over_20": int(error > 20.0),
            "critical_over_10": int(record.y <= critical_horizon and error > severe_delta),
            "sample_weight": record.sample_weight,
        }
        row.update(record.quality)
        rows.append(row)
    table = pd.DataFrame(rows)
    validate_risk_feature_groups()
    return table


def validate_risk_feature_groups() -> None:
    for group, columns in FEATURE_GROUPS.items():
        overlap = FORBIDDEN_RISK_COLUMNS.intersection(columns)
        if overlap:
            raise ValueError(f"Risk feature group {group} leaks labels: {sorted(overlap)}")


def assert_oof_units(table: pd.DataFrame, fold_training_units: dict[int, set[int]]) -> None:
    for fold_id, training_units in fold_training_units.items():
        predicted_units = set(table.loc[table["inner_fold_id"] == fold_id, "unit_id"])
        if predicted_units & training_units:
            raise AssertionError(f"OOF leakage in fold {fold_id}")


def _fit_reference(embeddings: np.ndarray) -> DistanceReference:
    if len(embeddings) < 2:
        raise ValueError("At least two embeddings are required for latent-distance fitting")
    estimator = LedoitWolf().fit(embeddings)
    delta = embeddings - estimator.location_
    distances = np.sqrt(
        np.maximum(np.einsum("ni,ij,nj->n", delta, estimator.precision_, delta), 0.0)
    )
    return DistanceReference(estimator.location_, estimator.precision_, np.sort(distances))
