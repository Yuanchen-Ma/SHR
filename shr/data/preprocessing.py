from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from shr.data.cmapss import SENSOR_COLUMNS, SETTING_COLUMNS


@dataclass
class ConditionStatistics:
    mean: np.ndarray
    std: np.ndarray


class FoldPreprocessor:
    """Fold-local condition encoder and sensor scaler.

    The object records its training units, making leakage audits straightforward.
    """

    def __init__(
        self,
        n_condition_clusters: int,
        variance_threshold: float = 1e-8,
        min_cluster_samples: int = 100,
        epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        self.n_condition_clusters = n_condition_clusters
        self.variance_threshold = variance_threshold
        self.min_cluster_samples = min_cluster_samples
        self.epsilon = epsilon
        self.seed = seed

    def fit(self, train_frame: pd.DataFrame) -> FoldPreprocessor:
        if train_frame.empty:
            raise ValueError("Cannot fit preprocessor on an empty frame")
        variances = train_frame[SENSOR_COLUMNS].var(ddof=0)
        self.active_sensors = variances[variances >= self.variance_threshold].index.tolist()
        if not self.active_sensors:
            raise ValueError("Variance filtering removed every sensor")
        self.training_units = tuple(sorted(int(unit) for unit in train_frame["unit_id"].unique()))
        self.setting_scaler = StandardScaler().fit(train_frame[SETTING_COLUMNS])
        scaled_settings = self.setting_scaler.transform(train_frame[SETTING_COLUMNS])
        n_clusters = min(self.n_condition_clusters, len(train_frame))
        self.condition_model = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        condition_ids = self.condition_model.fit_predict(scaled_settings)
        sensor_values = train_frame[self.active_sensors].to_numpy(dtype=float)
        global_std = np.maximum(sensor_values.std(axis=0), self.epsilon)
        self.global_statistics = ConditionStatistics(sensor_values.mean(axis=0), global_std)
        self.condition_statistics: dict[int, ConditionStatistics] = {}
        for condition_id in range(n_clusters):
            rows = condition_ids == condition_id
            if int(rows.sum()) < self.min_cluster_samples:
                self.condition_statistics[condition_id] = self.global_statistics
            else:
                values = sensor_values[rows]
                self.condition_statistics[condition_id] = ConditionStatistics(
                    values.mean(axis=0), np.maximum(values.std(axis=0), self.epsilon)
                )
        transformed = self.transform(train_frame)
        self.slope_mean, self.slope_std = _fit_slope_reference(
            transformed, self.active_sensors, epsilon=self.epsilon
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        float_columns = [*self.active_sensors, *SETTING_COLUMNS]
        result = frame.astype({column: "float64" for column in float_columns}, copy=True)
        scaled_settings = self.setting_scaler.transform(result[SETTING_COLUMNS])
        condition_ids = self.condition_model.predict(scaled_settings)
        sensors = result[self.active_sensors].to_numpy(dtype=float)
        normalized = np.empty_like(sensors)
        for condition_id, statistics in self.condition_statistics.items():
            rows = condition_ids == condition_id
            normalized[rows] = (sensors[rows] - statistics.mean) / statistics.std
        result.loc[:, self.active_sensors] = normalized
        result.loc[:, SETTING_COLUMNS] = scaled_settings
        result["condition_id"] = condition_ids.astype(int)
        return result

    @property
    def sensor_count(self) -> int:
        return len(self.active_sensors)

    def metadata(self) -> dict[str, object]:
        return {
            "active_sensors": self.active_sensors,
            "training_units": list(self.training_units),
            "n_condition_clusters": int(self.condition_model.n_clusters),
            "variance_threshold": self.variance_threshold,
            "min_cluster_samples": self.min_cluster_samples,
        }


def _fit_slope_reference(
    frame: pd.DataFrame, sensor_columns: list[str], window: int = 10, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    slopes: list[np.ndarray] = []
    time = np.arange(window, dtype=float)
    centered_time = time - time.mean()
    denominator = float(np.square(centered_time).sum())
    for _, unit in frame.groupby("unit_id", sort=False):
        values = unit[sensor_columns].to_numpy(dtype=float)
        for end in range(window, len(values) + 1, 5):
            segment = values[end - window : end]
            slopes.append(centered_time @ segment / denominator)
    if not slopes:
        return np.zeros(len(sensor_columns)), np.ones(len(sensor_columns))
    matrix = np.vstack(slopes)
    return matrix.mean(axis=0), np.maximum(matrix.std(axis=0), epsilon)
