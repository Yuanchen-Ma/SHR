"""Construct observable features and select causal historical query pairs."""
import numpy as np
import pandas as pd

from shr.predictors.query import FEATURE_GROUPS


STATIC = list(FEATURE_GROUPS["fusion_lite"]) + [
    "latent_mahalanobis_global", "ensemble_std", "prediction_disagreement",
    "real_missing_fraction", "log_missing_age",
]
HISTORY = [
    "anchor_available", "anchor_delta", "anchor_dispersion", "anchor_age",
    "anchor_effective_count", "anchor_confidence",
]
AGE = ["query_maturity", "query_log_age", "estimated_end"]
ANCHOR = ["empirical_mse", "empirical_positive"]
PAIR = ["pair_mu", "pair_mse", "pair_positive", "pair_spread", "pair_count"]
BASE = STATIC + HISTORY + AGE + ANCHOR + PAIR


def add_features(table: pd.DataFrame) -> pd.DataFrame:
    """Only current observed quantities and prior predictions; no truth or onset access."""
    f = table.copy().reset_index(drop=True)
    valid = np.minimum(f.cycle.to_numpy(float), 30)
    f['real_missing_fraction'] = np.clip((f.missing_fraction_all * 30 - (30 - valid)) / valid, 0, 1)
    f['log_missing_age'] = np.log1p(f.max_time_since_observed)
    f['history_available'] = 0.0
    f['upward_rul_change'] = 0.0
    f['elapsed_adjusted_change'] = 0.0
    for _, g in f.groupby(['unit_id', 'corruption_id'], sort=False):
        g = g.sort_values('cycle')
        pos = g.index.to_numpy()
        if len(pos) < 2:
            continue
        pred = g.rul_pred.to_numpy(float)
        dt = np.diff(g.cycle.to_numpy(float))
        # Capped healthy-life plateaus do not obey the uncapped countdown identity.
        eligible = (pred[:-1] < 120) & (pred[1:] < 120)
        f.loc[pos[1:], 'history_available'] = eligible.astype(float)
        f.loc[pos[1:], 'upward_rul_change'] = np.maximum(np.diff(pred), 0) / dt
        f.loc[pos[1:], 'elapsed_adjusted_change'] = np.where(eligible, (np.diff(pred) + dt) / dt, 0)
    return f


def age_features(f):
    h=add_features(f);h['query_maturity']=np.minimum(f.cycle.to_numpy()/30,1)
    h['query_log_age']=np.log1p(f.cycle.to_numpy())
    h['estimated_end']=(f.cycle.to_numpy()+f.rul_pred.to_numpy())/125
    return h


class HistoricalSummary:
    """Summarize mature past predictions with quality and time weights."""

    def __init__(self, config):
        self.config = dict(config)

    def fit(self, rows):
        frame = add_features(rows)
        mature = frame.cycle >= self.config["window_length"]
        eligible = mature & (frame.rul_pred < self.config["cap_guard"])
        if not eligible.any():
            eligible = mature
        if not eligible.any():
            raise ValueError("No mature risk-fit reference windows")
        self.scales = {
            name: max(float(frame.loc[eligible, name].quantile(.75)), 1e-3)
            for name in ("qr_width", "drift_score_mean")
        }
        return self

    def transform(self, rows):
        frame = add_features(rows)
        cfg = self.config
        width = frame.qr_width.to_numpy(float) / self.scales["qr_width"]
        drift = frame.drift_score_mean.to_numpy(float) / self.scales["drift_score_mean"]
        missing = frame.real_missing_fraction.to_numpy(float)
        gap = frame.max_time_since_observed.to_numpy(float)
        quality = np.clip(
            (1 - missing) ** 2 / ((1 + width ** 2 + drift ** 2) * (1 + gap / cfg["window_length"])),
            0, 1,
        )
        cycle_all = frame.cycle.to_numpy(float)
        pred_all = frame.rul_pred.to_numpy(float)
        features = np.zeros((len(frame), len(HISTORY)), float)
        for positions in frame.groupby(["unit_id", "corruption_id"], sort=False).indices.values():
            positions = positions[np.argsort(cycle_all[positions], kind="stable")]
            cycle = cycle_all[positions]
            pred = pred_all[positions]
            q = quality[positions]
            if np.any(np.diff(cycle) <= 0):
                raise ValueError("Repeated or non-increasing trajectory endpoints")
            mature = cycle >= cfg["window_length"]
            for index, position in enumerate(positions):
                elapsed = cycle[index] - cycle[:index]
                selected = np.flatnonzero(mature[:index] & (elapsed <= cfg["lookback"]))
                if not len(selected):
                    continue
                reliability = q[selected]
                weights = reliability * np.exp(-elapsed[selected] / cfg["decay_cycles"])
                if weights.sum() < 1e-12:
                    continue
                weights = weights / weights.sum()
                failure_time = cycle[selected] + pred[selected]
                center = float(weights @ failure_time)
                dispersion = float(np.sqrt(weights @ np.square(failure_time - center)))
                delta = float(np.clip(cycle[index] + pred[index] - center, -125, 125))
                age = float(weights @ elapsed[selected])
                confidence = float((weights @ reliability) * np.exp(-age / cfg["decay_cycles"]))
                features[position] = [1, delta, dispersion, age / cfg["lookback"],
                                      float(1 / np.square(weights).sum()), confidence]
        frame[HISTORY] = features
        return frame


def observed(history, rows):
    frame = history.transform(rows)
    frame[AGE] = age_features(rows)[AGE].to_numpy()
    return frame


def pair_data(frame):
    """Return the eligible past and current row indices and prediction-time displacement."""
    cycle = frame.cycle.to_numpy()
    prediction = frame.rul_pred.to_numpy()
    source = []
    current = []
    for positions in frame.groupby(["unit_id", "corruption_id"], sort=False).indices.values():
        positions = positions[np.argsort(cycle[positions])]
        for index, row in enumerate(positions):
            past = positions[:index]
            past = past[(cycle[row] - cycle[past] <= 90) & (cycle[past] >= 30) & (prediction[past] < 120)]
            source.extend(past)
            current.extend([row] * len(past))
    source = np.asarray(source, dtype=int)
    current = np.asarray(current, dtype=int)
    delta = cycle[current] + prediction[current] - cycle[source] - prediction[source]
    return source, current, delta


def pair_matrix(frame, source, current):
    features = age_features(frame)[STATIC + AGE].to_numpy(np.float32)
    lag = (frame.cycle.to_numpy()[current] - frame.cycle.to_numpy()[source]).astype(float)
    return np.column_stack([features[source], features[current], lag / 90])
