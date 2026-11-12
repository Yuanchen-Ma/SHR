"""Fit the empirical and historical reference models and cross-fit their features."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from shr.io import write
from shr.paths import model_dir
from shr.randomness import derive_seed

from .baselines import CCR, Epsilon, LossRejection, RcR
from .features import AGE, ANCHOR, HISTORY, PAIR, STATIC, HistoricalSummary, observed, pair_data, pair_matrix
from .moments import MomentHeads
from .settings import balanced_training, role_weights, settings, tree_options


class EmpiricalRisk:
    cols = STATIC + HISTORY + AGE

    def fit(self, features, rows, weights, config, seed):
        error = (rows.rul_pred - rows.rul_true).to_numpy()
        x = features[self.cols].to_numpy(np.float32)
        self.models = [
            XGBRegressor(objective="reg:squarederror", **tree_options(config, seed))
            .fit(x, target, sample_weight=weights)
            for target in (error ** 2, np.maximum(error, 0))
        ]
        return self

    def predict(self, features):
        x = features[self.cols].to_numpy(np.float32)
        return np.column_stack([np.maximum(model.predict(x), 0) for model in self.models])


class ResidualQuantiles:
    quantile_levels = np.arange(.05, 1, .1)

    def fit(self, rows, weights, config, seed):
        source, current, delta = pair_data(rows)
        x = pair_matrix(rows, source, current)
        count = np.bincount(current, minlength=len(rows))
        pair_weights = weights[current] / np.maximum(count[current], 1)
        target = (rows.rul_pred - rows.rul_true).to_numpy()[current] - delta
        if len(source) > 60000:
            selected = np.random.default_rng(seed).choice(len(source), 60000, replace=False)
            x, target, pair_weights = x[selected], target[selected], pair_weights[selected]
        pair_weights *= len(pair_weights) / pair_weights.sum()
        self.estimator = XGBRegressor(objective="reg:quantileerror", quantile_alpha=self.quantile_levels,
                                      **tree_options(config, seed, 100))
        self.estimator.fit(x, target, sample_weight=pair_weights)
        return self

    def predict(self, rows):
        source, current, delta = pair_data(rows)
        count = np.bincount(current, minlength=len(rows))
        result = np.zeros((len(rows), len(PAIR)))
        result[:, 4] = count
        if not len(source):
            return result
        x = pair_matrix(rows, source, current)
        prediction = rows.rul_pred.to_numpy()[current]
        quantiles = self.estimator.predict(x) + delta[:, None]
        quantiles = np.clip(quantiles, prediction[:, None] - 125, prediction[:, None])
        values = (quantiles.mean(axis=1), (quantiles ** 2).mean(axis=1),
                  np.maximum(quantiles, 0).mean(axis=1))
        denominator = np.maximum(count, 1)
        for index, value in enumerate(values):
            result[:, index] = np.bincount(current, weights=value, minlength=len(rows)) / denominator
        result[:, 3] = np.sqrt(np.maximum(result[:, 1] - result[:, 0] ** 2, 0))
        return result


def enrich(features, empirical, historical, rows):
    features = features.copy()
    features[ANCHOR] = empirical.predict(features)
    features[PAIR] = historical.predict(rows)
    empty = features.pair_count.eq(0)
    features.loc[empty, "pair_mse"] = features.loc[empty, "empirical_mse"]
    features.loc[empty, "pair_positive"] = features.loc[empty, "empirical_positive"]
    return features


class Auxiliary:
    def fit(self, oof, adapt, config, seed):
        rows, weights = balanced_training(oof, adapt)
        self.training_units = sorted(rows.unit_id.unique().tolist())
        self.kind = "empirical" if np.mean(rows.rul_true.to_numpy() >= 125 - 1e-6) >= .01 else "physical"
        self.history = HistoricalSummary(config["history"]).fit(rows)
        observed_features = observed(self.history, rows)
        model_config = config["model"]
        self.empirical = EmpiricalRisk().fit(observed_features, rows, weights, model_config, seed)
        self.historical = ResidualQuantiles().fit(rows, weights, model_config, seed)
        features = enrich(observed_features, self.empirical, self.historical, rows)

        rng = np.random.default_rng(seed)
        folds = {}
        for part in (oof, adapt):
            folds.update({unit: index % 3 for index, unit in enumerate(rng.permutation(sorted(part.unit_id.unique())))})
        assigned = rows.unit_id.map(folds).to_numpy()
        self.roles = []
        for fold in range(3):
            held = assigned == fold
            fitting_rows = rows.loc[~held].reset_index(drop=True)
            held_rows = rows.loc[held].reset_index(drop=True)
            assert not set(fitting_rows.unit_id) & set(held_rows.unit_id)
            fold_weights = role_weights(fitting_rows, adapt.unit_id.unique())
            fold_seed = seed + fold + 1
            empirical = EmpiricalRisk().fit(observed_features.loc[~held].reset_index(drop=True),
                                            fitting_rows, fold_weights, model_config, fold_seed)
            historical = ResidualQuantiles().fit(fitting_rows, fold_weights, model_config, fold_seed)
            held_features = enrich(observed_features.loc[held].reset_index(drop=True), empirical, historical, held_rows)
            features.loc[held, ANCHOR + PAIR] = held_features[ANCHOR + PAIR].to_numpy()
            self.roles.append({"fit": sorted(fitting_rows.unit_id.unique().tolist()),
                               "held": sorted(held_rows.unit_id.unique().tolist())})
        features["fit_weight"] = weights
        return features

    def features(self, rows, check=True):
        if check and set(rows.unit_id) & set(self.training_units):
            raise ValueError("Inference engine used in fitting")
        return enrich(observed(self.history, rows), self.empirical, self.historical, rows)


def save_model(model, path):
    temporary = path.with_suffix(".tmp")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)


def fit(job, raw_path: Path):
    """Prepare the cross-fitted risk inputs and fitted default candidates for one job."""
    dest = model_dir(job)
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "complete.json"
    if marker.exists():
        return
    config = settings()
    # Keep the experiment's seed namespace strings to preserve the derived seeds.
    seed = derive_seed(job["seed"], "v10_crossfit", job["protocol"])
    epsilon_seed = derive_seed(job["seed"], "literature_budget", job["protocol"])
    oof = pd.read_parquet(raw_path / "oof.parquet")
    adapt = pd.read_parquet(raw_path / "adapt.parquet")
    feature_marker = dest / "features_complete.json"
    if feature_marker.exists():
        aux = joblib.load(dest / "auxiliary.joblib")
        training = pd.read_parquet(dest / "training.parquet")
    else:
        aux = Auxiliary()
        training = aux.fit(oof, adapt, config, seed)
        save_model(aux, dest / "auxiliary.joblib")
        temporary = dest / "training.tmp"
        training.to_parquet(temporary, index=False)
        temporary.replace(dest / "training.parquet")
        write(feature_marker, {"roles": aux.roles})
    rows, weights = balanced_training(oof, adapt)
    baseline_features = observed(aux.history, rows)
    error = (rows.rul_pred - rows.rul_true).to_numpy()
    if job["architecture"] == "tcn":
        candidates = {
            "shr_heads": lambda: MomentHeads().fit(training, aux.kind, config["model"], seed),
            "epsilon": lambda: Epsilon().fit(baseline_features, error, weights, config["model"], epsilon_seed),
            "rcr": lambda: RcR().fit(baseline_features, error, weights, config["model"], seed),
            "ccr": lambda: CCR().fit(baseline_features, error, weights, seed),
        }
    else:
        candidates = {}
    candidates["loss_rejection"] = lambda: LossRejection().fit(baseline_features, error, weights, seed)
    for name, build in candidates.items():
        path = dest / f"{name}.joblib"
        if not path.is_file():
            save_model(build(), path)
    write(marker, {"job": job, "fit_seed": seed, "epsilon_seed": epsilon_seed})
