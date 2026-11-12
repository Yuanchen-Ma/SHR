"""Tune the baseline candidates on reserved engines and cache their evaluation features."""
import gc
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from shr.evaluation.metrics import area
from shr.io import key, now, read, sha, write
from shr.paths import CONFIGS, PRIOR_DIR, ROOT, model_dir, partition_path, prediction_dir
from shr.randomness import derive_seed

from . import learners as learner_module
from .methods import LEARNERS, RiskMethod


BASELINES = ("epsilon", "ccr", "rcr")


def save_features(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name, suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validation_inputs(job):
    models = model_dir(job)
    cache = PRIOR_DIR / "features" / key(job)
    selection_path = partition_path(job)
    selection = read(selection_path)
    training_path = models / "training.parquet"
    training = pd.read_parquet(training_path)
    marker = cache / "meta.json"
    if not marker.exists():
        aux = joblib.load(models / "auxiliary.joblib")
        calibration = pd.read_parquet(prediction_dir(job) / "calibration.parquet")
        validation = calibration[calibration.unit_id.isin(selection["validation_units"])].reset_index(drop=True)
        assert set(validation.unit_id) == set(selection["validation_units"])
        assert not set(validation.unit_id) & set(training.unit_id)
        save_features(aux.features(validation), cache / "validation.parquet")
        write(marker, dict(kind=aux.kind, validation_units=selection["validation_units"],
                           train_units=sorted(training.unit_id.unique().tolist()),
                           train_sha=sha(training_path), selection_sha=sha(selection_path)))
    meta = read(marker)
    if sha(training_path) != meta["train_sha"] or sha(selection_path) != meta["selection_sha"]:
        raise ValueError(f"Validation features were built using different inputs for {key(job)}")
    validation = pd.read_parquet(cache / "validation.parquet")
    assert not set(training.unit_id) & set(validation.unit_id)
    return training, validation, meta


def value(frame, score):
    error = (frame.rul_pred - frame.rul_true).to_numpy(float)
    loss = error ** 2 + 20 * np.maximum(error, 0)
    values = [area(score[indices], loss[indices])
              for indices in frame.groupby("dataset_id", sort=True).indices.values()]
    return float(np.mean(values))


def fit_seed(job, method):
    # Preserve the exact published seed derivation for each training objective.
    namespace = "literature_budget" if method == "epsilon" else "v10_crossfit"
    return derive_seed(job["seed"], namespace, job["protocol"])


class NativeAdapter:
    """Use the already fitted default candidate when its settings are identical."""

    def __init__(self, method, learner, candidate, job):
        name = "shr_heads" if method == "shr" else method
        path = model_dir(job) / f"{name}.joblib"
        self.model = joblib.load(path)
        if method == "shr":
            estimators = [self.model.sign] + self.model.heads["normalized_logistic"]
        else:
            estimators = [self.model.estimator]
        self.actual = []
        for estimator in estimators:
            if isinstance(estimator, float):
                self.actual.append({"constant": estimator})
                continue
            actual = estimator.get_params()
            for field, setting in candidate["parameters"].items():
                assert actual[field] == setting, (field, actual[field], setting)
            assert actual["random_state"] == fit_seed(job, method)
            actual.update(reused_from=path.relative_to(ROOT).as_posix(), source_sha=sha(path))
            self.actual.append(actual)

    def score(self, features):
        return self.model.score(features)


def fitted(job, training, meta, method, learner, candidate):
    native = (learner == "rf" and method == "ccr" and candidate["id"] == "rf_00") or (
        learner == "xgboost" and method in {"shr", "epsilon", "rcr"} and candidate["id"] == "xgboost_00"
    )
    if job["architecture"] == "tcn" and native:
        return NativeAdapter(method, learner, candidate, job)
    model = RiskMethod(method, learner, candidate["parameters"], fit_seed(job, method), meta["kind"])
    return model.fit(training)


def tune(job, training, validation, meta, method, learner, candidates):
    dest = PRIOR_DIR / "jobs" / key(job)
    selected = dest / f"{method}_{learner}_selected.json"
    if selected.exists():
        return read(selected)
    records = []
    learner_module.CACHE_NEIGHBORS = learner == "knn"
    learner_module.NEIGHBOR_CACHE.clear()
    try:
        for candidate in candidates:
            trial = dest / "trials" / f"{method}_{candidate['id']}.json"
            if trial.exists():
                records.append(read(trial))
                continue
            start = time.perf_counter()
            record = dict(method=method, learner=learner, **candidate, seed=fit_seed(job, method))
            if learner == "knn" and candidate["parameters"]["n_neighbors"] > len(training):
                record.update(status="invalid", reason="k exceeds training count")
            else:
                model = fitted(job, training, meta, method, learner, candidate)
                scores = model.score(validation)
                assert np.isfinite(scores).all()
                record.update(status="ok", validation_area=value(validation, scores),
                              seconds=time.perf_counter() - start, actual=model.actual)
                del model
                gc.collect()
            write(trial, record)
            records.append(record)
    finally:
        learner_module.CACHE_NEIGHBORS = False
        learner_module.NEIGHBOR_CACHE.clear()
    valid = [record for record in records if record["status"] == "ok"]
    best = min(valid, key=lambda record: (record["validation_area"], record["id"]))
    result = dict(best=best, valid_candidates=len(valid), total_candidates=len(candidates),
                  validation_units=meta["validation_units"])
    write(selected, result)
    return result


def outer_features(job, meta):
    raw = prediction_dir(job) / "evaluation"
    cache = PRIOR_DIR / "features" / key(job) / "outer"
    excluded = set(meta["train_units"]) | set(meta["validation_units"])
    aux = None
    frames = []
    for path in sorted(raw.glob("*/*/test.parquet")):
        dataset, scenario = path.parent.relative_to(raw).parts
        if scenario == "clean" or scenario.endswith("_high"):
            continue
        destination = cache / dataset / scenario / "test.parquet"
        if destination.exists():
            features = pd.read_parquet(destination)
        else:
            rows = pd.read_parquet(path)
            if aux is None:
                aux = joblib.load(model_dir(job) / "auxiliary.joblib")
            features = aux.features(rows)
            save_features(features, destination)
        assert not set(features.unit_id) & excluded
        frames.append((dataset, scenario, features))
    expected = 1 if job["benchmark"] == "cmapss" else 3 if job["cohort"] == "ds01_03" else 4
    assert len(frames) == expected, (key(job), len(frames), expected)
    return frames


@threadpool_limits.wrap(limits=4, user_api="openmp")
@threadpool_limits.wrap(limits=2, user_api="blas")
def run_job(job):
    dest = PRIOR_DIR / "jobs" / key(job)
    marker = dest / "complete.json"
    if marker.exists():
        return
    design = read(CONFIGS / "experiment.json")
    training, validation, meta = validation_inputs(job)
    selected = {}
    for learner in LEARNERS:
        for method in BASELINES:
            selected[method, learner] = tune(job, training, validation, meta, method, learner,
                                             design["candidates"][learner])["best"]
    frames = outer_features(job, meta)
    for (method, learner), candidate in selected.items():
        if learner == design["primary_learners"][method]:
            continue
        path = dest / f"{method}_{learner}_metrics.json"
        if path.exists():
            continue
        model = fitted(job, training, meta, method, learner, candidate)
        replay = value(validation, model.score(validation))
        np.testing.assert_allclose(replay, candidate["validation_area"], rtol=2e-6, atol=1e-6)
        metrics = []
        for dataset, scenario, features in frames:
            score = model.score(features)
            error = (features.rul_pred - features.rul_true).to_numpy(float)
            positive = area(score, np.maximum(error, 0))
            squared = area(score, error ** 2)
            metrics.append(dict(**job, dataset_id=dataset, scenario=scenario, method=method, learner=learner,
                                cost=20, positive_area=positive, mse_area=squared, joint_area=squared + 20 * positive,
                                candidate_id=candidate["id"], fit_seed=candidate["seed"],
                                n_queries=len(features), n_engines=features.unit_id.nunique()))
        write(path, dict(metrics=metrics, actual=model.actual, validation_replay=replay))
        del model
        gc.collect()
    write(marker, dict(job=job, configurations=len(frames), completed_at=now()))
