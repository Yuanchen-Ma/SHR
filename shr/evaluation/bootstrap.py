"""Paired physical-engine resampling, conditional on the fitted risk models."""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import numpy as np
import pandas as pd

from shr.io import key, now, read, write
from shr.paths import EXPERIMENTS, prediction_dir

from . import BOOTSTRAP, PRIMARY
from .metrics import area, weighted_area


METHODS = ("epsilon", "lossrej_linear", "selrej", "rcr", "ccr", "shr")
BASE_DRAWS = BOOTSTRAP / "base_draws.npz"
EXPECTED_CONFIGURATIONS = {"cmapss": 160, "ncmapss": 280}
DRAWS = 1000


def prepare(jobs):
    path = BOOTSTRAP / "design.json"
    if path.is_file():
        return read(path)
    engines = defaultdict(set)
    for job in jobs:
        folder = prediction_dir(job) / "evaluation"
        for source in folder.glob("*/*/test.parquet"):
            dataset, scenario = source.parent.relative_to(folder).parts
            if scenario == "clean" or scenario.endswith("_high"):
                continue
            units = pd.read_parquet(source, columns=["unit_id"]).unit_id.astype(int)
            engines[f"{job['benchmark']}|{dataset}"].update(units)
    if len(engines) != 11:
        raise ValueError(f"Expected 11 benchmark subsets for bootstrap, got {len(engines)}")
    design = {
        "seed": 2026100503,
        "draws": DRAWS,
        "methods": METHODS,
        "engines": {name: sorted(units) for name, units in sorted(engines.items())},
        "conditional_on_fitted_models": True,
        "paired_physical_engines": True,
        "pointwise_intervals": True,
    }
    write(path, design)
    return design


def weighted_job_scores(job, design, folder, methods):
    if not (folder / "complete.json").is_file():
        raise FileNotFoundError(f"Score evaluation must finish before bootstrap: {folder}")
    rng = np.random.default_rng(design["seed"])
    draws = design["draws"]
    indices = {}
    weights = {}
    for name, units in sorted(design["engines"].items()):
        indices[name] = {unit: i for i, unit in enumerate(units)}
        weights[name] = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=draws)
    total = np.zeros((draws, len(methods)))
    count = 0
    for source in sorted((folder / "scores").glob("*/*/queries.parquet")):
        dataset, scenario = source.parent.relative_to(folder / "scores").parts
        if scenario == "clean" or scenario.endswith("_high"):
            continue
        columns = ["unit_id", "rul_pred", "rul_true", *(f"score_{method}" for method in methods)]
        frame = pd.read_parquet(source, columns=columns)
        error = (frame.rul_pred - frame.rul_true).to_numpy(float)
        loss = error ** 2 + 20 * np.maximum(error, 0)
        name = f"{job['benchmark']}|{dataset}"
        engine = frame.unit_id.map(indices[name]).to_numpy()
        if pd.isna(engine).any():
            raise ValueError(f"Unknown bootstrap engine in {source}")
        for column, method in enumerate(methods):
            score = frame[f"score_{method}"].to_numpy()
            unweighted = weighted_area(score, loss, np.ones((1, len(frame))))[0]
            np.testing.assert_allclose(unweighted, area(score, loss), rtol=1e-10, atol=1e-8)
            for start in range(0, draws, 25):
                stop = start + 25
                total[start:stop, column] += weighted_area(score, loss, weights[name][start:stop, engine])
        count += 1
    if not count:
        raise ValueError(f"No main-scenario query scores found in {folder}")
    return total, count


def one(job, design):
    folder = BOOTSTRAP / key(job)
    marker = folder / "complete.json"
    if marker.is_file():
        return read(marker)
    draws, count = weighted_job_scores(job, design, PRIMARY / key(job), METHODS)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "draws.npz", metrics=draws)
    result = dict(job=job, configurations=count, finished_at=now())
    write(marker, result)
    return result


def aggregate(jobs, directory, n_methods):
    totals = {benchmark: np.zeros((DRAWS, n_methods)) for benchmark in EXPECTED_CONFIGURATIONS}
    counts = defaultdict(int)
    for job in jobs:
        folder = directory / key(job)
        benchmark = job["benchmark"]
        with np.load(folder / "draws.npz") as archive:
            totals[benchmark] += archive["metrics"]
        counts[benchmark] += read(folder / "complete.json")["configurations"]
    if dict(counts) != EXPECTED_CONFIGURATIONS:
        raise ValueError(f"Incomplete paired bootstrap: {dict(counts)}")
    return {benchmark: total / counts[benchmark] for benchmark, total in totals.items()}


def paired_intervals(paired_draws):
    rows = []
    for benchmark, values in paired_draws.items():
        for column, method in enumerate(METHODS[:-1]):
            change = 100 * (values[:, -1] / values[:, column] - 1)
            low, median, high = np.quantile(change, [.025, .5, .975])
            rows.append(dict(
                benchmark=benchmark, reference=method, lower_pct=low, median_pct=median,
                upper_pct=high, draws=len(values), configurations=EXPECTED_CONFIGURATIONS[benchmark],
            ))
    return pd.DataFrame(rows)


def run_jobs(jobs, directory, worker, design, workers, stage):
    pending = [job for job in jobs if not (directory / key(job) / "complete.json").is_file()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, job, design): job for job in pending}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception:
                write(EXPERIMENTS / f"{stage}_error.json", dict(job=futures[future], traceback=traceback.format_exc()))
                raise
            write(EXPERIMENTS / f"{stage}_status.json", dict(complete=done, total=len(pending), last=result))
            print(f"{stage}: {done}/{len(pending)}", flush=True)


def tcn_jobs():
    return [job for job in read(EXPERIMENTS / "design.json")["jobs"] if job["architecture"] == "tcn"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    jobs = tcn_jobs()
    design = prepare(jobs)
    run_jobs(jobs, BOOTSTRAP, one, design, args.workers, "bootstrap")
    np.savez_compressed(BASE_DRAWS, **aggregate(jobs, BOOTSTRAP, len(METHODS)))
    print("Paired bootstrap complete.", flush=True)


if __name__ == "__main__":
    main()
