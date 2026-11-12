"""Insert the validation-selected SelRej scores into the same paired bootstrap draws."""

import argparse

from shr.io import key, now, read, sha, write
from shr.paths import EXPERIMENTS

from . import BOOTSTRAP, DIAGNOSTICS, METRICS, SELREJ, SELREJ_BOOTSTRAP
from .bootstrap import BASE_DRAWS, METHODS, aggregate, paired_intervals, run_jobs, tcn_jobs, weighted_job_scores

import numpy as np


def one(job, design):
    folder = SELREJ_BOOTSTRAP / key(job)
    marker = folder / "complete.json"
    if marker.is_file():
        return read(marker)
    draws, count = weighted_job_scores(job, design, SELREJ / key(job), ("selrej",))
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "draws.npz", metrics=draws)
    result = dict(job=job, configurations=count, finished_at=now())
    write(marker, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not BASE_DRAWS.is_file():
        raise FileNotFoundError(f"Run the paired bootstrap first: {BASE_DRAWS}")
    design = read(BOOTSTRAP / "design.json")
    jobs = tcn_jobs()
    run_jobs(jobs, SELREJ_BOOTSTRAP, one, design, args.workers, "selrej_bootstrap")
    selrej = aggregate(jobs, SELREJ_BOOTSTRAP, 1)
    column = METHODS.index("selrej")
    with np.load(BASE_DRAWS) as archive:
        paired = {benchmark: archive[benchmark].copy() for benchmark in selrej}
    for benchmark, draws in paired.items():
        draws[:, column] = selrej[benchmark][:, 0]
    METRICS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(METRICS / "bootstrap_draws.npz", **paired)
    paired_intervals(paired).to_csv(METRICS / "paired_bootstrap_intervals.csv", index=False)
    registry = read(EXPERIMENTS / "design.json")["jobs"]
    write(DIAGNOSTICS / "selrej_bootstrap_verification.json", dict(
        complete=True, tcn_jobs=len(jobs),
        selected_selrej_jobs=sum((SELREJ / key(job) / "complete.json").is_file() for job in registry),
        same_physical_engine_draws=True, replaced_method="selrej", base_sha=sha(BASE_DRAWS),
        completed_at=now(),
    ))
    print("SelRej paired bootstrap complete.", flush=True)


if __name__ == "__main__":
    main()
