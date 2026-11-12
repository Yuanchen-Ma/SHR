"""Reproduce the experimental data for SHR."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys

from .io import key, read, write
from .paths import (
    CONFIGS, EXPERIMENTS, PRIOR_RESULTS, RESULTS, ROOT, check_run_settings,
    model_dir, official_path, partition_path, prediction_dir,
)


STAGES = {
    "select": "select",
    "evaluate": "evaluate",
    "learners": "learner_evaluate",
    "selrej": "selrej",
    "bootstrap": "bootstrap",
    "selrej-bootstrap": "selrej_bootstrap",
    "summarize": "summarize",
    "nasa": "nasa_decomposition",
    "causality": "verify_causality",
}
PARALLEL = {"select", "evaluate", "learners", "selrej", "bootstrap", "selrej-bootstrap"}


def plan():
    jobs = read(CONFIGS / "experiment.json")["jobs"]
    repetitions = sorted({job["repetition"] for job in jobs})
    print(f"{len(jobs)} outer jobs, repetitions {repetitions}.")
    counts = Counter((job["benchmark"], job["architecture"]) for job in jobs)
    for (benchmark, architecture), count in sorted(counts.items()):
        print(f"  {benchmark:8} {architecture:11} {count:3}")


def initialize(design):
    files = {
        "design.json": design,
        "selrej_design.json": read(CONFIGS / "selrej.json"),
    }
    for name, config in files.items():
        path = EXPERIMENTS / name
        if not path.exists():
            write(path, config)
        elif read(path) != config:
            raise SystemExit(f"{path.relative_to(ROOT)} was created with different settings. Use a clean checkout.")


def require_preparation(jobs):
    required = [PRIOR_RESULTS]
    for job in jobs:
        predictions = prediction_dir(job)
        models = model_dir(job)
        required.extend((models / "complete.json", predictions / "calibration.parquet",
                         predictions / "evaluation", partition_path(job)))
        if job["benchmark"] == "cmapss" and job["architecture"] == "tcn" and job["protocol"] == "matched":
            required.append(official_path(job))
    missing = next((path for path in required if not path.exists()), None)
    if missing is not None:
        raise SystemExit(f"Preparation is incomplete ({missing.relative_to(ROOT)}). Run 'upstream --stage all' first.")


def require_stage(stage, design):
    jobs = design["jobs"]
    tcn_jobs = [job for job in jobs if job["architecture"] == "tcn"]
    selected = EXPERIMENTS / "selection"
    primary = EXPERIMENTS / "primary"
    selrej = EXPERIMENTS / "selrej"
    if stage == "evaluate":
        required = [selected / key(job) / f"{method}_{learner}.json" for job in jobs
                    for method, learner in design["primary_learners"].items()]
    elif stage == "learners":
        required = [selected / key(job) / f"shr_{learner}.json" for job in tcn_jobs
                    for learner in ("rf", "logistic", "knn")]
    elif stage == "causality":
        required = [selected / key(job) / "shr_xgboost.json" for job in jobs]
    elif stage == "bootstrap":
        required = [primary / key(job) / "complete.json" for job in tcn_jobs]
    elif stage in {"selrej-bootstrap", "summarize", "nasa"}:
        required = [selrej / key(job) / "complete.json" for job in jobs]
        if stage == "selrej-bootstrap":
            required.append(EXPERIMENTS / "bootstrap/base_draws.npz")
        elif stage == "summarize":
            required.append(RESULTS / "diagnostics/selrej_bootstrap_verification.json")
        else:
            required.extend(primary / key(job) / "complete.json" for job in tcn_jobs)
            required.append(RESULTS / "metrics/offline.parquet")
    else:
        return
    missing = next((path for path in required if not path.exists()), None)
    if missing is not None:
        raise SystemExit(f"An earlier stage is incomplete ({missing.relative_to(ROOT)}). See README.md for the workflow.")


def run_stages(stage, workers):
    design = read(CONFIGS / "experiment.json")
    if stage in {"all", "select", "evaluate", "learners", "selrej", "causality"}:
        require_preparation(design["jobs"])
    require_stage("select" if stage == "all" else stage, design)
    try:
        check_run_settings()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    initialize(design)
    for step in STAGES if stage == "all" else (stage,):
        require_stage(step, design)
        print(f"SHR stage: {step}", flush=True)
        args = ["--workers", str(workers)] if step in PARALLEL else []
        subprocess.run([sys.executable, "-B", "-m", f"shr.evaluation.{STAGES[step]}", *args],
                       cwd=ROOT, check=True)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("workers must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan", help="List the experiment counts without starting training.")
    commands.add_parser("upstream", add_help=False, help="Prepare data, predictors and baseline models.")
    run = commands.add_parser("run", help="Run the evaluation stages after preparation.")
    run.add_argument("stage", choices=["all", *STAGES])
    run.add_argument("--workers", type=positive_int, default=1)
    reproduce = commands.add_parser("reproduce", help="Generate the experimental data from NASA benchmarks.")
    reproduce.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    reproduce.add_argument("--workers", type=positive_int, default=1)
    reproduce.add_argument("--cmapss-dir", type=Path)
    reproduce.add_argument("--ncmapss-dir", type=Path)
    args, rest = parser.parse_known_args(argv)
    if args.command == "upstream":
        from .upstream import main as upstream
        upstream(rest)
        return
    if rest:
        parser.error("unrecognized arguments: " + " ".join(rest))
    if args.command == "reproduce":
        from .upstream import main as upstream

        preparation = ["--stage", "all", "--device", args.device]
        for option, path in (("--cmapss-dir", args.cmapss_dir), ("--ncmapss-dir", args.ncmapss_dir)):
            if path is not None:
                preparation.extend((option, str(path)))
        upstream(preparation)
        run_stages("all", args.workers)
    elif args.command == "run":
        run_stages(args.stage, args.workers)
    else:
        plan()


if __name__ == "__main__":
    main()
