"""Regenerate predictor, query and risk-training inputs from the raw benchmarks."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path

from shr.paths import ROOT, check_run_settings

WORK = ROOT / "experiments" / "upstream"
CONFIG = ROOT / "configs" / "experiment.json"
COHORTS = ("FD001", "FD002", "FD003", "FD004", "ds01_03", "ds04_07")
PROTOCOLS = ("matched", "burst_missing", "channel_dropout", "gradual_bias")


def config_for(job, cmapss_dir):
    """Resolve the published network settings for a single experiment job."""
    from shr.io import load_config

    config = load_config(ROOT / "configs" / "prediction.yaml")
    overrides = load_config(ROOT / "configs" / "backbones.yaml")
    arch, cohort = job["architecture"], job["cohort"]
    config["model"]["architecture"] = arch
    config["experiment"].update(name=f"{cohort}_{arch}", seeds=[job["seed"]])
    if arch == "transformer":
        config["model"]["transformer"] = overrides["transformer"]
    if job["benchmark"] == "cmapss":
        settings = overrides["cmapss"]
        config["experiment"]["dataset_id"] = cohort
        config["paths"]["raw_data_dir"] = str(cmapss_dir)
        config["preprocessing"]["n_condition_clusters"] = settings["condition_clusters"][cohort]
        config["model"]["dropout"] = settings["dropout"][arch]
    return config


def flight_files(cohort, directory):
    from shr.data.ncmapss import CONFIRMATION_FILES, DEVELOPMENT_FILES

    files = enumerate(DEVELOPMENT_FILES, start=1) if cohort == "ds01_03" else enumerate(CONFIRMATION_FILES, start=4)
    resolved = [(index, directory / name) for index, name in files]
    missing = [str(path) for _, path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing N-CMAPSS HDF5 files: " + ", ".join(missing))
    return resolved


def check_data(jobs, cmapss_dir, ncmapss_dir):
    from shr.data.cmapss import add_test_rul, read_cmapss_table, read_test_rul, resolve_cmapss_paths
    from shr.data.ncmapss import validate_hdf5

    problems = []
    for benchmark, cohort in dict.fromkeys((job["benchmark"], job["cohort"]) for job in jobs):
        try:
            if benchmark == "cmapss":
                paths = resolve_cmapss_paths(cmapss_dir, cohort)
                read_cmapss_table(paths.train)
                add_test_rul(read_cmapss_table(paths.test), read_test_rul(paths.rul))
            else:
                for _, path in flight_files(cohort, ncmapss_dir):
                    validate_hdf5(path)
        except (OSError, ValueError) as exc:
            problems.append(f"{cohort}: {exc}")
    if problems:
        raise ValueError("Benchmark data check failed. See README.md for the expected files.\n"
                         + "\n".join(problems))


@lru_cache(maxsize=None)
def data_digest(path):
    from shr.io import sha

    return sha(path)


@lru_cache(maxsize=2)
def aggregate_flights(cohort, directory):
    import pandas as pd
    from shr.io import fingerprint
    from shr.data.ncmapss import aggregate_file

    files = flight_files(cohort, directory)
    signature = fingerprint({path.name: data_digest(path) for _, path in files})
    cache = WORK / "flights" / cohort / signature / "flights.parquet"
    if not cache.exists():
        frames = []
        for index, path in files:
            for role in ("dev", "test"):
                print(f"Aggregate {path.name} {role}", flush=True)
                frames.append(aggregate_file(path, index, role))
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        pd.concat(frames, ignore_index=True).to_parquet(temporary, index=False)
        temporary.replace(cache)
    return pd.read_parquet(cache), signature


def backbone_dir(job):
    from shr.io import key

    root = WORK / "predictors"
    if job["benchmark"] == "ncmapss":
        # Each N-CMAPSS predictor is reused across all four corruption protocols.
        return root / "ncmapss" / job["cohort"] / job["architecture"] / str(job["seed"])
    return root / key(job)


def ensure_backbone(job, device, cmapss_dir, ncmapss_dir):
    import torch
    import yaml

    from shr.data.cmapss import load_training_set, resolve_cmapss_paths
    from shr.data.ncmapss import SCENARIOS
    from shr.io import fingerprint, key, read, write
    from shr.predictors import cmapss, ncmapss, training

    config = config_for(job, cmapss_dir)
    dest = backbone_dir(job)
    if job["benchmark"] == "cmapss":
        inputs = resolve_cmapss_paths(cmapss_dir, job["cohort"]).as_list()
        data_signature = {path.name: data_digest(path) for path in inputs}
    else:
        flights, data_signature = aggregate_flights(job["cohort"], ncmapss_dir)
    signature = fingerprint(dict(config=config, data=data_signature, device=str(device),
                                 torch=torch.__version__))
    marker = dest / "upstream.json"
    if marker.exists():
        if read(marker)["signature"] != signature:
            raise ValueError(f"Data, configuration or runtime changed for {dest}. Use a clean experiment workspace.")
        return dest, signature

    print(f"Backbone {key(job)}", flush=True)
    torch.set_num_threads(4)
    progress = training.tqdm
    training.tqdm = training.QuietProgress
    try:
        if job["benchmark"] == "ncmapss":
            scenarios = SCENARIOS
            if job["architecture"] == "transformer":
                scenarios = [pair for pair in scenarios if pair[1] != "high"]
            ncmapss.fit_backbone(flights, job["architecture"], job["seed"], signature,
                                dest, config, device, scenarios)
        else:
            scene = "matched" if job["protocol"] == "matched" else "full_loto_" + job["protocol"]
            protocols = [cmapss.matched_protocol(config), *cmapss.held_out_protocols(config)]
            protocol = next(item for item in protocols if item.name == scene)
            raw = load_training_set(cmapss_dir, job["cohort"], config["experiment"]["rul_cap"])
            split = cmapss.make_split(raw, config, job["seed"], f"split_0_seed_{job['seed']}")
            (dest / "preprocessing").mkdir(parents=True, exist_ok=True)
            (dest / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            write(dest / "split_manifest.json", split.as_dict())
            cmapss.fit_backbone(config, raw, split, protocol, dest, device, signature)
        write(marker, {"signature": signature, "job": job})
    finally:
        training.tqdm = progress
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return dest, signature


def materialize(job, backbone, device, signature):
    from shr.data.preparation import prepare_predictions
    from shr.io import read, write
    from shr.paths import model_dir
    from shr.risk.auxiliary import fit
    from threadpoolctl import threadpool_limits

    marker = model_dir(job) / "upstream.json"
    if marker.exists():
        if read(marker)["signature"] != signature:
            raise ValueError(f"Auxiliary models use a different predictor signature: {marker.parent}")
        return
    raw = prepare_predictions(job, backbone, device, signature)
    with threadpool_limits(limits=4, user_api="openmp"), threadpool_limits(limits=2, user_api="blas"):
        fit(job, raw)
    write(marker, {"job": job, "signature": signature})



def prior_metrics(all_jobs, design):
    import pandas as pd
    from shr.paths import PRIOR_DIR, PRIOR_RESULTS
    from shr.risk.methods import LEARNERS
    from shr.io import key, read

    tcn = [job for job in all_jobs if job["architecture"] == "tcn"]
    if not all((PRIOR_DIR / "jobs" / key(job) / "complete.json").exists() for job in tcn):
        return
    records = []
    for job in tcn:
        folder = PRIOR_DIR / "jobs" / key(job)
        for method, primary in design["primary_learners"].items():
            if method != "shr":
                for learner in LEARNERS:
                    if learner != primary:
                        records.extend(read(folder / f"{method}_{learner}_metrics.json")["metrics"])
    frame = pd.DataFrame(records)
    fields = ["benchmark", "dataset_id", "architecture", "scenario", "repetition", "method", "learner"]
    if len(frame) != 3960 or frame.duplicated(fields).any():
        raise ValueError("Incomplete or duplicated configurations in the alternative baseline learners")
    PRIOR_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    temporary = PRIOR_RESULTS.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(PRIOR_RESULTS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "check-data", "backbones", "predict", "learners", "all"),
                        default="plan", help="Default: plan. Predict includes backbones; learners includes predict.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--architecture", choices=("both", "tcn", "transformer"), default="both")
    parser.add_argument("--benchmark", choices=("both", "cmapss", "ncmapss"), default="both")
    parser.add_argument("--cohort", choices=COHORTS)
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("--repetition", type=int, choices=range(11, 21))
    parser.add_argument("--cmapss-dir", type=Path, default=ROOT / "data" / "raw" / "CMAPSSData")
    parser.add_argument("--ncmapss-dir", type=Path, default=ROOT / "data" / "raw" / "NCMAPSSData")
    args = parser.parse_args(argv)
    design = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    jobs = [job for job in design["jobs"]
            if all(value in (None, "both", job[field]) for field, value in (
                ("architecture", args.architecture), ("benchmark", args.benchmark), ("cohort", args.cohort),
                ("protocol", args.protocol), ("repetition", args.repetition)))
            and (args.stage != "learners" or job["architecture"] == "tcn")]
    if not jobs:
        parser.error("No registered jobs match the filters")
    if args.stage == "plan":
        print(json.dumps({"jobs": len(jobs), "selected": jobs}, indent=2))
        return
    for name, value in (("CUBLAS_WORKSPACE_CONFIG", ":4096:8"), ("LOKY_MAX_CPU_COUNT", "4"),
                        ("OMP_NUM_THREADS", "4"), ("OPENBLAS_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2")):
        os.environ.setdefault(name, value)
    cmapss_dir, ncmapss_dir = args.cmapss_dir.resolve(), args.ncmapss_dir.resolve()
    try:
        check_data(jobs, cmapss_dir, ncmapss_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.stage == "check-data":
        print(f"Validated raw input files for {len(jobs)} registered jobs.")
        return

    from shr.io import key
    from shr.predictors.training import resolve_device

    device = resolve_device(args.device)
    try:
        check_run_settings()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    fit_baselines = args.stage in {"all", "learners"}
    if fit_baselines:
        from shr.risk.training import run_job

    for job in jobs:
        print(f"Upstream {key(job)}", flush=True)
        backbone, signature = ensure_backbone(job, device, cmapss_dir, ncmapss_dir)
        if args.stage != "backbones":
            materialize(job, backbone, device, signature)
        if fit_baselines and job["architecture"] == "tcn":
            run_job(job)
    if fit_baselines:
        prior_metrics(design["jobs"], design)


if __name__ == "__main__":
    main()
