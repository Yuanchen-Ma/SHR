"""Prepare disjoint training, calibration and evaluation queries for risk fitting."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from shr.data.cmapss import SETTING_COLUMNS, load_official_test_set
from shr.data.corruption import apply_corruption, causal_fill
from shr.data.ncmapss import adaptation_split, fitting_rows
from shr.data.splits import split_calibration_units
from shr.data.windows import build_windows, records_to_arrays
from shr.io import key, load_config, read, write
from shr.paths import ROOT, official_path, partition_path, prediction_dir
from shr.predictors.query import build_risk_table
from shr.predictors.replay import replay
from shr.predictors.training import load_models, predict_ensemble
from shr.randomness import derive_seed
from shr.schemas import PreparedTrajectory


def prediction_tables(job, backbone, device, signature):
    """Replay the C-MAPSS grid or read the N-CMAPSS flight predictions."""
    if job["benchmark"] == "ncmapss":
        return {role: pd.read_parquet(backbone / f"{role}.parquet")
                for role in ("oof", "calibration", "test")}

    cache = ROOT / "experiments" / "upstream" / "query_cache"
    source = {}
    files = {"oof": "oof_risk_table", "calibration": "calibration_table", "test": "locked_test_table"}
    for role, name in files.items():
        original = pd.read_parquet(backbone / f"{name}.parquet")
        source[role], _ = replay(backbone, original, job["cohort"], role, device, cache, signature)
        if role == "test" and job["architecture"] == "tcn" and job["protocol"] != "matched":
            source["high"], _ = replay(backbone, original, job["cohort"], role, device, cache, signature, high=True)
    return source


def write_partition(job, calibration):
    # The predictor-independent engine partition retains the TCN job key used
    # for the original validation/policy seed derivation.
    partition = partition_path(job)
    validation, policy = split_calibration_units(calibration, key(dict(job, architecture="tcn")))
    assigned = {"validation_units": validation, "policy_units": policy}
    if partition.exists():
        existing = read(partition)
        if any(existing[name] != units for name, units in assigned.items()):
            raise ValueError(f"Predictors use different calibration engine roles: {partition}")
    else:
        write(partition, assigned)


def prepare_predictions(job, backbone, device, signature):
    dest = prediction_dir(job)
    marker = dest / "complete.json"
    if marker.exists():
        if read(marker)["signature"] != signature:
            raise ValueError(f"Prepared predictions use a different predictor signature: {dest}")
        return dest

    source = prediction_tables(job, backbone, device, signature)
    oof, calibration, test = (source[role] for role in ("oof", "calibration", "test"))
    ncmapss = job["benchmark"] == "ncmapss"
    primary = job["architecture"] == "tcn"
    protocol = job["protocol"]
    if ncmapss:
        oof = fitting_rows(oof, protocol)
        calibration = fitting_rows(calibration, protocol)
        adapt_units, threshold_units = adaptation_split(calibration, job["seed"])
    else:
        units = np.random.default_rng(job["seed"] + 600).permutation(sorted(calibration.unit_id.unique()))
        adapt_units, threshold_units = units[:len(units) // 2], units[len(units) // 2:]
    if set(oof.unit_id) & set(calibration.unit_id) or set(test.unit_id) & (set(oof.unit_id) | set(calibration.unit_id)):
        raise ValueError(f"Engine role overlap for {key(job)}")
    if set(adapt_units) & set(threshold_units) or not len(adapt_units) or not len(threshold_units):
        raise ValueError(f"Invalid calibration and adaptation roles for {key(job)}")

    adapt = calibration[calibration.unit_id.isin(adapt_units)].reset_index(drop=True)
    threshold = calibration[calibration.unit_id.isin(threshold_units)].reset_index(drop=True)
    write_partition(job, threshold)
    dest.mkdir(parents=True, exist_ok=True)
    for name, frame in (("oof", oof), ("adapt", adapt), ("calibration", threshold)):
        frame.to_parquet(dest / f"{name}.parquet", index=False)

    if protocol == "matched":
        scenes = [("matched", test.corruption_severity.isin(["none", "medium"]))]
        if primary:
            scenes.append(("clean", test.corruption_type.eq("clean")))
    else:
        scene = "full_loto_" + protocol
        active = test.corruption_type.eq(protocol)
        scenes = [(scene, active & test.corruption_severity.eq("medium"))]
        if ncmapss and primary:
            scenes.append((scene + "_high", active & test.corruption_severity.eq("high")))
    evaluations = []

    def save_test(dataset, scene, frame):
        if frame.empty or not ((frame.cycle - 1) % 5 == 0).all():
            raise ValueError(f"Expected a nonempty, full stride-five query grid: {dataset}/{scene}")
        folder = dest / "evaluation" / dataset / scene
        folder.mkdir(parents=True, exist_ok=True)
        frame.reset_index(drop=True).to_parquet(folder / "test.parquet", index=False)
        evaluations.append([dataset, scene, len(frame)])

    for scene, selected in scenes:
        for dataset, frame in test.loc[selected].groupby("dataset_id", sort=True):
            save_test(dataset, scene, frame)
    if "high" in source:
        save_test(job["cohort"], "full_loto_" + protocol + "_high", source["high"])
    groups = 1 if not ncmapss else 3 if job["cohort"] == "ds01_03" else 4
    if len(evaluations) != groups * (2 if primary else 1):
        raise ValueError(f"Incomplete held-out scenarios for {key(job)}: {evaluations}")

    if primary and not ncmapss and protocol == "matched":
        official_predictions(job, backbone, device).to_parquet(official_path(job), index=False)
    write(marker, {"job": job, "signature": signature, "evaluations": evaluations,
                   "oof_units": sorted(oof.unit_id.unique().tolist()),
                   "adaptation_units": list(map(int, adapt_units)),
                   "threshold_units": list(map(int, threshold_units))})
    return dest


def official_predictions(job, backbone, device):
    """Score the NASA test grid and preserve the official terminal query."""
    config = load_config(backbone / "resolved_config.yaml")
    split = read(backbone / "split_manifest.json")
    preprocessor = joblib.load(backbone / "preprocessing" / "preprocessor.joblib")
    distance = joblib.load(backbone / "preprocessing" / "latent_distance.joblib")
    models = load_models(backbone / "checkpoints" / "final_base", config, device)
    raw = load_official_test_set(config["paths"]["raw_data_dir"], job["cohort"], 125)
    raw["unit_id"] += 100000
    transformed = preprocessor.transform(raw)
    trajectories = []
    for kind, severity in (("clean", "none"), ("burst_missing", "medium"),
                           ("channel_dropout", "medium"), ("gradual_bias", "medium")):
        for unit, group in transformed.groupby("unit_id", sort=True):
            group = group.sort_values("cycle")
            values = group[preprocessor.active_sensors].to_numpy(float)
            cycles = group.cycle.to_numpy(int)
            seed = derive_seed(job["seed"], "official", int(unit), kind, severity)
            corrupted, mask, manifest = apply_corruption(
                values, np.ones_like(values, bool), kind, severity, int(unit), cycles, seed,
            )
            filled, delta = causal_fill(corrupted, mask)
            trajectories.append(PreparedTrajectory(
                int(unit), cycles, group.rul.to_numpy(float), filled,
                group[SETTING_COLUMNS].to_numpy(float), mask, delta,
                group.condition_id.to_numpy(int), manifest,
            ))
    records = build_windows(trajectories, 30, 5, True, preprocessor.slope_mean,
                            preprocessor.slope_std, None, job["seed"])
    features, _, _ = records_to_arrays(records)
    predictions = predict_ensemble(models, features, 512, device)
    result = build_risk_table(records, predictions, distance, split["split_id"], -1,
                              split["development_units"], 10., 30., job["cohort"])
    terminal_cycles = raw.groupby("unit_id").cycle.max()
    failure_cycles = raw.groupby("unit_id").apply(lambda group: float(group.rul_raw.iloc[-1] + group.cycle.iloc[-1]))
    result["is_official_cutoff"] = result.cycle.eq(result.unit_id.map(terminal_cycles))
    result["rul_raw"] = result.unit_id.map(failure_cycles) - result.cycle
    result["local_unit_id"] = result.unit_id - 100000
    return result
