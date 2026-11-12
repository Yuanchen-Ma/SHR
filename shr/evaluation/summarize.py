"""Combine registered held-out results and check the manuscript's aggregation protocol."""

import json

import numpy as np
import pandas as pd

from shr.io import key, read, write
from shr.paths import EXPERIMENTS, PRIOR_RESULTS

from . import DIAGNOSTICS, LEARNERS, METRICS, PRIMARY, SELECTION, SELREJ


RESULT_TYPES = ("offline", "online", "ablation", "weights", "lambda", "official", "curves")
SELREJ_TYPES = {"offline", "online", "official", "curves"}
ALTERNATE_LEARNERS = ("rf", "logistic", "knn")
BENCHMARK_COUNTS = {"cmapss": 160, "ncmapss": 280}


def gather(name, jobs):
    frames = []
    for job in jobs:
        folder = PRIMARY / key(job)
        path = folder / f"{name}.parquet"
        if path.is_file():
            frame = pd.read_parquet(path)
            if name in SELREJ_TYPES:
                frame = frame[frame.variant.ne("selrej")]
            frames.append(frame)
        elif name in {"offline", "online"}:
            raise FileNotFoundError(path)
        if name in SELREJ_TYPES:
            neural = SELREJ / key(job) / f"{name}.parquet"
            if neural.is_file():
                frames.append(pd.read_parquet(neural))
            elif name in {"offline", "online"}:
                raise FileNotFoundError(neural)
    return pd.concat(frames, ignore_index=True)


def check_metrics(frames):
    offline = frames["offline"]
    assert len(offline) == 7920, len(offline)
    keys = ["architecture", "benchmark", "dataset_id", "scenario", "repetition", "variant"]
    assert not offline.duplicated(keys).any()
    np.testing.assert_allclose(offline.joint_area, offline.mse_area + 20 * offline.positive_area,
                               rtol=1e-10, atol=1e-8)
    online = frames["online"]
    assert len(online) == 5280, len(online)
    assert not online.duplicated(keys).any()
    assert online.rejection.between(0, .2 + 1e-12).all()
    np.testing.assert_allclose(online.joint, online.mse + 20 * online.positive,
                               rtol=1e-10, atol=1e-8)
    core = offline[offline.core]
    assert len(core) == 5280, len(core)
    for (_, benchmark, _), group in core.groupby(["architecture", "benchmark", "variant"]):
        assert len(group) == BENCHMARK_COUNTS[benchmark]
        assert set(group.repetition) == set(range(11, 21))
    assert len(frames["ablation"]) == 5720, len(frames["ablation"])
    assert len(frames["official"]) == 480, len(frames["official"])
    curves = frames["curves"]
    full = curves[np.isclose(curves.coverage, 1)].groupby(["benchmark", "variant", "metric"]).risk.mean()
    for _, group in full.groupby(level=["benchmark", "metric"]):
        assert np.ptp(group.to_numpy()) < 1e-8
    return core[core.architecture.eq("tcn")]


def gather_learners(jobs, primary_tcn, primary_learners):
    # Baseline validation selections were generated from the same training data and protocol.
    baseline = pd.read_parquet(PRIOR_RESULTS)
    baseline = baseline[baseline.architecture.eq("tcn") & baseline.method.ne("shr")].copy()
    baseline = baseline[baseline.learner.ne(baseline.method.map(primary_learners))]
    primary = primary_tcn[primary_tcn.variant.isin(primary_learners)].copy()
    primary["method"] = primary.variant
    primary["learner"] = primary.method.map(primary_learners)
    frames = [baseline, primary]
    for job in jobs:
        if job["architecture"] != "tcn":
            continue
        for learner in ALTERNATE_LEARNERS:
            folder = LEARNERS / key(job) / learner
            if not (folder / "complete.json").is_file():
                raise FileNotFoundError(f"Risk learner comparison is incomplete: {folder}")
            frames.append(pd.read_parquet(folder / "metrics.parquet"))
    learners = pd.concat(frames, ignore_index=True)
    assert len(learners) == 7040, len(learners)
    learner_means = learners.groupby(["benchmark", "method", "learner"]).joint_area.mean()
    primary_means = primary.groupby(["benchmark", "method"]).joint_area.mean()
    for benchmark in BENCHMARK_COUNTS:
        for method, learner in primary_learners.items():
            np.testing.assert_allclose(learner_means[benchmark, method, learner],
                                       primary_means[benchmark, method], rtol=0, atol=1e-9)
    return learners


def collect_diagnostics(jobs, primary_learners):
    references, selections, neural = [], [], []
    for job in jobs:
        folder = PRIMARY / key(job)
        ref = read(folder / "reference_diagnostic.json")
        expected = "empirical" if ref["capped_fraction"] >= .01 else "physical"
        if ref["kind"] != expected:
            raise ValueError(f"The fixed reference gate differs from training-label rule for {key(job)}")
        references.append(dict(
            **job, capped_fraction=ref["capped_fraction"], kind=ref["kind"], pairs=ref["pairs"],
            fraction_cap_adjustment=ref.get("fraction_cap_adjustment", 0),
            validation_empirical=ref["validation"]["empirical"],
            validation_historical=ref["validation"]["historical"],
        ))
        pairs = list(primary_learners.items())
        if job["architecture"] == "tcn":
            pairs.extend(("shr", learner) for learner in ALTERNATE_LEARNERS)
        for method, learner in pairs:
            chosen = read(SELECTION / key(job) / f"{method}_{learner}.json")["best"]
            selections.append(dict(
                **job, method=method, learner=learner, candidate=chosen["id"],
                validation_area=chosen["validation_area"],
                parameters=json.dumps(chosen["parameters"], sort_keys=True),
            ))
        selrej = read(SELREJ / key(job) / "selection.json")
        neural.append(dict(**job, **selrej["best"], canonical_replay_max_error=selrej["canonical_replay_max_error"]))
    return {
        "reference_diagnostics": pd.DataFrame(references),
        "selected_parameters": pd.DataFrame(selections),
        "selrej_selected_parameters": pd.DataFrame(neural),
    }


def main():
    design = read(EXPERIMENTS / "design.json")
    jobs = design["jobs"]
    primary_learners = design["primary_learners"]
    for directory in (PRIMARY, SELREJ):
        missing = next((key(job) for job in jobs if not (directory / key(job) / "complete.json").is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"Complete the {directory.name} stage before summarizing: {missing}")
    if not (METRICS / "bootstrap_draws.npz").is_file():
        raise FileNotFoundError("Complete the paired SelRej bootstrap before summarizing.")
    frames = {name: gather(name, jobs) for name in RESULT_TYPES}
    primary_tcn = check_metrics(frames)
    frames["learner_metrics"] = gather_learners(jobs, primary_tcn, primary_learners)
    diagnostics = collect_diagnostics(jobs, primary_learners)
    METRICS.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(METRICS / f"{name}.parquet", index=False)
    for name, frame in diagnostics.items():
        frame.to_csv(DIAGNOSTICS / f"{name}.csv", index=False)
    write(DIAGNOSTICS / "verification.json", dict(
        registered_jobs=len(jobs), offline_rows=len(frames["offline"]),
        primary_tcn_configurations=len(primary_tcn) // 6, learner_rows=len(frames["learner_metrics"]),
        all_ten_repeats=True, duplicate_configuration_rows=False, primary_learner_metrics_match=True,
        score_loss_decomposition_verified=True, reference_gate_matches_training_rule=True,
        fixed_predictor_shared_across_selectors=True, selrej_validation_selected=True,
    ))
    print("Held-out configuration metrics and diagnostics written.", flush=True)


if __name__ == "__main__":
    main()
