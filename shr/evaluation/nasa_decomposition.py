"""Recompute the manuscript's NASA error-tail analysis from saved query scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shr.paths import EXPERIMENTS

from . import DIAGNOSTICS, METRICS


METHODS = ("shr", "selrej", "ccr")
QUERY_COLUMNS = ("unit_id", "cycle", "corruption_id", "rul_pred", "rul_true")


def fractional_acceptance(score, coverage=.8):
    """Allocate fractional acceptance uniformly across a tie at the cutoff."""
    order = np.argsort(score, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
    sizes = np.diff(np.r_[starts, len(score)])
    fractions = np.clip((coverage * len(score) - starts) / sizes, 0, 1)
    weights = np.empty(len(score))
    weights[order] = np.repeat(fractions, sizes)
    np.testing.assert_allclose(weights.sum(), coverage * len(score))
    return weights


def decompose(primary, selrej, identity):
    fields = list(QUERY_COLUMNS)
    np.testing.assert_array_equal(primary[fields].to_numpy(), selrej[fields].to_numpy())
    error = (primary.rul_pred - primary.rul_true).to_numpy(float)
    rul = primary.rul_true.to_numpy(float)
    nasa = np.expm1(np.where(error < 0, -error / 13, error / 10))
    tails = {
        "e_lt_minus50": error < -50,
        "e_0_50": (error >= 0) & (error <= 50),
        "e_gt50": error > 50,
        "e_gt80": error > 80,
        "e_gt100": error > 100,
        "rul_0_25": rul <= 25,
        "rul_25_75": (rul > 25) & (rul <= 75),
        "rul_75_125": rul > 75,
        "rul_0_25_e_gt80": (rul <= 25) & (error > 80),
    }
    nasa_order = np.argsort(-nasa, kind="stable")
    for method in METHODS:
        scores = selrej if method == "selrej" else primary
        weights = fractional_acceptance(scores[f"score_{method}"].to_numpy())
        accepted = weights.sum()
        row = dict(**identity, method=method, n_queries=len(primary),
                   nasa=float(weights @ nasa / accepted),
                   mse=float(weights @ (error ** 2) / accepted),
                   positive=float(weights @ np.maximum(error, 0) / accepted))
        for name, mask in tails.items():
            row[f"{name}_fraction"] = float(weights @ mask / accepted)
            row[f"{name}_nasa"] = float(weights @ (nasa * mask) / accepted)
        sorted_weights = weights[nasa_order]
        top = np.clip(.01 * accepted - (np.cumsum(sorted_weights) - sorted_weights),
                      0, sorted_weights)
        row["top1pct_nasa"] = float(top @ nasa[nasa_order] / accepted)
        yield row


def collect(jobs, evaluation_dir, selrej_dir):
    rows = []
    for job in jobs:
        if job["architecture"] != "tcn":
            continue
        key = Path(*(str(job[k]) for k in ("benchmark", "cohort", "architecture", "seed", "protocol")))
        scores_dir = evaluation_dir / key / "scores"
        for query_file in sorted(scores_dir.glob("*/*/queries.parquet")):
            dataset_id, scenario = query_file.parent.relative_to(scores_dir).parts
            if scenario == "clean" or scenario.endswith("_high"):
                continue
            other_file = selrej_dir / key / "scores" / dataset_id / scenario / query_file.name
            primary = pd.read_parquet(query_file, columns=[*QUERY_COLUMNS, "score_shr", "score_ccr"])
            selrej = pd.read_parquet(other_file, columns=[*QUERY_COLUMNS, "score_selrej"])
            rows.extend(decompose(primary, selrej, dict(**job, dataset_id=dataset_id, scenario=scenario)))
    frame = pd.DataFrame(rows)
    if len(frame) != 1320:
        raise ValueError(f"Expected 1,320 method/configuration rows, found {len(frame)}")
    return frame


def verify_offline_metrics(decomposition, offline):
    """Check the independent tail calculation against the generated offline results."""
    keys = ["benchmark", "dataset_id", "scenario", "repetition"]
    matched = decomposition.merge(
        offline[offline.architecture.eq("tcn") & offline.core],
        left_on=[*keys, "method"], right_on=[*keys, "variant"], validate="one_to_one",
    )
    if len(matched) != len(decomposition):
        raise ValueError(f"Offline metrics match {len(matched)} of {len(decomposition)} decomposition rows")
    np.testing.assert_allclose(matched.nasa, matched.nasa_c80, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(matched.mse, matched.rmse_c80 ** 2, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(matched.positive, matched.positive_c80, rtol=1e-10, atol=1e-9)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENTS)
    parser.add_argument("--offline-metrics", type=Path, default=METRICS / "offline.parquet")
    parser.add_argument("--output", type=Path, default=DIAGNOSTICS / "nasa_decomposition.csv")
    args = parser.parse_args(argv)
    jobs = json.loads((args.experiment_dir / "design.json").read_text(encoding="utf-8"))["jobs"]
    frame = collect(jobs, args.experiment_dir / "primary", args.experiment_dir / "selrej")
    verify_offline_metrics(frame, pd.read_parquet(args.offline_metrics))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} NASA decomposition rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
