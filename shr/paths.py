"""Locations shared by preparation, training and evaluation."""
from pathlib import Path

from .io import key, read, sha, write


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
WORK = ROOT / "experiments"
EXPERIMENTS = WORK / "evaluation"
PRIOR_DIR = WORK / "learner_comparison"
PRIOR_RESULTS = PRIOR_DIR / "configuration_metrics.parquet"
RESULTS = ROOT / "results"


def check_run_settings():
    """Keep a resumed run tied to the same experimental configuration."""
    names = ("experiment.json", "selrej.json", "prediction.yaml", "backbones.yaml", "history.yaml")
    current = {name: sha(CONFIGS / name) for name in names}
    record = WORK / "run_settings.json"
    if record.is_file():
        previous = read(record)
        changed = [name for name in names if current[name] != previous.get(name)]
        if changed:
            raise ValueError(
                f"Configuration changed since this run started ({', '.join(changed)}). "
                "Restore the previous settings to resume, or run in a fresh checkout."
            )
    else:
        write(record, current)


def model_dir(job):
    return WORK / "risk_models" / key(job)


def prediction_dir(job):
    return WORK / "predictions" / key(job)


def partition_path(job):
    # Both architectures use the same reserved engines.
    return WORK / "partitions" / key(dict(job, architecture="tcn")) / "calibration.json"


def official_path(job):
    return prediction_dir(job) / "official.parquet"
