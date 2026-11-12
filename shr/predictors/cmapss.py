"""Fit C-MAPSS predictor ensembles and generate out-of-fold and held-out tables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from shr.data.preprocessing import FoldPreprocessor
from shr.data.splits import group_train_validation_split, grouped_kfold, make_unit_split
from shr.data.windows import build_windows, prepare_trajectories, records_to_arrays
from shr.predictors import training
from shr.predictors.cache import cached_trainer
from shr.predictors.query import ConditionedLatentDistance, assert_oof_units, build_risk_table
from shr.predictors.registry import architecture_name
from shr.predictors.training import QuietProgress, predict_ensemble, save_ensemble
from shr.randomness import derive_seed, seed_everything
from shr.schemas import WindowRecord


@dataclass(frozen=True)
class ScenarioProtocol:
    name: str
    train_types: tuple[str, ...]
    test_scenarios: tuple[tuple[str, str], ...]
    train_severities: tuple[str, ...] = ("low", "medium")
    calibration_scenarios: tuple[tuple[str, str], ...] | None = None
    held_out_type: str | None = None


def matched_protocol(config: dict[str, Any]) -> ScenarioProtocol:
    corruption_types = tuple(config["corruption"]["types"])
    return ScenarioProtocol(
        name="matched",
        train_types=corruption_types,
        test_scenarios=(
            ("clean", "none"),
            *((corruption_type, "medium") for corruption_type in corruption_types),
        ),
    )


def held_out_protocols(config: dict[str, Any]) -> list[ScenarioProtocol]:
    corruption_types = tuple(config["corruption"]["types"])
    seen = (("clean", "none"),) + tuple(
        (corruption_type, severity)
        for corruption_type in corruption_types
        for severity in config["corruption"]["seen_severities"]
    )
    protocols = []
    for held_out in corruption_types:
        remaining = tuple(value for value in corruption_types if value != held_out)
        calibration = tuple(
            (kind, severity) for kind, severity in seen if kind == "clean" or kind in remaining
        )
        protocols.append(
            ScenarioProtocol(
                name=f"full_loto_{held_out}",
                train_types=remaining,
                test_scenarios=((held_out, "medium"),),
                calibration_scenarios=calibration,
                held_out_type=held_out,
            )
        )
    return protocols


def fit_backbone(
    config: dict[str, Any],
    frame: pd.DataFrame,
    split: Any,
    protocol: ScenarioProtocol,
    output: Path,
    device: torch.device,
    signature: str,
) -> None:
    fit_ensemble = cached_trainer(output, training.train_ensemble, signature)
    fit_fixed_epochs = cached_trainer(output, training.train_ensemble_fixed_epochs, signature)
    seed = int(split.seed)
    seed_everything(seed)
    development = _select_units(frame, split.development_units)
    calibration = _select_units(frame, split.threshold_calibration_units)
    locked_test = _select_units(frame, split.outer_test_units)
    oof_tables: list[pd.DataFrame] = []
    fold_training_units: dict[int, set[int]] = {}
    train_scenarios = _risk_training_scenarios(protocol)
    context = f"{config['experiment']['dataset_id']}/{protocol.name}/{split.split_id}"

    folds = list(
        grouped_kfold(split.development_units, int(config["split"]["inner_folds"]), seed)
    )
    fold_progress = QuietProgress(
        enumerate(folds),
        total=len(folds),
        desc=f"{context} OOF folds",
        unit="fold",
        leave=False,
        dynamic_ncols=True,
    )
    for fold_id, (inner_train_units, oof_units) in fold_progress:
        fold_progress.set_postfix(stage="preprocess", fold=f"{fold_id + 1}/{len(folds)}")
        fold_seed = derive_seed(seed, "fold", fold_id)
        fold_training_units[fold_id] = {int(unit) for unit in inner_train_units}
        inner_train = _select_units(development, inner_train_units)
        oof = _select_units(development, oof_units)
        preprocessor = make_preprocessor(config, fold_seed).fit(inner_train)
        transformed_train = preprocessor.transform(inner_train)
        transformed_oof = preprocessor.transform(oof)
        fit_units, early_units = group_train_validation_split(
            inner_train_units, float(config["split"]["early_stop_fraction"]), fold_seed
        )
        train_records = _training_windows(
            _select_units(transformed_train, fit_units), preprocessor, protocol, config, fold_seed
        )
        validation_records, _ = _scenario_windows(
            _select_units(transformed_train, early_units),
            preprocessor,
            train_scenarios,
            config,
            derive_seed(fold_seed, "early"),
            False,
        )
        train_x, train_y, _ = records_to_arrays(train_records)
        validation_x, validation_y, _ = records_to_arrays(validation_records)
        fold_progress.set_postfix(
            stage=f"train {architecture_name(config).upper()}",
            fold=f"{fold_id + 1}/{len(folds)}",
        )
        ensemble = fit_ensemble(
            train_x,
            train_y,
            validation_x,
            validation_y,
            config,
            fold_seed,
            device,
            progress_label=f"{context} OOF{fold_id + 1}",
        )
        save_ensemble(
            ensemble,
            output / "checkpoints" / f"inner_fold_{fold_id}",
            {
                "architecture": architecture_name(config),
                "training_units": [int(unit) for unit in inner_train_units],
                "fold_id": fold_id,
            },
        )
        distance_model = fit_distance_model(
            transformed_train,
            preprocessor,
            ensemble,
            config,
            derive_seed(fold_seed, "distance"),
            device,
        )
        fold_progress.set_postfix(stage="OOF inference", fold=f"{fold_id + 1}/{len(folds)}")
        oof_records, _ = _scenario_windows(
            transformed_oof,
            preprocessor,
            train_scenarios,
            config,
            derive_seed(fold_seed, "oof"),
            False,
        )
        oof_x, _, _ = records_to_arrays(oof_records)
        predictions = predict_ensemble(
            [result.model for result in ensemble],
            oof_x,
            int(config["model"]["batch_size"]),
            device,
        )
        oof_tables.append(
            build_risk_table(
                oof_records,
                predictions,
                distance_model,
                split.split_id,
                fold_id,
                inner_train_units,
                float(config["risk"]["severe_delta"]),
                float(config["risk"]["critical_horizon"]),
                config["experiment"]["dataset_id"],
            )
        )

    oof_table = pd.concat(oof_tables, ignore_index=True)
    assert_oof_units(oof_table, fold_training_units)
    oof_table.to_parquet(output / "oof_risk_table.parquet", index=False)
    final_preprocessor = make_preprocessor(config, derive_seed(seed, "final")).fit(development)
    final_development = final_preprocessor.transform(development)
    fit_units, early_units = group_train_validation_split(
        split.development_units,
        float(config["split"]["early_stop_fraction"]),
        derive_seed(seed, "final_early"),
    )
    tuning_train = _training_windows(
        _select_units(final_development, fit_units),
        final_preprocessor,
        protocol,
        config,
        derive_seed(seed, "tune_train"),
    )
    tuning_validation, _ = _scenario_windows(
        _select_units(final_development, early_units),
        final_preprocessor,
        train_scenarios,
        config,
        derive_seed(seed, "tune_validation"),
        False,
    )
    tuning_x, tuning_y, _ = records_to_arrays(tuning_train)
    validation_x, validation_y, _ = records_to_arrays(tuning_validation)
    tuning_results = fit_ensemble(
        tuning_x,
        tuning_y,
        validation_x,
        validation_y,
        config,
        derive_seed(seed, "final_tune"),
        device,
        progress_label=f"{context} final epoch tuning",
    )
    full_training = _training_windows(
        final_development, final_preprocessor, protocol, config, derive_seed(seed, "full_train")
    )
    full_x, full_y, _ = records_to_arrays(full_training)
    final_ensemble = fit_fixed_epochs(
        full_x,
        full_y,
        [result.best_epoch for result in tuning_results],
        config,
        derive_seed(seed, "final_tune"),
        device,
        progress_label=f"{context} final all-dev fit",
    )
    save_ensemble(
        final_ensemble,
        output / "checkpoints" / "final_base",
        {
            "architecture": architecture_name(config),
            "training_units": list(split.development_units),
            "preprocessor": final_preprocessor.metadata(),
        },
    )
    joblib.dump(final_preprocessor, output / "preprocessing" / "preprocessor.joblib")
    final_distance = fit_distance_model(
        final_development,
        final_preprocessor,
        final_ensemble,
        config,
        derive_seed(seed, "final_distance"),
        device,
    )
    joblib.dump(final_distance, output / "preprocessing" / "latent_distance.joblib")

    calibration_scenarios = protocol.calibration_scenarios or train_scenarios
    transformed_calibration = final_preprocessor.transform(calibration)
    calibration_records, _ = _scenario_windows(
        transformed_calibration,
        final_preprocessor,
        list(calibration_scenarios),
        config,
        derive_seed(seed, "calibration"),
        False,
    )
    calibration_table = _predict_table(
        calibration_records,
        final_ensemble,
        final_distance,
        split,
        config,
        device,
        list(split.development_units),
    )
    calibration_table.to_parquet(output / "calibration_table.parquet", index=False)

    transformed_test = final_preprocessor.transform(locked_test)
    test_records, _ = _scenario_windows(
        transformed_test,
        final_preprocessor,
        list(protocol.test_scenarios),
        config,
        derive_seed(seed, "locked_test", protocol.name),
        False,
    )
    test_table = _predict_table(
        test_records,
        final_ensemble,
        final_distance,
        split,
        config,
        device,
        list(split.development_units),
    )
    test_table.to_parquet(output / "locked_test_table.parquet", index=False)


def _predict_table(
    records: list[WindowRecord],
    ensemble: list[Any],
    distance_model: ConditionedLatentDistance,
    split: Any,
    config: dict[str, Any],
    device: torch.device,
    training_units: list[int],
) -> pd.DataFrame:
    features, _, _ = records_to_arrays(records)
    predictions = predict_ensemble(
        [result.model for result in ensemble],
        features,
        int(config["model"]["batch_size"]),
        device,
    )
    return build_risk_table(
        records,
        predictions,
        distance_model,
        split.split_id,
        -1,
        training_units,
        float(config["risk"]["severe_delta"]),
        float(config["risk"]["critical_horizon"]),
        config["experiment"]["dataset_id"],
    )


def fit_distance_model(
    transformed_train: pd.DataFrame,
    preprocessor: FoldPreprocessor,
    ensemble: list[Any],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> ConditionedLatentDistance:
    records, _ = _scenario_windows(
        transformed_train, preprocessor, [("clean", "none")], config, seed, False
    )
    features, _, _ = records_to_arrays(records)
    predictions = predict_ensemble(
        [result.model for result in ensemble],
        features,
        int(config["model"]["batch_size"]),
        device,
    )
    return ConditionedLatentDistance().fit(
        predictions["embedding"],
        np.array([record.condition_id for record in records], dtype=int),
        min_samples=max(5, int(config["preprocessing"]["min_cluster_samples"]) // 5),
    )


def _training_windows(
    transformed: pd.DataFrame,
    preprocessor: FoldPreprocessor,
    protocol: ScenarioProtocol,
    config: dict[str, Any],
    seed: int,
) -> list[WindowRecord]:
    scenarios = _risk_training_scenarios(protocol)
    records, _ = _scenario_windows(transformed, preprocessor, scenarios, config, seed, True)
    proportions = {
        "clean": 0.50,
        "burst_missing": 0.15,
        "channel_dropout": 0.15,
        "gradual_bias": 0.20,
    }
    available_types = {record.corruption_type for record in records}
    total_probability = sum(proportions.get(kind, 0.0) for kind in available_types)
    max_per_unit = int(config["windowing"]["max_train_windows_per_unit"])
    selected: list[WindowRecord] = []
    rng = np.random.default_rng(seed)
    for unit_records in _group_records(records, "unit_id").values():
        for kind in sorted(available_types):
            candidates = [record for record in unit_records if record.corruption_type == kind]
            target = max(1, round(max_per_unit * proportions.get(kind, 0.0) / total_probability))
            if len(candidates) <= target:
                selected.extend(candidates)
            else:
                indices = rng.choice(len(candidates), target, replace=False)
                selected.extend(candidates[int(index)] for index in indices)
    return selected


def _scenario_windows(
    transformed: pd.DataFrame,
    preprocessor: FoldPreprocessor,
    scenarios: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    config: dict[str, Any],
    seed: int,
    training: bool,
) -> tuple[list[WindowRecord], list[dict[str, object]]]:
    trajectories = []
    manifests: list[dict[str, object]] = []
    for corruption_type, severity in scenarios:
        scenario_trajectories, scenario_manifests = prepare_trajectories(
            transformed,
            preprocessor.active_sensors,
            corruption_type,
            severity,
            derive_seed(seed, corruption_type, severity),
        )
        trajectories.extend(scenario_trajectories)
        manifests.extend(scenario_manifests)
    records = build_windows(
        trajectories,
        int(config["windowing"]["window_length"]),
        int(
            config["windowing"]["train_stride"]
            if training
            else config["windowing"]["inspection_stride"]
        ),
        bool(config["preprocessing"]["include_delta_t"]),
        preprocessor.slope_mean,
        preprocessor.slope_std,
        None if training else int(config["windowing"]["max_eval_windows_per_unit"]),
        seed,
    )
    return records, manifests


def _risk_training_scenarios(protocol: ScenarioProtocol) -> list[tuple[str, str]]:
    return [("clean", "none")] + [
        (corruption_type, severity)
        for corruption_type in protocol.train_types
        for severity in protocol.train_severities
    ]


def make_preprocessor(config: dict[str, Any], seed: int) -> FoldPreprocessor:
    preprocessing = config["preprocessing"]
    return FoldPreprocessor(
        n_condition_clusters=int(preprocessing["n_condition_clusters"]),
        variance_threshold=float(preprocessing["variance_threshold"]),
        min_cluster_samples=int(preprocessing["min_cluster_samples"]),
        epsilon=float(preprocessing["scaler_epsilon"]),
        seed=seed,
    )


def make_split(frame: pd.DataFrame, config: dict[str, Any], seed: int, split_id: str):
    split = config["split"]
    return make_unit_split(
        frame["unit_id"].unique(),
        float(split["outer_test_fraction"]),
        float(split["threshold_cal_fraction_of_remaining"]),
        seed,
        split_id,
    )


def _select_units(frame: pd.DataFrame, units: Any) -> pd.DataFrame:
    selected = frame[frame["unit_id"].isin(list(units))].copy()
    if selected.empty:
        raise ValueError("Unit selection produced an empty frame")
    return selected


def _group_records(records: list[WindowRecord], attribute: str) -> dict[object, list[WindowRecord]]:
    grouped: dict[object, list[WindowRecord]] = {}
    for record in records:
        grouped.setdefault(getattr(record, attribute), []).append(record)
    return grouped
