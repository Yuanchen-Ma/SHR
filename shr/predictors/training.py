from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from shr.predictors.losses import multi_head_loss
from shr.predictors.registry import architecture_config, architecture_name, make_rul_model
from shr.randomness import derive_seed, seed_everything


@dataclass
class TrainingResult:
    model: nn.Module
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float]]


def resolve_device(requested: str | None = None) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but the active PyTorch installation cannot access CUDA. "
            "Run with 'cpu' or install a CUDA-enabled PyTorch build in this environment."
        )
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(input_dim: int, config: dict[str, Any]) -> nn.Module:
    return make_rul_model(input_dim, config)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    progress_label: str | None = None,
) -> TrainingResult:
    seed_everything(seed)
    model = make_model(train_x.shape[-1], config).to(device)
    model_config = architecture_config(config)
    progress_label = progress_label or architecture_name(config).upper()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    train_loader = _loader(train_x, train_y, int(model_config["batch_size"]), True, seed)
    validation_loader = _loader(
        validation_x, validation_y, int(model_config["batch_size"]), False, seed
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, float]] = []
    epoch_progress = tqdm(
        range(1, int(model_config["max_epochs"]) + 1),
        desc=progress_label,
        unit="epoch",
        leave=False,
        dynamic_ncols=True,
    )
    for epoch in epoch_progress:
        train_loss = _train_epoch(model, train_loader, optimizer, config, device)
        validation_loss = _evaluate_loss(model, validation_loader, config, device)
        history.append(
            {"epoch": float(epoch), "train_loss": train_loss, "val_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        epoch_progress.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{validation_loss:.4f}",
            best=f"{best_loss:.4f}",
            patience=f"{patience}/{int(model_config['early_stop_patience'])}",
        )
        if patience >= int(model_config["early_stop_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(model, best_epoch, best_loss, history)


def train_ensemble(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    progress_label: str | None = None,
) -> list[TrainingResult]:
    ensemble_size = int(config["model"]["ensemble_size"])
    progress_label = progress_label or f"{architecture_name(config).upper()} ensemble"
    results: list[TrainingResult] = []
    for member in range(ensemble_size):
        results.append(
            train_model(
                train_x,
                train_y,
                validation_x,
                validation_y,
                config,
                derive_seed(seed, "ensemble", member),
                device,
                progress_label=f"{progress_label} M{member + 1}/{ensemble_size}",
            )
        )
    return results


def train_ensemble_fixed_epochs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: list[int],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    progress_label: str | None = None,
) -> list[TrainingResult]:
    results: list[TrainingResult] = []
    ensemble_size = len(epochs)
    progress_label = progress_label or f"Final {architecture_name(config).upper()} ensemble"
    for member, member_epochs in enumerate(epochs):
        member_seed = derive_seed(seed, "ensemble", member)
        seed_everything(member_seed)
        model = make_model(train_x.shape[-1], config).to(device)
        model_config = architecture_config(config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        loader = _loader(train_x, train_y, int(model_config["batch_size"]), True, member_seed)
        history: list[dict[str, float]] = []
        epoch_progress = tqdm(
            range(1, max(1, member_epochs) + 1),
            desc=f"{progress_label} M{member + 1}/{ensemble_size}",
            unit="epoch",
            leave=False,
            dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            loss = _train_epoch(model, loader, optimizer, config, device)
            history.append({"epoch": float(epoch), "train_loss": loss})
            epoch_progress.set_postfix(train=f"{loss:.4f}")
        results.append(TrainingResult(model, max(1, member_epochs), float("nan"), history))
    return results


def predict_ensemble(
    models: list[nn.Module],
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if not models:
        raise ValueError("Ensemble is empty")
    member_points: list[np.ndarray] = []
    member_lows: list[np.ndarray] = []
    member_highs: list[np.ndarray] = []
    member_embeddings: list[np.ndarray] = []
    tensor = torch.from_numpy(features.astype(np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    for model in models:
        model.to(device).eval()
        points: list[np.ndarray] = []
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                point, low, high, embedding = model(batch.to(device))
                points.append(point.cpu().numpy())
                lows.append(low.cpu().numpy())
                highs.append(high.cpu().numpy())
                embeddings.append(embedding.cpu().numpy())
        member_points.append(np.concatenate(points))
        member_lows.append(np.concatenate(lows))
        member_highs.append(np.concatenate(highs))
        member_embeddings.append(np.concatenate(embeddings))
    point_matrix = np.stack(member_points)
    return {
        "rul_pred": point_matrix.mean(axis=0),
        "q_low": np.stack(member_lows).mean(axis=0),
        "q_high": np.stack(member_highs).mean(axis=0),
        "ensemble_std": point_matrix.std(axis=0, ddof=1)
        if len(models) > 1
        else np.zeros(len(features)),
        "embedding": np.stack(member_embeddings).mean(axis=0),
    }


def save_ensemble(results: list[TrainingResult], directory: Path, metadata: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results):
        torch.save(
            {
                "state_dict": result.model.state_dict(),
                "input_dim": result.model.input_dim,
                "architecture": metadata.get("architecture"),
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "metadata": metadata,
            },
            directory / f"member_{index}.pt",
        )


def _loader(
    features: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)), torch.from_numpy(targets.astype(np.float32))
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        point, low, high, _ = model(features)
        loss = _loss(point, low, high, targets, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["model"]["gradient_clip_norm"])
        )
        optimizer.step()
        total += float(loss.detach()) * len(features)
        count += len(features)
    return total / count


def _evaluate_loss(
    model: nn.Module, loader: DataLoader, config: dict[str, Any], device: torch.device
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            point, low, high, _ = model(features)
            loss = _loss(point, low, high, targets, config)
            total += float(loss) * len(features)
            count += len(features)
    return total / count


def _loss(
    point: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    low_q, high_q = (float(value) for value in config["model"]["quantiles"])
    return multi_head_loss(
        point,
        low,
        high,
        targets,
        low_q,
        high_q,
        float(config["model"]["quantile_loss_weight"]),
    )


class QuietProgress:
    """Minimal progress iterator used by the benchmark workflows."""

    def __init__(self, items, **kwargs):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def set_postfix(self, **kwargs):
        pass


def load_models(folder: Path, config: dict[str, Any], device: torch.device) -> list[nn.Module]:
    models = []
    for path in sorted(folder.glob("member_*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = make_rul_model(state["input_dim"], config)
        model.load_state_dict(state["state_dict"])
        models.append(model.to(device))
    return models
