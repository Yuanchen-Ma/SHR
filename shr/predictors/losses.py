from __future__ import annotations

import torch
from torch.nn import functional as F


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    residual = target - prediction
    return torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()


def multi_head_loss(
    point: torch.Tensor,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    target: torch.Tensor,
    low_quantile: float,
    high_quantile: float,
    quantile_weight: float,
) -> torch.Tensor:
    return (
        F.huber_loss(point, target, delta=1.0)
        + quantile_weight * pinball_loss(q_low, target, low_quantile)
        + quantile_weight * pinball_loss(q_high, target, high_quantile)
    )
