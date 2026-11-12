from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left_padding = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(inputs, (left_padding, 0)))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(F.gelu(self.norm1(self.conv1(inputs))))
        hidden = self.dropout(F.gelu(self.norm2(self.conv2(hidden))))
        return inputs + hidden


class MultiHeadTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        rul_cap: int | None = 125,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.rul_cap = rul_cap
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualTCNBlock(channels, kernel_size, dilation, dropout) for dilation in dilations]
        )
        self.point_head = nn.Linear(channels, 1)
        self.quantile_head = nn.Linear(channels, 3)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.input_projection(inputs.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        embedding = hidden[:, :, -1]
        raw_point = self.point_head(embedding).squeeze(-1)
        raw_center, raw_lower, raw_upper = self.quantile_head(embedding).unbind(dim=-1)
        point = F.softplus(raw_point)
        center = F.softplus(raw_center)
        q_low = torch.clamp(center - F.softplus(raw_lower), min=0.0)
        q_high = center + F.softplus(raw_upper)
        if self.rul_cap is not None:
            point = torch.clamp(point, max=float(self.rul_cap))
            q_low = torch.clamp(q_low, max=float(self.rul_cap))
            q_high = torch.clamp(q_high, max=float(self.rul_cap))
        return point, q_low, q_high, embedding
