from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F


class MultiHeadTransformer(nn.Module):
    """Causal encoder over the observed window, with sinusoidal position encoding.

    The final observed token supplies the same three prediction heads and latent
    embedding as TCN/LSTM/GRU. Each encoder layer is constructed independently;
    no cloned layers share identical initial parameter values. Existing missing
    value masks and elapsed-time inputs remain model inputs, not padding masks.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        rul_cap: int | None = 125,
    ) -> None:
        super().__init__()
        if min(input_dim, d_model, nhead, num_layers, dim_feedforward) < 1:
            raise ValueError("Transformer dimensions and layer count must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.input_dim = input_dim
        self.d_model = d_model
        self.rul_cap = rul_cap
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.point_head = nn.Linear(d_model, 1)
        self.quantile_head = nn.Linear(d_model, 3)

    def _position_encoding(self, length: int, reference: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(length, device=reference.device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=reference.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = positions * frequencies
        encoding = torch.empty(length, self.d_model, device=reference.device)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :self.d_model // 2].cos()
        return encoding.to(dtype=reference.dtype)

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] == 0 or inputs.shape[2] != self.input_dim:
            raise ValueError("Expected nonempty [batch, time, input_dim] inputs")
        hidden = self.input_projection(inputs)
        hidden = self.input_dropout(hidden + self._position_encoding(inputs.shape[1], hidden))
        mask = torch.ones(
            inputs.shape[1], inputs.shape[1], device=inputs.device, dtype=torch.bool
        ).triu(diagonal=1)
        # The fused memory-efficient CUDA backward can remain nondeterministic
        # under the shared trainer's warn_only=True setting. These short windows
        # use math SDPA to preserve reproducibility for the fixed paper seeds.
        with sdpa_kernel(SDPBackend.MATH):
            for layer in self.layers:
                hidden = layer(hidden, src_mask=mask)
        return self.output_norm(hidden)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.output_dropout(self.encode_sequence(inputs)[:, -1, :])
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
