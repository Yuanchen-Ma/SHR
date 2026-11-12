from __future__ import annotations

from typing import Any

from torch import nn

from shr.predictors.tcn import MultiHeadTCN
from shr.predictors.transformer import MultiHeadTransformer

SUPPORTED_RUL_ARCHITECTURES = ("tcn", "transformer")


def architecture_name(config: dict[str, Any]) -> str:
    name = str(config["model"].get("architecture", "tcn")).lower()
    if name not in SUPPORTED_RUL_ARCHITECTURES:
        raise ValueError(
            f"Unknown RUL architecture {name!r}; expected one of {SUPPORTED_RUL_ARCHITECTURES}"
        )
    return name


def architecture_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return common model settings overlaid by the selected architecture section.

    Legacy flat TCN configurations remain valid. New configurations may place
    architecture-specific values under ``model.<architecture>``.
    """

    name = architecture_name(config)
    values = dict(config["model"])
    nested = values.get(name, {})
    if nested is not None and not isinstance(nested, dict):
        raise TypeError(f"model.{name} must be a mapping")
    values.update(nested or {})
    return values


def make_rul_model(input_dim: int, config: dict[str, Any]) -> nn.Module:
    name = architecture_name(config)
    values = architecture_config(config)
    rul_cap = config["experiment"].get("rul_cap")
    if name == "tcn":
        return MultiHeadTCN(
            input_dim=input_dim,
            channels=int(values["channels"]),
            kernel_size=int(values["kernel_size"]),
            dilations=tuple(int(value) for value in values["dilations"]),
            dropout=float(values["dropout"]),
            rul_cap=rul_cap,
        )
    if name == "transformer":
        return MultiHeadTransformer(
            input_dim=input_dim,
            d_model=int(values["d_model"]),
            nhead=int(values["nhead"]),
            num_layers=int(values["num_layers"]),
            dim_feedforward=int(values["dim_feedforward"]),
            dropout=float(values["dropout"]),
            rul_cap=rul_cap,
        )
    raise AssertionError(name)



def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
