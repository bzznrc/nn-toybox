"""Tiny classifier used by grad."""

from __future__ import annotations

import torch
from torch import nn


def activation_layer(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'.")


class TinyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 32,
        hidden_layers: int = 2,
        num_classes: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(max(1, int(hidden_layers))):
            layers.append(nn.Linear(in_dim, int(hidden_size)))
            layers.append(activation_layer(activation))
            in_dim = int(hidden_size)
        layers.append(nn.Linear(in_dim, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(config: dict[str, object], num_classes: int) -> TinyMLP:
    return TinyMLP(
        hidden_size=int(config.get("hidden_size", 32)),
        hidden_layers=int(config.get("hidden_layers", 2)),
        num_classes=int(num_classes),
        activation=str(config.get("activation", "relu")),
    )

