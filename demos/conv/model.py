"""Tiny CNN used by the conv demo."""

from __future__ import annotations

import torch
from torch import nn


class TinyConvNet(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(1, c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.head = nn.Linear(c * 8 * 8, 10)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f1 = torch.relu(self.conv1(x))
        f2 = torch.relu(self.conv2(f1))
        logits = self.head(f2.reshape(f2.shape[0], -1))
        return {
            "conv1": f1,
            "conv2": f2,
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["logits"]
