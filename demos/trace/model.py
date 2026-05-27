"""Tiny MLP used by the trace demo."""

from __future__ import annotations

import torch
from torch import nn


class TinyTraceMLP(nn.Module):
    def __init__(self, hidden1: int = 32, hidden2: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, int(hidden1))
        self.fc2 = nn.Linear(int(hidden1), int(hidden2))
        self.fc3 = nn.Linear(int(hidden2), 10)

    def forward_with_activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = x.reshape(x.shape[0], -1)
        z1 = self.fc1(flat)
        h1 = torch.relu(z1)
        z2 = self.fc2(h1)
        h2 = torch.relu(z2)
        logits = self.fc3(h2)
        return {
            "input": flat,
            "hidden1": h1,
            "hidden2": h2,
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_activations(x)["logits"]
