"""Small PyTorch helpers shared by trainers."""

from __future__ import annotations

import torch


def torch_device(name: str) -> torch.device:
    if str(name).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
