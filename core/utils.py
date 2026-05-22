"""Small general-purpose helpers."""

from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
    except Exception:
        return


def minibatches(count: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    indices = rng.permutation(int(count))
    size = max(1, int(batch_size))
    for start in range(0, int(count), size):
        yield indices[start : start + size]


def compact_float(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def as_path_string(value: object) -> str:
    return str(value).replace("\\", "/")

