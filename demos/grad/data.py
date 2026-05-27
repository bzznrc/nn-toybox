"""Generated 2D datasets for the grad module."""

from __future__ import annotations

import numpy as np

from core.generated_2d import CLASSIFICATION_DATASETS, make_2d_classification


DATASETS = CLASSIFICATION_DATASETS


def make_dataset(
    name: str = "moons",
    n: int = 512,
    noise: float = 0.08,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    return make_2d_classification(name=name, n=int(n), noise=float(noise), seed=int(seed))
