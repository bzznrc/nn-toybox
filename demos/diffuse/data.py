"""Generated point-cloud datasets for diffuse."""

from __future__ import annotations

import numpy as np

from core.generated_2d import DIFFUSION_DATASETS, make_point_cloud


DATASETS = DIFFUSION_DATASETS


def make_dataset(
    name: str = "gaussian_mixtures",
    n: int = 1024,
    noise: float = 0.04,
    seed: int = 0,
) -> np.ndarray:
    return make_point_cloud(name=name, n=int(n), noise=float(noise), seed=int(seed))
