"""Generated icon dataset for autoencode."""

from __future__ import annotations

import numpy as np

from core.generated_icons import SHAPE_NAMES, make_icons


DATASETS = ("icons",)


def make_dataset(
    name: str = "icons",
    n: int = 512,
    size: int = 16,
    noise: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if str(name) != "icons":
        raise ValueError("V1 autoencode only ships the generated icons dataset.")
    return make_icons(n=int(n), size=int(size), seed=int(seed), noise=float(noise))


__all__ = ["DATASETS", "SHAPE_NAMES", "make_dataset"]

