"""Generated image datasets for autoencode."""

from __future__ import annotations

import numpy as np

from core.datasets import IMAGE_DATASET_KEYS, dataset_display_names, dataset_key
from core.generated_icons import PATTERN_NAMES, SHAPE_NAMES, make_icons, make_patterns


DATASETS = dataset_display_names(IMAGE_DATASET_KEYS)


def make_dataset(
    name: str = "Images - Icons",
    n: int = 512,
    size: int = 16,
    noise: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    key = dataset_key(name)
    if key == "icons":
        return make_icons(n=int(n), size=int(size), seed=int(seed), noise=float(noise))
    if key == "patterns":
        return make_patterns(n=int(n), size=int(size), seed=int(seed), noise=float(noise))
    valid = ", ".join(DATASETS)
    raise ValueError(f"Unknown autoencode dataset '{name}'. Valid: {valid}")


__all__ = ["DATASETS", "PATTERN_NAMES", "SHAPE_NAMES", "make_dataset"]
