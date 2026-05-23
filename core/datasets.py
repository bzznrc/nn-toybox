"""Canonical dataset names and backwards-compatible aliases."""

from __future__ import annotations

import re
from typing import Iterable


DISTRIBUTION_DATASET_KEYS = (
    "blobs",
    "gaussian_mixtures",
    "circles",
    "rings",
    "moons",
    "xor",
    "checkerboard",
    "spirals",
)
TEXT_DATASET_KEYS = ("tiny_semantics", "big_tiny_semantics")
IMAGE_DATASET_KEYS = ("icons", "patterns")
DATASET_KEYS = DISTRIBUTION_DATASET_KEYS + TEXT_DATASET_KEYS + IMAGE_DATASET_KEYS


DATASET_DISPLAY_NAMES: dict[str, str] = {
    "blobs": "Distributions - Blobs",
    "gaussian_mixtures": "Distributions - Gaussian Mixtures",
    "circles": "Distributions - Circles",
    "rings": "Distributions - Rings",
    "moons": "Distributions - Moons",
    "xor": "Distributions - XOR",
    "checkerboard": "Distributions - Checkerboard",
    "spirals": "Distributions - Spiral",
    "tiny_semantics": "Text - Tiny Semantics",
    "big_tiny_semantics": "Text - Big Tiny Semantics",
    "icons": "Images - Icons",
    "patterns": "Images - Patterns",
}


def _alias_key(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


DATASET_ALIASES: dict[str, str] = {
    "moon": "moons",
    "moons": "moons",
    "distribution_moon": "moons",
    "distribution_moons": "moons",
    "distributions_moon": "moons",
    "distributions_moons": "moons",
    "circle": "circles",
    "circles": "circles",
    "distribution_circle": "circles",
    "distribution_circles": "circles",
    "distributions_circle": "circles",
    "distributions_circles": "circles",
    "spiral": "spirals",
    "spirals": "spirals",
    "distribution_spiral": "spirals",
    "distribution_spirals": "spirals",
    "distributions_spiral": "spirals",
    "distributions_spirals": "spirals",
    "xor": "xor",
    "distribution_xor": "xor",
    "distributions_xor": "xor",
    "blob": "blobs",
    "blobs": "blobs",
    "distribution_blob": "blobs",
    "distribution_blobs": "blobs",
    "distributions_blob": "blobs",
    "distributions_blobs": "blobs",
    "checkerboard": "checkerboard",
    "checker_board": "checkerboard",
    "distribution_checkerboard": "checkerboard",
    "distribution_checker_board": "checkerboard",
    "distributions_checkerboard": "checkerboard",
    "distributions_checker_board": "checkerboard",
    "ring": "rings",
    "rings": "rings",
    "distribution_ring": "rings",
    "distribution_rings": "rings",
    "distributions_ring": "rings",
    "distributions_rings": "rings",
    "gaussian_mixture": "gaussian_mixtures",
    "gaussian_mixtures": "gaussian_mixtures",
    "distribution_gaussian_mixture": "gaussian_mixtures",
    "distribution_gaussian_mixtures": "gaussian_mixtures",
    "distributions_gaussian_mixture": "gaussian_mixtures",
    "distributions_gaussian_mixtures": "gaussian_mixtures",
    "tiny_semantic": "tiny_semantics",
    "tiny_semantics": "tiny_semantics",
    "text_tiny_semantic": "tiny_semantics",
    "text_tiny_semantics": "tiny_semantics",
    "big_tiny_semantic": "big_tiny_semantics",
    "big_tiny_semantics": "big_tiny_semantics",
    "big_tiny": "big_tiny_semantics",
    "text_big_tiny": "big_tiny_semantics",
    "text_big_tiny_semantic": "big_tiny_semantics",
    "text_big_tiny_semantics": "big_tiny_semantics",
    "large_tiny_semantic": "big_tiny_semantics",
    "large_tiny_semantics": "big_tiny_semantics",
    "expanded_tiny_semantic": "big_tiny_semantics",
    "expanded_tiny_semantics": "big_tiny_semantics",
    "icon": "icons",
    "icons": "icons",
    "image_icon": "icons",
    "image_icons": "icons",
    "images_icon": "icons",
    "images_icons": "icons",
    "generated_icon": "icons",
    "generated_icons": "icons",
    "pattern": "patterns",
    "patterns": "patterns",
    "image_pattern": "patterns",
    "image_patterns": "patterns",
    "images_pattern": "patterns",
    "images_patterns": "patterns",
    "generated_pattern": "patterns",
    "generated_patterns": "patterns",
}


def dataset_key(value: object) -> str:
    key = _alias_key(value)
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    valid = ", ".join(DATASET_DISPLAY_NAMES.values())
    raise ValueError(f"Unknown dataset '{value}'. Valid: {valid}")


def dataset_display_name(value: object) -> str:
    return DATASET_DISPLAY_NAMES[dataset_key(value)]


def dataset_display_names(keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(dataset_display_name(key) for key in keys)
