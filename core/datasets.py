"""Canonical dataset names."""

from __future__ import annotations

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
DIGIT_DATASET_KEYS = ("digits8_mini",)
SYNTHETIC_TASK_KEYS = ("subject_verb_agreement", "loss_landscapes")
DATASET_KEYS = DISTRIBUTION_DATASET_KEYS + TEXT_DATASET_KEYS + IMAGE_DATASET_KEYS + DIGIT_DATASET_KEYS + SYNTHETIC_TASK_KEYS


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
    "digits8_mini": "Digits - 8x8 Mini",
    "subject_verb_agreement": "Text - Subject Verb Agreement",
    "loss_landscapes": "Optimization - Loss Landscapes",
}


DATASET_KEYS_BY_DISPLAY_NAME: dict[str, str] = {name: key for key, name in DATASET_DISPLAY_NAMES.items()}


def dataset_key(value: object) -> str:
    name = str(value).strip()
    if name in DATASET_KEYS_BY_DISPLAY_NAME:
        return DATASET_KEYS_BY_DISPLAY_NAME[name]
    valid = ", ".join(DATASET_DISPLAY_NAMES.values())
    raise ValueError(f"Unknown dataset '{value}'. Use exactly one of: {valid}")


def dataset_display_name(key: object) -> str:
    key_text = str(key).strip()
    if key_text not in DATASET_DISPLAY_NAMES:
        valid = ", ".join(DATASET_KEYS)
        raise ValueError(f"Unknown dataset key '{key}'. Valid keys: {valid}")
    return DATASET_DISPLAY_NAMES[key_text]


def canonical_dataset_name(value: object) -> str:
    return DATASET_DISPLAY_NAMES[dataset_key(value)]


def dataset_display_names(keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(dataset_display_name(key) for key in keys)
