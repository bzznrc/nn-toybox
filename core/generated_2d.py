"""Deterministic generated 2D datasets used by grad and diffuse."""

from __future__ import annotations

import math

import numpy as np


DISTRIBUTION_VARIANT_KEYS = (
    "blobs",
    "gaussian_mixtures",
    "circles",
    "rings",
    "moons",
    "xor",
    "checkerboard",
    "spirals",
)
DIFFUSION_VARIANT_KEYS = tuple(
    key for key in DISTRIBUTION_VARIANT_KEYS if key in {"gaussian_mixtures", "rings", "moons", "checkerboard", "spirals"}
)
CLASSIFICATION_DATASET_KEYS = DISTRIBUTION_VARIANT_KEYS
DIFFUSION_DATASET_KEYS = DIFFUSION_VARIANT_KEYS
CLASSIFICATION_DATASETS = DISTRIBUTION_VARIANT_KEYS
DIFFUSION_DATASETS = DIFFUSION_VARIANT_KEYS


def normalize_distribution_variant(value: object, *, choices: tuple[str, ...] = DISTRIBUTION_VARIANT_KEYS) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in choices:
        valid = ", ".join(choices)
        raise ValueError(f"Unknown distribution variant '{value}'. Valid: {valid}")
    return key


def _standardize(points: np.ndarray, scale: float = 1.3) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.percentile(np.linalg.norm(centered, axis=1), 95)
    if radius > 0:
        centered = centered / radius * float(scale)
    return centered.astype(np.float32)


def _add_noise(points: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    if float(noise) <= 0:
        return points
    return points + rng.normal(0.0, float(noise), size=points.shape).astype(np.float32)


def make_moons(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n0 = int(n) // 2
    n1 = int(n) - n0
    t0 = rng.uniform(0.0, math.pi, size=n0)
    t1 = rng.uniform(0.0, math.pi, size=n1)
    moon0 = np.column_stack([np.cos(t0), np.sin(t0)])
    moon1 = np.column_stack([1.0 - np.cos(t1), 0.45 - np.sin(t1)])
    points = np.vstack([moon0, moon1]).astype(np.float32)
    labels = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.int64)
    points = _standardize(_add_noise(points, noise, rng))
    return points, labels


def make_circles(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n0 = int(n) // 2
    n1 = int(n) - n0
    angles0 = rng.uniform(0.0, 2.0 * math.pi, size=n0)
    angles1 = rng.uniform(0.0, 2.0 * math.pi, size=n1)
    inner = np.column_stack([0.48 * np.cos(angles0), 0.48 * np.sin(angles0)])
    outer = np.column_stack([1.0 * np.cos(angles1), 1.0 * np.sin(angles1)])
    points = np.vstack([inner, outer]).astype(np.float32)
    labels = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.int64)
    points = _standardize(_add_noise(points, noise, rng))
    return points, labels


def make_spirals(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n0 = int(n) // 2
    n1 = int(n) - n0
    r0 = np.linspace(0.08, 1.0, n0)
    r1 = np.linspace(0.08, 1.0, n1)
    t0 = np.linspace(0.0, 3.5 * math.pi, n0) + rng.normal(0.0, 0.08, size=n0)
    t1 = np.linspace(0.0, 3.5 * math.pi, n1) + math.pi + rng.normal(0.0, 0.08, size=n1)
    arm0 = np.column_stack([r0 * np.cos(t0), r0 * np.sin(t0)])
    arm1 = np.column_stack([r1 * np.cos(t1), r1 * np.sin(t1)])
    points = np.vstack([arm0, arm1]).astype(np.float32)
    labels = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.int64)
    points = _standardize(_add_noise(points, noise, rng))
    return points, labels


def make_xor(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    points = rng.uniform(-1.0, 1.0, size=(int(n), 2)).astype(np.float32)
    labels = ((points[:, 0] * points[:, 1]) > 0.0).astype(np.int64)
    points = _standardize(_add_noise(points, noise, rng))
    return points, labels


def make_blobs(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    centers = np.asarray([[-0.85, -0.55], [0.75, -0.45], [0.05, 0.85]], dtype=np.float32)
    labels = rng.integers(0, len(centers), size=int(n), endpoint=False)
    points = centers[labels] + rng.normal(0.0, 0.18 + float(noise), size=(int(n), 2)).astype(np.float32)
    return _standardize(points), labels.astype(np.int64)


def make_checkerboard(n: int, noise: float, seed: int, cells: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    points = rng.uniform(-1.0, 1.0, size=(int(n), 2)).astype(np.float32)
    ix = np.floor((points[:, 0] + 1.0) * float(cells) / 2.0).astype(np.int64)
    iy = np.floor((points[:, 1] + 1.0) * float(cells) / 2.0).astype(np.int64)
    labels = ((ix + iy) % 2).astype(np.int64)
    points = _standardize(_add_noise(points, noise, rng), scale=1.15)
    return points, labels


def make_rings(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    radii = np.asarray([0.35, 0.72, 1.08], dtype=np.float32)
    labels = rng.integers(0, len(radii), size=int(n), endpoint=False)
    angles = rng.uniform(0.0, 2.0 * math.pi, size=int(n))
    radius = radii[labels] + rng.normal(0.0, 0.025 + float(noise), size=int(n))
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]).astype(np.float32)
    return _standardize(points), labels.astype(np.int64)


def make_gaussian_mixtures(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    centers = np.asarray(
        [[-0.8, -0.7], [0.75, -0.65], [0.0, 0.0], [-0.55, 0.7], [0.65, 0.65]],
        dtype=np.float32,
    )
    labels = rng.integers(0, len(centers), size=int(n), endpoint=False)
    points = centers[labels] + rng.normal(0.0, 0.12 + float(noise), size=(int(n), 2)).astype(np.float32)
    return _standardize(points), labels.astype(np.int64)


def make_2d_classification(
    name: str,
    n: int = 512,
    noise: float = 0.08,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    key = normalize_distribution_variant(name, choices=CLASSIFICATION_DATASET_KEYS)
    if key == "blobs":
        return make_blobs(n, noise, seed)
    if key == "gaussian_mixtures":
        return make_gaussian_mixtures(n, noise, seed)
    if key == "circles":
        return make_circles(n, noise, seed)
    if key == "rings":
        return make_rings(n, noise, seed)
    if key == "moons":
        return make_moons(n, noise, seed)
    if key == "xor":
        return make_xor(n, noise, seed)
    if key == "checkerboard":
        return make_checkerboard(n, noise, seed)
    if key == "spirals":
        return make_spirals(n, noise, seed)
    valid = ", ".join(CLASSIFICATION_DATASETS)
    raise ValueError(f"Unknown classification distribution '{name}'. Valid: {valid}")


def make_point_cloud(
    name: str,
    n: int = 1024,
    noise: float = 0.04,
    seed: int = 0,
) -> np.ndarray:
    key = normalize_distribution_variant(name, choices=DIFFUSION_DATASET_KEYS)
    if key == "gaussian_mixtures":
        points, _labels = make_gaussian_mixtures(n, noise, seed)
        return points
    if key == "rings":
        points, _labels = make_rings(n, noise, seed)
        return points
    if key == "moons":
        points, _labels = make_moons(n, noise, seed)
        return points
    if key == "checkerboard":
        points, _labels = make_checkerboard(n, noise, seed)
        return points
    if key == "spirals":
        points, _labels = make_spirals(n, noise, seed)
        return points
    valid = ", ".join(DIFFUSION_DATASETS)
    raise ValueError(f"Unknown point-cloud distribution '{name}'. Valid: {valid}")
