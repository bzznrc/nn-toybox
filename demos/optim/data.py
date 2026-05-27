"""Analytic loss landscapes for the optim demo."""

from __future__ import annotations

import numpy as np


LANDSCAPES = ("bowl", "ravine", "saddle", "two_basins", "noisy_bowl")
XLIM = (-2.0, 2.0)
YLIM = (-2.0, 2.0)


def loss_and_grad(landscape: str, point: np.ndarray) -> tuple[float, np.ndarray]:
    name = str(landscape).strip().lower().replace("-", "_")
    x, y = float(point[0]), float(point[1])
    if name == "bowl":
        value = 0.5 * (x * x + y * y)
        grad = np.asarray([x, y], dtype=np.float32)
    elif name == "ravine":
        value = 0.04 * x * x + 1.15 * y * y
        grad = np.asarray([0.08 * x, 2.30 * y], dtype=np.float32)
    elif name == "saddle":
        value = 0.35 * x * x - 0.35 * y * y + 0.08 * (x**4 + y**4)
        grad = np.asarray([0.70 * x + 0.32 * x**3, -0.70 * y + 0.32 * y**3], dtype=np.float32)
    elif name == "two_basins":
        f1 = (x + 0.95) ** 2 + 0.55 * (y + 0.55) ** 2
        f2 = 0.70 * (x - 1.05) ** 2 + 1.15 * (y - 0.45) ** 2 + 0.25
        sharp = 5.0
        weights = np.exp(-sharp * np.asarray([f1, f2], dtype=np.float64))
        weights = weights / max(1e-12, float(np.sum(weights)))
        value = float(-np.log(max(1e-12, float(np.sum(np.exp(-sharp * np.asarray([f1, f2])))))) / sharp)
        g1 = np.asarray([2.0 * (x + 0.95), 1.10 * (y + 0.55)], dtype=np.float64)
        g2 = np.asarray([1.40 * (x - 1.05), 2.30 * (y - 0.45)], dtype=np.float64)
        grad = (weights[0] * g1 + weights[1] * g2).astype(np.float32)
    elif name == "noisy_bowl":
        value = 0.35 * (x * x + y * y) + 0.10 * np.sin(4.0 * x + 1.3) * np.cos(3.5 * y - 0.2)
        grad = np.asarray(
            [
                0.70 * x + 0.40 * np.cos(4.0 * x + 1.3) * np.cos(3.5 * y - 0.2),
                0.70 * y - 0.35 * np.sin(4.0 * x + 1.3) * np.sin(3.5 * y - 0.2),
            ],
            dtype=np.float32,
        )
    else:
        valid = ", ".join(LANDSCAPES)
        raise ValueError(f"Unknown landscape '{landscape}'. Valid: {valid}")
    return float(value), grad


def landscape_grid(landscape: str, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res = max(16, int(resolution))
    xs = np.linspace(XLIM[0], XLIM[1], res, dtype=np.float32)
    ys = np.linspace(YLIM[0], YLIM[1], res, dtype=np.float32)
    values = np.empty((res, res), dtype=np.float32)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            values[row, col] = loss_and_grad(landscape, np.asarray([x, y], dtype=np.float32))[0]
    return xs, ys, values
