"""Analytic loss landscapes for the optim demo."""

from __future__ import annotations

import numpy as np


LANDSCAPES = (
    "bowl",
    "ravine",
    "saddle",
    "two_basins",
    "noisy_bowl",
    "hidden_well",
    "narrow_pass",
    "ripple_traps",
)
XLIM = (-2.0, 2.0)
YLIM = (-2.0, 2.0)


def _negative_gaussian_well(
    x: float,
    y: float,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    amp: float,
) -> tuple[float, np.ndarray]:
    q = ((x - cx) ** 2) / float(sx) + ((y - cy) ** 2) / float(sy)
    e = float(np.exp(-q))
    value = -float(amp) * e
    grad = np.asarray(
        [
            float(amp) * e * 2.0 * (x - cx) / float(sx),
            float(amp) * e * 2.0 * (y - cy) / float(sy),
        ],
        dtype=np.float32,
    )
    return value, grad


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
    elif name == "hidden_well":
        value = 0.12 * (x * x + y * y) + 0.07 * x - 0.05 * y
        grad = np.asarray([0.24 * x + 0.07, 0.24 * y - 0.05], dtype=np.float32)
        local_value, local_grad = _negative_gaussian_well(x, y, 0.95, -0.90, 0.58, 0.46, 0.55)
        global_value, global_grad = _negative_gaussian_well(x, y, -1.35, 1.25, 0.10, 0.13, 1.35)
        value = value + local_value + global_value
        grad = grad + local_grad + global_grad
    elif name == "narrow_pass":
        value = 0.18 * ((x + 1.45) ** 2 + 0.75 * (y - 1.25) ** 2)
        grad = np.asarray([0.36 * (x + 1.45), 0.27 * (y - 1.25)], dtype=np.float32)
        rx = float(np.exp(-((x + 0.03) ** 2) / 0.035))
        gap = float(np.exp(-((y + 0.55) ** 2) / 0.10))
        ridge = 0.72 * rx * (1.0 - gap)
        ridge_grad = np.asarray(
            [
                0.72 * (-2.0 * (x + 0.03) / 0.035) * rx * (1.0 - gap),
                0.72 * rx * (2.0 * (y + 0.55) / 0.10) * gap,
            ],
            dtype=np.float32,
        )
        local_value, local_grad = _negative_gaussian_well(x, y, 1.15, -1.10, 0.52, 0.38, 0.22)
        value = value + ridge + local_value
        grad = grad + ridge_grad + local_grad
    elif name == "ripple_traps":
        sx = np.sin(3.7 * x + 0.4)
        sy = np.sin(3.3 * y - 0.2)
        cx = np.cos(3.7 * x + 0.4)
        cy = np.cos(3.3 * y - 0.2)
        value = 0.10 * (x * x + y * y) + 0.16 * sx * sy + 0.05 * x - 0.04 * y
        grad = np.asarray(
            [
                0.20 * x + 0.16 * 3.7 * cx * sy + 0.05,
                0.20 * y + 0.16 * 3.3 * sx * cy - 0.04,
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
