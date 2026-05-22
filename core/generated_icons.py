"""Generated 16x16 icon dataset for autoencoding."""

from __future__ import annotations

import numpy as np


SHAPE_NAMES = ("circle", "square", "triangle", "cross", "ring", "arrow", "glyph")


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(int(size), dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _draw_glyph(mask: np.ndarray, digit: int, cx: float, cy: float, scale: float) -> None:
    size = mask.shape[0]
    x0 = int(max(1, round(cx - 3.0 * scale)))
    x1 = int(min(size - 2, round(cx + 3.0 * scale)))
    y0 = int(max(1, round(cy - 5.0 * scale)))
    y1 = int(min(size - 2, round(cy + 5.0 * scale)))
    ym = int(round((y0 + y1) * 0.5))
    thick = max(1, int(round(scale)))
    segments = {
        0: "abcfed",
        1: "bc",
        2: "abged",
        3: "abgcd",
        4: "fgbc",
        5: "afgcd",
        6: "afgecd",
        7: "abc",
        8: "abcdefg",
        9: "abfgcd",
    }[int(digit) % 10]

    def hline(y: int) -> None:
        mask[max(0, y - thick) : min(size, y + thick + 1), x0:x1 + 1] = True

    def vline(x: int, ya: int, yb: int) -> None:
        mask[ya:yb + 1, max(0, x - thick) : min(size, x + thick + 1)] = True

    if "a" in segments:
        hline(y0)
    if "g" in segments:
        hline(ym)
    if "d" in segments:
        hline(y1)
    if "f" in segments:
        vline(x0, y0, ym)
    if "e" in segments:
        vline(x0, ym, y1)
    if "b" in segments:
        vline(x1, y0, ym)
    if "c" in segments:
        vline(x1, ym, y1)


def _render_icon(shape: str, size: int, rng: np.random.Generator) -> np.ndarray:
    xx, yy = _grid(size)
    cx = (size - 1) * 0.5 + rng.uniform(-1.0, 1.0)
    cy = (size - 1) * 0.5 + rng.uniform(-1.0, 1.0)
    radius = rng.uniform(size * 0.24, size * 0.34)
    thickness = rng.uniform(1.2, 2.1)
    mask = np.zeros((size, size), dtype=bool)

    if shape == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    elif shape == "square":
        half = radius
        mask = (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)
    elif shape == "triangle":
        top = cy - radius
        bottom = cy + radius
        progress = np.clip((yy - top) / max(1e-6, bottom - top), 0.0, 1.0)
        half_width = progress * radius
        mask = (yy >= top) & (yy <= bottom) & (np.abs(xx - cx) <= half_width)
    elif shape == "cross":
        span = radius * 1.05
        mask = (
            (np.abs(xx - cx) <= thickness) & (np.abs(yy - cy) <= span)
        ) | (
            (np.abs(yy - cy) <= thickness) & (np.abs(xx - cx) <= span)
        )
    elif shape == "ring":
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = np.abs(dist - radius) <= thickness
    elif shape == "arrow":
        shaft = (np.abs(yy - cy) <= thickness) & (xx >= cx - radius) & (xx <= cx + radius * 0.35)
        head = (xx >= cx) & (xx <= cx + radius) & (np.abs(yy - cy) <= (cx + radius - xx))
        mask = shaft | head
    elif shape == "glyph":
        _draw_glyph(mask, int(rng.integers(0, 10)), cx, cy, rng.uniform(0.8, 1.1))
    else:
        raise ValueError(f"Unsupported shape '{shape}'.")

    image = mask.astype(np.float32)
    if rng.random() < 0.35:
        image = np.roll(image, shift=int(rng.integers(-1, 2)), axis=int(rng.integers(0, 2)))
    return image


def make_icons(
    n: int = 512,
    size: int = 16,
    seed: int = 0,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(int(seed))
    labels = rng.integers(0, len(SHAPE_NAMES), size=int(n), endpoint=False)
    images = np.zeros((int(n), 1, int(size), int(size)), dtype=np.float32)
    for idx, label in enumerate(labels):
        image = _render_icon(SHAPE_NAMES[int(label)], int(size), rng)
        if float(noise) > 0:
            image = image + rng.normal(0.0, float(noise), size=image.shape).astype(np.float32)
        images[idx, 0] = np.clip(image, 0.0, 1.0)
    return images.astype(np.float32), labels.astype(np.int64), SHAPE_NAMES

