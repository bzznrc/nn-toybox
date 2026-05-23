"""Generated 16x16 image datasets for autoencoding."""

from __future__ import annotations

import numpy as np


SHAPE_NAMES = ("circle", "square", "triangle", "cross", "ring", "arrow", "glyph")
PATTERN_NAMES = (
    "checkerboard",
    "vertical_stripes",
    "horizontal_stripes",
    "diagonal_stripes",
    "reverse_diagonal_stripes",
    "grid",
    "dots",
    "border",
)


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
    cx = (size - 1) * 0.5
    cy = (size - 1) * 0.5
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

    return mask.astype(np.float32)


def _render_pattern(pattern: str, size: int, rng: np.random.Generator) -> np.ndarray:
    xx, yy = _grid(size)
    period = int(rng.integers(3, 6))
    stripe_width = max(1, period // 2)

    if pattern == "checkerboard":
        block = int(rng.integers(2, 5))
        image = (((xx // block) + (yy // block)) % 2 == 0)
    elif pattern == "vertical_stripes":
        image = (xx % period) < stripe_width
    elif pattern == "horizontal_stripes":
        image = (yy % period) < stripe_width
    elif pattern == "diagonal_stripes":
        image = ((xx + yy) % period) < stripe_width
    elif pattern == "reverse_diagonal_stripes":
        image = ((xx - yy) % period) < stripe_width
    elif pattern == "grid":
        spacing = int(rng.integers(4, 7))
        image = ((xx % spacing) == 0) | ((yy % spacing) == 0)
    elif pattern == "dots":
        spacing = int(rng.integers(4, 6))
        radius = rng.uniform(0.9, 1.4)
        cx = (np.round(xx / spacing) * spacing).astype(np.float32)
        cy = (np.round(yy / spacing) * spacing).astype(np.float32)
        image = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    elif pattern == "border":
        thickness = int(rng.integers(1, 3))
        image = (xx < thickness) | (yy < thickness) | (xx >= size - thickness) | (yy >= size - thickness)
    else:
        raise ValueError(f"Unsupported pattern '{pattern}'.")

    return image.astype(np.float32)


def _apply_foreground_noise(image: np.ndarray, noise: float, seed: int) -> np.ndarray:
    if float(noise) <= 0:
        return image
    noise_rng = np.random.default_rng(int(seed))
    jitter = noise_rng.normal(0.0, float(noise), size=image.shape).astype(np.float32)
    return image + jitter * image


def make_icons(
    n: int = 512,
    size: int = 16,
    seed: int = 0,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(int(seed))
    labels = rng.integers(0, len(SHAPE_NAMES), size=int(n), endpoint=False)
    shape_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=int(n), dtype=np.uint32)
    noise_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=int(n), dtype=np.uint32)
    images = np.zeros((int(n), 1, int(size), int(size)), dtype=np.float32)
    for idx, label in enumerate(labels):
        shape_rng = np.random.default_rng(int(shape_seeds[idx]))
        image = _render_icon(SHAPE_NAMES[int(label)], int(size), shape_rng)
        image = _apply_foreground_noise(image, float(noise), int(noise_seeds[idx]))
        images[idx, 0] = np.clip(image, 0.0, 1.0)
    return images.astype(np.float32), labels.astype(np.int64), SHAPE_NAMES


def make_patterns(
    n: int = 512,
    size: int = 16,
    seed: int = 0,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(int(seed))
    labels = rng.integers(0, len(PATTERN_NAMES), size=int(n), endpoint=False)
    pattern_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=int(n), dtype=np.uint32)
    noise_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=int(n), dtype=np.uint32)
    images = np.zeros((int(n), 1, int(size), int(size)), dtype=np.float32)
    for idx, label in enumerate(labels):
        pattern_rng = np.random.default_rng(int(pattern_seeds[idx]))
        image = _render_pattern(PATTERN_NAMES[int(label)], int(size), pattern_rng)
        image = _apply_foreground_noise(image, float(noise), int(noise_seeds[idx]))
        images[idx, 0] = np.clip(image, 0.0, 1.0)
    return images.astype(np.float32), labels.astype(np.int64), PATTERN_NAMES
