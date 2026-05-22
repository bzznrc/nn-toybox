"""Arcade renderer for live autoencode snapshots."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import COLOR_AQUA, COLOR_CORAL, COLOR_LEAF_GREEN
from core.arcade_view import draw_panel_outline, draw_pixel_image, with_alpha
from demos.autoencode.config import AutoencodeConfig


class AutoencodeRenderer:
    def __init__(self, config: AutoencodeConfig) -> None:
        self.config = config

    @staticmethod
    def _split_three(rect: tuple[float, float, float, float]) -> tuple[tuple[float, float, float, float], ...]:
        left, bottom, width, height = rect
        gap = 14.0
        cell_width = (width - 2.0 * gap) / 3.0
        return (
            (left, bottom, cell_width, height),
            (left + cell_width + gap, bottom, cell_width, height),
            (left + 2.0 * (cell_width + gap), bottom, cell_width, height),
        )

    @staticmethod
    def _draw_image_in_rect(image: np.ndarray, rect: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
        left, bottom, width, height = rect
        img = np.asarray(image, dtype=np.float32)
        size = int(img.shape[-1])
        scale = max(1.0, min((width - 20.0) / max(1, size), (height - 20.0) / max(1, size)))
        draw_width = size * scale
        draw_pixel_image(
            img,
            left + (width - draw_width) * 0.5,
            bottom + (height - draw_width) * 0.5,
            scale=scale,
            on_color=color,
        )

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        latent_rect = layout.secondary if layout.secondary is not None else layout.text
        image_rects = self._split_three(main_rect)
        for rect in (*image_rects, latent_rect):
            draw_panel_outline(rect, "")
        metrics = dict(snapshot.get("metrics", {}))
        images = np.asarray(snapshot["images"], dtype=np.float32)
        recons = np.asarray(snapshot["recons"], dtype=np.float32)
        latents = np.asarray(snapshot["latents"], dtype=np.float32)
        idx = (int(metrics.get("step", 0)) // 20) % max(1, len(images))
        image = images[idx]
        recon = recons[idx]
        err = np.abs(image - recon)
        self._draw_image_in_rect(image, image_rects[0], COLOR_AQUA)
        self._draw_image_in_rect(recon, image_rects[1], COLOR_CORAL)
        self._draw_image_in_rect(err / max(1e-6, float(err.max())), image_rects[2], COLOR_LEAF_GREEN)
        latent = latents[idx]
        left, bottom, width, height = latent_rect
        if latent.size:
            lo = float(np.min(latent))
            hi = float(np.max(latent))
            span = max(1e-6, hi - lo)
            bar_w = width / latent.size
            for j, value in enumerate(latent):
                ratio = (float(value) - lo) / span
                bar_h = 8.0 + ratio * (height - 24.0)
                arcade.draw_lbwh_rectangle_filled(left + j * bar_w + 3, bottom + 8, max(2, bar_w - 6), bar_h, with_alpha(COLOR_AQUA, 220))
        labels = np.asarray(snapshot["labels"], dtype=np.int64)
        names = [str(name) for name in snapshot["names"]]
        label = int(labels[idx]) if labels.size else 0
        shape_name = names[label] if 0 <= label < len(names) else str(label)
        errors = np.asarray(snapshot["error"], dtype=np.float32)
        error = float(errors[idx]) if errors.size else 0.0
        window.draw_info(
            snapshot,
            secondary=True,
            extra=(
                f"sample: {idx}",
                f"shape: {shape_name}",
                f"sample error: {error:.5f}",
                f"latent dim: {int(self.config.latent_dim)}",
            ),
        )
