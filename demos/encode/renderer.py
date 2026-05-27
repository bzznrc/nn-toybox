"""Arcade renderer for live encode snapshots."""

from __future__ import annotations

import numpy as np

from core.arcade_style import COLOR_AQUA, COLOR_CORAL
from core.arcade_view import clipped_rect, draw_pixel_image, draw_vertical_bars, split_columns
from demos.encode.config import EncodeConfig


class EncodeRenderer:
    def __init__(self, config: EncodeConfig) -> None:
        self.config = config

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
        image_rects = split_columns(main_rect, 2)
        metrics = dict(snapshot.get("metrics", {}))
        images = np.asarray(snapshot["images"], dtype=np.float32)
        recons = np.asarray(snapshot["recons"], dtype=np.float32)
        latents = np.asarray(snapshot["latents"], dtype=np.float32)
        idx = (int(metrics.get("step", 0)) // 20) % max(1, len(images))
        image = images[idx]
        recon = recons[idx]
        with clipped_rect(main_rect):
            self._draw_image_in_rect(image, image_rects[0], COLOR_AQUA)
            self._draw_image_in_rect(recon, image_rects[1], COLOR_CORAL)
        latent = latents[idx]
        draw_vertical_bars(latent, latent_rect)
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
