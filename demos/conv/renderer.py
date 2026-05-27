"""Arcade renderer for convolution snapshots."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_DARK_NEUTRAL,
    COLOR_LIGHT_NEUTRAL,
    COLOR_SLATE_GRAY,
)
from core.arcade_view import clipped_rect, draw_pixel_image_in_rect
from demos.conv.config import ConvConfig


class ConvRenderer:
    def __init__(self, config: ConvConfig) -> None:
        self.config = config

    def on_key_press(self, symbol: int, modifiers: int, *, window: object) -> bool:
        del modifiers
        trainer = window.trainer
        if symbol == arcade.key.RIGHT:
            trainer.next_variation(1)
            return True
        if symbol == arcade.key.LEFT:
            trainer.next_variation(-1)
            return True
        if symbol == arcade.key.UP:
            trainer.cycle_digit(1)
            return True
        if symbol == arcade.key.DOWN:
            trainer.cycle_digit(-1)
            return True
        if symbol == arcade.key.G:
            trainer.random_example()
            return True
        if symbol == arcade.key.V:
            trainer.toggle_feature_layer()
            return True
        if symbol == arcade.key.M:
            trainer.cycle_noise()
            return True
        return False

    @staticmethod
    def _normalize_map(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        return ((arr - lo) / max(1e-6, hi - lo)).astype(np.float32)

    def _draw_map_grid(self, maps: np.ndarray, rect: tuple[float, float, float, float]) -> None:
        arr = np.asarray(maps, dtype=np.float32)
        count = min(arr.shape[0], 8)
        cols = 4
        rows = 2
        left, bottom, width, height = rect
        pad = min(width, height) * 0.035
        gap = min(width, height) * 0.035
        grid_left = left + pad
        grid_bottom = bottom + pad
        grid_width = max(1.0, width - 2.0 * pad)
        grid_height = max(1.0, height - 2.0 * pad)
        cell_w = (grid_width - gap * (cols - 1)) / cols
        cell_h = (grid_height - gap * (rows - 1)) / rows
        for idx in range(count):
            row = idx // cols
            col = idx % cols
            x = grid_left + col * (cell_w + gap)
            y = grid_bottom + grid_height - (row + 1) * cell_h - row * gap
            cell = (x, y, cell_w, cell_h)
            draw_pixel_image_in_rect(
                self._normalize_map(arr[idx]),
                cell,
                on_color=COLOR_AQUA,
                off_color=COLOR_DARK_NEUTRAL,
                border_color=COLOR_SLATE_GRAY,
                padding=0.0,
            )

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        preview_rect = layout.secondary if layout.secondary is not None else layout.text
        feature_key = "conv2" if int(self.config.selected_feature_layer) == 2 else "conv1"
        draw_pixel_image_in_rect(
            np.asarray(snapshot["image"], dtype=np.float32),
            preview_rect,
            on_color=COLOR_LIGHT_NEUTRAL,
            off_color=COLOR_DARK_NEUTRAL,
            border_color=COLOR_SLATE_GRAY,
            padding=0.0,
        )
        with clipped_rect(main_rect):
            self._draw_map_grid(np.asarray(snapshot[feature_key], dtype=np.float32), main_rect)
        status = "correct" if bool(snapshot["correct"]) else "misclassified"
        extra = (
            f"label: {int(snapshot['label'])}",
            f"prediction: {int(snapshot['pred'])}",
            f"result: {status}",
            f"variation: {int(snapshot.get('variation_index', 0)) + 1}",
            f"feature layer: {feature_key}",
            f"noise: {float(self.config.noise_amount):.2f}",
        )
        window.draw_info(snapshot, secondary=True, extra=extra, compact=True)
