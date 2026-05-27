"""Arcade renderer for optimizer motion snapshots."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import COLOR_AQUA, COLOR_CORAL, COLOR_DARK_NEUTRAL, COLOR_FOG_GRAY, COLOR_LIGHT_NEUTRAL, COLOR_SLATE_GRAY
from core.arcade_view import clipped_rect, draw_curve, draw_diamond_node, with_alpha
from demos.optim.config import OptimConfig
from demos.optim.data import XLIM, YLIM


class OptimRenderer:
    def __init__(self, config: OptimConfig) -> None:
        self.config = config

    def on_key_press(self, symbol: int, modifiers: int, *, window: object) -> bool:
        del modifiers
        trainer = window.trainer
        if symbol == arcade.key.O:
            trainer.cycle_optimizer(1)
            return True
        if symbol == arcade.key.L:
            trainer.cycle_landscape(1)
            return True
        if symbol == arcade.key.G:
            trainer.new_start()
            return True
        plus_keys = {arcade.key.EQUAL, getattr(arcade.key, "NUM_ADD", arcade.key.EQUAL)}
        minus_keys = {arcade.key.MINUS, getattr(arcade.key, "NUM_SUBTRACT", arcade.key.MINUS)}
        if symbol in plus_keys:
            trainer.scale_lr(1.25)
            return True
        if symbol in minus_keys:
            trainer.scale_lr(0.80)
            return True
        return False

    @staticmethod
    def _world_to_rect(points: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
        left, bottom, width, height = rect
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        x = (arr[:, 0] - XLIM[0]) / max(1e-6, XLIM[1] - XLIM[0])
        y = (arr[:, 1] - YLIM[0]) / max(1e-6, YLIM[1] - YLIM[0])
        return np.column_stack([left + x * width, bottom + y * height]).astype(np.float32)

    def _draw_landscape(self, values: np.ndarray, rect: tuple[float, float, float, float]) -> None:
        arr = np.asarray(values, dtype=np.float32)
        left, bottom, width, height = rect
        rows, cols = arr.shape
        lo = float(np.percentile(arr, 3))
        hi = float(np.percentile(arr, 97))
        cell_w = width / max(1, cols)
        cell_h = height / max(1, rows)
        for row in range(rows):
            for col in range(cols):
                ratio = (float(arr[row, col]) - lo) / max(1e-6, hi - lo)
                ratio = max(0.0, min(1.0, ratio))
                color = (
                    int(COLOR_DARK_NEUTRAL[0] + (COLOR_SLATE_GRAY[0] - COLOR_DARK_NEUTRAL[0]) * ratio),
                    int(COLOR_DARK_NEUTRAL[1] + (COLOR_AQUA[1] - COLOR_DARK_NEUTRAL[1]) * ratio),
                    int(COLOR_DARK_NEUTRAL[2] + (COLOR_CORAL[2] - COLOR_DARK_NEUTRAL[2]) * ratio),
                )
                arcade.draw_lbwh_rectangle_filled(left + col * cell_w, bottom + row * cell_h, cell_w + 0.5, cell_h + 0.5, color)

    def _draw_trail(self, trail: np.ndarray, point: np.ndarray, rect: tuple[float, float, float, float]) -> None:
        xy = self._world_to_rect(trail, rect)
        if len(xy) >= 2:
            for idx, (p0, p1) in enumerate(zip(xy, xy[1:])):
                alpha = int(45 + 175 * (idx + 1) / max(1, len(xy) - 1))
                arcade.draw_line(float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), with_alpha(COLOR_FOG_GRAY, alpha), 2.0)
        current = self._world_to_rect(np.asarray(point, dtype=np.float32), rect)[0]
        draw_diamond_node(
            float(current[0]),
            float(current[1]),
            radius=12.0,
            fill_color=COLOR_CORAL,
            outline_color=COLOR_FOG_GRAY,
            alpha=115,
            outline_alpha=170,
            outline_width=1.5,
            inner_radius=6.5,
            inner_color=COLOR_CORAL,
            inner_alpha=245,
        )

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        secondary_rect = layout.secondary if layout.secondary is not None else layout.text
        with clipped_rect(main_rect):
            self._draw_landscape(np.asarray(snapshot["values"], dtype=np.float32), main_rect)
            self._draw_trail(np.asarray(snapshot["trail"], dtype=np.float32), np.asarray(snapshot["point"], dtype=np.float32), main_rect)
            arcade.draw_lbwh_rectangle_outline(*main_rect, with_alpha(COLOR_LIGHT_NEUTRAL, 115), 1.0)
        draw_curve(np.asarray(snapshot["losses"], dtype=np.float32), secondary_rect, color=COLOR_FOG_GRAY)
        metrics = dict(snapshot.get("metrics", {}))
        point = np.asarray(snapshot["point"], dtype=np.float32)
        extra = (
            f"landscape: {metrics.get('landscape', self.config.landscape)}",
            f"optimizer: {metrics.get('optimizer', self.config.optimizer)}",
            f"lr: {float(self.config.lr):.4f}",
            f"point: {float(point[0]):.2f}, {float(point[1]):.2f}",
            "keys: O optimizer, L landscape, G start, +/- lr",
        )
        window.draw_info(snapshot, secondary=True, extra=extra)
