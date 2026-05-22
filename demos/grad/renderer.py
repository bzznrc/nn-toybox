"""Arcade renderer for live grad snapshots."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import CLASS_COLOR_PAIRS, COLOR_CORAL, COLOR_FOG_GRAY
from core.arcade_view import draw_curve, draw_panel_outline, draw_points, with_alpha, world_to_panel
from demos.grad.config import GradConfig


class GradRenderer:
    def __init__(self, config: GradConfig) -> None:
        self.config = config

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        loss_rect = layout.secondary if layout.secondary is not None else layout.text
        draw_panel_outline(main_rect, "")
        draw_panel_outline(loss_rect, "")
        grid = np.asarray(snapshot["boundary_grid"], dtype=np.float32)
        pred = np.asarray(snapshot["boundary_pred"], dtype=np.int64)
        resolution = max(1, int(snapshot.get("boundary_resolution", self.config.boundary_resolution)))
        xy = world_to_panel(grid, main_rect)
        cell = main_rect[2] / float(resolution) + 1.0
        for label, (x, y) in zip(pred, xy):
            outer, _inner = CLASS_COLOR_PAIRS[int(label) % len(CLASS_COLOR_PAIRS)]
            arcade.draw_lbwh_rectangle_filled(float(x) - cell * 0.5, float(y) - cell * 0.5, cell, cell, with_alpha(outer, 36))
        draw_points(
            np.asarray(snapshot["points"], dtype=np.float32),
            np.asarray(snapshot["labels"], dtype=np.int64),
            main_rect,
            size=7.0,
        )
        losses = np.asarray(snapshot["losses"], dtype=np.float32)
        draw_curve(losses, loss_rect, color=COLOR_FOG_GRAY)
        if losses.size >= 2:
            left, bottom, width, height = loss_rect
            x = left + width
            lo = float(np.nanmin(losses))
            hi = float(np.nanmax(losses))
            y = bottom + ((float(losses[-1]) - lo) / max(1e-6, hi - lo)) * height
            arcade.draw_line(x, bottom, x, bottom + height, with_alpha(COLOR_CORAL, 150), 1.0)
            arcade.draw_circle_filled(x, y, 4.0, COLOR_CORAL)
        window.draw_info(snapshot, secondary=True, extra=(f"boundary resolution: {resolution}",))
