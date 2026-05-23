"""Arcade renderer for live grad snapshots."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import CLASS_COLOR_PAIRS, COLOR_BRICK_RED, COLOR_CORAL, COLOR_FOG_GRAY
from core.arcade_view import clamp_point_to_rect, draw_curve, draw_diamond_marker, draw_points, marker_outer_radius, with_alpha
from demos.grad.config import GradConfig


class GradRenderer:
    def __init__(self, config: GradConfig) -> None:
        self.config = config

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        loss_rect = layout.secondary if layout.secondary is not None else layout.text
        pred = np.asarray(snapshot["boundary_pred"], dtype=np.int64)
        resolution = max(1, int(snapshot.get("boundary_resolution", self.config.boundary_resolution)))
        left, bottom, width, height = main_rect
        cell_w = width / float(resolution)
        cell_h = height / float(resolution)
        for idx, label in enumerate(pred[: resolution * resolution]):
            col = idx % resolution
            row = idx // resolution
            outer, _inner = CLASS_COLOR_PAIRS[int(label) % len(CLASS_COLOR_PAIRS)]
            arcade.draw_lbwh_rectangle_filled(
                left + col * cell_w,
                bottom + row * cell_h,
                cell_w,
                cell_h,
                with_alpha(outer, 36),
            )
        draw_points(
            np.asarray(snapshot["points"], dtype=np.float32),
            np.asarray(snapshot["labels"], dtype=np.int64),
            main_rect,
            marker="small",
        )
        losses = np.asarray(snapshot["losses"], dtype=np.float32)
        draw_curve(losses, loss_rect, color=COLOR_FOG_GRAY)
        if losses.size >= 2:
            left, bottom, width, height = loss_rect
            x = left + width - 1.0
            lo = float(np.nanmin(losses))
            hi = float(np.nanmax(losses))
            y = bottom + ((float(losses[-1]) - lo) / max(1e-6, hi - lo)) * height
            arcade.draw_line(x, bottom, x, bottom + height, with_alpha(COLOR_CORAL, 150), 1.0)
            marker_radius = marker_outer_radius("small")
            marker_x, marker_y = clamp_point_to_rect(x, y, loss_rect, inset=marker_radius)
            draw_diamond_marker(marker_x, marker_y, outer_color=COLOR_CORAL, inner_color=COLOR_BRICK_RED, marker="small", alpha=245)
        window.draw_info(snapshot, secondary=True, extra=(f"boundary resolution: {resolution}",))
