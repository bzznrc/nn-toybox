"""Arcade renderer for live embed snapshots."""

from __future__ import annotations

import numpy as np

import arcade

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BLUE,
    COLOR_BRICK_RED,
    COLOR_CORAL,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_PURPLE,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_LIGHT_NEUTRAL,
    COLOR_NAVY,
    COLOR_PURPLE,
    COLOR_SLATE_GRAY,
    TOYBOX_CHART_FILL_ALPHA,
    TOYBOX_CHART_TRACK_ALPHA,
)
from core.arcade_view import clipped_rect, clamp_point_to_rect, draw_diamond_marker, draw_diamond_node, fit_points_to_panel, with_alpha
from demos.embed.config import EmbedConfig


class EmbedRenderer:
    GROUP_COLOR_PAIRS = {
        "animals": (COLOR_AQUA, COLOR_DEEP_TEAL),
        "vehicles": (COLOR_CORAL, COLOR_BRICK_RED),
        "food": (COLOR_BLUE, COLOR_NAVY),
        "tools": (COLOR_PURPLE, COLOR_DEEP_PURPLE),
    }

    def __init__(self, config: EmbedConfig) -> None:
        self.config = config
        self.mouse_xy: tuple[float, float] | None = None
        self.hover_radius = 14.0
        self.selected_index = 0

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int, *, window: object) -> bool:
        del dx, dy, window
        self.mouse_xy = (float(x), float(y))
        return False

    def on_key_press(self, symbol: int, modifiers: int, *, window: object) -> bool:
        del modifiers
        trainer = window.trainer
        if symbol == arcade.key.UP:
            trainer.cycle_dataset(1)
            return True
        if symbol == arcade.key.DOWN:
            trainer.cycle_dataset(-1)
            return True
        if symbol == arcade.key.RIGHT:
            trainer.cycle_embedding_dim(1)
            return True
        if symbol == arcade.key.LEFT:
            trainer.cycle_embedding_dim(-1)
            return True
        return False

    @staticmethod
    def _nearest(coords: np.ndarray, selected: int, *, count: int = 5) -> list[tuple[float, int]]:
        delta = np.asarray(coords, dtype=np.float32) - np.asarray(coords[int(selected)], dtype=np.float32)
        distances = np.linalg.norm(delta, axis=1)
        order = [int(idx) for idx in np.argsort(distances) if int(idx) != int(selected)]
        if not order:
            return []
        local = order[: max(1, int(count) + 1)]
        radius = float(distances[local[-1]])
        radius = max(radius, 1e-6)
        nearest = []
        for idx in order[: int(count)]:
            score = 1.0 - float(distances[idx]) / radius
            nearest.append((max(0.05, min(1.0, score)), int(idx)))
        return nearest

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        secondary_rect = layout.secondary if layout.secondary is not None else layout.text
        tokens = [str(token) for token in snapshot["tokens"]]
        groups = [str(group) for group in snapshot.get("token_groups", [""] * len(tokens))]
        coords = np.asarray(snapshot["coords"], dtype=np.float32)
        xy = fit_points_to_panel(coords, main_rect, fill=0.95)
        hovered = self._hovered_index(xy, main_rect)
        if hovered is not None:
            self.selected_index = int(hovered)
        selected = max(0, min(int(self.selected_index), len(tokens) - 1))
        nearest: list[tuple[float, int]] = []
        selected_xy = xy[selected]
        nearest = self._nearest(coords, selected)
        with clipped_rect(main_rect):
            for score, idx in nearest[:3]:
                alpha = int(55 + max(0.0, score) * 150)
                arcade.draw_line(selected_xy[0], selected_xy[1], xy[idx, 0], xy[idx, 1], with_alpha(COLOR_FOG_GRAY, alpha), 1.5)
            for idx, point in enumerate(xy):
                is_selected = idx == selected
                outer, inner = self.GROUP_COLOR_PAIRS.get(groups[idx], (COLOR_SLATE_GRAY, COLOR_DARK_NEUTRAL))
                if is_selected:
                    draw_diamond_node(
                        float(point[0]),
                        float(point[1]),
                        radius=13.0,
                        fill_color=COLOR_FOG_GRAY,
                        outline_color=COLOR_FOG_GRAY,
                        alpha=42,
                        outline_alpha=0,
                        outline_width=0.0,
                    )
                draw_diamond_marker(
                    float(point[0]),
                    float(point[1]),
                    outer_color=outer,
                    inner_color=inner,
                    marker="regular",
                    alpha=245 if groups[idx] else 175,
                )
            label_width = window.text_cache.measure_width(tokens[selected], font_size=12, anchor_y="center")
            label_x, label_y = clamp_point_to_rect(
                float(selected_xy[0]) + 12.0,
                float(selected_xy[1]) + 12.0,
                (
                    main_rect[0],
                    main_rect[1] + 6.0,
                    max(1.0, main_rect[2] - label_width),
                    max(1.0, main_rect[3] - 12.0),
                ),
            )
            window.text_cache.draw(
                tokens[selected],
                label_x,
                label_y,
                color=COLOR_FOG_GRAY,
                font_size=12,
                anchor_y="center",
            )

        left, bottom, width, height = secondary_rect
        bar_gap = 8.0
        bar_count = max(1, len(nearest))
        bar_height = max(4.0, (height - bar_gap * (bar_count + 1)) / bar_count)
        for row, (score, idx) in enumerate(nearest):
            y = bottom + height - bar_gap - (row + 1) * bar_height - row * bar_gap
            fill_width = max(2.0, width * max(0.0, min(1.0, score)))
            arcade.draw_lbwh_rectangle_filled(left, y, width, bar_height, with_alpha(COLOR_SLATE_GRAY, TOYBOX_CHART_TRACK_ALPHA))
            arcade.draw_lbwh_rectangle_filled(left, y, fill_width, bar_height, with_alpha(COLOR_SLATE_GRAY, TOYBOX_CHART_FILL_ALPHA))
            window.text_cache.draw(
                tokens[idx],
                left + 8.0,
                y + bar_height * 0.5,
                color=COLOR_LIGHT_NEUTRAL,
                font_size=11,
                anchor_y="center",
                bold=True,
            )

        extra = [
            f"selected: {tokens[selected]}",
            f"embedding dim: {int(self.config.embedding_dim)}",
            "keys: up/down dataset, left/right embedding dim",
        ]
        window.draw_info(snapshot, secondary=True, extra=tuple(extra))

    def _hovered_index(self, xy: np.ndarray, rect: tuple[float, float, float, float]) -> int | None:
        if self.mouse_xy is None or xy.size == 0:
            return None
        mx, my = self.mouse_xy
        left, bottom, width, height = rect
        if not (left <= mx <= left + width and bottom <= my <= bottom + height):
            return None
        delta = xy - np.asarray([[mx, my]], dtype=np.float32)
        dist2 = np.sum(delta * delta, axis=1)
        idx = int(np.argmin(dist2))
        if float(dist2[idx]) > self.hover_radius * self.hover_radius:
            return None
        return idx
