"""Arcade renderer for live embed snapshots."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import COLOR_AQUA, COLOR_CORAL, COLOR_FOG_GRAY, COLOR_LIGHT_NEUTRAL, COLOR_SLATE_GRAY
from core.arcade_view import draw_panel_outline, with_alpha, world_to_panel
from demos.embed.config import EmbedConfig


class EmbedRenderer:
    def __init__(self, config: EmbedConfig) -> None:
        self.config = config

    @staticmethod
    def _nearest(embeddings: np.ndarray, selected: int) -> list[tuple[float, int]]:
        vector = embeddings[selected]
        denom = np.linalg.norm(embeddings, axis=1) * max(1e-6, float(np.linalg.norm(vector)))
        sim = (embeddings @ vector) / np.maximum(denom, 1e-6)
        order = np.argsort(-sim)
        return [(float(sim[int(idx)]), int(idx)) for idx in order if int(idx) != selected][:5]

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        secondary_rect = layout.secondary if layout.secondary is not None else layout.text
        draw_panel_outline(main_rect, "")
        draw_panel_outline(secondary_rect, "")
        tokens = [str(token) for token in snapshot["tokens"]]
        embeddings = np.asarray(snapshot["embeddings"], dtype=np.float32)
        coords = np.asarray(snapshot["coords"], dtype=np.float32)
        metrics = dict(snapshot.get("metrics", {}))
        selected = (int(metrics.get("step", 0)) // 25) % max(1, len(tokens))
        xy = world_to_panel(coords, main_rect, xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
        selected_xy = xy[selected]
        nearest = self._nearest(embeddings, selected)
        for sim, idx in nearest[:3]:
            alpha = int(60 + max(0.0, sim) * 130)
            arcade.draw_line(selected_xy[0], selected_xy[1], xy[idx, 0], xy[idx, 1], with_alpha(COLOR_FOG_GRAY, alpha), 1.5)
        for idx, (token, point) in enumerate(zip(tokens, xy)):
            is_selected = idx == selected
            color = COLOR_CORAL if is_selected else COLOR_AQUA
            radius = 9 if is_selected else 5
            arcade.draw_circle_filled(float(point[0]), float(point[1]), radius, color)
            outline = COLOR_LIGHT_NEUTRAL if is_selected else with_alpha(COLOR_SLATE_GRAY, 120)
            arcade.draw_circle_outline(float(point[0]), float(point[1]), radius + 2, outline, 1.0)

        left, bottom, width, height = secondary_rect
        bar_gap = 8.0
        bar_count = max(1, len(nearest))
        bar_height = max(4.0, (height - bar_gap * (bar_count + 1)) / bar_count)
        for row, (sim, _idx) in enumerate(nearest):
            y = bottom + height - bar_gap - (row + 1) * bar_height - row * bar_gap
            fill_width = max(2.0, width * max(0.0, min(1.0, (sim + 1.0) * 0.5)))
            arcade.draw_lbwh_rectangle_filled(left, y, width, bar_height, with_alpha(COLOR_SLATE_GRAY, 70))
            arcade.draw_lbwh_rectangle_filled(left, y, fill_width, bar_height, with_alpha(COLOR_AQUA, 210))

        extra = [f"selected: {tokens[selected]}"]
        for sim, idx in nearest:
            extra.append(f"nearest: {tokens[idx]} ({sim:.2f})")
        window.draw_info(snapshot, secondary=True, extra=tuple(extra))
