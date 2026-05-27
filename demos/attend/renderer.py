"""Arcade renderer for agreement self-attention snapshots."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CORAL,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_LIGHT_NEUTRAL,
    COLOR_SLATE_GRAY,
    TOYBOX_CHART_FILL_ALPHA,
    TOYBOX_CHART_TRACK_ALPHA,
    TOYBOX_CONNECTION_LINE_WIDTH,
)
from core.arcade_view import clipped_rect, draw_diamond_node, with_alpha
from demos.attend.config import AttendConfig


class AttendRenderer:
    def __init__(self, config: AttendConfig) -> None:
        self.config = config

    def on_key_press(self, symbol: int, modifiers: int, *, window: object) -> bool:
        del modifiers
        if symbol == arcade.key.G:
            window.trainer.new_preview()
            return True
        if symbol in {arcade.key.H, arcade.key.TAB}:
            window.trainer.cycle_attention_row(1)
            return True
        return False

    @staticmethod
    def _sub_rect(
        rect: tuple[float, float, float, float],
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> tuple[float, float, float, float]:
        left, bottom, width, height = rect
        return left + width * x, bottom + height * y, width * w, height * h

    @staticmethod
    def _curve_points(start: tuple[float, float], end: tuple[float, float], lift: float) -> list[tuple[float, float]]:
        sx, sy = start
        ex, ey = end
        cx = (sx + ex) * 0.5
        cy = max(sy, ey) + float(lift)
        points: list[tuple[float, float]] = []
        for step in range(22):
            t = step / 21.0
            omt = 1.0 - t
            x = omt * omt * sx + 2.0 * omt * t * cx + t * t * ex
            y = omt * omt * sy + 2.0 * omt * t * cy + t * t * ey
            points.append((x, y))
        return points

    def _token_positions(self, rect: tuple[float, float, float, float], count: int) -> list[tuple[float, float]]:
        left, bottom, width, height = rect
        pad_x = width * 0.08
        usable = max(1.0, width - 2.0 * pad_x)
        y = bottom + height * 0.58
        return [(left + pad_x + usable * idx / max(1, count - 1), y) for idx in range(count)]

    def _draw_connections(
        self,
        positions: list[tuple[float, float]],
        row: np.ndarray,
        source_index: int,
    ) -> None:
        source = positions[int(source_index)]
        strongest = int(np.argmax(row)) if row.size else -1
        for idx, value in enumerate(row):
            if idx == int(source_index):
                continue
            weight = max(0.0, min(1.0, float(value)))
            alpha = int(18 + 220 * weight)
            if idx == strongest:
                alpha = max(alpha, 185)
            span = abs(idx - int(source_index))
            lift = 28.0 + 9.0 * span
            points = self._curve_points(source, positions[idx], lift)
            for start, end in zip(points, points[1:]):
                arcade.draw_line(
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                    with_alpha(COLOR_SLATE_GRAY, alpha),
                    TOYBOX_CONNECTION_LINE_WIDTH,
                )

    def _draw_nodes(
        self,
        tokens: tuple[str, ...],
        positions: list[tuple[float, float]],
        *,
        highlighted_row: int,
        subject_index: int,
        distractor_index: int,
        mask_index: int,
        strongest_index: int,
        window: object,
    ) -> None:
        for idx, (token, (x, y)) in enumerate(zip(tokens, positions)):
            radius = 16.0
            fill = COLOR_DARK_NEUTRAL
            outline = COLOR_SLATE_GRAY
            inner = None
            inner_radius = None
            alpha = 220
            outline_alpha = 185
            if idx == subject_index:
                fill = COLOR_AQUA
                outline = COLOR_DEEP_TEAL
                inner = COLOR_DEEP_TEAL
                inner_radius = 7.0
            if idx == distractor_index:
                fill = COLOR_CORAL
                outline = COLOR_BRICK_RED
                inner = COLOR_BRICK_RED
                inner_radius = 7.0
            if idx == mask_index:
                fill = COLOR_LIGHT_NEUTRAL
                outline = COLOR_AQUA
                alpha = 235
            if idx == highlighted_row:
                radius = 18.0
                outline = COLOR_FOG_GRAY
                outline_alpha = 235
            if idx == strongest_index:
                outline_alpha = 255
            draw_diamond_node(
                x,
                y,
                radius=radius,
                fill_color=fill,
                outline_color=outline,
                alpha=alpha,
                outline_alpha=outline_alpha,
                outline_width=TOYBOX_CONNECTION_LINE_WIDTH,
                inner_radius=inner_radius,
                inner_color=inner,
                inner_alpha=205,
            )
            text_color = COLOR_LIGHT_NEUTRAL if idx != mask_index else COLOR_DARK_NEUTRAL
            window.text_cache.draw(
                token,
                x,
                y - 34.0,
                color=text_color if idx == mask_index else COLOR_FOG_GRAY,
                font_size=12,
                anchor_x="center",
                anchor_y="center",
                bold=idx in {subject_index, distractor_index, mask_index},
            )

    def _draw_attention_map(self, snapshot: dict[str, object], rect: tuple[float, float, float, float], window: object) -> None:
        tokens = tuple(str(token) for token in snapshot["tokens"])
        attention = np.asarray(snapshot["attention"], dtype=np.float32)
        highlighted_row = int(snapshot["highlighted_row"])
        subject_index = int(snapshot["subject_index"])
        distractor_index = int(snapshot["distractor_index"])
        mask_index = int(snapshot["mask_index"])
        row = attention[highlighted_row] if attention.ndim == 2 else np.zeros(len(tokens), dtype=np.float32)
        strongest_index = int(np.argmax(row)) if row.size else -1
        positions = self._token_positions(rect, len(tokens))
        self._draw_connections(positions, row, highlighted_row)
        self._draw_nodes(
            tokens,
            positions,
            highlighted_row=highlighted_row,
            subject_index=subject_index,
            distractor_index=distractor_index,
            mask_index=mask_index,
            strongest_index=strongest_index,
            window=window,
        )
        left, bottom, width, height = rect
        for idx, value in enumerate(row):
            x, _y = positions[idx]
            bar_h = max(3.0, height * 0.18 * max(0.0, min(1.0, float(value))))
            arcade.draw_lbwh_rectangle_filled(
                x - 9.0,
                bottom + height * 0.18,
                18.0,
                height * 0.18,
                with_alpha(COLOR_SLATE_GRAY, TOYBOX_CHART_TRACK_ALPHA),
            )
            arcade.draw_lbwh_rectangle_filled(
                x - 9.0,
                bottom + height * 0.18,
                18.0,
                bar_h,
                with_alpha(COLOR_AQUA if idx == strongest_index else COLOR_SLATE_GRAY, TOYBOX_CHART_FILL_ALPHA),
            )

    def _draw_output(self, snapshot: dict[str, object], rect: tuple[float, float, float, float], window: object) -> None:
        left, bottom, width, height = rect
        probs = np.asarray(snapshot["probs"], dtype=np.float32).reshape(-1)
        labels = tuple(str(label) for label in snapshot["labels"])
        target = int(snapshot["target"])
        pred = int(snapshot["pred"])
        pad = 18.0
        gap = 14.0
        bar_w = max(1.0, (width - 2.0 * pad - gap) / 2.0)
        track_h = max(1.0, height - 54.0)
        for idx, label in enumerate(labels[:2]):
            x = left + pad + idx * (bar_w + gap)
            y = bottom + 30.0
            value = max(0.0, min(1.0, float(probs[idx])))
            color = COLOR_FOG_GRAY if idx == pred else COLOR_SLATE_GRAY
            arcade.draw_lbwh_rectangle_filled(x, y, bar_w, track_h, with_alpha(COLOR_SLATE_GRAY, TOYBOX_CHART_TRACK_ALPHA))
            arcade.draw_lbwh_rectangle_filled(x, y, bar_w, track_h * value, with_alpha(color, TOYBOX_CHART_FILL_ALPHA))
            if idx == target:
                arcade.draw_lbwh_rectangle_outline(x, y, bar_w, track_h, with_alpha(COLOR_FOG_GRAY, 210), TOYBOX_CONNECTION_LINE_WIDTH)
            window.text_cache.draw(
                label,
                x + bar_w * 0.5,
                bottom + 15.0,
                color=COLOR_LIGHT_NEUTRAL,
                font_size=13,
                anchor_x="center",
                anchor_y="center",
                bold=idx == pred,
            )
            window.text_cache.draw(
                f"{value:.0%}",
                x + bar_w * 0.5,
                y + track_h + 12.0,
                color=COLOR_FOG_GRAY,
                font_size=11,
                anchor_x="center",
                anchor_y="center",
            )

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        secondary_rect = layout.secondary if layout.secondary is not None else layout.text
        with clipped_rect(main_rect, inset=2.0):
            self._draw_attention_map(snapshot, self._sub_rect(main_rect, 0.03, 0.08, 0.94, 0.84), window)
        with clipped_rect(secondary_rect, inset=2.0):
            self._draw_output(snapshot, secondary_rect, window)
        status = "yes" if bool(snapshot["correct"]) else "no"
        trap = "yes" if bool(snapshot["trap"]) else "no"
        extra = (
            ("template", snapshot["template_id"]),
            ("target", snapshot["target_label"]),
            ("prediction", snapshot["predicted_label"]),
            ("correct", status),
            ("subject", snapshot["subject"]),
            ("distractor", snapshot["distractor"]),
            ("trap", trap),
            ("selected", snapshot["highlighted_token"]),
            ("controls", "G new, H row"),
        )
        window.draw_info(snapshot, secondary=True, extra=extra, compact=True)
