"""Arcade renderer for trace snapshots."""

from __future__ import annotations

import arcade
import numpy as np

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_LIGHT_NEUTRAL,
    COLOR_SLATE_GRAY,
    TOYBOX_CONNECTION_LINE_WIDTH,
)
from core.arcade_view import clipped_rect, draw_diamond_node, draw_pixel_image_in_rect, marker_outer_radius, with_alpha
from demos.trace.config import TraceConfig


class TraceRenderer:
    def __init__(self, config: TraceConfig) -> None:
        self.config = config
        self.show_edges = True
        self.zoom = 1.0
        self.pan = np.zeros(2, dtype=np.float32)

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
        if symbol == arcade.key.E:
            self.show_edges = not self.show_edges
            return True
        if symbol == arcade.key.LEFT_BRACKET:
            trainer.cycle_digit(-1)
            return True
        if symbol == arcade.key.RIGHT_BRACKET:
            trainer.cycle_digit(1)
            return True
        if symbol == arcade.key.BACKSPACE:
            trainer.clear_digit_filter()
            return True
        return False

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int, *, window: object) -> bool:
        del scroll_x
        layout = getattr(window, "_last_layout", None)
        if layout is None:
            return False
        left, bottom, width, height = layout.main
        if not (left <= float(x) <= left + width and bottom <= float(y) <= bottom + height):
            return False
        factor = 1.12 if int(scroll_y) > 0 else 1.0 / 1.12
        self.zoom = float(np.clip(self.zoom * factor, 0.75, 3.0))
        return True

    def on_mouse_drag(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        buttons: int,
        modifiers: int,
        *,
        window: object,
    ) -> bool:
        del modifiers
        layout = getattr(window, "_last_layout", None)
        if layout is None or not (int(buttons) & arcade.MOUSE_BUTTON_LEFT):
            return False
        left, bottom, width, height = layout.main
        if not (left <= float(x) <= left + width and bottom <= float(y) <= bottom + height):
            return False
        self.pan += np.asarray([float(dx), float(dy)], dtype=np.float32)
        return True

    @staticmethod
    def _rect_part(rect: tuple[float, float, float, float], x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        left, bottom, width, height = rect
        return left + width * x, bottom + height * y, width * w, height * h

    @staticmethod
    def _zoom_positions(
        positions: list[np.ndarray],
        rect: tuple[float, float, float, float],
        zoom: float,
        pan: np.ndarray,
    ) -> list[np.ndarray]:
        left, bottom, width, height = rect
        center = np.asarray([left + width * 0.5, bottom + height * 0.5], dtype=np.float32)
        safe_zoom = max(0.1, float(zoom))
        offset = np.asarray(pan, dtype=np.float32).reshape(2)
        return [((np.asarray(layer, dtype=np.float32) - center) * safe_zoom + center + offset).astype(np.float32) for layer in positions]

    @staticmethod
    def _point_visible(point: np.ndarray, rect: tuple[float, float, float, float], *, pad: float = 28.0) -> bool:
        left, bottom, width, height = rect
        x, y = float(point[0]), float(point[1])
        return left - pad <= x <= left + width + pad and bottom - pad <= y <= bottom + height + pad

    def _layer_positions(self, rect: tuple[float, float, float, float], sizes: list[int]) -> list[np.ndarray]:
        left, bottom, width, height = rect
        x_values = np.linspace(left + width * 0.08, left + width * 0.92, len(sizes), dtype=np.float32)
        layers: list[np.ndarray] = []
        for layer_idx, size in enumerate(sizes):
            if layer_idx == 0:
                xs = []
                ys = []
                grid_size = min(width * 0.14, height * 0.56)
                cell = grid_size / 7.0
                x0 = float(x_values[layer_idx]) - grid_size * 0.5
                y0 = bottom + height * 0.5 + grid_size * 0.5
                for row in range(8):
                    for col in range(8):
                        xs.append(x0 + col * cell)
                        ys.append(y0 - row * cell)
                layers.append(np.column_stack([xs, ys]).astype(np.float32))
            else:
                ys = np.linspace(bottom + height * 0.08, bottom + height * 0.92, size, dtype=np.float32)
                xs = np.full(size, x_values[layer_idx], dtype=np.float32)
                layers.append(np.column_stack([xs, ys]).astype(np.float32))
        return layers

    def _draw_edges(self, positions: list[np.ndarray], edges: list[np.ndarray], rect: tuple[float, float, float, float]) -> None:
        if not self.show_edges:
            return
        for layer_idx, layer_edges in enumerate(edges):
            arr = np.asarray(layer_edges, dtype=np.float32)
            if arr.size == 0:
                continue
            max_strength = max(1e-6, float(np.max(arr[:, 3])))
            for src, dst, _signed, strength in arr:
                src_i = int(src)
                dst_i = int(dst)
                if src_i >= len(positions[layer_idx]) or dst_i >= len(positions[layer_idx + 1]):
                    continue
                ratio = max(0.0, min(1.0, float(strength) / max_strength))
                p0 = positions[layer_idx][src_i]
                p1 = positions[layer_idx + 1][dst_i]
                if not (self._point_visible(p0, rect) or self._point_visible(p1, rect)):
                    continue
                if ratio < 0.25:
                    alpha = 0
                elif ratio < 0.50:
                    alpha = 64
                elif ratio < 0.75:
                    alpha = 128
                else:
                    alpha = 192
                if alpha <= 0:
                    continue
                arcade.draw_line(
                    float(p0[0]),
                    float(p0[1]),
                    float(p1[0]),
                    float(p1[1]),
                    with_alpha(COLOR_SLATE_GRAY, alpha),
                    TOYBOX_CONNECTION_LINE_WIDTH,
                )

    def _draw_nodes(
        self,
        positions: list[np.ndarray],
        activations: list[np.ndarray],
        pred: int,
        rect: tuple[float, float, float, float],
    ) -> None:
        radius_scale = float(np.clip(self.zoom, 0.75, 3.0))
        normal_radius = marker_outer_radius("small") * radius_scale
        selected_radius = marker_outer_radius("regular") * radius_scale
        for layer_idx, (xy, values) in enumerate(zip(positions, activations)):
            vals = np.asarray(values, dtype=np.float32).reshape(-1)
            max_value = float(np.max(np.abs(vals))) if vals.size else 1.0
            for idx, (x, y) in enumerate(xy):
                point = np.asarray([x, y], dtype=np.float32)
                if not self._point_visible(point, rect, pad=18.0):
                    continue
                value = float(vals[idx]) if idx < vals.size else 0.0
                if layer_idx == len(positions) - 1:
                    ratio = max(0.0, min(1.0, value))
                    radius = selected_radius if idx == pred else normal_radius
                    fill_alpha = 75 + int(150 * ratio)
                    outline = COLOR_DEEP_TEAL
                else:
                    ratio = 0.0 if max_value <= 1e-8 else max(0.0, min(1.0, abs(value) / max_value))
                    radius = normal_radius
                    fill_alpha = 38 + int(170 * ratio)
                    outline = COLOR_DEEP_TEAL
                draw_diamond_node(
                    float(x),
                    float(y),
                    radius=radius,
                    fill_color=COLOR_AQUA,
                    outline_color=outline,
                    alpha=fill_alpha,
                    outline_alpha=225,
                    outline_width=TOYBOX_CONNECTION_LINE_WIDTH,
                )

    def _draw_preview_panel(self, snapshot: dict[str, object], rect: tuple[float, float, float, float], window: object) -> None:
        del window
        draw_pixel_image_in_rect(
            np.asarray(snapshot["image"], dtype=np.float32),
            rect,
            on_color=COLOR_LIGHT_NEUTRAL,
            off_color=COLOR_DARK_NEUTRAL,
            border_color=COLOR_SLATE_GRAY,
            padding=min(rect[2], rect[3]) * 0.09,
        )

    def draw(self, snapshot: dict[str, object], window: object) -> None:
        layout = window.layout(secondary=True)
        main_rect = layout.main
        preview_rect = layout.secondary if layout.secondary is not None else layout.text
        graph_rect = self._rect_part(main_rect, 0.03, 0.04, 0.94, 0.92)
        layers = [
            np.asarray(snapshot["input"], dtype=np.float32),
            np.asarray(snapshot["hidden1"], dtype=np.float32),
            np.asarray(snapshot["hidden2"], dtype=np.float32),
            np.asarray(snapshot["probs"], dtype=np.float32),
        ]
        base_positions = self._layer_positions(graph_rect, [64, len(layers[1]), len(layers[2]), 10])
        positions = self._zoom_positions(base_positions, graph_rect, self.zoom, self.pan)
        with clipped_rect(main_rect):
            self._draw_edges(positions, [np.asarray(arr, dtype=np.float32) for arr in snapshot["top_edges"]], main_rect)
            self._draw_nodes(positions, layers, int(snapshot["pred"]), main_rect)
        self._draw_preview_panel(snapshot, preview_rect, window)
        status = "correct" if bool(snapshot["correct"]) else "misclassified"
        extra = [
            f"label: {int(snapshot['label'])}",
            f"prediction: {int(snapshot['pred'])}",
            f"result: {status}",
            f"variation: {int(snapshot.get('variation_index', 0)) + 1}",
            f"zoom: {self.zoom:.2f}x",
        ]
        window.draw_info(snapshot, secondary=True, extra=tuple(extra), compact=True)
