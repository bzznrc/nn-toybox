"""Arcade drawing helpers shared by interactive viewers."""

from __future__ import annotations

import array
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import re
from typing import Iterable, Protocol, Sequence

import arcade
from arcade.types import Color
from arcade.window_commands import get_window
import numpy as np

from core.arcade_style import (
    CLASS_COLOR_PAIRS,
    COLOR_AQUA,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_LIGHT_NEUTRAL,
    COLOR_SLATE_GRAY,
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_CELL_INSET,
    DEFAULT_STATUS_BAR_FONT_SIZE,
    GAME_TITLE_FONT_NAME,
    GAME_UI_FONT_NAME,
    POINT_MARKERS,
    TOYBOX_GAP,
    TOYBOX_INFO_LINE_HEIGHT,
    TOYBOX_INNER_GAP,
    TOYBOX_MAIN_WIDTH_FRACTION,
    TOYBOX_MARGIN,
    TOYBOX_PANEL_OUTLINE_ALPHA,
    TOYBOX_PANEL_OUTLINE_WIDTH,
    TOYBOX_SECONDARY_HEIGHT_FRACTION,
    TOYBOX_TEXT_RIGHT_PADDING,
    TOYBOX_TEXT_SCROLL_WIDTH,
    TOYBOX_CHART_FILL_ALPHA,
    TOYBOX_CHART_TRACK_ALPHA,
)
from core.shared_config import (
    PLAYFIELD_HEIGHT,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from core.checkpoints import create_run_from_config, write_json
from core.config import CommonConfig, DisplayConfig, to_dict


def _env_visible_default(default: bool = True) -> bool:
    raw = os.getenv("NN_TOYBOX_RENDER_VISIBLE")
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def with_alpha(color: tuple[int, int, int] | tuple[int, int, int, int], alpha: int | float) -> tuple[int, int, int, int]:
    return int(color[0]), int(color[1]), int(color[2]), int(max(0, min(255, round(float(alpha)))))


def camel_case_text(value: object) -> str:
    """Format UI text as spaced Camel Case while preserving numbers and acronyms."""
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+-\s+", " - ", text)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) > 1 and token.isupper():
            return token
        return token[:1].upper() + token[1:].lower()

    return re.sub(r"[A-Za-z][A-Za-z0-9]*", replace, text)


def format_info_line(item: object) -> str:
    if isinstance(item, tuple) and len(item) == 2:
        key, value = item
        return f"{camel_case_text(key)}: {camel_case_text(value)}"
    text = str(item).strip()
    if ":" in text:
        key, value = text.split(":", 1)
        return f"{camel_case_text(key)}: {camel_case_text(value)}"
    return camel_case_text(text)


class TextCache:
    """Reusable cache of arcade.Text objects."""

    def __init__(self, max_entries: int = 1024) -> None:
        self._cached_text = lru_cache(maxsize=max(1, int(max_entries)))(self._build_text)

    @staticmethod
    def _font_tuple(font_name: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(font_name, str):
            return (font_name,)
        return tuple(str(item) for item in font_name)

    @staticmethod
    def _rgba(color: tuple[int, int, int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if len(color) == 4:
            return int(color[0]), int(color[1]), int(color[2]), int(color[3])
        return int(color[0]), int(color[1]), int(color[2]), 255

    @staticmethod
    def _build_text(
        text: str,
        color: tuple[int, int, int, int],
        font_size: int,
        font_name: tuple[str, ...],
        anchor_x: str,
        anchor_y: str,
        bold: bool,
    ) -> arcade.Text:
        return arcade.Text(
            text=text,
            x=0,
            y=0,
            color=color,
            font_size=font_size,
            font_name=font_name,
            bold=bold,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )

    def draw(
        self,
        text: str,
        x: float,
        y: float,
        *,
        color: tuple[int, int, int] | tuple[int, int, int, int] = COLOR_FOG_GRAY,
        font_size: int | float = DEFAULT_STATUS_BAR_FONT_SIZE,
        font_name: str | Iterable[str] = GAME_UI_FONT_NAME,
        anchor_x: str = "left",
        anchor_y: str = "baseline",
        bold: bool = False,
    ) -> None:
        text_obj = self._cached_text(
            str(text),
            self._rgba(color),
            int(font_size),
            self._font_tuple(font_name),
            str(anchor_x),
            str(anchor_y),
            bool(bold),
        )
        text_obj.x = float(x)
        text_obj.y = float(y)
        text_obj.draw()

    def measure_width(
        self,
        text: str,
        *,
        color: tuple[int, int, int] | tuple[int, int, int, int] = COLOR_FOG_GRAY,
        font_size: int | float = DEFAULT_STATUS_BAR_FONT_SIZE,
        font_name: str | Iterable[str] = GAME_UI_FONT_NAME,
        anchor_x: str = "left",
        anchor_y: str = "baseline",
        bold: bool = False,
    ) -> float:
        text_obj = self._cached_text(
            str(text),
            self._rgba(color),
            int(font_size),
            self._font_tuple(font_name),
            str(anchor_x),
            str(anchor_y),
            bool(bold),
        )
        try:
            return float(text_obj.content_width)
        except RuntimeError:
            weight = 0.64 if bold else 0.58
            return float(len(str(text)) * int(font_size) * weight)


class DemoWindow(arcade.Window):
    """Small Arcade window with rl-toybox-inspired defaults."""

    def __init__(self, title: str, *, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT) -> None:
        arcade.close_window()
        super().__init__(
            int(width),
            int(height),
            str(title),
            vsync=False,
            visible=_env_visible_default(True),
        )
        self.text_cache = TextCache()
        self.background_color = COLOR_DARK_NEUTRAL

    @property
    def playfield_bottom(self) -> float:
        return float(DEFAULT_BOTTOM_BAR_HEIGHT)

    @property
    def playfield_top(self) -> float:
        return float(self.height)

    def draw_status_bar(self, entries: Sequence[tuple[str, object]]) -> None:
        arcade.draw_lbwh_rectangle_filled(0, 0, self.width, DEFAULT_BOTTOM_BAR_HEIGHT, COLOR_DARK_NEUTRAL)
        x = 8.0
        y = DEFAULT_BOTTOM_BAR_HEIGHT * 0.5 - 1.0
        for key, value in entries:
            segment = format_info_line((key, value))
            self.text_cache.draw(segment, x, y, color=COLOR_FOG_GRAY, font_size=12, anchor_y="center")
            x += max(86.0, len(segment) * 7.0 + 18.0)

    def draw_title(self, title: str, subtitle: str = "") -> None:
        self.text_cache.draw(
            title,
            18,
            self.height - 22,
            color=COLOR_LIGHT_NEUTRAL,
            font_size=16,
            font_name=GAME_TITLE_FONT_NAME,
            anchor_y="center",
        )
        if subtitle:
            self.text_cache.draw(
                subtitle,
                18,
                self.height - 42,
                color=COLOR_SLATE_GRAY,
                font_size=11,
                anchor_y="center",
            )


def panel(left: float, bottom: float, width: float, height: float) -> tuple[float, float, float, float]:
    return float(left), float(bottom), float(width), float(height)


@dataclass(frozen=True)
class ToyboxLayout:
    """Shared live-demo layout: square main canvas plus right sidebar regions."""

    main: tuple[float, float, float, float]
    secondary: tuple[float, float, float, float] | None
    text: tuple[float, float, float, float]
    right: tuple[float, float, float, float]


def fit_rect(
    rect: tuple[float, float, float, float],
    *,
    aspect: float = 1.0,
    align_x: str = "center",
    align_y: str = "center",
) -> tuple[float, float, float, float]:
    left, bottom, width, height = rect
    target_aspect = max(1e-6, float(aspect))
    if width / max(1e-6, height) > target_aspect:
        fitted_height = height
        fitted_width = fitted_height * target_aspect
    else:
        fitted_width = width
        fitted_height = fitted_width / target_aspect

    if align_x == "left":
        fitted_left = left
    elif align_x == "right":
        fitted_left = left + width - fitted_width
    else:
        fitted_left = left + (width - fitted_width) * 0.5

    if align_y == "top":
        fitted_bottom = bottom + height - fitted_height
    elif align_y == "bottom":
        fitted_bottom = bottom
    else:
        fitted_bottom = bottom + (height - fitted_height) * 0.5

    return panel(fitted_left, fitted_bottom, fitted_width, fitted_height)


def toybox_layout(
    width: int | float,
    height: int | float,
    *,
    secondary: bool = False,
    margin: float = TOYBOX_MARGIN,
    gap: float = TOYBOX_GAP,
) -> ToyboxLayout:
    outer_width = float(width)
    outer_height = float(height)
    safe_margin = max(0.0, float(margin))
    safe_gap = max(0.0, float(gap))
    content_width = max(1.0, outer_width - 2.0 * safe_margin)
    content_height = max(1.0, outer_height - 2.0 * safe_margin)
    main_fraction = max(0.05, min(0.95, float(TOYBOX_MAIN_WIDTH_FRACTION)))
    sidebar_ratio = (1.0 - main_fraction) / main_fraction
    main_size = min(content_height, max(1.0, (content_width - safe_gap) / (1.0 + sidebar_ratio)))
    right_width = max(1.0, main_size * sidebar_ratio)
    block_width = main_size + safe_gap + right_width
    block_left = safe_margin + max(0.0, (content_width - block_width) * 0.5)
    block_bottom = safe_margin + max(0.0, (content_height - main_size) * 0.5)

    main = panel(block_left, block_bottom, main_size, main_size)
    right = panel(block_left + main_size + safe_gap, block_bottom, right_width, main_size)

    secondary_rect = None
    if secondary:
        secondary_height = min(right_width, main[3] * float(TOYBOX_SECONDARY_HEIGHT_FRACTION))
        secondary_rect = panel(right[0], main[1] + main[3] - secondary_height, right_width, secondary_height)
        text_top = secondary_rect[1] - float(gap)
    else:
        text_top = right[1] + right[3]

    text_bottom = main[1]
    text_height = max(1.0, text_top - text_bottom)
    text_rect = panel(right[0], text_bottom, right_width, text_height)
    return ToyboxLayout(main=main, secondary=secondary_rect, text=text_rect, right=right)


def split_columns(
    rect: tuple[float, float, float, float],
    count: int,
    *,
    gap: float = TOYBOX_INNER_GAP,
) -> tuple[tuple[float, float, float, float], ...]:
    left, bottom, width, height = rect
    columns = max(1, int(count))
    total_gap = float(gap) * float(columns - 1)
    column_width = max(1.0, (float(width) - total_gap) / float(columns))
    return tuple(
        panel(float(left) + idx * (column_width + float(gap)), float(bottom), column_width, float(height))
        for idx in range(columns)
    )


def _split_long_token(
    text_cache: TextCache,
    token: str,
    max_width: float,
    *,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    font_size: int,
    bold: bool = False,
) -> list[str]:
    if text_cache.measure_width(token, color=color, font_size=font_size, bold=bold) <= max_width:
        return [token]
    parts: list[str] = []
    current = ""
    for char in token:
        candidate = current + char
        if current and text_cache.measure_width(candidate, color=color, font_size=font_size, bold=bold) > max_width:
            parts.append(current)
            current = char
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [token]


def _wrap_text(
    text_cache: TextCache,
    text: str,
    first_width: float,
    next_width: float | None = None,
    *,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    font_size: int,
    bold: bool = False,
) -> list[str]:
    widths = (max(1.0, float(first_width)), max(1.0, float(first_width if next_width is None else next_width)))
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    line_width = widths[0]
    for word in words:
        pieces = _split_long_token(text_cache, word, line_width, color=color, font_size=font_size, bold=bold)
        for piece in pieces:
            candidate = piece if not current else f"{current} {piece}"
            if current and text_cache.measure_width(candidate, color=color, font_size=font_size, bold=bold) > line_width:
                lines.append(current)
                current = piece
                line_width = widths[1]
            else:
                current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _wrapped_text_rows(
    text_cache: TextCache,
    lines: Sequence[str],
    max_width: float,
    *,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    font_size: int,
) -> list[str]:
    rows: list[str] = []
    usable_width = max(1.0, float(max_width))
    for text in lines:
        rows.extend(_wrap_text(text_cache, text, usable_width, color=color, font_size=font_size))
    return rows


def draw_text_area(
    text_cache: TextCache,
    lines: Sequence[object],
    rect: tuple[float, float, float, float],
    *,
    scroll: int = 0,
    color: tuple[int, int, int] | tuple[int, int, int, int] = COLOR_FOG_GRAY,
    muted_color: tuple[int, int, int] | tuple[int, int, int, int] = COLOR_SLATE_GRAY,
    font_size: int = 12,
    line_height: float = TOYBOX_INFO_LINE_HEIGHT,
) -> int:
    left, bottom, width, height = rect
    line_count = max(1, int(height // float(line_height)))
    texts = [str(line) for line in lines if str(line)]
    text_right_pad = float(TOYBOX_TEXT_RIGHT_PADDING) + float(TOYBOX_TEXT_SCROLL_WIDTH)
    usable_width = max(1.0, float(width) - text_right_pad)
    rows = _wrapped_text_rows(text_cache, texts, usable_width, color=color, font_size=font_size)
    max_scroll = max(0, len(rows) - line_count)
    offset = max(0, min(int(scroll), max_scroll))
    top = bottom + height
    for idx, text in enumerate(rows[offset : offset + line_count]):
        y = top - 4.0 - idx * float(line_height)
        if ":" in text:
            key, value = text.split(":", 1)
            key_text = f"{key}:"
            key_width = text_cache.measure_width(key_text, color=color, font_size=font_size, bold=True)
            text_cache.draw(key_text, left, y, color=color, font_size=font_size, anchor_y="top", bold=True)
            if value:
                text_cache.draw(
                    value.lstrip(),
                    left + key_width + 4.0,
                    y,
                    color=color,
                    font_size=font_size,
                    anchor_y="top",
                )
        else:
            text_cache.draw(text, left, y, color=color, font_size=font_size, anchor_y="top")

    if max_scroll > 0:
        track_x = left + width - float(TOYBOX_TEXT_SCROLL_WIDTH) * 0.5
        arcade.draw_line(track_x, bottom, track_x, bottom + height, with_alpha(muted_color, 80), 1.0)
        thumb_height = max(18.0, height * (line_count / max(1, len(rows))))
        thumb_span = max(1.0, height - thumb_height)
        thumb_bottom = bottom + thumb_span * (1.0 - offset / max(1, max_scroll))
        arcade.draw_line(track_x, thumb_bottom, track_x, thumb_bottom + thumb_height, with_alpha(color, 170), 2.0)
    return max_scroll


def draw_panel_outline(rect: tuple[float, float, float, float], title: str, text_cache: TextCache | None = None) -> None:
    left, bottom, width, height = rect
    arcade.draw_lbwh_rectangle_outline(
        left,
        bottom,
        width,
        height,
        with_alpha(COLOR_FOG_GRAY, TOYBOX_PANEL_OUTLINE_ALPHA),
        TOYBOX_PANEL_OUTLINE_WIDTH,
    )
    if text_cache is not None:
        text_cache.draw(
            camel_case_text(title),
            left + 8,
            bottom + height - 18,
            color=COLOR_FOG_GRAY,
            font_size=11,
            anchor_y="center",
        )


def draw_toybox_frame(layout: ToyboxLayout) -> None:
    draw_panel_outline(layout.main, "")


def _scissor_tuple(rect: tuple[float, float, float, float], *, inset: float = 0.0) -> tuple[int, int, int, int]:
    left, bottom, width, height = rect
    pad = max(0.0, float(inset))
    x = int(round(float(left) + pad))
    y = int(round(float(bottom) + pad))
    w = int(round(max(0.0, float(width) - 2.0 * pad)))
    h = int(round(max(0.0, float(height) - 2.0 * pad)))
    return x, y, w, h


def _intersect_scissor(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    bottom = max(ay, by)
    right = min(ax + aw, bx + bw)
    top = min(ay + ah, by + bh)
    return left, bottom, max(0, right - left), max(0, top - bottom)


@contextmanager
def clipped_rect(rect: tuple[float, float, float, float], *, inset: float = 0.0):
    """Clip Arcade drawing to a panel rectangle.

    Renderers should wrap main-panel drawing with this helper so zoomed or large
    visuals cannot leak into neighboring panels.
    """

    window = get_window()
    ctx = window.ctx
    previous = ctx.scissor
    scissor = _scissor_tuple(rect, inset=inset)
    if previous is not None:
        scissor = _intersect_scissor(tuple(int(v) for v in previous), scissor)
    ctx.scissor = scissor
    try:
        yield
    finally:
        ctx.scissor = previous


def world_to_panel(
    points: np.ndarray,
    rect: tuple[float, float, float, float],
    *,
    xlim: tuple[float, float] = (-1.6, 1.6),
    ylim: tuple[float, float] = (-1.6, 1.6),
) -> np.ndarray:
    left, bottom, width, height = rect
    pts = np.asarray(points, dtype=np.float32)
    x = (pts[:, 0] - float(xlim[0])) / max(1e-6, float(xlim[1] - xlim[0]))
    y = (pts[:, 1] - float(ylim[0])) / max(1e-6, float(ylim[1] - ylim[0]))
    return np.column_stack([left + x * width, bottom + y * height]).astype(np.float32)


def fit_points_to_panel(
    points: np.ndarray,
    rect: tuple[float, float, float, float],
    *,
    fill: float = 0.95,
) -> np.ndarray:
    left, bottom, width, height = rect
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    xy = pts[:, :2]
    lo = np.min(xy, axis=0)
    hi = np.max(xy, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    target = np.asarray([width, height], dtype=np.float32) * float(max(0.01, min(1.0, fill)))
    scale = float(np.min(target / span))
    center = (lo + hi) * 0.5
    panel_center = np.asarray([left + width * 0.5, bottom + height * 0.5], dtype=np.float32)
    return ((xy - center) * scale + panel_center).astype(np.float32)


def marker_outer_radius(marker: str) -> float:
    marker_spec = POINT_MARKERS.get(str(marker))
    if marker_spec is None:
        valid = ", ".join(sorted(POINT_MARKERS))
        raise ValueError(f"Unknown point marker '{marker}'. Valid: {valid}")
    return float(marker_spec["outer_radius"])


def clamp_point_to_rect(
    x: float,
    y: float,
    rect: tuple[float, float, float, float],
    *,
    inset: float = 0.0,
) -> tuple[float, float]:
    left, bottom, width, height = rect
    pad = max(0.0, float(inset))
    min_x = float(left) + pad
    max_x = float(left) + max(pad, float(width) - pad)
    min_y = float(bottom) + pad
    max_y = float(bottom) + max(pad, float(height) - pad)
    return (
        max(min_x, min(max_x, float(x))),
        max(min_y, min(max_y, float(y))),
    )


def diamond_points(x: float, y: float, radius: float) -> list[tuple[float, float]]:
    cx = float(x)
    cy = float(y)
    r = float(radius)
    return [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)]


def draw_diamond_node(
    x: float,
    y: float,
    *,
    radius: float,
    fill_color: tuple[int, int, int] | tuple[int, int, int, int],
    outline_color: tuple[int, int, int] | tuple[int, int, int, int] = COLOR_DEEP_TEAL,
    alpha: int = 230,
    outline_alpha: int = 230,
    outline_width: float = 1.5,
    inner_radius: float | None = None,
    inner_color: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
    inner_alpha: int | None = None,
) -> None:
    """Draw the shared diamond-square visual primitive used for nodes/markers."""

    outer = diamond_points(float(x), float(y), float(radius))
    arcade.draw_polygon_filled(outer, with_alpha(fill_color, alpha))
    if float(outline_width) > 0.0 and int(outline_alpha) > 0:
        arcade.draw_polygon_outline(outer, with_alpha(outline_color, outline_alpha), float(outline_width))
    if inner_radius is not None and inner_color is not None:
        arcade.draw_polygon_filled(
            diamond_points(float(x), float(y), float(inner_radius)),
            with_alpha(inner_color, alpha if inner_alpha is None else inner_alpha),
        )


def draw_diamond_marker(
    x: float,
    y: float,
    *,
    outer_color: tuple[int, int, int] | tuple[int, int, int, int],
    inner_color: tuple[int, int, int] | tuple[int, int, int, int],
    marker: str = "regular",
    alpha: int = 230,
) -> None:
    cx = float(x)
    cy = float(y)
    outer_radius = marker_outer_radius(marker)
    marker_spec = POINT_MARKERS[str(marker)]
    inner_radius = float(marker_spec["inner_radius"])
    draw_diamond_node(
        cx,
        cy,
        radius=outer_radius,
        fill_color=outer_color,
        outline_color=outer_color,
        alpha=alpha,
        outline_alpha=0,
        outline_width=0.0,
        inner_radius=inner_radius,
        inner_color=inner_color,
        inner_alpha=alpha,
    )


def _draw_diamond_layer(
    centers: Sequence[tuple[float, float]] | np.ndarray,
    *,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    marker: str = "regular",
    alpha: int = 230,
    inner: bool = False,
) -> None:
    pts = np.asarray(centers, dtype=np.float32)
    if pts.size == 0:
        return
    pts = pts.reshape(-1, 2)
    marker_spec = POINT_MARKERS.get(str(marker))
    if marker_spec is None:
        valid = ", ".join(sorted(POINT_MARKERS))
        raise ValueError(f"Unknown point marker '{marker}'. Valid: {valid}")
    radius_key = "inner_radius" if bool(inner) else "outer_radius"
    radius = float(marker_spec[radius_key])
    size = radius * math.sqrt(2.0)

    window = get_window()
    ctx = window.ctx
    program = ctx.shape_rectangle_filled_unbuffered_program  # type: ignore[attr-defined]
    geometry = ctx.shape_rectangle_filled_unbuffered_geometry
    buffer = ctx.shape_rectangle_filled_unbuffered_buffer  # type: ignore[attr-defined]
    point_array = array.array("f", pts[:, :2].reshape(-1).tolist())

    buffer.orphan(size=len(pts) * 8)
    ctx.enable(ctx.BLEND)
    program["color"] = Color.from_iterable(with_alpha(color, alpha)).normalized
    program["shape"] = size, size, 45.0
    buffer.write(data=point_array)
    geometry.render(program, instances=len(pts))
    ctx.disable(ctx.BLEND)


def draw_points(
    points: np.ndarray,
    labels: np.ndarray | None,
    rect: tuple[float, float, float, float],
    *,
    marker: str = "small",
    alpha: int = 230,
    limit: int | None = None,
    xlim: tuple[float, float] = (-1.6, 1.6),
    ylim: tuple[float, float] = (-1.6, 1.6),
    color_pair: tuple[
        tuple[int, int, int] | tuple[int, int, int, int],
        tuple[int, int, int] | tuple[int, int, int, int],
    ]
    | None = None,
    batched: bool = True,
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    label_arr = None if labels is None else np.asarray(labels)
    if limit is not None:
        pts = pts[: int(limit)]
        if label_arr is not None:
            label_arr = label_arr[: int(limit)]
    if pts.size == 0:
        return
    xy_pts = pts[:, :2]
    finite = np.isfinite(xy_pts).all(axis=1)
    in_bounds = (
        finite
        & (xy_pts[:, 0] >= float(xlim[0]))
        & (xy_pts[:, 0] <= float(xlim[1]))
        & (xy_pts[:, 1] >= float(ylim[0]))
        & (xy_pts[:, 1] <= float(ylim[1]))
    )
    if not np.any(in_bounds):
        return
    pts = pts[in_bounds]
    if label_arr is not None:
        label_arr = label_arr[in_bounds]
    radius = marker_outer_radius(marker)
    xy = world_to_panel(pts, rect, xlim=xlim, ylim=ylim)
    centers = np.empty((len(xy), 2), dtype=np.float32)
    pair_keys: list[
        tuple[
            tuple[int, int, int] | tuple[int, int, int, int],
            tuple[int, int, int] | tuple[int, int, int, int],
        ]
    ] = []
    for idx, (x, y) in enumerate(xy):
        centers[idx] = clamp_point_to_rect(float(x), float(y), rect, inset=radius)
        label = int(label_arr[idx]) if label_arr is not None else 0
        pair_keys.append(color_pair if color_pair is not None else CLASS_COLOR_PAIRS[label % len(CLASS_COLOR_PAIRS)])

    if batched:
        groups: dict[
            tuple[
                tuple[int, int, int] | tuple[int, int, int, int],
                tuple[int, int, int] | tuple[int, int, int, int],
            ],
            list[tuple[float, float]],
        ] = {}
        for center, pair in zip(centers, pair_keys):
            groups.setdefault(pair, []).append((float(center[0]), float(center[1])))
        for (outer, inner), group_centers in groups.items():
            _draw_diamond_layer(group_centers, color=outer, marker=marker, alpha=alpha)
            _draw_diamond_layer(group_centers, color=inner, marker=marker, alpha=alpha, inner=True)
        return

    for center, (outer, inner) in zip(centers, pair_keys):
        draw_diamond_marker(
            float(center[0]),
            float(center[1]),
            outer_color=outer,
            inner_color=inner,
            marker=marker,
            alpha=alpha,
        )


def draw_curve(values: Sequence[float], rect: tuple[float, float, float, float], color: tuple[int, int, int] = COLOR_FOG_GRAY) -> None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 2:
        return
    left, bottom, width, height = rect
    pad = 1.0
    left = left + pad
    bottom = bottom + pad
    width = max(1.0, width - 2.0 * pad)
    height = max(1.0, height - 2.0 * pad)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = max(1e-6, hi - lo)
    coords = []
    for idx, value in enumerate(arr):
        x = left + (idx / max(1, arr.size - 1)) * width
        y = bottom + ((float(value) - lo) / span) * height
        coords.append((x, y))
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        arcade.draw_line(x0, y0, x1, y1, color, 2.0)


def draw_vertical_bars(
    values: Sequence[float],
    rect: tuple[float, float, float, float],
    *,
    color: tuple[int, int, int] = COLOR_SLATE_GRAY,
    gap: float = 6.0,
    pad: float = 8.0,
    min_height: float = 8.0,
) -> None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return
    left, bottom, width, height = rect
    inner_left = left + float(pad)
    inner_bottom = bottom + float(pad)
    inner_width = max(1.0, width - 2.0 * float(pad))
    inner_height = max(1.0, height - 2.0 * float(pad))
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = max(1e-6, hi - lo)
    bar_w = max(2.0, (inner_width - float(gap) * max(0, arr.size - 1)) / float(arr.size))
    for idx, value in enumerate(arr):
        x = inner_left + idx * (bar_w + float(gap))
        ratio = (float(value) - lo) / span
        bar_h = float(min_height) + ratio * max(1.0, inner_height - float(min_height))
        arcade.draw_lbwh_rectangle_filled(x, inner_bottom, bar_w, inner_height, with_alpha(color, TOYBOX_CHART_TRACK_ALPHA))
        arcade.draw_lbwh_rectangle_filled(x, inner_bottom, bar_w, bar_h, with_alpha(color, TOYBOX_CHART_FILL_ALPHA))


def draw_pixel_image(
    image: np.ndarray,
    left: float,
    bottom: float,
    *,
    scale: float = 12.0,
    on_color: tuple[int, int, int] = COLOR_AQUA,
    off_color: tuple[int, int, int] = COLOR_DARK_NEUTRAL,
    border_color: tuple[int, int, int] | None = None,
) -> None:
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img[0]
    rows, cols = img.shape
    for row in range(rows):
        for col in range(cols):
            value = float(np.clip(img[row, col], 0.0, 1.0))
            color = (
                int(off_color[0] + (on_color[0] - off_color[0]) * value),
                int(off_color[1] + (on_color[1] - off_color[1]) * value),
                int(off_color[2] + (on_color[2] - off_color[2]) * value),
            )
            x = float(left) + col * float(scale)
            y = float(bottom) + (rows - 1 - row) * float(scale)
            arcade.draw_lbwh_rectangle_filled(x, y, float(scale), float(scale), color)
    if border_color is not None:
        arcade.draw_lbwh_rectangle_outline(left, bottom, cols * scale, rows * scale, border_color, 1.0)


class Trainer(Protocol):
    step_count: int
    metrics: dict[str, object]

    def step(self, n_steps: int = 1) -> None: ...

    def snapshot(self) -> dict[str, object]: ...

    def reset(self, seed: int | None = None) -> None: ...

    def save(self, run_paths: object) -> None: ...


class Renderer(Protocol):
    def draw(self, snapshot: dict[str, object], window: "LiveTrainingWindow") -> None: ...


class LiveTrainingWindow(DemoWindow):
    """Shared window that owns timing and controls, while trainers own learning."""

    def __init__(
        self,
        *,
        trainer: Trainer,
        renderer: Renderer,
        config: CommonConfig,
        display_config: DisplayConfig,
        title: str,
    ) -> None:
        super().__init__(title, width=int(display_config.window_width), height=int(display_config.window_height))
        self.trainer = trainer
        self.renderer = renderer
        self.toybox_config = config
        self.display_config = display_config
        self.paused = bool(display_config.paused)
        self.speed_index = 0
        self.speed_values = (max(1, int(display_config.steps_per_frame)), 4, 16)
        self.info_scroll = 0
        self.background_color = tuple(display_config.background_color)
        self._last_layout: ToyboxLayout | None = None
        arcade.set_background_color(COLOR_DARK_NEUTRAL)
        if int(display_config.fps) > 0:
            self.set_update_rate(1.0 / float(display_config.fps))

    @property
    def steps_per_frame(self) -> int:
        return int(self.speed_values[self.speed_index])

    def on_update(self, delta_time: float) -> None:
        del delta_time
        if self.paused:
            return
        self.trainer.step(self.steps_per_frame)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if self._renderer_mouse_event("on_key_press", symbol, modifiers):
            return
        if symbol == arcade.key.SPACE:
            self.paused = not self.paused
        elif symbol == arcade.key.R:
            self.trainer.reset(int(self.toybox_config.seed))
        elif symbol == arcade.key.N:
            self.toybox_config.seed = int(self.toybox_config.seed) + 1
            self.trainer.reset(int(self.toybox_config.seed))
        elif symbol == arcade.key.S:
            self._save_display_snapshot()
        elif symbol == arcade.key.KEY_1:
            self.speed_index = 0
        elif symbol == arcade.key.KEY_2:
            self.speed_index = 1
        elif symbol == arcade.key.KEY_3:
            self.speed_index = 2
        elif symbol == arcade.key.ESCAPE:
            self.close()

    def _renderer_mouse_event(self, name: str, *args: object) -> bool:
        handler = getattr(self.renderer, name, None)
        if not callable(handler):
            return False
        return bool(handler(*args, window=self))

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self._renderer_mouse_event("on_mouse_motion", x, y, dx, dy)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        self._renderer_mouse_event("on_mouse_drag", x, y, dx, dy, buttons, modifiers)

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        if self._renderer_mouse_event("on_mouse_scroll", x, y, scroll_x, scroll_y):
            return
        self.info_scroll = max(0, int(self.info_scroll) - int(scroll_y))

    def _save_display_snapshot(self) -> Path:
        run_paths = create_run_from_config(self.toybox_config)
        write_json(run_paths.config_path, to_dict(self.toybox_config))
        self.trainer.save(run_paths)
        write_json(run_paths.metrics_path, self.trainer.metrics)
        return run_paths.run_dir

    def on_draw(self) -> None:
        self.clear(self.background_color)
        self._last_layout = None
        snapshot = self.trainer.snapshot()
        self.renderer.draw(snapshot, self)
        if self._last_layout is not None:
            draw_toybox_frame(self._last_layout)

    def layout(self, *, secondary: bool = False):
        self._last_layout = toybox_layout(self.width, self.height, secondary=bool(secondary))
        return self._last_layout

    def info_lines(
        self,
        snapshot: dict[str, object],
        extra: list[str] | tuple[str, ...] = (),
        *,
        compact: bool = False,
    ) -> list[str]:
        metrics = dict(snapshot.get("metrics", self.trainer.metrics))
        loss = metrics.get("loss")
        loss_text = "N/A" if loss is None else f"{float(loss):.4f}"
        head = [] if compact else [
            ("demo", self.toybox_config.demo),
            ("dataset", self.toybox_config.dataset),
        ]
        lines = [
            ("step", int(self.trainer.step_count)),
            ("loss", loss_text),
        ]
        if not compact:
            lines.append(("speed", f"{self.steps_per_frame} step/frame"))
        if "accuracy" in metrics:
            lines.append(("accuracy", f"{float(metrics['accuracy']):.1%}"))
        if "mean_error" in metrics:
            lines.append(("mean error", f"{float(metrics['mean_error']):.5f}"))
        if "tokens" in metrics:
            lines.append(("tokens", int(metrics["tokens"])))
        if "positive_pairs" in metrics:
            lines.append(("positive pairs", int(metrics["positive_pairs"])))
        if "timesteps" in metrics:
            lines.append(("timesteps", int(metrics["timesteps"])))
        if self.paused:
            lines.append(("paused", "yes"))

        selected_lines = []
        extra_lines = []
        for item in extra:
            if not str(item):
                continue
            formatted = format_info_line(item)
            key = formatted.split(":", 1)[0].strip().lower() if ":" in formatted else ""
            if key == "selected":
                selected_lines.append(item)
            else:
                extra_lines.append(item)
        ordered = [*head, *selected_lines, *lines, *extra_lines]
        return [line for line in (format_info_line(item) for item in ordered) if line]

    def draw_info(
        self,
        snapshot: dict[str, object],
        *,
        secondary: bool = False,
        extra: list[str] | tuple[str, ...] = (),
        compact: bool = False,
    ) -> None:
        layout = self.layout(secondary=secondary)
        max_scroll = draw_text_area(
            self.text_cache,
            self.info_lines(snapshot, extra, compact=compact),
            layout.text,
            scroll=int(self.info_scroll),
        )
        self.info_scroll = min(int(self.info_scroll), int(max_scroll))


def run_window(window: arcade.Window) -> None:
    del window
    arcade.run()


def require_run_file(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Expected run artifact at {target}")
    return target
