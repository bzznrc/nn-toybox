"""Arcade drawing helpers shared by interactive viewers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Iterable, Protocol, Sequence

import arcade
import numpy as np

from core.arcade_style import (
    CLASS_COLOR_PAIRS,
    COLOR_AQUA,
    COLOR_DARK_NEUTRAL,
    COLOR_FOG_GRAY,
    COLOR_LIGHT_NEUTRAL,
    COLOR_SLATE_GRAY,
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_CELL_INSET,
    DEFAULT_STATUS_BAR_FONT_SIZE,
    GAME_TITLE_FONT_NAME,
    GAME_UI_FONT_NAME,
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
    text = re.sub(r"[_-]+", " ", text)

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
    ) -> arcade.Text:
        return arcade.Text(
            text=text,
            x=0,
            y=0,
            color=color,
            font_size=font_size,
            font_name=font_name,
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
    ) -> None:
        text_obj = self._cached_text(
            str(text),
            self._rgba(color),
            int(font_size),
            self._font_tuple(font_name),
            str(anchor_x),
            str(anchor_y),
        )
        text_obj.x = float(x)
        text_obj.y = float(y)
        text_obj.draw()


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
    margin: float = 24.0,
    gap: float = 18.0,
) -> ToyboxLayout:
    content_left = float(margin)
    content_bottom = float(margin)
    content_width = max(1.0, float(width) - 2.0 * float(margin) - float(gap))
    content_height = max(1.0, float(height) - 2.0 * float(margin))
    left_width = content_width * 0.75
    right_width = content_width * 0.25

    left_area = panel(content_left, content_bottom, left_width, content_height)
    main = fit_rect(left_area, aspect=1.0, align_x="center", align_y="top")
    right = panel(content_left + left_width + float(gap), main[1], right_width, main[3])

    secondary_rect = None
    if secondary:
        secondary_height = min(right_width, main[3] * 0.42)
        secondary_rect = panel(right[0], main[1] + main[3] - secondary_height, right_width, secondary_height)
        text_top = secondary_rect[1] - float(gap)
    else:
        text_top = right[1] + right[3]

    text_bottom = main[1]
    text_height = max(1.0, text_top - text_bottom)
    text_rect = panel(right[0], text_bottom, right_width, text_height)
    return ToyboxLayout(main=main, secondary=secondary_rect, text=text_rect, right=right)


def draw_text_area(
    text_cache: TextCache,
    lines: Sequence[object],
    rect: tuple[float, float, float, float],
    *,
    scroll: int = 0,
    color: tuple[int, int, int] | tuple[int, int, int, int] = COLOR_FOG_GRAY,
    muted_color: tuple[int, int, int] | tuple[int, int, int, int] = COLOR_SLATE_GRAY,
    font_size: int = 12,
    line_height: float = 18.0,
) -> int:
    left, bottom, width, height = rect
    line_count = max(1, int(height // float(line_height)))
    texts = [str(line) for line in lines if str(line)]
    max_scroll = max(0, len(texts) - line_count)
    offset = max(0, min(int(scroll), max_scroll))
    top = bottom + height
    for idx, text in enumerate(texts[offset : offset + line_count]):
        y = top - 4.0 - idx * float(line_height)
        text_cache.draw(text, left, y, color=color, font_size=font_size, anchor_y="top")

    if max_scroll > 0:
        track_x = left + width - 3.0
        arcade.draw_line(track_x, bottom, track_x, bottom + height, with_alpha(muted_color, 80), 1.0)
        thumb_height = max(18.0, height * (line_count / max(1, len(texts))))
        thumb_span = max(1.0, height - thumb_height)
        thumb_bottom = bottom + thumb_span * (1.0 - offset / max(1, max_scroll))
        arcade.draw_line(track_x, thumb_bottom, track_x, thumb_bottom + thumb_height, with_alpha(color, 170), 2.0)
    return max_scroll


def draw_panel_outline(rect: tuple[float, float, float, float], title: str, text_cache: TextCache | None = None) -> None:
    left, bottom, width, height = rect
    arcade.draw_lbwh_rectangle_outline(left, bottom, width, height, with_alpha(COLOR_FOG_GRAY, 90), 1.0)
    if text_cache is not None:
        text_cache.draw(
            camel_case_text(title),
            left + 8,
            bottom + height - 18,
            color=COLOR_FOG_GRAY,
            font_size=11,
            anchor_y="center",
        )


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


def draw_points(
    points: np.ndarray,
    labels: np.ndarray | None,
    rect: tuple[float, float, float, float],
    *,
    size: float = 5.0,
    alpha: int = 230,
    limit: int | None = None,
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if limit is not None:
        pts = pts[: int(limit)]
        if labels is not None:
            labels = labels[: int(limit)]
    xy = world_to_panel(pts, rect)
    for idx, (x, y) in enumerate(xy):
        label = int(labels[idx]) if labels is not None else 0
        outer, inner = CLASS_COLOR_PAIRS[label % len(CLASS_COLOR_PAIRS)]
        outer_size = float(size)
        inner_size = max(1.0, outer_size - 2.0)
        arcade.draw_lbwh_rectangle_filled(
            float(x) - outer_size * 0.5,
            float(y) - outer_size * 0.5,
            outer_size,
            outer_size,
            with_alpha(outer, alpha),
        )
        arcade.draw_lbwh_rectangle_filled(
            float(x) - inner_size * 0.5,
            float(y) - inner_size * 0.5,
            inner_size,
            inner_size,
            with_alpha(inner, alpha),
        )


def draw_curve(values: Sequence[float], rect: tuple[float, float, float, float], color: tuple[int, int, int] = COLOR_FOG_GRAY) -> None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 2:
        return
    left, bottom, width, height = rect
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


def draw_pixel_image(
    image: np.ndarray,
    left: float,
    bottom: float,
    *,
    scale: float = 12.0,
    on_color: tuple[int, int, int] = COLOR_AQUA,
    off_color: tuple[int, int, int] = COLOR_DARK_NEUTRAL,
    border_color: tuple[int, int, int] = COLOR_SLATE_GRAY,
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
        del modifiers
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

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        del x, y, scroll_x
        self.info_scroll = max(0, int(self.info_scroll) - int(scroll_y))

    def _save_display_snapshot(self) -> Path:
        run_paths = create_run_from_config(self.toybox_config)
        write_json(run_paths.config_path, to_dict(self.toybox_config))
        self.trainer.save(run_paths)
        write_json(run_paths.metrics_path, self.trainer.metrics)
        return run_paths.run_dir

    def on_draw(self) -> None:
        self.clear(self.background_color)
        snapshot = self.trainer.snapshot()
        self.renderer.draw(snapshot, self)

    def layout(self, *, secondary: bool = False):
        return toybox_layout(self.width, self.height, secondary=bool(secondary))

    def info_lines(self, snapshot: dict[str, object], extra: list[str] | tuple[str, ...] = ()) -> list[str]:
        metrics = dict(snapshot.get("metrics", self.trainer.metrics))
        loss = metrics.get("loss")
        loss_text = "N/A" if loss is None else f"{float(loss):.4f}"
        lines = [
            ("demo", self.toybox_config.demo),
            ("dataset", self.toybox_config.dataset),
            ("step", int(self.trainer.step_count)),
            ("loss", loss_text),
            ("speed", f"{self.steps_per_frame} step/frame"),
        ]
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
        lines.extend(item for item in extra if str(item))
        return [line for line in (format_info_line(item) for item in lines) if line]

    def draw_info(self, snapshot: dict[str, object], *, secondary: bool = False, extra: list[str] | tuple[str, ...] = ()) -> None:
        layout = self.layout(secondary=secondary)
        max_scroll = draw_text_area(
            self.text_cache,
            self.info_lines(snapshot, extra),
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
