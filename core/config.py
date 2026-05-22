"""Shared project paths plus compatibility exports."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence, TypeVar

from core.arcade_style import (
    CLASS_COLOR_PAIRS,
    COLOR_AQUA,
    COLOR_BARK,
    COLOR_BLUE,
    COLOR_BRICK_RED,
    COLOR_CORAL,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_PURPLE,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_FOREST_GREEN,
    COLOR_LEAF_GREEN,
    COLOR_LIGHT_NEUTRAL,
    COLOR_NAVY,
    COLOR_OCHRE,
    COLOR_PURPLE,
    COLOR_SAND,
    COLOR_SLATE_GRAY,
    COLOR_WALNUT,
    DEFAULT_BOTTOM_BAR_HEIGHT,
    DEFAULT_CELL_INSET,
    DEFAULT_GRID_COLUMNS,
    DEFAULT_GRID_ROWS,
    DEFAULT_STATUS_BAR_FONT_SIZE,
    DEFAULT_TILE_SIZE,
    GAME_TITLE_FONT_NAME,
    GAME_UI_FONT_NAME,
    INTER_FONT_NAME,
    screen_height,
    screen_width,
)
from core.shared_config import PLAYFIELD_HEIGHT, SCREEN_HEIGHT, SCREEN_WIDTH, SHOW_FPS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
MEDIA_ROOT = PROJECT_ROOT / "media"


@dataclass
class CommonConfig:
    demo: str
    dataset: str
    seed: int = 0
    device: str = "cpu"
    steps: int = 1000
    batch_size: int = 128
    lr: float = 0.01
    hidden_dim: int = 32
    output_dir: str = "runs"
    run_name: str | None = None
    save_every: int = 250
    log_every: int = 25
    smoke: bool = False


@dataclass
class DisplayConfig:
    window_width: int = SCREEN_WIDTH
    window_height: int = SCREEN_HEIGHT
    fps: int = 60
    steps_per_frame: int = 1
    paused: bool = False
    background_color: tuple[int, int, int] = COLOR_DARK_NEUTRAL
    text_color: tuple[int, int, int] = COLOR_FOG_GRAY
    overlay_color: tuple[int, int, int] = COLOR_SLATE_GRAY


ConfigT = TypeVar("ConfigT", bound=CommonConfig)


def add_common_args(parser: argparse.ArgumentParser, demo_choices: Sequence[str], *, require_demo: bool = True) -> None:
    parser.add_argument("--demo", required=bool(require_demo), choices=tuple(demo_choices))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--output-dir", "--runs-dir", dest="output_dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")


def add_display_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-width", type=int, default=SCREEN_WIDTH)
    parser.add_argument("--window-height", type=int, default=SCREEN_HEIGHT)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--steps-per-frame", type=int, default=1)
    parser.add_argument("--paused", action="store_true")


def display_config_from_args(args: argparse.Namespace) -> DisplayConfig:
    return DisplayConfig(
        window_width=int(args.window_width),
        window_height=int(args.window_height),
        fps=int(args.fps),
        steps_per_frame=max(1, int(args.steps_per_frame)),
        paused=bool(args.paused),
    )


def normalize_device(device: str) -> str:
    key = str(device).strip().lower()
    if key in {"", "cpu"}:
        return "cpu"
    if key in {"cuda", "gpu"}:
        return "cuda"
    if key == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    raise ValueError(f"Unsupported device '{device}'. Expected cpu, cuda, or auto.")


def config_from_args(config_cls: type[ConfigT], args: argparse.Namespace, *, default_dataset: str) -> ConfigT:
    payload: dict[str, Any] = {}
    for field in fields(config_cls):
        if not hasattr(args, field.name):
            continue
        value = getattr(args, field.name)
        if value is not None:
            payload[field.name] = value
    payload["demo"] = str(args.demo)
    payload["dataset"] = str(args.dataset or default_dataset)
    payload["device"] = normalize_device(str(payload.get("device", "cpu")))
    if bool(payload.get("smoke", False)):
        payload["steps"] = min(int(payload.get("steps", 1000)), 5)
        payload["save_every"] = max(1, min(int(payload.get("save_every", 250)), int(payload["steps"])))
        payload["log_every"] = max(1, min(int(payload.get("log_every", 25)), int(payload["steps"])))
    return config_cls(**payload)


def to_dict(config: CommonConfig | DisplayConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data
