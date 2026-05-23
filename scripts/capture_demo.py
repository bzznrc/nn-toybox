"""Capture a short GIF from the shared live Arcade display."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import arcade
from PIL import Image

from core.arcade_view import LiveTrainingWindow
from core.config import MEDIA_ROOT, add_common_args, add_display_args, config_from_args, display_config_from_args
from core.registry import DEMO_ORDER, get_demo_spec


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    add_common_args(base, DEMO_ORDER, require_demo=False)
    add_display_args(base)
    known, _unknown = base.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Capture an nn-toybox live display GIF")
    add_common_args(parser, DEMO_ORDER)
    add_display_args(parser)
    if known.demo is not None:
        get_demo_spec(known.demo).add_cli_args(parser)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--visible", action="store_true", help="Show the Arcade window while capturing")
    return parser.parse_args(argv)


def _output_path(demo: str) -> Path:
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    return MEDIA_ROOT / f"{demo}.gif"


def _capture_frame(window: arcade.Window) -> Image.Image:
    return arcade.get_image(
        x=0,
        y=0,
        width=int(window.width),
        height=int(window.height),
        components=3,
    ).convert("RGB")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not bool(args.visible):
        os.environ["NN_TOYBOX_RENDER_VISIBLE"] = "0"

    spec = get_demo_spec(args.demo)
    config = config_from_args(spec.config_cls, args, default_dataset=spec.default_dataset)
    display_config = display_config_from_args(args)
    display_config.fps = max(1, int(args.fps))
    window = LiveTrainingWindow(
        trainer=spec.trainer_cls()(config),
        renderer=spec.renderer_cls()(config),
        config=config,
        display_config=display_config,
        title=f"nn-toybox / {config.demo}",
    )
    frames: list[Image.Image] = []
    frame_count = max(1, int(round(float(args.seconds) * int(args.fps))))
    dt = 1.0 / float(max(1, int(args.fps)))
    try:
        for _idx in range(frame_count):
            window.switch_to()
            window.dispatch_events()
            window.on_update(dt)
            window.on_draw()
            frames.append(_capture_frame(window))
            window.flip()
    finally:
        window.close()

    output_path = _output_path(args.demo)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000.0 / float(max(1, int(args.fps)))))
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f">>> Capture: {output_path}")


if __name__ == "__main__":
    main()
