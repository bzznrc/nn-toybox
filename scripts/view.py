"""Shared Arcade live-training entrypoint."""

from __future__ import annotations

import argparse
from typing import Sequence

import arcade

from core.arcade_view import LiveTrainingWindow
from core.config import add_common_args, add_display_args, config_from_args, display_config_from_args
from core.registry import DEMO_ORDER, get_demo_spec


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    add_common_args(base, DEMO_ORDER, require_demo=False)
    add_display_args(base)
    known, _unknown = base.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Display live nn-toybox training")
    add_common_args(parser, DEMO_ORDER)
    add_display_args(parser)
    if known.demo is not None:
        spec = get_demo_spec(known.demo)
        spec.add_cli_args(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    spec = get_demo_spec(args.demo)
    config = config_from_args(spec.config_cls, args, default_dataset=spec.default_dataset)
    display_config = display_config_from_args(args)
    trainer = spec.trainer_cls()(config)
    renderer = spec.renderer_cls()(config)
    LiveTrainingWindow(
        trainer=trainer,
        renderer=renderer,
        config=config,
        display_config=display_config,
        title=f"nn-toybox / {config.demo}",
    )
    arcade.run()


if __name__ == "__main__":
    main()
