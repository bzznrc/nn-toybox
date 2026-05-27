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
from core.registry import DEMO_ORDER, get_demo_spec, validate_demo_dataset


ACCURACY_THRESHOLDS = {
    "grad": 0.78,
    "trace": 0.35,
    "conv": 0.25,
    "attend": 0.65,
}
MEAN_ERROR_THRESHOLDS = {
    "encode": 0.22,
}
LOSS_DROP_RATIOS = {
    "embed": 0.995,
    "diffuse": 0.94,
    "optim": 0.80,
}


def _float_metric(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_simple_convergence(demo: str, metrics: dict[str, object], baseline_loss: float | None) -> bool:
    accuracies = [
        value
        for value in (_float_metric(metrics, "accuracy"), _float_metric(metrics, "train_accuracy"))
        if value is not None
    ]
    if accuracies and max(accuracies) >= float(ACCURACY_THRESHOLDS.get(str(demo), 0.70)):
        return True

    mean_error = _float_metric(metrics, "mean_error")
    if mean_error is not None and mean_error <= float(MEAN_ERROR_THRESHOLDS.get(str(demo), 0.08)):
        return True

    loss = _float_metric(metrics, "loss")
    if loss is None or baseline_loss is None or baseline_loss <= 0.0:
        return False
    return loss <= baseline_loss * float(LOSS_DROP_RATIOS.get(str(demo), 0.80))


class AutoCycleController:
    """Capture-only trigger that sends the shared UP action after simple convergence."""

    def __init__(
        self,
        *,
        demo: str,
        fps: int,
        min_seconds: float = 1.0,
        cooldown_seconds: float = 2.0,
        max_cycles: int = 2,
    ) -> None:
        self.demo = str(demo)
        self.min_frame = max(0, int(round(float(min_seconds) * int(fps))))
        self.cooldown_frames = max(1, int(round(float(cooldown_seconds) * int(fps))))
        self.max_cycles = max(0, int(max_cycles))
        self.cycles = 0
        self.last_cycle_frame = -self.cooldown_frames
        self.baseline_loss: float | None = None

    def should_cycle(self, frame_index: int, metrics: dict[str, object]) -> bool:
        if self.cycles >= self.max_cycles or int(frame_index) < self.min_frame:
            return False
        if int(frame_index) - self.last_cycle_frame < self.cooldown_frames:
            return False

        loss = _float_metric(metrics, "loss")
        if self.baseline_loss is None and loss is not None:
            self.baseline_loss = loss
        return _has_simple_convergence(self.demo, metrics, self.baseline_loss)

    def record_cycle(self, frame_index: int) -> None:
        self.cycles += 1
        self.last_cycle_frame = int(frame_index)
        self.baseline_loss = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    add_common_args(base, DEMO_ORDER, require_demo=False)
    add_display_args(base)
    known, _unknown = base.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Capture an nn-toybox live display GIF")
    add_common_args(parser, DEMO_ORDER)
    add_display_args(parser)
    if known.demo is not None:
        try:
            spec = get_demo_spec(known.demo)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        spec.add_cli_args(parser)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--visible", action="store_true", help="Show the Arcade window while capturing")
    parser.add_argument("--no-auto-cycle", action="store_true", help="Disable convergence-triggered UP key changes")
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
    args.demo = spec.name
    try:
        config = config_from_args(spec.config_cls, args, default_dataset=spec.default_dataset)
        config.dataset = validate_demo_dataset(spec, config.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
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
    auto_cycle = AutoCycleController(demo=config.demo, fps=int(args.fps))
    try:
        for idx in range(frame_count):
            window.switch_to()
            window.dispatch_events()
            window.on_update(dt)
            if not bool(args.no_auto_cycle) and auto_cycle.should_cycle(idx, dict(window.trainer.metrics)):
                window.on_key_press(arcade.key.UP, 0)
                auto_cycle.record_cycle(idx)
                print(f">>> Capture auto-cycle: demo={config.demo} frame={idx} key=UP")
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
