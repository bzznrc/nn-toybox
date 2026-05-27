"""Conv demo configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.config import CommonConfig


@dataclass
class ConvConfig(CommonConfig):
    lr: float = 0.006
    batch_size: int = 64
    channels: int = 8
    selected_feature_layer: int = 1
    noise_amount: float = 0.0
    shift_x: int = 0
    shift_y: int = 0
    example_split: str = "inference"
    checkpoint_path: str = "assets/conv_cnn_checkpoint.pt"


def add_conv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--selected-feature-layer", type=int, choices=[1, 2], default=None)
    parser.add_argument("--noise-amount", type=float, default=None)
    parser.add_argument("--shift-x", type=int, default=None)
    parser.add_argument("--shift-y", type=int, default=None)
    parser.add_argument("--example-split", choices=["train", "inference"], default=None)
    parser.add_argument("--checkpoint-path", default=None)
