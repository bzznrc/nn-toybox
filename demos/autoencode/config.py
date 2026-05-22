"""Autoencode demo configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.config import CommonConfig


@dataclass
class AutoencodeConfig(CommonConfig):
    lr: float = 0.003
    hidden_dim: int = 64
    image_size: int = 16
    n_samples: int = 512
    shape_types: str = "all"
    latent_dim: int = 8
    noise: float = 0.02


def add_autoencode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--shape-types", default="all")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.02)
