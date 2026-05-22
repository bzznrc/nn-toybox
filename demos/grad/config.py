"""Grad demo configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.config import CommonConfig


@dataclass
class GradConfig(CommonConfig):
    n_points: int = 512
    noise: float = 0.08
    layers: int = 2
    activation: str = "relu"
    optimizer: str = "adam"
    boundary_resolution: int = 72


def add_grad_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh", "sigmoid", "gelu"])
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--boundary-resolution", type=int, default=72)
