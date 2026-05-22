"""Diffuse demo configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.config import CommonConfig


@dataclass
class DiffuseConfig(CommonConfig):
    lr: float = 0.002
    hidden_dim: int = 64
    batch_size: int = 256
    n_points: int = 1024
    noise: float = 0.04
    timesteps: int = 32
    noise_schedule: str = "linear"
    sample_count: int = 512
    sample_refresh_every: int = 100
    time_dim: int = 16


def add_diffuse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--noise", type=float, default=0.04)
    parser.add_argument("--timesteps", type=int, default=32)
    parser.add_argument("--noise-schedule", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--sample-refresh-every", type=int, default=100)
    parser.add_argument("--time-dim", type=int, default=16)
