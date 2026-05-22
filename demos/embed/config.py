"""Embed demo configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.config import CommonConfig


@dataclass
class EmbedConfig(CommonConfig):
    lr: float = 0.05
    embedding_dim: int = 2
    negative_samples: int = 3
    noise_pairs: int = 0


def add_embed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-dim", type=int, default=2)
    parser.add_argument("--negative-samples", type=int, default=3)
    parser.add_argument("--noise-pairs", type=int, default=0)
