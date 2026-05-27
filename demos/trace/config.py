"""Trace demo configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.config import CommonConfig


@dataclass
class TraceConfig(CommonConfig):
    lr: float = 0.01
    batch_size: int = 64
    hidden1: int = 32
    hidden2: int = 16
    checkpoint_path: str = "assets/trace_mlp_checkpoint.pt"
    top_k_edges: int = 120
    example_split: str = "inference"

    @staticmethod
    def resolve_payload(payload: dict[str, object]) -> dict[str, object]:
        resolved = dict(payload)
        if "hidden_dim" in resolved and "hidden1" not in resolved:
            resolved["hidden1"] = int(resolved["hidden_dim"])
        return resolved


def add_trace_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden1", type=int, default=None)
    parser.add_argument("--hidden2", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--top-k-edges", type=int, default=None)
    parser.add_argument("--example-split", choices=["train", "inference"], default=None)
