"""Optim demo configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.config import CommonConfig
from demos.optim.data import LANDSCAPES


@dataclass
class OptimConfig(CommonConfig):
    steps: int = 500
    lr: float = 0.06
    landscape: str = "ravine"
    optimizer: str = "adam"
    momentum: float = 0.9
    start_point: str = ""
    start_x: float = 1.45
    start_y: float = -1.25
    trail_length: int = 180
    landscape_resolution: int = 58

    @staticmethod
    def resolve_payload(payload: dict[str, object]) -> dict[str, object]:
        resolved = dict(payload)
        start_point = str(resolved.get("start_point", "") or "").strip()
        if start_point:
            parts = [part.strip() for part in start_point.replace(";", ",").split(",") if part.strip()]
            if len(parts) != 2:
                raise ValueError("--start-point must look like x,y")
            resolved["start_x"] = float(parts[0])
            resolved["start_y"] = float(parts[1])
        resolved["landscape"] = str(resolved.get("landscape", "ravine")).strip().lower().replace("-", "_")
        resolved["optimizer"] = str(resolved.get("optimizer", "adam")).strip().lower()
        return resolved


def add_optim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--landscape", choices=list(LANDSCAPES), default=None)
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--start-point", default=None)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--trail-length", type=int, default=None)
    parser.add_argument("--landscape-resolution", type=int, default=None)
