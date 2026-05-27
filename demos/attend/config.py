"""Attend demo configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.config import CommonConfig
from demos.attend.data import MAX_SENTENCE_LENGTH


@dataclass
class AttendConfig(CommonConfig):
    lr: float = 0.005
    batch_size: int = 16
    task: str = "subject_verb_agreement"
    sequence_length: int = MAX_SENTENCE_LENGTH
    embedding_dim: int = 24
    attention_dim: int = 24
    trap_rate: float = 0.75
    display_refresh_every: int = 60

    @staticmethod
    def resolve_payload(payload: dict[str, object]) -> dict[str, object]:
        resolved = dict(payload)
        if str(resolved.get("task", "subject_verb_agreement")).strip() != "subject_verb_agreement":
            raise ValueError("Attend currently supports task=subject_verb_agreement")
        resolved["task"] = "subject_verb_agreement"
        resolved["sequence_length"] = MAX_SENTENCE_LENGTH
        resolved["trap_rate"] = max(0.0, min(1.0, float(resolved.get("trap_rate", 0.75))))
        resolved["display_refresh_every"] = max(0, int(resolved.get("display_refresh_every", 60)))
        return resolved


def add_attend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default=None, choices=["subject_verb_agreement"])
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--attention-dim", type=int, default=None)
    parser.add_argument("--trap-rate", type=float, default=None)
    parser.add_argument("--display-refresh-every", type=int, default=None)
