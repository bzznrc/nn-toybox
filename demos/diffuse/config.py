"""Diffuse demo configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.config import CommonConfig
from core.datasets import dataset_key
from core.generated_2d import DIFFUSION_DATASET_KEYS, normalize_distribution_variant


@dataclass
class DiffuseConfig(CommonConfig):
    preset: str = "fast"
    lr: float = 0.002
    hidden_dim: int = 64
    batch_size: int = 128
    distribution: str = "gaussian_mixtures"
    n_points: int = 512
    noise: float = 0.04
    timesteps: int = 16
    sample_timesteps: int = 8
    noise_schedule: str = "linear"
    sample_count: int = 64
    sample_refresh_every: int = 500
    max_clean_points: int = 128
    max_noised_points: int = 128
    max_generated_points: int = 64
    time_dim: int = 16

    @staticmethod
    def resolve_payload(payload: dict[str, object]) -> dict[str, object]:
        preset = str(payload.get("preset", "fast")).strip().lower()
        resolved = dict(payload)
        if preset == "nice":
            nice_defaults: dict[str, object] = {
                "n_points": 512,
                "timesteps": 16,
                "sample_timesteps": 16,
                "sample_count": 192,
                "sample_refresh_every": 200,
                "max_clean_points": 384,
                "max_noised_points": 384,
                "max_generated_points": 192,
                "hidden_dim": 64,
            }
            for key, value in nice_defaults.items():
                resolved.setdefault(key, value)
        elif preset != "fast":
            raise ValueError(f"Unknown diffuse preset '{preset}'. Valid: fast, nice")
        resolved["preset"] = preset
        dataset = dataset_key(resolved.get("dataset", "Distributions - Gaussian Mixtures"))
        if dataset in DIFFUSION_DATASET_KEYS:
            resolved["distribution"] = dataset
        resolved["distribution"] = normalize_distribution_variant(
            resolved.get("distribution", "gaussian_mixtures"),
            choices=DIFFUSION_DATASET_KEYS,
        )
        return resolved


def add_diffuse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=["fast", "nice"], default=None)
    parser.add_argument("--distribution", choices=list(DIFFUSION_DATASET_KEYS), default=None)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--sample-timesteps", type=int, default=None)
    parser.add_argument("--noise-schedule", default=None, choices=["linear", "cosine"])
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--sample-refresh-every", type=int, default=None)
    parser.add_argument("--max-clean-points", type=int, default=None)
    parser.add_argument("--max-noised-points", type=int, default=None)
    parser.add_argument("--max-generated-points", type=int, default=None)
    parser.add_argument("--time-dim", type=int, default=None)
