"""Tiny denoising model and DDPM helpers."""

from __future__ import annotations

import math

import torch
from torch import nn


class TinyDenoiser(nn.Module):
    def __init__(self, hidden_size: int = 64, time_dim: int = 16) -> None:
        super().__init__()
        self.time_dim = int(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + self.time_dim, int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.SiLU(),
            nn.Linear(int(hidden_size), 2),
        )

    def _time_features(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(-1, 1)
        half = max(1, self.time_dim // 2)
        freqs = torch.exp(torch.linspace(0.0, math.log(100.0), half, device=t.device)).view(1, -1)
        angles = t * freqs * math.pi * 2.0
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if features.shape[1] < self.time_dim:
            features = torch.cat([features, t], dim=1)
        return features[:, : self.time_dim]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self._time_features(t)], dim=1))


def make_schedule(steps: int, schedule: str = "linear") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = max(2, int(steps))
    if str(schedule).lower() == "cosine":
        t = torch.linspace(0, math.pi / 2.0, count)
        betas = 0.0005 + (torch.sin(t) ** 2) * 0.035
    else:
        betas = torch.linspace(0.0008, 0.035, count)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def q_sample(x0: torch.Tensor, t_idx: torch.Tensor, alphas_bar: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    a = alphas_bar[t_idx].view(-1, 1).to(x0.device)
    return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise


@torch.no_grad()
def sample_points(
    model: TinyDenoiser,
    count: int,
    steps: int,
    schedule: str = "linear",
    seed: int = 0,
    keep_frames: int = 18,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    betas, alphas, alphas_bar = make_schedule(steps, schedule)
    x = torch.randn((int(count), 2), generator=generator)
    frames: list[torch.Tensor] = [x.clone()]
    keep_every = max(1, int(steps) // max(1, int(keep_frames) - 1))

    for idx in reversed(range(int(steps))):
        t = torch.full((int(count),), float(idx) / max(1, int(steps) - 1))
        eps = model(x, t)
        beta = betas[idx]
        alpha = alphas[idx]
        alpha_bar = alphas_bar[idx]
        mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha)
        if idx > 0:
            z = torch.randn(x.shape, generator=generator)
            x = mean + torch.sqrt(beta) * z
        else:
            x = mean
        if idx % keep_every == 0 or idx == 0:
            frames.append(x.clone())
    return x, torch.stack(frames, dim=0)


def build_model(config: dict[str, object]) -> TinyDenoiser:
    return TinyDenoiser(hidden_size=int(config.get("hidden_size", 64)), time_dim=int(config.get("time_dim", 16)))

