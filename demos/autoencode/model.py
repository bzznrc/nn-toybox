"""Tiny fully connected autoencoder."""

from __future__ import annotations

import torch
from torch import nn


class TinyAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 256, latent_dim: int = 8, hidden_size: int = 64) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), self.input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z).view_as(x)
        return recon, z


def build_model(config: dict[str, object]) -> TinyAutoencoder:
    size = int(config.get("size", 16))
    return TinyAutoencoder(
        input_dim=size * size,
        latent_dim=int(config.get("latent_dim", 8)),
        hidden_size=int(config.get("hidden_size", 64)),
    )

