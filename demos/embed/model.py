"""Tiny embedding model."""

from __future__ import annotations

import torch
from torch import nn


class TinyEmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(int(vocab_size), int(dim))
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.15)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        a = self.emb(left)
        b = self.emb(right)
        return (a * b).sum(dim=-1)

    def embeddings(self) -> torch.Tensor:
        return self.emb.weight

