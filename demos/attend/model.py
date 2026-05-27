"""Tiny single-head self-attention model for agreement."""

from __future__ import annotations

import math

import torch
from torch import nn


class AgreementSelfAttentionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        embedding_dim: int = 24,
        attention_dim: int = 24,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(int(vocab_size), int(embedding_dim))
        self.position_embedding = nn.Embedding(int(sequence_length), int(embedding_dim))
        self.query_proj = nn.Linear(int(embedding_dim), int(attention_dim), bias=False)
        self.key_proj = nn.Linear(int(embedding_dim), int(attention_dim), bias=False)
        self.value_proj = nn.Linear(int(embedding_dim), int(attention_dim), bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(int(attention_dim), int(embedding_dim)),
            nn.ReLU(),
            nn.Linear(int(embedding_dim), 2),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        mask_indices: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        queries = self.query_proj(hidden)
        keys = self.key_proj(hidden)
        values = self.value_proj(hidden)
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(max(1, keys.shape[-1]))
        if token_mask is not None:
            scores = scores.masked_fill(~token_mask.bool().unsqueeze(1), -1.0e9)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, values)
        row_index = mask_indices.view(-1, 1, 1).expand(-1, 1, context.shape[-1])
        mask_context = context.gather(1, row_index).squeeze(1)
        return self.classifier(mask_context), attention
