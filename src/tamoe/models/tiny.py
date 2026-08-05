"""Tiny frozen representation and equal-architecture residual experts."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TinyBackbone(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def freeze(self) -> None:
        self.requires_grad_(False)
        self.eval()


class ResidualExpert(nn.Module):
    def __init__(self, embedding_dim: int, rank: int = 4) -> None:
        super().__init__()
        self.down = nn.Linear(embedding_dim, rank, bias=False)
        self.up = nn.Linear(rank, embedding_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, embeddings: Tensor) -> Tensor:
        return embeddings + self.up(torch.relu(self.down(embeddings)))
