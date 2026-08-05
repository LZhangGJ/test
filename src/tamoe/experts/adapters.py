"""Residual bottleneck adapters used for all M2 expert variants."""

from __future__ import annotations

from torch import Tensor, nn


class ResidualAdapter(nn.Module):
    def __init__(self, embedding_dim: int, rank: int) -> None:
        super().__init__()
        if embedding_dim <= 0 or rank <= 0:
            raise ValueError("embedding_dim and rank must be positive")
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.down = nn.Linear(embedding_dim, rank, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(rank, embedding_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, embeddings: Tensor) -> Tensor:
        return embeddings + self.up(self.activation(self.down(self.norm(embeddings))))

    @property
    def activated_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def approximate_flops_per_sample(self) -> int:
        return 4 * self.embedding_dim * self.rank
