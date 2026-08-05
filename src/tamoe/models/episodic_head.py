"""Episode-local prototype classification for variable label spaces."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def prototype_logits(
    support_embeddings: Tensor,
    support_labels: Tensor,
    query_embeddings: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if support_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("support and query embeddings must be rank-two tensors")
    if support_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("support/query embedding dimensions differ")
    labels = torch.unique(support_labels, sorted=True)
    expected = torch.arange(len(labels), device=labels.device)
    if not torch.equal(labels, expected):
        raise ValueError("support labels must be contiguous episode-local IDs starting at zero")
    prototypes = torch.stack(
        [support_embeddings[support_labels == label].mean(dim=0) for label in labels]
    )
    prototypes = functional.normalize(prototypes, dim=-1)
    queries = functional.normalize(query_embeddings, dim=-1)
    return queries @ prototypes.transpose(0, 1) / temperature
