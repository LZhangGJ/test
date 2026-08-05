"""Explicit analysis-only episode oracle utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class OracleResult:
    expert_index: int
    metric: float


def episode_oracle(expert_metrics: Tensor) -> OracleResult:
    """Select the best expert using query metrics; never use this for routing."""

    if expert_metrics.ndim != 1 or expert_metrics.numel() == 0:
        raise ValueError("expert_metrics must be a non-empty one-dimensional tensor")
    if not torch.isfinite(expert_metrics).all():
        raise ValueError("expert_metrics must be finite")
    index = int(torch.argmax(expert_metrics).item())
    return OracleResult(expert_index=index, metric=float(expert_metrics[index].item()))
