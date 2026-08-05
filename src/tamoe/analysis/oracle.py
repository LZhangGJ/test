"""Explicit analysis-only episode oracle utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class OracleResult:
    expert_index: int
    metric: float


@dataclass(frozen=True, slots=True)
class NamedOracleResult:
    """Deterministic query-label analysis result for Gate 1R."""

    expert_name: str
    accuracy: float
    query_nll: float


def episode_oracle(expert_metrics: Tensor) -> OracleResult:
    """Select the best expert using query metrics; never use this for routing."""

    if expert_metrics.ndim != 1 or expert_metrics.numel() == 0:
        raise ValueError("expert_metrics must be a non-empty one-dimensional tensor")
    if not torch.isfinite(expert_metrics).all():
        raise ValueError("expert_metrics must be finite")
    index = int(torch.argmax(expert_metrics).item())
    return OracleResult(expert_index=index, metric=float(expert_metrics[index].item()))


def deterministic_accuracy_oracle(
    expert_names: Sequence[str],
    accuracies: Tensor,
    query_nlls: Tensor,
    *,
    accuracy_tie_tolerance: float = 1e-12,
) -> NamedOracleResult:
    """Apply frozen accuracy, NLL, then name tie-breaking for analysis only."""

    if accuracy_tie_tolerance < 0:
        raise ValueError("accuracy_tie_tolerance cannot be negative")
    if accuracies.ndim != 1 or query_nlls.ndim != 1:
        raise ValueError("accuracies and query_nlls must be one-dimensional")
    if len(expert_names) == 0 or len(expert_names) != len(accuracies):
        raise ValueError("expert_names and metric tensors must have equal non-zero length")
    if len(query_nlls) != len(accuracies):
        raise ValueError("accuracy and query-NLL counts differ")
    if len(set(expert_names)) != len(expert_names):
        raise ValueError("expert names must be unique")
    if not torch.isfinite(accuracies).all() or not torch.isfinite(query_nlls).all():
        raise ValueError("oracle metrics must be finite")
    maximum = float(accuracies.max().item())
    eligible = [
        index
        for index, accuracy in enumerate(accuracies)
        if maximum - float(accuracy.item()) <= accuracy_tie_tolerance
    ]
    selected = min(
        eligible,
        key=lambda index: (float(query_nlls[index].item()), str(expert_names[index])),
    )
    return NamedOracleResult(
        expert_name=str(expert_names[selected]),
        accuracy=float(accuracies[selected].item()),
        query_nll=float(query_nlls[selected].item()),
    )


def epsilon_optimal_experts(
    expert_names: Sequence[str], accuracies: Tensor, *, epsilon_accuracy: float = 0.01
) -> tuple[str, ...]:
    """Return lexicographically sorted experts within epsilon of maximum accuracy."""

    if epsilon_accuracy < 0:
        raise ValueError("epsilon_accuracy cannot be negative")
    if accuracies.ndim != 1 or len(expert_names) == 0 or len(expert_names) != len(accuracies):
        raise ValueError("expert_names and accuracies must have equal non-zero length")
    if not torch.isfinite(accuracies).all():
        raise ValueError("accuracies must be finite")
    maximum = float(accuracies.max().item())
    return tuple(
        sorted(
            str(expert_names[index])
            for index, accuracy in enumerate(accuracies)
            if maximum - float(accuracy.item()) <= epsilon_accuracy + 1e-12
        )
    )
