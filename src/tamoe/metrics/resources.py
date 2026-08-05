"""Capacity- and compute-matched accounting primitives."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from tamoe.experts.adapters import ResidualAdapter


@dataclass(frozen=True, slots=True)
class ResourceCounts:
    total_parameters: int
    trainable_parameters: int
    activated_parameters_per_query: int
    approximate_adapter_flops_per_query: int
    expert_bank_size: int


def count_resources(backbone: nn.Module, experts: list[ResidualAdapter]) -> ResourceCounts:
    if not experts:
        raise ValueError("at least one expert is required")
    total = sum(parameter.numel() for parameter in backbone.parameters()) + sum(
        parameter.numel() for expert in experts for parameter in expert.parameters()
    )
    trainable = sum(
        parameter.numel()
        for module in [backbone, *experts]
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    backbone_parameters = sum(parameter.numel() for parameter in backbone.parameters())
    activated = backbone_parameters + experts[0].activated_parameters
    return ResourceCounts(
        total_parameters=total,
        trainable_parameters=trainable,
        activated_parameters_per_query=activated,
        approximate_adapter_flops_per_query=experts[0].approximate_flops_per_sample,
        expert_bank_size=len(experts),
    )
