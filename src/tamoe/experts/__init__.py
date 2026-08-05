"""Fixed, equal-architecture expert modules."""

from tamoe.experts.adapters import ResidualAdapter
from tamoe.experts.bank import ExpertDefinition, build_expert_definitions

__all__ = ["ExpertDefinition", "ResidualAdapter", "build_expert_definitions"]
