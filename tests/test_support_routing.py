from __future__ import annotations

import torch

from tamoe.experts.adapters import ResidualAdapter
from tamoe.routing.support import (
    evaluate_external_weights,
    evaluate_routing_episode,
    mixture_log_probabilities,
)


def _identity_adapter() -> ResidualAdapter:
    adapter = ResidualAdapter(embedding_dim=2, rank=1)
    with torch.no_grad():
        adapter.down.weight.zero_()
        adapter.up.weight.zero_()
    return adapter.eval()


def test_mixture_weights_are_applied_in_probability_space() -> None:
    probabilities = torch.tensor(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.3, 0.7], [0.6, 0.4]],
        ]
    )
    result = mixture_log_probabilities(probabilities.log(), torch.tensor([0.25, 0.75]))
    expected = 0.25 * probabilities[0] + 0.75 * probabilities[1]
    assert torch.allclose(result.exp(), expected)


def test_support_router_is_finite_and_excludes_single_from_candidates() -> None:
    experts = {
        "shared": _identity_adapter(),
        "single": _identity_adapter(),
        "source_a": _identity_adapter(),
        "source_b": _identity_adapter(),
    }
    references = {
        "shared": torch.tensor([0.5, 0.5]),
        "source_a": torch.tensor([1.0, 0.0]),
        "source_b": torch.tensor([0.0, 1.0]),
    }
    features = torch.tensor([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
    labels = torch.tensor([0] * 5 + [1] * 5)
    rows, routed = evaluate_routing_episode(
        experts,
        references,
        features,
        labels,
        shots=1,
        queries_per_class=2,
        seed=1,
        repetition=0,
        prototype_temperature=0.1,
        route_temperature=0.25,
        device=torch.device("cpu"),
        shared_fallback_weight=0.5,
    )
    assert routed.candidate_names == ("shared", "source_a", "source_b")
    assert torch.isfinite(routed.support_weights).all()
    assert torch.isclose(routed.support_weights.sum(), torch.tensor(1.0))
    methods = {row["method"] for row in rows}
    assert "support_conditioned_soft_mixture" in methods
    assert "capacity_matched_single" in methods
    external = evaluate_external_weights(routed, torch.tensor([1.0, 0.0, 0.0]))
    assert 0 <= external["accuracy"] <= 1
