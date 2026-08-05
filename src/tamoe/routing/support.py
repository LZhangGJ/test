"""No-parameter query and support weighting for Gate 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from tamoe.analysis.gate1 import macro_f1
from tamoe.episodes.sampler import index_labels, sample_episode_indices
from tamoe.experts.adapters import ResidualAdapter
from tamoe.models.episodic_head import prototype_logits


@dataclass(frozen=True, slots=True)
class RoutedEpisode:
    episode_hash: str
    candidate_names: tuple[str, ...]
    query_labels: Tensor
    candidate_log_probabilities: Tensor
    support_weights: Tensor
    query_weights: Tensor
    label_removed_weights: Tensor


def _metrics(log_probabilities: Tensor, labels: Tensor) -> dict[str, float]:
    prediction = log_probabilities.argmax(dim=-1)
    return {
        "accuracy": float((prediction == labels).float().mean().item()),
        "macro_f1": macro_f1(prediction, labels),
        "loss": float(functional.nll_loss(log_probabilities, labels).item()),
    }


def _standardize(values: Tensor) -> Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)


def _weight_fields(weights: Tensor) -> dict[str, Any]:
    values = weights.detach().cpu().clamp_min(1e-12)
    return {
        "route_entropy": float(-(values * values.log()).sum().item()),
        "top1_route_weight": float(values.max().item()),
        "route_weights": "|".join(f"{float(value):.12g}" for value in values),
    }


def mixture_log_probabilities(log_probabilities: Tensor, weights: Tensor) -> Tensor:
    if log_probabilities.ndim != 3 or weights.ndim != 1:
        raise ValueError("mixture expects [experts, queries, classes] and [experts]")
    if len(weights) != len(log_probabilities):
        raise ValueError("mixture expert and weight counts differ")
    normalized = weights / weights.sum().clamp_min(1e-12)
    return torch.logsumexp(
        log_probabilities + normalized.clamp_min(1e-12).log().view(-1, 1, 1), dim=0
    )


def evaluate_routing_episode(
    experts: dict[str, ResidualAdapter],
    references: dict[str, Tensor],
    features: Tensor,
    labels: Tensor,
    *,
    shots: int,
    queries_per_class: int,
    seed: int,
    repetition: int,
    prototype_temperature: float,
    route_temperature: float,
    device: torch.device,
    shared_fallback_weight: float,
) -> tuple[list[dict[str, Any]], RoutedEpisode]:
    """Evaluate metadata-free baselines; query labels are used only for final metrics."""

    if not 0 <= shared_fallback_weight <= 1:
        raise ValueError("shared_fallback_weight must lie in [0, 1]")
    episode = sample_episode_indices(
        index_labels(labels.tolist()),
        shots=shots,
        queries_per_class=queries_per_class,
        seed=seed,
        repetition=repetition,
    )
    support_index = torch.tensor(episode.support_indices, dtype=torch.long)
    query_index = torch.tensor(episode.query_indices, dtype=torch.long)
    support = features[support_index].to(device)
    query = features[query_index].to(device)
    support_labels = torch.tensor(episode.support_labels, dtype=torch.long, device=device)
    query_labels = torch.tensor(episode.query_labels, dtype=torch.long, device=device)
    candidates = tuple(
        name
        for name in sorted(experts)
        if name == "shared" or name.startswith("source_")
    )
    if "shared" not in candidates or "single" not in experts:
        raise ValueError("Gate 2 bank lacks shared or capacity-matched single")
    if set(candidates) != set(references):
        raise ValueError("reference signatures do not match router candidates")
    support_embeddings: list[Tensor] = []
    query_embeddings: list[Tensor] = []
    log_probabilities: list[Tensor] = []
    support_nlls: list[Tensor] = []
    support_reference_similarities: list[Tensor] = []
    query_reference_similarities: list[Tensor] = []
    fixed_metrics: dict[str, dict[str, float]] = {}
    fixed_log_probabilities: dict[str, Tensor] = {}
    with torch.inference_mode():
        for name in sorted(experts):
            adapted_support = experts[name](support)
            adapted_query = experts[name](query)
            logits = prototype_logits(
                adapted_support,
                support_labels,
                adapted_query,
                temperature=prototype_temperature,
            )
            log_probs = functional.log_softmax(logits, dim=-1)
            fixed_log_probabilities[name] = log_probs
            fixed_metrics[name] = _metrics(log_probs, query_labels)
            if name not in candidates:
                continue
            self_logits = prototype_logits(
                adapted_support,
                support_labels,
                adapted_support,
                temperature=prototype_temperature,
            )
            support_nlls.append(functional.cross_entropy(self_logits, support_labels))
            reference = functional.normalize(references[name].to(device), dim=0)
            support_centroid = functional.normalize(adapted_support.mean(dim=0), dim=0)
            query_centroid = functional.normalize(adapted_query.mean(dim=0), dim=0)
            support_reference_similarities.append(support_centroid @ reference)
            query_reference_similarities.append(query_centroid @ reference)
            support_embeddings.append(adapted_support)
            query_embeddings.append(adapted_query)
            log_probabilities.append(log_probs)
    stacked_log_probs = torch.stack(log_probabilities)
    support_signal = _standardize(-torch.stack(support_nlls)) + _standardize(
        torch.stack(support_reference_similarities)
    )
    query_signal = _standardize(torch.stack(query_reference_similarities))
    label_removed_signal = _standardize(torch.stack(support_reference_similarities))
    support_weights = functional.softmax(support_signal / route_temperature, dim=0)
    query_weights = functional.softmax(query_signal / route_temperature, dim=0)
    label_removed_weights = functional.softmax(label_removed_signal / route_temperature, dim=0)

    support_mixture = mixture_log_probabilities(stacked_log_probs, support_weights)
    query_mixture = mixture_log_probabilities(stacked_log_probs, query_weights)
    label_removed_mixture = mixture_log_probabilities(
        stacked_log_probs, label_removed_weights
    )
    mixed_support = torch.stack(support_embeddings).mul(
        support_weights.view(-1, 1, 1)
    ).sum(dim=0)
    mixed_query = torch.stack(query_embeddings).mul(
        support_weights.view(-1, 1, 1)
    ).sum(dim=0)
    support_prototype_logits = prototype_logits(
        mixed_support,
        support_labels,
        mixed_query,
        temperature=prototype_temperature,
    )
    support_prototype_log_probs = functional.log_softmax(
        support_prototype_logits, dim=-1
    )
    shared_probability = fixed_log_probabilities["shared"].exp()
    fallback_probability = (
        (1 - shared_fallback_weight) * support_mixture.exp()
        + shared_fallback_weight * shared_probability
    )
    fallback_log_probs = fallback_probability.clamp_min(1e-12).log()
    top1_index = int(support_weights.argmax().item())
    rows = [
        {
            "method": name,
            "selected_expert": name,
            "router_input": "none",
            "route_entropy": float("nan"),
            "top1_route_weight": float("nan"),
            "route_weights": "",
            **fixed_metrics[name],
        }
        for name in sorted(experts)
    ]
    random_expected = {
        metric: sum(fixed_metrics[name][metric] for name in candidates) / len(candidates)
        for metric in ("accuracy", "macro_f1", "loss")
    }
    rows.extend(
        [
            {
                "method": "random_expected",
                "selected_expert": "exact_mean",
                "router_input": "none",
                "route_entropy": float("nan"),
                "top1_route_weight": float("nan"),
                "route_weights": "",
                **random_expected,
            },
            {
                "method": "query_only_weighting",
                "selected_expert": candidates[int(query_weights.argmax().item())],
                "router_input": "query_embeddings_only",
                **_weight_fields(query_weights),
                **_metrics(query_mixture, query_labels),
            },
            {
                "method": "support_prototype_weighting",
                "selected_expert": candidates[top1_index],
                "router_input": "support_embeddings_and_labels",
                **_weight_fields(support_weights),
                **_metrics(support_prototype_log_probs, query_labels),
            },
            {
                "method": "support_conditioned_soft_mixture",
                "selected_expert": candidates[top1_index],
                "router_input": "support_embeddings_and_labels",
                **_weight_fields(support_weights),
                **_metrics(support_mixture, query_labels),
            },
            {
                "method": "support_label_removal",
                "selected_expert": candidates[int(label_removed_weights.argmax().item())],
                "router_input": "support_embeddings_without_labels",
                **_weight_fields(label_removed_weights),
                **_metrics(label_removed_mixture, query_labels),
            },
            {
                "method": "shared_fallback",
                "selected_expert": "support_soft_plus_shared",
                "router_input": "support_embeddings_and_labels",
                **_weight_fields(support_weights),
                **_metrics(fallback_log_probs, query_labels),
            },
            {
                "method": "compute_matched_support_top1_diagnostic",
                "selected_expert": candidates[top1_index],
                "router_input": "support_embeddings_and_labels",
                **_weight_fields(support_weights),
                **fixed_metrics[candidates[top1_index]],
            },
            {
                "method": "capacity_matched_single",
                "selected_expert": "single",
                "router_input": "none",
                "route_entropy": float("nan"),
                "top1_route_weight": float("nan"),
                "route_weights": "",
                **fixed_metrics["single"],
            },
        ]
    )
    routed = RoutedEpisode(
        episode_hash=episode.episode_hash,
        candidate_names=candidates,
        query_labels=query_labels.detach().cpu(),
        candidate_log_probabilities=stacked_log_probs.detach().cpu(),
        support_weights=support_weights.detach().cpu(),
        query_weights=query_weights.detach().cpu(),
        label_removed_weights=label_removed_weights.detach().cpu(),
    )
    return rows, routed


def evaluate_external_weights(routed: RoutedEpisode, weights: Tensor) -> dict[str, float]:
    log_probabilities = mixture_log_probabilities(
        routed.candidate_log_probabilities, weights.cpu()
    )
    return _metrics(log_probabilities, routed.query_labels)
