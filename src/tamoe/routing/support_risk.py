"""Direct cross-validated support-risk routing without learned parameters."""

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
from tamoe.routing.support import evaluate_routing_episode, mixture_log_probabilities


@dataclass(frozen=True, slots=True)
class SupportRiskEpisode:
    episode_hash: str
    candidate_names: tuple[str, ...]
    query_labels: Tensor
    candidate_log_probabilities: Tensor
    shrinkage_weights: Tensor
    shared_index: int
    risk_statistics: tuple[dict[str, float | str], ...]


def calibration_metrics(
    log_probabilities: Tensor, labels: Tensor, *, bins: int = 15
) -> dict[str, float]:
    probabilities = log_probabilities.exp()
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(labels).float()
    ece = confidence.new_zeros(())
    boundaries = torch.linspace(0, 1, bins + 1, device=confidence.device)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece = ece + selected.float().mean() * (
                correct[selected].mean() - confidence[selected].mean()
            ).abs()
    one_hot = functional.one_hot(labels, num_classes=probabilities.shape[-1]).float()
    brier = ((probabilities - one_hot) ** 2).sum(dim=-1).mean()
    return {"ece": float(ece.item()), "brier": float(brier.item())}


def probability_metrics(log_probabilities: Tensor, labels: Tensor) -> dict[str, float]:
    prediction = log_probabilities.argmax(dim=-1)
    return {
        "accuracy": float(prediction.eq(labels).float().mean().item()),
        "macro_f1": macro_f1(prediction, labels),
        "loss": float(functional.nll_loss(log_probabilities, labels).item()),
        **calibration_metrics(log_probabilities, labels),
    }


def _risk_statistics(
    adapted_support: Tensor,
    support_labels: Tensor,
    *,
    temperature: float,
) -> dict[str, float]:
    naive_logits = prototype_logits(
        adapted_support, support_labels, adapted_support, temperature=temperature
    )
    naive_loss = functional.cross_entropy(naive_logits, support_labels)
    sample_count, embedding_dim = adapted_support.shape
    class_count = int(support_labels.max().item()) + 1
    class_sums = adapted_support.new_zeros((class_count, embedding_dim))
    class_sums.index_add_(0, support_labels, adapted_support)
    class_counts = torch.bincount(support_labels, minlength=class_count).to(adapted_support)
    leave_one_out_sums = class_sums.unsqueeze(0).expand(sample_count, -1, -1).clone()
    row_index = torch.arange(sample_count, device=adapted_support.device)
    leave_one_out_sums[row_index, support_labels] -= adapted_support
    leave_one_out_counts = class_counts.unsqueeze(0).expand(sample_count, -1).clone()
    leave_one_out_counts[row_index, support_labels] -= 1
    if (leave_one_out_counts <= 0).any():
        raise ValueError("leave-one-out risk requires at least two samples per class")
    leave_one_out_prototypes = functional.normalize(
        leave_one_out_sums / leave_one_out_counts.unsqueeze(-1), dim=-1
    )
    held_out_queries = functional.normalize(adapted_support, dim=-1)
    held_out_logits = torch.einsum(
        "nd,ncd->nc", held_out_queries, leave_one_out_prototypes
    ) / temperature
    loss_values = functional.cross_entropy(
        held_out_logits, support_labels, reduction="none"
    )
    correct_values = held_out_logits.argmax(dim=-1).eq(support_labels).float()
    return {
        "naive_support_loss": float(naive_loss.item()),
        "loo_cross_entropy": float(loss_values.mean().item()),
        "loo_accuracy": float(correct_values.mean().item()),
        "loo_loss_variance": float(loss_values.var(unbiased=False).item()),
        "support_sample_count": float(len(loss_values)),
    }


def shrink_cross_validated_risks(
    mean_risks: Tensor,
    risk_variances: Tensor,
    sample_count: int,
    *,
    prior_strength: float,
    variance_scale: float,
) -> Tensor:
    if prior_strength <= 0 or variance_scale <= 0 or sample_count <= 0:
        raise ValueError("shrinkage parameters and sample count must be positive")
    pooled = mean_risks.mean()
    effective_count = sample_count / (1 + risk_variances / variance_scale)
    return (
        effective_count * mean_risks + prior_strength * pooled
    ) / (effective_count + prior_strength)


def _weight_fields(weights: Tensor) -> dict[str, Any]:
    values = weights.detach().cpu().clamp_min(1e-12)
    return {
        "route_entropy": float(-(values * values.log()).sum().item()),
        "top1_route_weight": float(values.max().item()),
        "route_weights": "|".join(f"{float(value):.12g}" for value in values),
    }


def evaluate_support_risk_episode(
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
    original_route_temperature: float,
    risk_temperature: float,
    prior_strength: float,
    variance_scale: float,
    shared_fallback_weight: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], SupportRiskEpisode]:
    if shots <= 1:
        raise ValueError("leave-one-out support risk requires at least two shots per class")
    if risk_temperature <= 0:
        raise ValueError("risk temperature must be positive")
    if not 0 <= shared_fallback_weight <= 1:
        raise ValueError("shared fallback weight must lie in [0, 1]")
    original_rows, original = evaluate_routing_episode(
        experts,
        references,
        features,
        labels,
        shots=shots,
        queries_per_class=queries_per_class,
        seed=seed,
        repetition=repetition,
        prototype_temperature=prototype_temperature,
        route_temperature=original_route_temperature,
        device=device,
        shared_fallback_weight=shared_fallback_weight,
    )
    episode = sample_episode_indices(
        index_labels(labels.tolist()),
        shots=shots,
        queries_per_class=queries_per_class,
        seed=seed,
        repetition=repetition,
    )
    if episode.episode_hash != original.episode_hash:
        raise RuntimeError("support-risk and Gate 2 evaluators sampled different episodes")
    support = features[torch.tensor(episode.support_indices, dtype=torch.long)].to(device)
    query = features[torch.tensor(episode.query_indices, dtype=torch.long)].to(device)
    support_labels = torch.tensor(episode.support_labels, dtype=torch.long, device=device)
    query_labels = torch.tensor(episode.query_labels, dtype=torch.long, device=device)
    candidates = original.candidate_names
    candidate_logs = original.candidate_log_probabilities.to(device)
    fixed_logs: dict[str, Tensor] = {}
    adapted_support: dict[str, Tensor] = {}
    adapted_query: dict[str, Tensor] = {}
    statistics: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for name in sorted(experts):
            support_embedding = experts[name](support)
            query_embedding = experts[name](query)
            adapted_support[name] = support_embedding
            adapted_query[name] = query_embedding
            logits = prototype_logits(
                support_embedding,
                support_labels,
                query_embedding,
                temperature=prototype_temperature,
            )
            fixed_logs[name] = functional.log_softmax(logits, dim=-1)
        for name in candidates:
            statistics.append(
                {
                    "expert": name,
                    **_risk_statistics(
                        adapted_support[name],
                        support_labels,
                        temperature=prototype_temperature,
                    ),
                }
            )
    mean_risks = torch.tensor(
        [float(record["loo_cross_entropy"]) for record in statistics], device=device
    )
    variances = torch.tensor(
        [float(record["loo_loss_variance"]) for record in statistics], device=device
    )
    naive_risks = torch.tensor(
        [float(record["naive_support_loss"]) for record in statistics], device=device
    )
    sample_count = int(float(statistics[0]["support_sample_count"]))
    shrunk_risks = shrink_cross_validated_risks(
        mean_risks,
        variances,
        sample_count,
        prior_strength=prior_strength,
        variance_scale=variance_scale,
    )
    loo_weights = functional.softmax(-mean_risks / risk_temperature, dim=0)
    shrinkage_weights = functional.softmax(-shrunk_risks / risk_temperature, dim=0)
    for record, shrunk_risk in zip(statistics, shrunk_risks, strict=True):
        record["shrinkage_cross_entropy_risk"] = float(shrunk_risk.item())
    naive_index = min(
        range(len(candidates)), key=lambda index: (float(naive_risks[index]), candidates[index])
    )
    loo_index = min(
        range(len(candidates)), key=lambda index: (float(mean_risks[index]), candidates[index])
    )
    loo_mixture = mixture_log_probabilities(candidate_logs, loo_weights)
    shrinkage_mixture = mixture_log_probabilities(candidate_logs, shrinkage_weights)
    shared_index = candidates.index("shared")
    shared_logs = candidate_logs[shared_index]
    fallback_logs = (
        (1 - shared_fallback_weight) * shrinkage_mixture.exp()
        + shared_fallback_weight * shared_logs.exp()
    ).clamp_min(1e-12).log()

    original_support = original.support_weights.to(device)
    mixed_support = torch.stack([adapted_support[name] for name in candidates]).mul(
        original_support.view(-1, 1, 1)
    ).sum(dim=0)
    mixed_query = torch.stack([adapted_query[name] for name in candidates]).mul(
        original_support.view(-1, 1, 1)
    ).sum(dim=0)
    original_prototype_logs = functional.log_softmax(
        prototype_logits(
            mixed_support,
            support_labels,
            mixed_query,
            temperature=prototype_temperature,
        ),
        dim=-1,
    )
    original_soft_logs = mixture_log_probabilities(candidate_logs, original_support)
    label_removed_weights = original.label_removed_weights.to(device)
    label_removed_logs = (
        (1 - shared_fallback_weight)
        * mixture_log_probabilities(candidate_logs, label_removed_weights).exp()
        + shared_fallback_weight * shared_logs.exp()
    ).clamp_min(1e-12).log()

    fixed_metrics = {name: probability_metrics(logs, query_labels) for name, logs in fixed_logs.items()}
    oracle_name = min(
        candidates,
        key=lambda name: (-fixed_metrics[name]["accuracy"], fixed_metrics[name]["loss"], name),
    )
    rows: list[dict[str, Any]] = []

    def append(
        method: str,
        logs: Tensor,
        selected: str,
        router_input: str,
        weights: Tensor | None = None,
    ) -> None:
        fields = (
            _weight_fields(weights)
            if weights is not None
            else {"route_entropy": float("nan"), "top1_route_weight": float("nan"),
                  "route_weights": ""}
        )
        rows.append(
            {
                "method": method,
                "selected_expert": selected,
                "router_input": router_input,
                **fields,
                **probability_metrics(logs, query_labels),
            }
        )

    append("shared", shared_logs, "shared", "none")
    append("capacity_matched_single", fixed_logs["single"], "single", "none")
    append("oracle_analysis_only", fixed_logs[oracle_name], oracle_name, "query_labels_analysis_only")
    append(
        "naive_support_loss_top1",
        candidate_logs[naive_index],
        candidates[naive_index],
        "support_embeddings_and_labels",
    )
    append(
        "leave_one_out_support_risk_top1",
        candidate_logs[loo_index],
        candidates[loo_index],
        "support_embeddings_and_labels",
    )
    append(
        "leave_one_out_support_risk_soft_mixture",
        loo_mixture,
        candidates[int(loo_weights.argmax().item())],
        "support_embeddings_and_labels",
        loo_weights,
    )
    append(
        "shrinkage_support_risk_mixture",
        shrinkage_mixture,
        candidates[int(shrinkage_weights.argmax().item())],
        "support_embeddings_and_labels",
        shrinkage_weights,
    )
    append(
        "shrinkage_support_risk_with_shared_fallback",
        fallback_logs,
        candidates[int(shrinkage_weights.argmax().item())],
        "support_embeddings_and_labels",
        shrinkage_weights,
    )
    append(
        "original_support_prototype",
        original_prototype_logs,
        candidates[int(original_support.argmax().item())],
        "support_embeddings_and_labels",
        original_support,
    )
    append(
        "original_support_soft_mixture",
        original_soft_logs,
        candidates[int(original_support.argmax().item())],
        "support_embeddings_and_labels",
        original_support,
    )
    append(
        "support_label_removal",
        label_removed_logs,
        candidates[int(label_removed_weights.argmax().item())],
        "support_embeddings_without_labels",
        label_removed_weights,
    )
    random_values = {
        metric: sum(fixed_metrics[name][metric] for name in candidates) / len(candidates)
        for metric in ("accuracy", "macro_f1", "loss", "ece", "brier")
    }
    rows.append(
        {
            "method": "random_expected",
            "selected_expert": "exact_candidate_mean",
            "router_input": "none",
            "route_entropy": float("nan"),
            "top1_route_weight": float("nan"),
            "route_weights": "",
            **random_values,
        }
    )
    expected_original = {row["method"]: row for row in original_rows}[
        "support_conditioned_soft_mixture"
    ]
    reproduced_original = {row["method"]: row for row in rows}[
        "original_support_soft_mixture"
    ]
    if any(
        abs(expected_original[metric] - reproduced_original[metric]) > 1e-7
        for metric in ("accuracy", "macro_f1", "loss")
    ):
        raise RuntimeError("original Gate 2 soft-mixture evaluation was not reproduced")
    return rows, SupportRiskEpisode(
        episode_hash=episode.episode_hash,
        candidate_names=candidates,
        query_labels=query_labels.detach().cpu(),
        candidate_log_probabilities=candidate_logs.detach().cpu(),
        shrinkage_weights=shrinkage_weights.detach().cpu(),
        shared_index=shared_index,
        risk_statistics=tuple(statistics),
    )


def evaluate_risk_ablation(
    recipient: SupportRiskEpisode,
    donor_weights: Tensor,
    *,
    shared_fallback_weight: float,
) -> dict[str, Any]:
    logs = mixture_log_probabilities(
        recipient.candidate_log_probabilities, donor_weights.cpu()
    )
    shared_logs = recipient.candidate_log_probabilities[recipient.shared_index]
    fallback_logs = (
        (1 - shared_fallback_weight) * logs.exp()
        + shared_fallback_weight * shared_logs.exp()
    ).clamp_min(1e-12).log()
    return {
        **_weight_fields(donor_weights),
        **probability_metrics(fallback_logs, recipient.query_labels),
    }
