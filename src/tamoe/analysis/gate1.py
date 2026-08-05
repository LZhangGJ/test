"""Analysis-only Gate 1 episode evaluation and preregistered decision rules."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional

from tamoe.episodes.sampler import index_labels, sample_episode_indices
from tamoe.experts.adapters import ResidualAdapter
from tamoe.models.episodic_head import prototype_logits


@dataclass(frozen=True, slots=True)
class Gate1Thresholds:
    """Numerical Gate 1 rules fixed before looking at pilot outcomes."""

    minimum_accuracy_gap: float = 0.01
    maximum_global_best_frequency: float = 0.90
    minimum_passing_split_count: int = 2
    minimum_passing_task_split_count: int = 2
    minimum_intervention_task_split_count: int = 2


def macro_f1(prediction: Tensor, target: Tensor) -> float:
    values: list[float] = []
    for label in torch.unique(target, sorted=True):
        true_positive = ((prediction == label) & (target == label)).sum().float()
        false_positive = ((prediction == label) & (target != label)).sum().float()
        false_negative = ((prediction != label) & (target == label)).sum().float()
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(float((2 * true_positive / denominator).item()) if denominator else 0.0)
    return float(np.mean(values))


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    prediction = logits.argmax(dim=-1)
    return {
        "accuracy": float((prediction == labels).float().mean().item()),
        "macro_f1": macro_f1(prediction, labels),
        "loss": float(functional.cross_entropy(logits, labels).item()),
    }


def oracle_convex_mixture(
    logits_by_expert: Tensor, labels: Tensor, *, steps: int = 75
) -> tuple[dict[str, float], list[float]]:
    """Fit analysis-only convex prediction weights using query labels."""

    if logits_by_expert.ndim != 3:
        raise ValueError("logits_by_expert must have shape [experts, queries, classes]")
    log_probabilities = functional.log_softmax(logits_by_expert.detach(), dim=-1)
    raw_weights = torch.zeros(
        logits_by_expert.shape[0], device=logits_by_expert.device, requires_grad=True
    )
    optimizer = torch.optim.Adam([raw_weights], lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        log_weights = functional.log_softmax(raw_weights, dim=0).view(-1, 1, 1)
        mixture = torch.logsumexp(log_probabilities + log_weights, dim=0)
        loss = functional.nll_loss(mixture, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weights = raw_weights.softmax(dim=0)
        mixture = torch.logsumexp(
            log_probabilities + weights.log().view(-1, 1, 1), dim=0
        )
    return _metrics(mixture, labels), [float(value) for value in weights.cpu()]


def sample_oracle_metrics(logits_by_expert: Tensor, labels: Tensor) -> dict[str, float]:
    """Query-label upper bound, reported only as secondary analysis."""

    losses = torch.stack(
        [functional.cross_entropy(logits, labels, reduction="none") for logits in logits_by_expert]
    )
    selected = losses.argmin(dim=0)
    query_index = torch.arange(len(labels), device=labels.device)
    chosen_logits = logits_by_expert[selected, query_index]
    return _metrics(chosen_logits, labels)


def evaluate_episode(
    experts: dict[str, ResidualAdapter],
    features: Tensor,
    labels: Tensor,
    *,
    shots: int,
    queries_per_class: int,
    seed: int,
    repetition: int,
    temperature: float,
    device: torch.device,
    random_repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate fixed experts and query-label analysis-only upper bounds."""

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
    expert_names = sorted(experts)
    logits: list[Tensor] = []
    rows: list[dict[str, Any]] = []
    latencies: dict[str, float] = {}
    with torch.inference_mode():
        for name in expert_names:
            start = time.perf_counter()
            adapted_support = experts[name](support)
            adapted_query = experts[name](query)
            expert_logits = prototype_logits(
                adapted_support, support_labels, adapted_query, temperature=temperature
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies[name] = (time.perf_counter() - start) * 1000
            logits.append(expert_logits)
            rows.append({"method": name, "selected_expert": name, **_metrics(expert_logits, query_labels)})
    stacked = torch.stack(logits)
    accuracies = torch.tensor([row["accuracy"] for row in rows])
    losses = torch.tensor([row["loss"] for row in rows])
    best_index = int(torch.argmax(accuracies).item())
    worst_index = int(torch.argmin(accuracies).item())
    if accuracies[best_index] == accuracies[worst_index]:
        best_index = int(torch.argmin(losses).item())
        worst_index = int(torch.argmax(losses).item())
    best_name = expert_names[best_index]
    worst_name = expert_names[worst_index]
    rows.append(
        {
            "method": "episode_oracle",
            "selected_expert": best_name,
            **_metrics(stacked[best_index], query_labels),
        }
    )
    mixture_metrics, mixture_weights = oracle_convex_mixture(stacked, query_labels)
    rows.append(
        {"method": "oracle_mixture", "selected_expert": "convex", **mixture_metrics}
    )
    rows.append(
        {
            "method": "sample_oracle",
            "selected_expert": "per_sample",
            **sample_oracle_metrics(stacked, query_labels),
        }
    )
    rows.append(
        {
            "method": "forced_worst",
            "selected_expert": worst_name,
            **_metrics(stacked[worst_index], query_labels),
        }
    )
    swap_index = (best_index + 1) % len(expert_names)
    rows.append(
        {
            "method": "swap",
            "selected_expert": expert_names[swap_index],
            **_metrics(stacked[swap_index], query_labels),
        }
    )
    random_generator = random.Random(f"gate1:{seed}:{repetition}:{shots}")
    for random_repeat in range(random_repeats):
        index = random_generator.randrange(len(expert_names))
        rows.append(
            {
                "method": f"random_{random_repeat:02d}",
                "selected_expert": expert_names[index],
                **_metrics(stacked[index], query_labels),
            }
        )
    audit = {
        "episode_hash": episode.episode_hash,
        "class_ids": list(episode.class_ids),
        "expert_names": expert_names,
        "expert_metrics": {row["method"]: row for row in rows[: len(expert_names)]},
        "mixture_weights": dict(zip(expert_names, mixture_weights, strict=True)),
        "latency_ms": latencies,
    }
    return rows, audit


def add_global_interventions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add most-used-expert mask and task-modal assignment permutation pilots."""

    oracle = frame[frame["method"] == "episode_oracle"]
    most_used = str(oracle["selected_expert"].value_counts().index[0])
    episode_columns = [
        "split_seed", "train_seed", "task", "shots", "support_resample", "episode_hash"
    ]
    fixed_mask = frame.apply(
        lambda row: str(row["method"]) in str(row["expert_name_set"]).split("|"), axis=1
    )
    fixed = frame[fixed_mask]
    modal = (
        oracle.groupby(["split_seed", "task"])["selected_expert"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )
    modal_values = sorted(set(modal.values()))
    permutation = {
        name: modal_values[(index + 1) % len(modal_values)]
        for index, name in enumerate(modal_values)
    }
    additions: list[dict[str, Any]] = []
    for key, group in fixed.groupby(episode_columns, sort=False):
        records = {str(row.method): row for row in group.itertuples(index=False)}
        candidates = [record for name, record in records.items() if name != most_used]
        mask_record = max(candidates, key=lambda record: (record.accuracy, -record.loss))
        base = dict(zip(episode_columns, key, strict=True))
        common = {
            column: getattr(mask_record, column)
            for column in frame.columns
            if column not in {*episode_columns, "method", "selected_expert", "accuracy", "macro_f1", "loss"}
        }
        additions.append(
            {
                **base, **common, "method": "mask_most_used", "selected_expert": mask_record.method,
                "accuracy": mask_record.accuracy, "macro_f1": mask_record.macro_f1, "loss": mask_record.loss,
            }
        )
        assigned = modal[(int(base["split_seed"]), str(base["task"]))]
        permuted = permutation.get(assigned, assigned)
        permutation_record = records.get(permuted, records[assigned])
        additions.append(
            {
                **base, **common, "method": "permuted_task_assignment",
                "selected_expert": permutation_record.method,
                "accuracy": permutation_record.accuracy, "macro_f1": permutation_record.macro_f1,
                "loss": permutation_record.loss,
            }
        )
    return pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)


def bootstrap_mean_ci(values: np.ndarray, *, repeats: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n": int(len(values)),
    }


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def decide_gate1(
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int,
    thresholds: Gate1Thresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or Gate1Thresholds()
    key = ["split_seed", "train_seed", "task", "shots", "support_resample", "episode_hash"]
    mean_random = (
        frame[frame["method"].str.startswith("random_")]
        .groupby(key, as_index=False)[["accuracy", "macro_f1", "loss"]]
        .mean()
        .assign(method="random_mean")
    )
    analysis = pd.concat([frame, mean_random], ignore_index=True)
    wide = analysis.pivot_table(index=key, columns="method", values="accuracy", aggfunc="first")
    required = {"episode_oracle", "shared", "single"}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"Gate 1 frame lacks required methods: {sorted(missing)}")
    comparisons: dict[str, list[dict[str, Any]]] = {}
    for baseline in ("shared", "single"):
        rows = []
        for split_seed, group in wide.groupby(level="split_seed"):
            delta = (group["episode_oracle"] - group[baseline]).to_numpy()
            rows.append(
                {
                    "split_seed": int(split_seed),
                    "baseline": baseline,
                    **bootstrap_mean_ci(
                        delta, repeats=bootstrap_repeats,
                        seed=_stable_seed(f"split:{split_seed}:{baseline}"),
                    ),
                }
            )
        comparisons[baseline] = rows
    task_split_effects: list[dict[str, Any]] = []
    for (split_seed, task), group in wide.groupby(level=["split_seed", "task"]):
        for baseline in ("shared", "single"):
            delta = (group["episode_oracle"] - group[baseline]).to_numpy()
            task_split_effects.append(
                {
                    "split_seed": int(split_seed), "task": str(task), "baseline": baseline,
                    **bootstrap_mean_ci(
                        delta, repeats=bootstrap_repeats,
                        seed=_stable_seed(f"task:{split_seed}:{task}:{baseline}"),
                    ),
                }
            )
    oracle = frame[frame["method"] == "episode_oracle"]
    frequencies = (oracle["selected_expert"].value_counts(normalize=True).sort_index()).to_dict()
    maximum_frequency = max(frequencies.values())
    interventions = [
        "forced_worst", "random_mean", "swap", "mask_most_used", "permuted_task_assignment"
    ]
    intervention_effects: list[dict[str, Any]] = []
    for (split_seed, task), group in wide.groupby(level=["split_seed", "task"]):
        for method in interventions:
            if method not in group:
                continue
            delta = (group["episode_oracle"] - group[method]).dropna().to_numpy()
            intervention_effects.append(
                {
                    "split_seed": int(split_seed), "task": str(task), "intervention": method,
                    **bootstrap_mean_ci(
                        delta, repeats=bootstrap_repeats,
                        seed=_stable_seed(f"intervention:{split_seed}:{task}:{method}"),
                    ),
                }
            )
    passing_splits = 0
    for split_seed in sorted(wide.index.get_level_values("split_seed").unique()):
        if all(
            next(row for row in comparisons[baseline] if row["split_seed"] == split_seed)["ci_low"]
            > thresholds.minimum_accuracy_gap
            for baseline in ("shared", "single")
        ):
            passing_splits += 1
    passing_task_splits = len(
        {
            (row["split_seed"], row["task"])
            for row in task_split_effects
            if row["ci_low"] > thresholds.minimum_accuracy_gap
            and all(
                any(
                    other["split_seed"] == row["split_seed"]
                    and other["task"] == row["task"]
                    and other["baseline"] == baseline
                    and other["ci_low"] > thresholds.minimum_accuracy_gap
                    for other in task_split_effects
                )
                for baseline in ("shared", "single")
            )
        }
    )
    intervention_groups = {
        (row["split_seed"], row["task"])
        for row in intervention_effects
        if row["mean"] > thresholds.minimum_accuracy_gap
        and all(
            any(
                other["split_seed"] == row["split_seed"]
                and other["task"] == row["task"]
                and other["intervention"] == method
                and other["mean"] > thresholds.minimum_accuracy_gap
                for other in intervention_effects
            )
            for method in interventions
        )
    }
    criteria = {
        "stable_split_oracle_gap": passing_splits >= thresholds.minimum_passing_split_count,
        "multiple_task_split_oracle_gap": passing_task_splits
        >= thresholds.minimum_passing_task_split_count,
        "no_global_expert_dominance": maximum_frequency
        < thresholds.maximum_global_best_frequency,
        "task_specific_intervention_effects": len(intervention_groups)
        >= thresholds.minimum_intervention_task_split_count,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "analysis_only_query_label_use": [
            "episode_oracle", "oracle_mixture", "sample_oracle", "forced_worst"
        ],
        "thresholds": asdict(thresholds),
        "criteria": criteria,
        "passing_split_count": passing_splits,
        "passing_task_split_count": passing_task_splits,
        "passing_intervention_task_split_count": len(intervention_groups),
        "maximum_global_best_frequency": float(maximum_frequency),
        "best_expert_frequencies": {str(key): float(value) for key, value in frequencies.items()},
        "split_comparisons": comparisons,
        "task_split_effects": task_split_effects,
        "intervention_effects": intervention_effects,
    }
