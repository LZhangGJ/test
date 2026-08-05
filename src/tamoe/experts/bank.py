"""Capacity-matched shared, single, and source-group expert banks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from tamoe.data.feature_cache import FeatureSet
from tamoe.data.medmnist_tasks import grouped_tasks
from tamoe.data.task_splits import TaskSplit
from tamoe.experts.adapters import ResidualAdapter
from tamoe.experts.training import ExpertTrainConfig, TrainResult, train_expert
from tamoe.utils.atomic_io import atomic_write_json


@dataclass(frozen=True, slots=True)
class ExpertDefinition:
    name: str
    kind: str
    source_tasks: tuple[str, ...]
    schedule: str


@dataclass(frozen=True, slots=True)
class TrainedExpert:
    definition: ExpertDefinition
    checkpoint: str
    train_result: TrainResult
    parameter_count: int


def build_expert_definitions(split: TaskSplit) -> tuple[ExpertDefinition, ...]:
    train_tasks = tuple(sorted(split.meta_train))
    definitions = [
        ExpertDefinition(
            name="shared",
            kind="shared",
            source_tasks=train_tasks,
            schedule="balanced_round_robin",
        ),
        ExpertDefinition(
            name="single",
            kind="equal_parameter_single",
            source_tasks=train_tasks,
            schedule="pooled_random",
        ),
    ]
    groups = grouped_tasks()
    for group_id in sorted(split.meta_train_groups):
        source_tasks = tuple(task for task in groups[group_id] if task in train_tasks)
        definitions.append(
            ExpertDefinition(
                name=f"source_{group_id}",
                kind="source_group",
                source_tasks=source_tasks,
                schedule="balanced_round_robin",
            )
        )
    return tuple(definitions)


def train_expert_bank(
    feature_sets: dict[str, FeatureSet],
    split: TaskSplit,
    base_config: ExpertTrainConfig,
    output_directory: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, ResidualAdapter], tuple[TrainedExpert, ...]]:
    definitions = build_expert_definitions(split)
    experts: dict[str, ResidualAdapter] = {}
    records = []
    expected_parameter_count: int | None = None
    for definition in definitions:
        missing = set(definition.source_tasks) - set(feature_sets)
        if missing:
            raise KeyError(f"missing feature sets for {definition.name}: {sorted(missing)}")
        config = replace(base_config, schedule=definition.schedule)
        checkpoint = output_directory / definition.name / "checkpoint.pt"
        expert, train_result = train_expert(
            {task: feature_sets[task] for task in definition.source_tasks},
            config,
            checkpoint,
            checkpoint.parent / "metrics.jsonl",
            device=device,
        )
        parameter_count = sum(parameter.numel() for parameter in expert.parameters())
        if expected_parameter_count is None:
            expected_parameter_count = parameter_count
        elif parameter_count != expected_parameter_count:
            raise RuntimeError("expert bank contains unequal parameter counts")
        experts[definition.name] = expert.eval()
        records.append(
            TrainedExpert(
                definition=definition,
                checkpoint=str(checkpoint),
                train_result=train_result,
                parameter_count=parameter_count,
            )
        )
    atomic_write_json(
        output_directory / "expert_bank_manifest.json",
        {
            "schema_version": 1,
            "split_seed": split.seed,
            "split_hash": split.split_hash,
            "base_config": asdict(base_config),
            "experts": [asdict(record) for record in records],
        },
    )
    return experts, tuple(records)
