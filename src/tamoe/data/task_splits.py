"""Deterministic group-aware task meta-splits."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tamoe.data.medmnist_tasks import MEDMNIST_TASKS, TaskSpec, grouped_tasks
from tamoe.utils.atomic_io import atomic_write_json


@dataclass(frozen=True, slots=True)
class TaskSplit:
    seed: int
    meta_train: tuple[str, ...]
    meta_validation: tuple[str, ...]
    meta_test: tuple[str, ...]
    meta_train_groups: tuple[str, ...]
    meta_validation_groups: tuple[str, ...]
    meta_test_groups: tuple[str, ...]
    split_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        task_sets = [set(self.meta_train), set(self.meta_validation), set(self.meta_test)]
        if any(left & right for index, left in enumerate(task_sets) for right in task_sets[index + 1 :]):
            raise ValueError("task meta-splits overlap")
        group_sets = [
            set(self.meta_train_groups),
            set(self.meta_validation_groups),
            set(self.meta_test_groups),
        ]
        if any(
            left & right
            for index, left in enumerate(group_sets)
            for right in group_sets[index + 1 :]
        ):
            raise ValueError("task groups cross meta-split boundaries")

    def write(self, path: Path) -> None:
        self.validate()
        atomic_write_json(path, self.to_dict())


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_task_split(
    seed: int,
    *,
    tasks: tuple[TaskSpec, ...] = MEDMNIST_TASKS,
    validation_group_count: int = 1,
    test_group_count: int = 2,
) -> TaskSplit:
    """Split whole task groups; group counts are explicit for the eight-group registry."""

    groups = grouped_tasks(tasks)
    group_ids = sorted(groups)
    if validation_group_count <= 0 or test_group_count <= 0:
        raise ValueError("validation and test group counts must be positive")
    if validation_group_count + test_group_count >= len(group_ids):
        raise ValueError("at least one meta-training group is required")
    random.Random(seed).shuffle(group_ids)
    train_end = len(group_ids) - validation_group_count - test_group_count
    validation_end = len(group_ids) - test_group_count
    train_groups = tuple(sorted(group_ids[:train_end]))
    validation_groups = tuple(sorted(group_ids[train_end:validation_end]))
    test_groups = tuple(sorted(group_ids[validation_end:]))

    def expand(group_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(task for group in group_names for task in groups[group]))

    payload = {
        "seed": seed,
        "meta_train": expand(train_groups),
        "meta_validation": expand(validation_groups),
        "meta_test": expand(test_groups),
        "meta_train_groups": train_groups,
        "meta_validation_groups": validation_groups,
        "meta_test_groups": test_groups,
    }
    split = TaskSplit(**payload, split_hash=_hash_payload(payload))
    split.validate()
    return split


def validate_split_suite(splits: tuple[TaskSplit, ...]) -> None:
    if len({split.seed for split in splits}) != len(splits):
        raise ValueError("task split seeds must be unique")
    expected = {task.key for task in MEDMNIST_TASKS}
    observed = set().union(*(set(split.meta_test) for split in splits))
    missing = expected - observed
    if missing:
        raise ValueError(f"tasks never used as meta-test across split suite: {sorted(missing)}")
