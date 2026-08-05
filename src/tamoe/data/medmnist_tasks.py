"""Audited MedMNIST v2 registry for the initial single-label protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ObjectiveType(StrEnum):
    BINARY_CLASS = "binary-class"
    MULTI_CLASS = "multi-class"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Metadata used by data preparation, never passed into a router."""

    key: str
    python_class: str
    objective: ObjectiveType
    num_classes: int
    channels: int
    group_id: str


def _task(
    key: str,
    python_class: str,
    objective: ObjectiveType,
    num_classes: int,
    channels: int,
    group_id: str | None = None,
) -> TaskSpec:
    return TaskSpec(
        key=key,
        python_class=python_class,
        objective=objective,
        num_classes=num_classes,
        channels=channels,
        group_id=group_id or key,
    )


MEDMNIST_TASKS: tuple[TaskSpec, ...] = (
    _task("pathmnist", "PathMNIST", ObjectiveType.MULTI_CLASS, 9, 3),
    _task("dermamnist", "DermaMNIST", ObjectiveType.MULTI_CLASS, 7, 3),
    _task("octmnist", "OCTMNIST", ObjectiveType.MULTI_CLASS, 4, 1),
    _task("pneumoniamnist", "PneumoniaMNIST", ObjectiveType.BINARY_CLASS, 2, 1),
    _task("breastmnist", "BreastMNIST", ObjectiveType.BINARY_CLASS, 2, 1),
    _task("bloodmnist", "BloodMNIST", ObjectiveType.MULTI_CLASS, 8, 3),
    _task("tissuemnist", "TissueMNIST", ObjectiveType.MULTI_CLASS, 8, 1),
    _task(
        "organamnist",
        "OrganAMNIST",
        ObjectiveType.MULTI_CLASS,
        11,
        1,
        "organmnist",
    ),
    _task(
        "organcmnist",
        "OrganCMNIST",
        ObjectiveType.MULTI_CLASS,
        11,
        1,
        "organmnist",
    ),
    _task(
        "organsmnist",
        "OrganSMNIST",
        ObjectiveType.MULTI_CLASS,
        11,
        1,
        "organmnist",
    ),
)

_TASK_BY_KEY = {task.key: task for task in MEDMNIST_TASKS}


def get_task(key: str) -> TaskSpec:
    try:
        return _TASK_BY_KEY[key.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(_TASK_BY_KEY))
        raise KeyError(f"unknown MedMNIST task {key!r}; allowed: {allowed}") from exc


def grouped_tasks(tasks: tuple[TaskSpec, ...] = MEDMNIST_TASKS) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for task in tasks:
        groups.setdefault(task.group_id, []).append(task.key)
    return {group_id: tuple(sorted(keys)) for group_id, keys in sorted(groups.items())}
