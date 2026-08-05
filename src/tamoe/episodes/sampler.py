"""Deterministic N-way K-shot sampling with auditable local sample IDs."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tamoe.utils.atomic_io import atomic_write_json


@dataclass(frozen=True, slots=True)
class EpisodeIndices:
    seed: int
    repetition: int
    class_ids: tuple[int, ...]
    support_indices: tuple[int, ...]
    support_labels: tuple[int, ...]
    query_indices: tuple[int, ...]
    query_labels: tuple[int, ...]
    episode_hash: str

    def validate(self) -> None:
        if set(self.support_indices) & set(self.query_indices):
            raise ValueError("support and query indices overlap")
        if len(self.support_indices) != len(self.support_labels):
            raise ValueError("support indices and labels differ in length")
        if len(self.query_indices) != len(self.query_labels):
            raise ValueError("query indices and labels differ in length")
        expected_local_labels = set(range(len(self.class_ids)))
        if set(self.support_labels) != expected_local_labels:
            raise ValueError("support does not contain every episode-local class")
        if not set(self.query_labels).issubset(expected_local_labels):
            raise ValueError("query contains a label outside the episode-local space")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        self.validate()
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class RouterInput:
    """The only episode fields a task-ID-free router may receive."""

    support_images: Tensor
    support_labels: Tensor
    query_images: Tensor


@dataclass(frozen=True, slots=True)
class MaterializedEpisode:
    router_input: RouterInput
    query_labels: Tensor
    audit_indices: EpisodeIndices


def _episode_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def index_labels(labels: Sequence[int]) -> dict[int, tuple[int, ...]]:
    indexed: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        indexed.setdefault(int(label), []).append(index)
    return {label: tuple(indices) for label, indices in sorted(indexed.items())}


def sample_episode_indices(
    class_to_indices: Mapping[int, Sequence[int]],
    *,
    shots: int,
    queries_per_class: int,
    seed: int,
    repetition: int = 0,
    n_way: int | None = None,
) -> EpisodeIndices:
    if shots <= 0 or queries_per_class <= 0:
        raise ValueError("shots and queries_per_class must be positive")
    available_classes = sorted(int(label) for label in class_to_indices)
    ways = len(available_classes) if n_way is None else n_way
    if ways <= 0 or ways > len(available_classes):
        raise ValueError("n_way must be positive and cannot exceed available classes")
    combined_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{repetition}".encode()).digest()[:8], "big"
    )
    random_generator = random.Random(combined_seed)
    class_ids = tuple(sorted(random_generator.sample(available_classes, ways)))
    support_indices: list[int] = []
    support_labels: list[int] = []
    query_indices: list[int] = []
    query_labels: list[int] = []
    required = shots + queries_per_class
    for local_label, class_id in enumerate(class_ids):
        candidates = list(map(int, class_to_indices[class_id]))
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"class {class_id} contains duplicate sample indices")
        if len(candidates) < required:
            raise ValueError(f"class {class_id} has {len(candidates)} samples; {required} required")
        selected = random_generator.sample(candidates, required)
        support_indices.extend(selected[:shots])
        support_labels.extend([local_label] * shots)
        query_indices.extend(selected[shots:])
        query_labels.extend([local_label] * queries_per_class)
    payload = {
        "seed": seed,
        "repetition": repetition,
        "class_ids": class_ids,
        "support_indices": tuple(support_indices),
        "support_labels": tuple(support_labels),
        "query_indices": tuple(query_indices),
        "query_labels": tuple(query_labels),
    }
    episode = EpisodeIndices(**payload, episode_hash=_episode_hash(payload))
    episode.validate()
    return episode


def materialize_episode(dataset: Sequence[Any], episode: EpisodeIndices) -> MaterializedEpisode:
    """Materialize `(image, label)` items without exposing task metadata to a router."""

    def image_at(index: int) -> Tensor:
        item = dataset[index]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image if isinstance(image, Tensor) else torch.as_tensor(image)

    support_images = torch.stack([image_at(index) for index in episode.support_indices])
    query_images = torch.stack([image_at(index) for index in episode.query_indices])
    return MaterializedEpisode(
        router_input=RouterInput(
            support_images=support_images,
            support_labels=torch.tensor(episode.support_labels, dtype=torch.long),
            query_images=query_images,
        ),
        query_labels=torch.tensor(episode.query_labels, dtype=torch.long),
        audit_indices=episode,
    )


def assert_router_input_is_metadata_free(router_input: RouterInput) -> None:
    field_names = set(router_input.__dataclass_fields__)
    forbidden_fragments = {
        "task",
        "dataset",
        "path",
        "split",
        "source",
        "group",
        "metadata",
        "sample_id",
        "index",
    }
    leaking = {
        name
        for name in field_names
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    }
    if leaking:
        raise ValueError(f"router input contains forbidden metadata fields: {sorted(leaking)}")
