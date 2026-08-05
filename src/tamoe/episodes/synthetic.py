"""Deterministic synthetic episodes used by M0 tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Episode:
    support_images: Tensor
    support_labels: Tensor
    query_images: Tensor
    query_labels: Tensor
    support_indices: Tensor
    query_indices: Tensor

    def validate(self) -> None:
        overlap = set(self.support_indices.tolist()) & set(self.query_indices.tolist())
        if overlap:
            raise ValueError(f"support/query indices overlap: {sorted(overlap)}")
        if self.support_images.shape[0] != self.support_labels.shape[0]:
            raise ValueError("support image/label counts differ")
        if self.query_images.shape[0] != self.query_labels.shape[0]:
            raise ValueError("query image/label counts differ")


def make_synthetic_episode(
    *,
    num_classes: int,
    samples_per_class: int,
    shots: int,
    queries_per_class: int,
    seed: int,
) -> Episode:
    if samples_per_class < shots + queries_per_class:
        raise ValueError("insufficient unique samples for disjoint support and query")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = []
    labels = []
    for label in range(num_classes):
        center = torch.full((1, 8, 8), float(label) / max(num_classes - 1, 1))
        noise = torch.randn((samples_per_class, 1, 8, 8), generator=generator) * 0.05
        images.append(center + noise)
        labels.append(torch.full((samples_per_class,), label, dtype=torch.long))
    all_images = torch.cat(images)
    all_labels = torch.cat(labels)

    support_indices: list[int] = []
    query_indices: list[int] = []
    for label in range(num_classes):
        offset = label * samples_per_class
        permutation = torch.randperm(samples_per_class, generator=generator).tolist()
        support_indices.extend(offset + index for index in permutation[:shots])
        query_indices.extend(
            offset + index for index in permutation[shots : shots + queries_per_class]
        )
    support_tensor = torch.tensor(support_indices, dtype=torch.long)
    query_tensor = torch.tensor(query_indices, dtype=torch.long)
    episode = Episode(
        support_images=all_images[support_tensor],
        support_labels=all_labels[support_tensor],
        query_images=all_images[query_tensor],
        query_labels=all_labels[query_tensor],
        support_indices=support_tensor,
        query_indices=query_tensor,
    )
    episode.validate()
    return episode
