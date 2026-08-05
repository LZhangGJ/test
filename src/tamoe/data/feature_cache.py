"""Validated frozen-feature caches with content-derived keys."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from tamoe.data.medmnist_dataset import MedMNISTTensorDataset, sha256_file
from tamoe.models.backbones import FrozenBackbone, normalize_for_backbone
from tamoe.utils.atomic_io import atomic_torch_save


@dataclass(frozen=True, slots=True)
class FeatureCacheSpec:
    task: str
    split: str
    dataset_sha256: str
    backbone: str
    weights: str
    preprocess_version: str
    code_version: str
    max_samples_per_class: int | None
    seed: int

    @property
    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureSet:
    features: Tensor
    labels: Tensor
    sample_indices: Tensor
    cache_key: str

    def validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("cached features must be rank two")
        if len(self.features) != len(self.labels) or len(self.labels) != len(self.sample_indices):
            raise ValueError("feature, label, and sample-index counts differ")


def _balanced_indices(
    labels: tuple[int, ...], max_samples_per_class: int | None, seed: int
) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    selected = []
    for label, indices in sorted(by_class.items()):
        if max_samples_per_class is None or len(indices) <= max_samples_per_class:
            selected.extend(indices)
            continue
        generator = random.Random(f"{seed}:{label}")
        selected.extend(generator.sample(indices, max_samples_per_class))
    return sorted(selected)


def build_cache_spec(
    dataset: MedMNISTTensorDataset,
    backbone: FrozenBackbone,
    *,
    code_version: str,
    max_samples_per_class: int | None,
    seed: int,
) -> FeatureCacheSpec:
    return FeatureCacheSpec(
        task=dataset.task.key,
        split=dataset.split,
        dataset_sha256=sha256_file(dataset.path),
        backbone=backbone.info.name,
        weights=backbone.info.weights,
        preprocess_version=backbone.info.preprocess_version,
        code_version=code_version,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )


def extract_or_load_features(
    dataset: MedMNISTTensorDataset,
    backbone: FrozenBackbone,
    cache_root: Path,
    *,
    device: torch.device,
    batch_size: int,
    code_version: str,
    max_samples_per_class: int | None,
    seed: int,
    num_workers: int = 0,
) -> FeatureSet:
    spec = build_cache_spec(
        dataset,
        backbone,
        code_version=code_version,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )
    cache_path = cache_root / spec.backbone / spec.task / f"{spec.split}_{spec.key}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("spec") != asdict(spec):
            raise ValueError(f"feature cache metadata mismatch: {cache_path}")
        result = FeatureSet(
            features=payload["features"],
            labels=payload["labels"],
            sample_indices=payload["sample_indices"],
            cache_key=spec.key,
        )
        result.validate()
        return result

    selected = _balanced_indices(dataset.labels, max_samples_per_class, seed)
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    backbone = backbone.to(device).eval()
    features: list[Tensor] = []
    labels: list[Tensor] = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = normalize_for_backbone(images.to(device), backbone.info.name)
            features.append(backbone(images).cpu())
            labels.append(batch_labels.cpu())
    result = FeatureSet(
        features=torch.cat(features),
        labels=torch.cat(labels).long(),
        sample_indices=torch.tensor(selected, dtype=torch.long),
        cache_key=spec.key,
    )
    result.validate()
    atomic_torch_save(
        cache_path,
        {
            "schema_version": 1,
            "spec": asdict(spec),
            "features": result.features,
            "labels": result.labels,
            "sample_indices": result.sample_indices,
        },
    )
    return result
