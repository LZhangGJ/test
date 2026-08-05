"""MedMNIST preparation and tensor access, kept separate from training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from tamoe.data.medmnist_tasks import TaskSpec, get_task


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    task: str
    split: str
    size: int
    sample_count: int
    label_counts: dict[int, int]
    source_file: str
    source_sha256: str


class MedMNISTTensorDataset(Sequence[tuple[Tensor, int]]):
    """Load official NPZ arrays into normalized three-channel tensors."""

    def __init__(self, task: TaskSpec, root: Path, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        path = root / f"{task.key}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"prepared MedMNIST file is missing: {path}; run prepare_medmnist.py"
            )
        with np.load(path, allow_pickle=False) as archive:
            self._images = np.asarray(archive[f"{split}_images"])
            labels = np.asarray(archive[f"{split}_labels"])
        self._labels = labels.reshape(-1).astype(np.int64, copy=False)
        if len(self._images) != len(self._labels):
            raise ValueError(f"image/label count mismatch in {path} split {split}")
        self.task = task
        self.split = split
        self.path = path

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(map(int, self._labels))

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        array = self._images[index]
        tensor = torch.from_numpy(np.array(array, copy=True)).float().div_(255.0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"unsupported image rank {tensor.ndim} at index {index}")
        if tensor.shape[0] == 1:
            tensor = tensor.expand(3, -1, -1)
        if tensor.shape[0] != 3:
            raise ValueError(f"expected one or three channels, found {tensor.shape[0]}")
        return tensor, int(self._labels[index])


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_task(task_key: str, root: Path, *, size: int = 28) -> list[PreparedDataset]:
    """Download through MedMNIST's public API and record the resulting NPZ hash."""

    task = get_task(task_key)
    if size not in {28, 64, 128, 224}:
        raise ValueError("MedMNIST size must be one of 28, 64, 128, or 224")
    root.mkdir(parents=True, exist_ok=True)
    import medmnist

    dataset_class = getattr(medmnist, task.python_class)
    prepared: list[PreparedDataset] = []
    for split in ("train", "val", "test"):
        dataset = dataset_class(
            split=split,
            root=str(root),
            download=True,
            size=size,
            as_rgb=False,
        )
        labels = np.asarray(dataset.labels).reshape(-1)
        unique, counts = np.unique(labels, return_counts=True)
        source_file = root / f"{task.key}{f'_{size}' if size != 28 else ''}.npz"
        if not source_file.exists() and size == 28:
            source_file = root / f"{task.key}.npz"
        prepared.append(
            PreparedDataset(
                task=task.key,
                split=split,
                size=size,
                sample_count=len(labels),
                label_counts={int(label): int(count) for label, count in zip(unique, counts)},
                source_file=str(source_file),
                source_sha256=sha256_file(source_file),
            )
        )
    return prepared


def write_preparation_manifest(path: Path, datasets: Sequence[PreparedDataset]) -> None:
    from tamoe.utils.atomic_io import atomic_write_text

    payload = [asdict(dataset) for dataset in datasets]
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
