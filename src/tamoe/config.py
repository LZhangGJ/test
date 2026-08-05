"""Typed configuration loading and stable hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    """Configuration for the cross-platform synthetic M0 smoke test."""

    study: str = "M0"
    experiment_name: str = "synthetic_cpu_smoke"
    seed: int = 20260805
    device: str = "cpu"
    num_workers: int = 0
    num_classes: int = 3
    samples_per_class: int = 6
    shots: int = 2
    queries_per_class: int = 2
    embedding_dim: int = 16
    learning_rate: float = 1e-3

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        for field_name in (
            "num_classes",
            "samples_per_class",
            "shots",
            "queries_per_class",
            "embedding_dim",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        required = self.shots + self.queries_per_class
        if self.samples_per_class < required:
            raise ValueError("samples_per_class must be at least shots + queries_per_class")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, path: Path) -> SmokeConfig:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"configuration file does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON configuration {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"configuration root must be an object: {path}")
        try:
            config = cls(**payload)
        except TypeError as exc:
            raise ValueError(f"invalid smoke configuration fields in {path}: {exc}") from exc
        config.validate()
        return config
