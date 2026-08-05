"""Resumable episodic expert training over frozen feature caches."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional

from tamoe.data.feature_cache import FeatureSet
from tamoe.episodes.sampler import index_labels, sample_episode_indices
from tamoe.experts.adapters import ResidualAdapter
from tamoe.models.episodic_head import prototype_logits
from tamoe.utils.atomic_io import atomic_torch_save
from tamoe.utils.jsonl import JsonlLogger
from tamoe.utils.reproducibility import seed_everything


@dataclass(frozen=True, slots=True)
class ExpertTrainConfig:
    embedding_dim: int
    rank: int
    steps: int
    shots: int
    queries_per_class: int
    n_way: int
    learning_rate: float
    weight_decay: float
    temperature: float
    seed: int
    schedule: str

    def validate(self) -> None:
        if self.steps <= 0 or self.rank <= 0 or self.embedding_dim <= 0:
            raise ValueError("steps, rank, and embedding_dim must be positive")
        if self.schedule not in {"balanced_round_robin", "pooled_random"}:
            raise ValueError("unsupported training schedule")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainResult:
    checkpoint: str
    start_step: int
    final_step: int
    final_loss: float
    final_accuracy: float
    resumed: bool
    config_hash: str


def load_expert_checkpoint(
    checkpoint_path: Path, *, device: torch.device
) -> tuple[ResidualAdapter, ExpertTrainConfig]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ExpertTrainConfig(**payload["config"])
    if payload.get("config_hash") != config.config_hash:
        raise ValueError(f"checkpoint configuration hash is invalid: {checkpoint_path}")
    expert = ResidualAdapter(config.embedding_dim, config.rank).to(device)
    expert.load_state_dict(payload["model"])
    expert.eval()
    return expert, config


def _choose_task(
    task_names: list[str], feature_sets: dict[str, FeatureSet], config: ExpertTrainConfig, step: int
) -> str:
    if config.schedule == "balanced_round_robin":
        return task_names[step % len(task_names)]
    weights = [len(feature_sets[name].labels) for name in task_names]
    return random.Random(f"{config.seed}:{step}").choices(task_names, weights=weights, k=1)[0]


def _episode_tensors(
    feature_set: FeatureSet, config: ExpertTrainConfig, step: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mapping = index_labels(feature_set.labels.tolist())
    n_way = min(config.n_way, len(mapping)) if config.n_way > 0 else None
    episode = sample_episode_indices(
        mapping,
        shots=config.shots,
        queries_per_class=config.queries_per_class,
        seed=config.seed,
        repetition=step,
        n_way=n_way,
    )
    support_index = torch.tensor(episode.support_indices, dtype=torch.long)
    query_index = torch.tensor(episode.query_indices, dtype=torch.long)
    return (
        feature_set.features[support_index].to(device),
        torch.tensor(episode.support_labels, dtype=torch.long, device=device),
        feature_set.features[query_index].to(device),
        torch.tensor(episode.query_labels, dtype=torch.long, device=device),
    )


def train_expert(
    feature_sets: dict[str, FeatureSet],
    config: ExpertTrainConfig,
    checkpoint_path: Path,
    log_path: Path,
    *,
    device: torch.device,
) -> tuple[ResidualAdapter, TrainResult]:
    config.validate()
    if not feature_sets:
        raise ValueError("expert training requires at least one task feature set")
    seed_everything(config.seed)
    expert = ResidualAdapter(config.embedding_dim, config.rank).to(device)
    optimizer = torch.optim.AdamW(
        expert.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_step = 0
    resumed = False
    final_loss = float("nan")
    final_accuracy = float("nan")
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("config_hash") != config.config_hash:
            raise ValueError(f"checkpoint config mismatch: {checkpoint_path}")
        expert.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        resumed = start_step > 0
        final_loss = float(payload.get("final_loss", final_loss))
        final_accuracy = float(payload.get("final_accuracy", final_accuracy))
    logger = JsonlLogger(log_path)
    task_names = sorted(feature_sets)
    for step in range(start_step, config.steps):
        task = _choose_task(task_names, feature_sets, config, step)
        support, support_labels, query, query_labels = _episode_tensors(
            feature_sets[task], config, step, device
        )
        optimizer.zero_grad(set_to_none=True)
        adapted_support = expert(support)
        adapted_query = expert(query)
        logits = prototype_logits(
            adapted_support,
            support_labels,
            adapted_query,
            temperature=config.temperature,
        )
        loss = functional.cross_entropy(logits, query_labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        final_accuracy = float((logits.argmax(dim=-1) == query_labels).float().mean().cpu())
        logger.log(
            "train_step",
            step=step + 1,
            task=task,
            loss=final_loss,
            accuracy=final_accuracy,
        )
        if (step + 1) % 50 == 0 or step + 1 == config.steps:
            atomic_torch_save(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "config": asdict(config),
                    "config_hash": config.config_hash,
                    "step": step + 1,
                    "final_loss": final_loss,
                    "final_accuracy": final_accuracy,
                    "model": expert.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "feature_cache_keys": {
                        name: feature_set.cache_key for name, feature_set in feature_sets.items()
                    },
                },
            )
    return expert, TrainResult(
        checkpoint=str(checkpoint_path),
        start_step=start_step,
        final_step=config.steps,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        resumed=resumed,
        config_hash=config.config_hash,
    )
