"""Run a frozen-ResNet feature-cache and adapter training M2 smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tamoe.data.feature_cache import extract_or_load_features
from tamoe.data.medmnist_dataset import MedMNISTTensorDataset
from tamoe.data.medmnist_tasks import get_task
from tamoe.experts.adapters import ResidualAdapter
from tamoe.experts.training import (
    ExpertTrainConfig,
    load_expert_checkpoint,
    train_expert,
)
from tamoe.metrics.resources import count_resources
from tamoe.models.backbones import build_backbone
from tamoe.utils.atomic_io import atomic_write_json, atomic_write_text


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.as_posix()}",
            "-C",
            str(project_root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-markdown", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    raw_config = json.loads(arguments.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone(raw_config["backbone"], pretrained=raw_config["pretrained"])
    dataset = MedMNISTTensorDataset(get_task("breastmnist"), arguments.data_root, "train")
    start = time.perf_counter()
    feature_set = extract_or_load_features(
        dataset,
        backbone,
        arguments.cache_root,
        device=device,
        batch_size=64,
        code_version=_git_commit(arguments.project_root),
        max_samples_per_class=50,
        seed=0,
        num_workers=0,
    )
    training_config = ExpertTrainConfig(
        embedding_dim=backbone.info.embedding_dim,
        rank=int(raw_config["adapter_rank"]),
        steps=10,
        shots=2,
        queries_per_class=2,
        n_way=2,
        learning_rate=float(raw_config["learning_rate"]),
        weight_decay=float(raw_config["weight_decay"]),
        temperature=float(raw_config["temperature"]),
        seed=0,
        schedule="balanced_round_robin",
    )
    checkpoint = arguments.run_root / "m2_smoke" / "expert.pt"
    expert, train_result = train_expert(
        {"breastmnist": feature_set},
        training_config,
        checkpoint,
        checkpoint.parent / "metrics.jsonl",
        device=device,
    )
    loaded, loaded_config = load_expert_checkpoint(checkpoint, device=device)
    sample = feature_set.features[:4].to(device)
    if not torch.allclose(expert.eval()(sample), loaded(sample)):
        raise RuntimeError("saved and loaded expert outputs differ")
    resources = count_resources(
        backbone, [ResidualAdapter(backbone.info.embedding_dim, raw_config["adapter_rank"])]
    )
    summary = {
        "schema_version": 1,
        "status": "SUCCEEDED",
        "experiment_id": "M2_SMOKE",
        "device": str(device),
        "backbone": asdict(backbone.info),
        "feature_cache_key": feature_set.cache_key,
        "feature_shape": list(feature_set.features.shape),
        "checkpoint": str(checkpoint),
        "train_result": asdict(train_result),
        "loaded_config_hash": loaded_config.config_hash,
        "resource_counts": asdict(resources),
        "elapsed_seconds": time.perf_counter() - start,
    }
    atomic_write_json(arguments.report_json, summary)
    markdown = [
        "# M2 frozen-feature expert smoke",
        "",
        f"- Status: **{summary['status']}**",
        f"- Device: `{device}`",
        f"- Backbone: `{backbone.info.name}` / `{backbone.info.weights}` (frozen)",
        f"- Feature shape: `{summary['feature_shape']}`",
        f"- Cache key: `{feature_set.cache_key}`",
        f"- Adapter rank: `{training_config.rank}`",
        f"- Training steps: `{training_config.steps}`",
        "- Save/load output equality: `true`",
        f"- Resource counts: `{summary['resource_counts']}`",
        "",
    ]
    atomic_write_text(arguments.report_markdown, "\n".join(markdown))
    print(json.dumps({"status": "SUCCEEDED", "device": str(device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
