"""End-to-end synthetic CPU smoke test for M0."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as functional

from tamoe.analysis.oracle import episode_oracle
from tamoe.config import SmokeConfig
from tamoe.episodes.synthetic import make_synthetic_episode
from tamoe.models.tiny import ResidualExpert, TinyBackbone
from tamoe.utils.atomic_io import atomic_write_json
from tamoe.utils.jsonl import JsonlLogger
from tamoe.utils.reproducibility import seed_everything
from tamoe.utils.run_identity import make_run_id


def _git_state(project_root: Path) -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        command = [
            "git",
            "-c",
            f"safe.directory={project_root.as_posix()}",
            "-C",
            str(project_root),
            *arguments,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _prototype_logits(
    support: Tensor,
    support_labels: Tensor,
    query: Tensor,
    num_classes: int,
) -> Tensor:
    prototypes = torch.stack(
        [support[support_labels == label].mean(dim=0) for label in range(num_classes)]
    )
    support_norm = functional.normalize(prototypes, dim=-1)
    query_norm = functional.normalize(query, dim=-1)
    return query_norm @ support_norm.transpose(0, 1)


def run_smoke(config: SmokeConfig, output_root: Path, project_root: Path) -> Path:
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configuration requests CUDA, but torch.cuda.is_available() is false")
    seed_state = seed_everything(config.seed)
    run_id = make_run_id(config.study, config.experiment_name, config.config_hash)
    run_directory = output_root / config.study / run_id
    if run_directory.exists():
        raise FileExistsError(f"refusing to reuse existing run directory: {run_directory}")
    run_directory.mkdir(parents=True)
    logger = JsonlLogger(run_directory / "metrics.jsonl")
    logger.log("run_started", run_id=run_id, config_hash=config.config_hash)

    atomic_write_json(run_directory / "config_resolved.json", config.to_dict())
    atomic_write_json(run_directory / "git_state.json", _git_state(project_root))
    atomic_write_json(
        run_directory / "env.json",
        {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "seed_state": asdict(seed_state),
        },
    )

    start = time.perf_counter()
    episode = make_synthetic_episode(
        num_classes=config.num_classes,
        samples_per_class=config.samples_per_class,
        shots=config.shots,
        queries_per_class=config.queries_per_class,
        seed=config.seed,
    )
    device = torch.device(config.device)
    backbone = TinyBackbone(config.embedding_dim).to(device)
    backbone.freeze()
    experts = torch.nn.ModuleList(
        [ResidualExpert(config.embedding_dim).to(device) for _ in range(3)]
    )
    optimizer = torch.optim.Adam(experts[0].parameters(), lr=config.learning_rate)

    with torch.no_grad():
        support_base = backbone(episode.support_images.to(device))
        query_base = backbone(episode.query_images.to(device))

    metrics: list[float] = []
    losses: list[Tensor] = []
    for expert in experts:
        support_embeddings = expert(support_base)
        query_embeddings = expert(query_base)
        logits = _prototype_logits(
            support_embeddings,
            episode.support_labels.to(device),
            query_embeddings,
            config.num_classes,
        )
        losses.append(functional.cross_entropy(logits, episode.query_labels.to(device)))
        metrics.append(
            float((logits.argmax(dim=-1).cpu() == episode.query_labels).float().mean().item())
        )

    optimizer.zero_grad(set_to_none=True)
    losses[0].backward()
    optimizer.step()
    oracle = episode_oracle(torch.tensor(metrics))
    elapsed = time.perf_counter() - start
    summary = {
        "schema_version": 1,
        "status": "SUCCEEDED",
        "experiment_id": "M0_ENV",
        "run_id": run_id,
        "config_hash": config.config_hash,
        "device": str(device),
        "num_workers": config.num_workers,
        "support_indices": episode.support_indices.tolist(),
        "query_indices": episode.query_indices.tolist(),
        "support_query_disjoint": True,
        "expert_metrics": metrics,
        "analysis_only_episode_oracle": asdict(oracle),
        "optimizer_step_completed": True,
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(run_directory / "summary.json", summary)
    atomic_write_json(run_directory / "COMPLETED.json", {"status": "SUCCEEDED"})
    logger.log("run_completed", elapsed_seconds=elapsed)
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = SmokeConfig.from_json(arguments.config)
    run_directory = run_smoke(
        config,
        arguments.output_root.resolve(),
        arguments.project_root.resolve(),
    )
    print(json.dumps({"status": "SUCCEEDED", "run_directory": str(run_directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
