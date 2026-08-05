"""Train the fixed M2 banks and execute the preregistered M3 Gate 1 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from tamoe.analysis.gate1 import (
    Gate1Thresholds,
    add_global_interventions,
    decide_gate1,
    evaluate_episode,
)
from tamoe.data.feature_cache import FeatureSet, extract_or_load_features
from tamoe.data.medmnist_dataset import MedMNISTTensorDataset
from tamoe.data.medmnist_tasks import get_task
from tamoe.data.task_splits import TaskSplit
from tamoe.experts.bank import train_expert_bank
from tamoe.experts.training import ExpertTrainConfig
from tamoe.metrics.resources import count_resources
from tamoe.models.backbones import build_backbone
from tamoe.utils.atomic_io import atomic_write_json, atomic_write_text


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={project_root.as_posix()}", "-C", str(project_root),
         "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_split(path: Path) -> TaskSplit:
    split = TaskSplit(**json.loads(path.read_text(encoding="utf-8")))
    split.validate()
    return split


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".parquet")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _features(
    tasks: tuple[str, ...], split_name: str, *, data_root: Path, cache_root: Path,
    backbone: torch.nn.Module, device: torch.device, config: dict[str, Any], code_version: str,
    max_per_class: int,
) -> dict[str, FeatureSet]:
    result: dict[str, FeatureSet] = {}
    for task_name in tasks:
        dataset = MedMNISTTensorDataset(get_task(task_name), data_root, split_name)
        result[task_name] = extract_or_load_features(
            dataset, backbone, cache_root, device=device,
            batch_size=int(config["feature_batch_size"]), code_version=code_version,
            max_samples_per_class=max_per_class, seed=0, num_workers=0,
        )
    return result


def _report(decision: dict[str, Any], *, metadata: dict[str, Any], frame: pd.DataFrame) -> str:
    lines = [
        "# Gate 1 fixed-expert oracle pilot", "",
        f"- Decision: **{decision['status']}**",
        f"- Experiment ID: `{metadata['experiment_id']}`",
        f"- Commit: `{metadata['git_commit']}`",
        f"- Host / device: `{metadata['host']}` / `{metadata['device']}`",
        f"- Episodes: `{frame[frame['method'] == 'episode_oracle']['episode_hash'].nunique()}`",
        f"- Split seeds: `{metadata['task_split_seeds']}`",
        f"- Train seeds: `{metadata['train_seeds']}`",
        f"- Shots: `{metadata['evaluation_shots']}`",
        "- Query-label use is analysis-only for episode oracle, convex oracle mixture, "
        "sample oracle, and forced-worst controls.", "", "## Preregistered criteria", "",
    ]
    for name, passed in decision["criteria"].items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend([
        "", "## Global dominance", "",
        f"- Maximum episode-best frequency: `{decision['maximum_global_best_frequency']:.4f}`",
        f"- Frequencies: `{decision['best_expert_frequencies']}`", "",
        "## Split-level paired oracle gaps (accuracy)", "",
        "| Baseline | Split | Mean | 95% CI | N |", "|---|---:|---:|---:|---:|",
    ])
    for baseline, records in decision["split_comparisons"].items():
        for record in records:
            lines.append(
                f"| {baseline} | {record['split_seed']} | {record['mean']:.4f} | "
                f"[{record['ci_low']:.4f}, {record['ci_high']:.4f}] | {record['n']} |"
            )
    if decision["status"] == "FAIL":
        lines.extend([
            "", "## Stop decision", "",
            "Gate 1 did not satisfy every preregistered criterion. Learned routing development "
            "is stopped; this report preserves the negative pilot rather than adding router complexity.",
        ])
    else:
        lines.extend([
            "", "## Continue decision", "",
            "All Gate 1 criteria passed. M4 support-conditioned routing is authorized by the protocol.",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    config_hash = _config_hash(config)
    code_version = _git_commit(arguments.project_root)
    device = torch.device(arguments.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal Gate 1 pilot requires an available CUDA GPU")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    backbone = build_backbone(config["backbone"], pretrained=bool(config["pretrained"]))
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    run_directory = arguments.run_root / arguments.experiment_id
    for split_seed in config["task_split_seeds"]:
        task_split = _load_split(
            arguments.project_root / "configs" / "task_splits" / f"medmnist_seed{split_seed}.json"
        )
        train_features = _features(
            task_split.meta_train, "train", data_root=arguments.data_root,
            cache_root=arguments.cache_root, backbone=backbone, device=device, config=config,
            code_version=code_version, max_per_class=int(config["max_train_samples_per_class"]),
        )
        evaluation_features = _features(
            task_split.meta_test, "test", data_root=arguments.data_root,
            cache_root=arguments.cache_root, backbone=backbone, device=device, config=config,
            code_version=code_version, max_per_class=int(config["max_evaluation_samples_per_class"]),
        )
        for train_seed in config["train_seeds"]:
            bank_started = time.perf_counter()
            training_config = ExpertTrainConfig(
                embedding_dim=backbone.info.embedding_dim, rank=int(config["adapter_rank"]),
                steps=int(config["training_steps"]), shots=int(config["training_shots"]),
                queries_per_class=int(config["training_queries_per_class"]),
                n_way=int(config["training_n_way"]), learning_rate=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]), temperature=float(config["temperature"]),
                seed=int(train_seed), schedule="balanced_round_robin",
            )
            bank_directory = run_directory / f"split_{split_seed}" / f"train_seed_{train_seed}"
            experts, records = train_expert_bank(
                train_features, task_split, training_config, bank_directory, device=device
            )
            resource = count_resources(backbone, list(experts.values()))
            resource_rows.append({
                "experiment_id": arguments.experiment_id, "split_seed": split_seed,
                "train_seed": train_seed, **asdict(resource),
                "wall_seconds_train_and_load": time.perf_counter() - bank_started,
                "gpu_hours_train_and_load": (time.perf_counter() - bank_started) / 3600,
                "max_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "device_name": torch.cuda.get_device_name(device),
                "checkpoint_count": len(records),
            })
            expert_name_set = "|".join(sorted(experts))
            for task_name, feature_set in evaluation_features.items():
                for shots in config["evaluation_shots"]:
                    for support_resample in range(int(config["support_resamples"])):
                        rows, audit = evaluate_episode(
                            experts, feature_set.features, feature_set.labels,
                            shots=int(shots), queries_per_class=int(config["query_per_class"]),
                            seed=int(split_seed) * 100_000 + int(train_seed) * 1_000 + int(shots),
                            repetition=support_resample, temperature=float(config["temperature"]),
                            device=device, random_repeats=int(config["random_router_repeats"]),
                        )
                        common = {
                            "experiment_id": arguments.experiment_id, "config_hash": config_hash,
                            "git_commit": code_version, "split_seed": split_seed,
                            "split_hash": task_split.split_hash, "train_seed": train_seed,
                            "task": task_name, "shots": shots, "support_resample": support_resample,
                            "episode_hash": audit["episode_hash"], "expert_name_set": expert_name_set,
                            "query_labels_analysis_only": True,
                        }
                        all_rows.extend([{**common, **row} for row in rows])
    frame = add_global_interventions(pd.DataFrame(all_rows))
    decision = decide_gate1(
        frame, bootstrap_repeats=int(config["bootstrap_repeats"]),
        thresholds=Gate1Thresholds(),
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "experiment_id": arguments.experiment_id, "config_hash": config_hash,
        "git_commit": code_version, "host": platform.node(), "device": str(device),
        "device_name": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "task_split_seeds": config["task_split_seeds"],
        "train_seeds": config["train_seeds"], "evaluation_shots": config["evaluation_shots"],
        "support_resamples": config["support_resamples"], "elapsed_seconds": elapsed,
        "max_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    decision["metadata"] = metadata
    results_root = arguments.project_root / "results"
    reports_root = arguments.project_root / "reports"
    _atomic_parquet(frame, results_root / "gate1_episode_metrics.parquet")
    fixed_names = sorted({name for names in frame["expert_name_set"] for name in names.split("|")})
    matrix = (
        frame[frame["method"].isin(fixed_names)]
        .groupby(["split_seed", "train_seed", "task", "method"], as_index=False)
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
             macro_f1_mean=("macro_f1", "mean"), loss_mean=("loss", "mean"),
             episode_count=("episode_hash", "nunique"))
    )
    matrix.to_csv(results_root / "expert_task_matrix.csv", index=False)
    pd.DataFrame(resource_rows).to_csv(results_root / "resource_usage.csv", index=False)
    atomic_write_json(reports_root / "gate1_decision.json", decision)
    atomic_write_text(reports_root / "gate1_report.md", _report(decision, metadata=metadata, frame=frame))
    atomic_write_json(run_directory / "completion.json", {"status": "SUCCEEDED", **metadata,
                                                            "gate1_decision": decision["status"]})
    print(json.dumps({"status": "SUCCEEDED", "gate1": decision["status"], **metadata}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
