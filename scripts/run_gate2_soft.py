"""Run the PASS_SOFT-authorized Gate 2 support-routing experiment."""

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
from torch import Tensor
from torch.nn import functional

from tamoe.analysis.gate1 import feasible_query_count
from tamoe.analysis.gate1r import EPISODE_KEYS, hierarchical_ci
from tamoe.data.feature_cache import FeatureSet, extract_or_load_features
from tamoe.data.medmnist_dataset import MedMNISTTensorDataset, sha256_file
from tamoe.data.medmnist_tasks import get_task
from tamoe.data.task_splits import TaskSplit
from tamoe.experts.bank import TrainedExpert, train_expert_bank
from tamoe.experts.training import ExpertTrainConfig
from tamoe.metrics.resources import count_resources
from tamoe.models.backbones import build_backbone
from tamoe.routing.support import RoutedEpisode, evaluate_external_weights, evaluate_routing_episode
from tamoe.utils.atomic_io import atomic_write_json, atomic_write_text

GATE1R_DECISION_SHA256 = "779ae2fd3a45623370fc4c9fdbe23d02c49b90254ee29795271f840c1a7c59ad"
REQUIRED_OUTPUTS = (
    "reports/gate2_report.md",
    "reports/gate2_decision.json",
    "results/gate2_episode_metrics.parquet",
    "results/gate2_router_comparison.csv",
    "results/gate2_support_ablation.csv",
    "results/gate2_resource_usage.csv",
)


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={project_root.as_posix()}", "-C", str(project_root),
         *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def _atomic_frame(frame: pd.DataFrame, path: Path, *, parquet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if parquet:
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_split(path: Path) -> TaskSplit:
    split = TaskSplit(**json.loads(path.read_text(encoding="utf-8")))
    split.validate()
    return split


def _features(
    tasks: tuple[str, ...] | list[str], split_name: str, *, data_root: Path,
    cache_root: Path, backbone: torch.nn.Module, device: torch.device,
    model_config: dict[str, Any], code_version: str, max_per_class: int,
) -> dict[str, FeatureSet]:
    return {
        task: extract_or_load_features(
            MedMNISTTensorDataset(get_task(task), data_root, split_name),
            backbone,
            cache_root,
            device=device,
            batch_size=int(model_config["feature_batch_size"]),
            code_version=code_version,
            max_samples_per_class=max_per_class,
            seed=0,
            num_workers=0,
        )
        for task in tasks
    }


def _references(
    experts: dict[str, torch.nn.Module], records: tuple[TrainedExpert, ...],
    features: dict[str, FeatureSet], device: torch.device,
) -> dict[str, Tensor]:
    definitions = {record.definition.name: record.definition for record in records}
    references: dict[str, Tensor] = {}
    with torch.inference_mode():
        for name in sorted(experts):
            if name == "single":
                continue
            chunks = []
            for task in definitions[name].source_tasks:
                task_features = features[task].features
                for start in range(0, len(task_features), 2048):
                    adapted = experts[name](task_features[start : start + 2048].to(device))
                    chunks.append(functional.normalize(adapted, dim=-1).sum(dim=0).cpu())
            count = sum(len(features[task].features) for task in definitions[name].source_tasks)
            references[name] = torch.stack(chunks).sum(dim=0) / count
    return references


def _route_entropy(weights: Tensor) -> float:
    values = weights.clamp_min(1e-12)
    return float(-(values * values.log()).sum().item())


def _paired(
    frame: pd.DataFrame, baseline: str, *, repeats: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary = frame[frame["method"] == "support_conditioned_soft_mixture"]
    selected = frame[frame["method"] == baseline]
    joined = primary[EPISODE_KEYS + ["accuracy", "loss"]].merge(
        selected[EPISODE_KEYS + ["accuracy", "loss"]],
        on=EPISODE_KEYS,
        suffixes=("_primary", "_baseline"),
        validate="one_to_one",
    )
    joined["accuracy_delta"] = joined["accuracy_primary"] - joined["accuracy_baseline"]
    joined["nll_delta"] = joined["loss_baseline"] - joined["loss_primary"]
    joined["_support_episode"] = (
        joined["shots"].astype(str) + ":" + joined["support_resample"].astype(str)
    )
    rows: list[dict[str, Any]] = []
    split_stats: dict[str, Any] = {}
    for split_seed, group in joined.groupby("split_seed", sort=True):
        stats = hierarchical_ci(
            group,
            "accuracy_delta",
            ["task", "train_seed", "_support_episode"],
            repeats=repeats,
            seed=int.from_bytes(
                hashlib.sha256(f"gate2:{baseline}:{split_seed}".encode()).digest()[:4], "big"
            ),
        )
        split_stats[str(int(split_seed))] = stats
        rows.append(
            {
                "comparison": baseline,
                "scope": "split",
                "split_seed": int(split_seed),
                "task": "ALL",
                "mean_accuracy_delta": stats["mean"],
                "accuracy_delta_ci_low": stats["ci_low"],
                "accuracy_delta_ci_high": stats["ci_high"],
                "mean_nll_improvement": float(group["nll_delta"].mean()),
                "episode_count": stats["n"],
            }
        )
    task_means: dict[str, float] = {}
    for (split_seed, task), group in joined.groupby(["split_seed", "task"], sort=True):
        stats = hierarchical_ci(
            group,
            "accuracy_delta",
            ["train_seed", "_support_episode"],
            repeats=repeats,
            seed=int.from_bytes(
                hashlib.sha256(f"gate2:{baseline}:{split_seed}:{task}".encode()).digest()[:4],
                "big",
            ),
        )
        task_means[f"{split_seed}:{task}"] = float(stats["mean"])
        rows.append(
            {
                "comparison": baseline,
                "scope": "task",
                "split_seed": int(split_seed),
                "task": str(task),
                "mean_accuracy_delta": stats["mean"],
                "accuracy_delta_ci_low": stats["ci_low"],
                "accuracy_delta_ci_high": stats["ci_high"],
                "mean_nll_improvement": float(group["nll_delta"].mean()),
                "episode_count": stats["n"],
            }
        )
    return rows, {"split": split_stats, "task_means": task_means}


def _decision(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    repeats = int(config["bootstrap_repeats"])
    required = config["gate2_requirements"]["required_primary_comparisons"]
    comparison_rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    criteria: dict[str, bool] = {}
    for baseline in required:
        rows, values = _paired(frame, baseline, repeats=repeats)
        comparison_rows.extend(rows)
        raw[baseline] = values
        split_values = values["split"].values()
        task_values = values["task_means"].values()
        criteria[f"{baseline}_each_split_mean"] = all(
            value["mean"] > config["gate2_requirements"]["each_split_mean_delta_gt"]
            for value in split_values
        )
        criteria[f"{baseline}_each_split_ci"] = all(
            value["ci_low"]
            > config["gate2_requirements"]["each_split_hierarchical_ci_low_gt"]
            for value in split_values
        )
        criteria[f"{baseline}_positive_tasks"] = sum(value > 0 for value in task_values) >= int(
            config["gate2_requirements"]["minimum_positive_task_count"]
        )
    diagnostic = [
        "wrong_support_task",
        "support_prototype_weighting",
        "shared",
        "shared_fallback",
        "capacity_matched_single",
        "compute_matched_support_top1_diagnostic",
    ]
    for baseline in diagnostic:
        rows, values = _paired(frame, baseline, repeats=repeats)
        comparison_rows.extend(rows)
        raw[baseline] = values
    comparison = pd.DataFrame(comparison_rows)
    ablation = comparison[
        comparison["comparison"].isin(
            ["support_query_shuffle", "support_label_removal", "wrong_support_task"]
        )
    ].copy()
    primary = frame[frame["method"] == "support_conditioned_soft_mixture"]
    oracle = frame[frame["method"] == "router_candidate_oracle"]
    shared = frame[frame["method"] == "shared"]
    joined = primary[EPISODE_KEYS + ["accuracy"]].merge(
        oracle[EPISODE_KEYS + ["accuracy"]], on=EPISODE_KEYS,
        suffixes=("_primary", "_oracle"), validate="one_to_one",
    ).merge(
        shared[EPISODE_KEYS + ["accuracy"]], on=EPISODE_KEYS, validate="one_to_one"
    )
    numerator = float((joined["accuracy_primary"] - joined["accuracy"]).mean())
    denominator = float((joined["accuracy_oracle"] - joined["accuracy"]).mean())
    raw["oracle_gap_recovery"] = numerator / denominator if denominator > 1e-12 else None
    raw["primary_mean_accuracy"] = float(primary["accuracy"].mean())
    raw["primary_mean_nll"] = float(primary["loss"].mean())
    raw["primary_route_entropy_mean"] = float(primary["route_entropy"].mean())
    return {
        "outcome": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
        "raw_values": raw,
    }, comparison, ablation


def _report(decision: dict[str, Any]) -> str:
    lines = [
        "# Gate 2 support-routing report",
        "",
        f"- Outcome: **{decision['outcome']}**",
        f"- Experiment: `{decision['experiment_id']}`",
        "- Gate 1R authorization: `PASS_SOFT`; hard top-1 is diagnostic only.",
        "- Primary method: `support_conditioned_soft_mixture`.",
        "- Router inputs contain tensors only and receive no task/dataset identity.",
        "",
        "## Frozen Gate 2 criteria",
        "",
    ]
    lines.extend(
        f"- `{name}`: **{'PASS' if passed else 'FAIL'}**"
        for name, passed in decision["criteria"].items()
    )
    lines.extend(["", "## Core results", ""])
    raw = decision["raw_values"]
    lines.append(f"- Primary mean accuracy: `{raw['primary_mean_accuracy']:.4f}`")
    lines.append(f"- Primary mean NLL: `{raw['primary_mean_nll']:.4f}`")
    recovery = raw["oracle_gap_recovery"]
    lines.append(
        f"- Oracle-gap recovery over shared: `{recovery:.4f}`"
        if recovery is not None
        else "- Oracle-gap recovery: not interpretable (near-zero denominator)."
    )
    lines.extend(["", "## Decision consequence", ""])
    if decision["outcome"] == "PASS":
        lines.append("Support value passed every frozen comparison; M5 bootstrap uncertainty is authorized.")
    else:
        lines.append(
            "At least one support-value requirement failed. Learned routing terminates and M5 is not run."
        )
    return "\n".join(lines) + "\n"


def _termination(decision: dict[str, Any]) -> str:
    failed = [name for name, value in decision["criteria"].items() if not value]
    return "\n".join(
        [
            "# H1 learned-routing termination report",
            "",
            "## Evidence sequence",
            "",
            "- Canonical Gate 1 remains FAIL and is unchanged.",
            "- M3A was exploratory and motivated a fresh confirmatory test only.",
            "- Gate 1R was PASS_SOFT: expert diversity/headroom passed, but hard identity stability failed.",
            "- Gate 2 is FAIL: support-conditioned soft routing did not satisfy every support-value control.",
            "",
            "## Failed Gate 2 criteria",
            "",
            *[f"- `{name}`" for name in failed],
            "",
            "## Research conclusion",
            "",
            "The fixed bank retains oracle, ensemble, and possible calibration value, but the available "
            "few-shot support statistic does not justify a learned-routing claim. H1 router and uncertainty "
            "development stop here; no uncertainty head is added to mask the negative result.",
            "",
            "A later project may study ensemble/calibration value, alternative medical tasks, or H2 only "
            "as a separately preregistered direction; it is not authorized by this H1 result.",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-count", type=int, required=True)
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    config_path = arguments.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    gate1r_path = project_root / "reports" / "gate1r_decision.json"
    if sha256_file(gate1r_path) != GATE1R_DECISION_SHA256:
        raise RuntimeError("Gate 1R decision hash differs from Gate 2 authorization")
    gate1r = json.loads(gate1r_path.read_text(encoding="utf-8"))
    if gate1r["outcome"] != "PASS_SOFT":
        raise RuntimeError("Gate 2 soft branch requires PASS_SOFT")
    runtime_sha = _git(project_root, "rev-parse", "HEAD")
    if _git(project_root, "status", "--porcelain"):
        raise RuntimeError("formal Gate 2 requires a clean checkout")
    existing = [relative for relative in REQUIRED_OUTPUTS if (project_root / relative).exists()]
    if existing:
        raise RuntimeError(f"Gate 2 outputs already exist: {existing}")
    gate1_config = json.loads(
        (project_root / "configs" / "m3_gate1r_confirmatory.json").read_text(encoding="utf-8")
    )
    model_config = gate1_config["model_and_training"]
    device = torch.device(arguments.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal Gate 2 requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    backbone = build_backbone(model_config["backbone"], pretrained=model_config["pretrained"])
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    routed_records: list[tuple[dict[str, Any], RoutedEpisode]] = []
    resources: list[dict[str, Any]] = []
    run_directory = arguments.run_root / arguments.experiment_id
    for split_seed in config["fresh_task_split_seeds"]:
        split = _load_split(
            project_root / "configs" / "task_splits" / f"medmnist_seed{split_seed}.json"
        )
        train_features = _features(
            split.meta_train, "train", data_root=arguments.data_root,
            cache_root=arguments.cache_root, backbone=backbone, device=device,
            model_config=model_config, code_version=runtime_sha,
            max_per_class=int(model_config["max_train_samples_per_class"]),
        )
        evaluation_features = _features(
            split.meta_test, "test", data_root=arguments.data_root,
            cache_root=arguments.cache_root, backbone=backbone, device=device,
            model_config=model_config, code_version=runtime_sha,
            max_per_class=int(model_config["max_evaluation_samples_per_class"]),
        )
        for train_seed in config["train_seeds"]:
            training_config = ExpertTrainConfig(
                embedding_dim=backbone.info.embedding_dim,
                rank=int(model_config["adapter_rank"]),
                steps=int(model_config["training_steps"]),
                shots=int(model_config["training_shots"]),
                queries_per_class=int(model_config["training_queries_per_class"]),
                n_way=int(model_config["training_n_way"]),
                learning_rate=float(model_config["learning_rate"]),
                weight_decay=float(model_config["weight_decay"]),
                temperature=float(model_config["temperature"]),
                seed=int(train_seed),
                schedule="balanced_round_robin",
            )
            bank_started = time.perf_counter()
            experts, records = train_expert_bank(
                train_features, split, training_config,
                run_directory / f"split_{split_seed}" / f"train_seed_{train_seed}",
                device=device,
            )
            references = _references(experts, records, train_features, device)
            bank_elapsed = time.perf_counter() - bank_started
            counts = count_resources(backbone, list(experts.values()))
            resources.append(
                {
                    "record_type": "expert_bank",
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    **asdict(counts),
                    "router_parameters": 0,
                    "wall_seconds_train_and_reference": bank_elapsed,
                    "gpu_hours_train_and_reference": bank_elapsed / 3600,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "gpu_name": torch.cuda.get_device_name(device),
                }
            )
            for task, feature_set in evaluation_features.items():
                for shots in config["shots"]:
                    queries = feasible_query_count(
                        feature_set.labels, shots=int(shots),
                        desired_queries=int(config["query_per_class"]),
                    )
                    for resample in range(int(config["support_resamples"])):
                        episode_started = time.perf_counter()
                        rows, routed = evaluate_routing_episode(
                            experts, references, feature_set.features, feature_set.labels,
                            shots=int(shots), queries_per_class=queries,
                            seed=int(split_seed) * 100_000 + int(train_seed) * 1_000 + int(shots),
                            repetition=resample,
                            prototype_temperature=float(model_config["temperature"]),
                            route_temperature=float(config["routing"]["route_temperature"]),
                            device=device,
                            shared_fallback_weight=float(
                                config["routing"]["shared_fallback_weight"]
                            ),
                        )
                        elapsed_episode = time.perf_counter() - episode_started
                        candidates = routed.candidate_names
                        fixed = {row["method"]: row for row in rows if row["method"] in candidates}
                        oracle_name = min(
                            candidates,
                            key=lambda name: (-fixed[name]["accuracy"], fixed[name]["loss"], name),
                        )
                        rows.append(
                            {
                                "method": "router_candidate_oracle",
                                "selected_expert": oracle_name,
                                "router_input": "analysis_only_query_labels",
                                "route_entropy": float("nan"),
                                "top1_route_weight": float("nan"),
                                "route_weights": "",
                                **{key: fixed[oracle_name][key] for key in ("accuracy", "macro_f1", "loss")},
                            }
                        )
                        common = {
                            "experiment_id": arguments.experiment_id,
                            "config_hash": config_hash,
                            "runtime_git_sha": runtime_sha,
                            "split_seed": split_seed,
                            "split_hash": split.split_hash,
                            "train_seed": train_seed,
                            "task": task,
                            "shots": shots,
                            "support_resample": resample,
                            "episode_hash": routed.episode_hash,
                            "queries_per_class": queries,
                            "candidate_name_set": "|".join(candidates),
                            "route_entropy": _route_entropy(routed.support_weights),
                            "top1_route_weight": float(routed.support_weights.max().item()),
                            "episode_wall_seconds": elapsed_episode,
                            "task_id_visible_to_router": False,
                        }
                        all_rows.extend([{**common, **row} for row in rows])
                        routed_records.append((common, routed))
    bundles = {
        (
            common["split_seed"], common["train_seed"], common["task"],
            common["shots"], common["support_resample"],
        ): routed
        for common, routed in routed_records
    }
    tasks_by_split = {
        split: sorted({key[2] for key in bundles if key[0] == split})
        for split in config["fresh_task_split_seeds"]
    }
    for common, routed in routed_records:
        split_seed = int(common["split_seed"])
        task = str(common["task"])
        tasks = tasks_by_split[split_seed]
        other_task = tasks[(tasks.index(task) + 1) % len(tasks)]
        wrong_key = (
            split_seed, common["train_seed"], other_task, common["shots"],
            common["support_resample"],
        )
        shuffle_key = (
            split_seed, common["train_seed"], other_task, common["shots"],
            (int(common["support_resample"]) + 1) % int(config["support_resamples"]),
        )
        for method, donor in (
            ("wrong_support_task", bundles[wrong_key]),
            ("support_query_shuffle", bundles[shuffle_key]),
        ):
            metrics = evaluate_external_weights(routed, donor.support_weights)
            all_rows.append(
                {
                    **common,
                    "method": method,
                    "selected_expert": routed.candidate_names[
                        int(donor.support_weights.argmax().item())
                    ],
                    "router_input": "mismatched_support_embeddings_and_labels",
                    "route_entropy": _route_entropy(donor.support_weights),
                    "top1_route_weight": float(donor.support_weights.max().item()),
                    "route_weights": "|".join(
                        f"{float(value):.12g}" for value in donor.support_weights
                    ),
                    **metrics,
                }
            )
    frame = pd.DataFrame(all_rows)
    decision_core, comparison, ablation = _decision(frame, config)
    elapsed = time.perf_counter() - started
    decision = {
        "schema_version": 1,
        **decision_core,
        "thresholds": config["gate2_requirements"],
        "experiment_id": arguments.experiment_id,
        "config_hash": config_hash,
        "runtime_git_sha": runtime_sha,
        "gate1r_outcome": "PASS_SOFT",
        "gate1r_decision_sha256": GATE1R_DECISION_SHA256,
        "host_gpu": {
            "host": platform.node(), "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "elapsed_time": elapsed,
        "gpu_hours": elapsed / 3600,
        "test_count": arguments.test_count,
        "schema_validation": "VALID",
    }
    schema = json.loads(
        (project_root / "schemas" / "gate2_decision.schema.json").read_text(encoding="utf-8")
    )
    required = set(schema["required"])
    if required - set(decision) or decision["outcome"] not in {"PASS", "FAIL"}:
        raise RuntimeError("Gate 2 decision schema validation failed")
    candidate_count = int(frame["candidate_name_set"].iloc[0].count("|") + 1)
    adapter_flops = resources[0]["approximate_adapter_flops_per_query"]
    for method, active_paths in {
        "shared": 1,
        "capacity_matched_single": 1,
        "compute_matched_support_top1_diagnostic": 1,
        "random_expected": candidate_count,
        "query_only_weighting": candidate_count,
        "support_prototype_weighting": candidate_count,
        "support_conditioned_soft_mixture": candidate_count,
        "shared_fallback": candidate_count,
    }.items():
        resources.append(
            {
                "record_type": "method",
                "method": method,
                "router_parameters": 0,
                "activated_expert_paths_per_query": active_paths,
                "approximate_adapter_flops_per_query": adapter_flops * active_paths,
                "mean_episode_wall_seconds_shared_evaluator": float(
                    frame[frame["method"] == method]["episode_wall_seconds"].mean()
                ),
                "hyperparameter_trials": int(config["routing"]["hyperparameter_trials"]),
            }
        )
    results_root = project_root / "results"
    reports_root = project_root / "reports"
    _atomic_frame(frame, results_root / "gate2_episode_metrics.parquet", parquet=True)
    _atomic_frame(comparison, results_root / "gate2_router_comparison.csv")
    _atomic_frame(ablation, results_root / "gate2_support_ablation.csv")
    _atomic_frame(pd.DataFrame(resources), results_root / "gate2_resource_usage.csv")
    atomic_write_json(reports_root / "gate2_decision.json", decision)
    atomic_write_text(reports_root / "gate2_report.md", _report(decision))
    if decision["outcome"] == "FAIL":
        atomic_write_text(reports_root / "h1_termination_report.md", _termination(decision))
    missing = [relative for relative in REQUIRED_OUTPUTS if not (project_root / relative).exists()]
    if missing:
        raise RuntimeError(f"Gate 2 outputs missing: {missing}")
    atomic_write_json(
        run_directory / "completion.json",
        {
            "status": "SUCCEEDED", "outcome": decision["outcome"],
            "runtime_git_sha": runtime_sha, "elapsed_time": elapsed,
        },
    )
    print(json.dumps({"status": "SUCCEEDED", "outcome": decision["outcome"],
                      "experiment_id": arguments.experiment_id, "elapsed_time": elapsed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
