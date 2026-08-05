"""Execute the frozen fresh-task Gate 1R confirmatory experiment."""

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

import numpy as np
import pandas as pd
import torch

from tamoe.analysis.gate1 import feasible_query_count
from tamoe.analysis.gate1r import (
    EPISODE_KEYS,
    add_confirmatory_interventions,
    decide_gate1r,
    evaluate_confirmatory_episode,
    hierarchical_ci,
    intervention_effects,
    leave_one_out,
    ranking_stability,
)
from tamoe.data.feature_cache import FeatureSet, extract_or_load_features
from tamoe.data.medmnist_dataset import MedMNISTTensorDataset, sha256_file
from tamoe.data.medmnist_tasks import get_task
from tamoe.data.task_splits import TaskSplit
from tamoe.experts.bank import train_expert_bank
from tamoe.experts.training import ExpertTrainConfig
from tamoe.metrics.resources import count_resources
from tamoe.models.backbones import build_backbone
from tamoe.utils.atomic_io import atomic_write_json, atomic_write_text

PREREGISTRATION_COMMIT = "761d1b4e0004e2636e3d4e21cc6cd95e07196774"
PREREGISTRATION_TAG = "gate1r-preregistered-v1"
FROZEN_CONFIG_SHA256 = "de435ad914d6af604185bdab2109f8517f97c3c59d77ede4770201c151e7f1f2"
REQUIRED_OUTPUTS = (
    "reports/gate1r_report.md",
    "reports/gate1r_decision.json",
    "results/gate1r_episode_metrics.parquet",
    "results/gate1r_expert_task_matrix.csv",
    "results/gate1r_intervention_effects.csv",
    "results/gate1r_ranking_stability.csv",
    "results/gate1r_epsilon_optimal_sets.csv",
    "results/gate1r_leave_one_expert_out.csv",
    "results/gate1r_resource_usage.csv",
)


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.as_posix()}",
            "-C",
            str(project_root),
            *arguments,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def _load_split(path: Path) -> TaskSplit:
    split = TaskSplit(**json.loads(path.read_text(encoding="utf-8")))
    split.validate()
    return split


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


def _features(
    tasks: tuple[str, ...],
    split_name: str,
    *,
    data_root: Path,
    cache_root: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    model_config: dict[str, Any],
    code_version: str,
    max_per_class: int,
) -> dict[str, FeatureSet]:
    result: dict[str, FeatureSet] = {}
    for task_name in tasks:
        dataset = MedMNISTTensorDataset(get_task(task_name), data_root, split_name)
        result[task_name] = extract_or_load_features(
            dataset,
            backbone,
            cache_root,
            device=device,
            batch_size=int(model_config["feature_batch_size"]),
            code_version=code_version,
            max_samples_per_class=max_per_class,
            seed=0,
            num_workers=0,
        )
    return result


def _ordinary_ci(values: pd.Series, *, repeats: int, seed: int) -> dict[str, Any]:
    array = values.to_numpy(dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = generator.choice(array, size=(repeats, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "n": int(len(array)),
    }


def _paired_reporting(frame: pd.DataFrame, *, repeats: int) -> dict[str, Any]:
    primary = frame[frame["method"] == "router_candidate_oracle"]
    reporting: dict[str, Any] = {}
    for baseline in ("shared", "single", "random_expected"):
        selected = frame[frame["method"] == baseline]
        joined = primary[EPISODE_KEYS + ["accuracy"]].merge(
            selected[EPISODE_KEYS + ["accuracy"]],
            on=EPISODE_KEYS,
            suffixes=("_oracle", "_baseline"),
            validate="one_to_one",
        )
        joined["delta"] = joined["accuracy_oracle"] - joined["accuracy_baseline"]
        joined["_support_episode"] = (
            joined["shots"].astype(str) + ":" + joined["support_resample"].astype(str)
        )
        levels: dict[str, Any] = {
            "episode": joined[EPISODE_KEYS + ["delta"]].to_dict(orient="records"),
            "task": [
                {"split_seed": int(split), "task": str(task), "mean": float(group["delta"].mean())}
                for (split, task), group in joined.groupby(["split_seed", "task"], sort=True)
            ],
            "split": [
                {"split_seed": int(split), "mean": float(group["delta"].mean())}
                for split, group in joined.groupby("split_seed", sort=True)
            ],
            "train_seed": [
                {
                    "split_seed": int(split),
                    "task": str(task),
                    "train_seed": int(train_seed),
                    "mean": float(group["delta"].mean()),
                }
                for (split, task, train_seed), group in joined.groupby(
                    ["split_seed", "task", "train_seed"], sort=True
                )
            ],
            "support_resample": [
                {
                    "split_seed": int(split),
                    "task": str(task),
                    "train_seed": int(train_seed),
                    "shots": int(shots),
                    "support_resample": int(resample),
                    "delta": float(group["delta"].mean()),
                }
                for (split, task, train_seed, shots, resample), group in joined.groupby(
                    ["split_seed", "task", "train_seed", "shots", "support_resample"],
                    sort=True,
                )
            ],
        }
        levels["ordinary_episode_bootstrap"] = _ordinary_ci(
            joined["delta"], repeats=repeats, seed=_stable_seed(f"ordinary:{baseline}")
        )
        levels["global_hierarchical_bootstrap"] = hierarchical_ci(
            joined,
            "delta",
            ["split_seed", "task", "train_seed", "_support_episode"],
            repeats=repeats,
            seed=_stable_seed(f"global:{baseline}"),
        )
        reporting[baseline] = levels
    return reporting


def _entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy(dtype=np.float64)
    return float(-(probabilities * np.log(probabilities)).sum())


def _report(decision: dict[str, Any], *, episode_count: int) -> str:
    lines = [
        "# Gate 1R confirmatory report",
        "",
        f"- Frozen outcome: **{decision['outcome']}**",
        f"- Experiment ID: `{decision['experiment_id']}`",
        f"- Fresh episodes: `{episode_count}`",
        "- Confirmatory split sample size: `2` (seeds 6 and 36).",
        f"- Runtime commit: `{decision['runtime_git_sha']}`",
        f"- Preregistration: `{decision['preregistration_tag']}` at "
        f"`{decision['preregistration_commit_sha']}`",
        "- Query labels are used only by the explicitly analysis-only oracle and intervention "
        "diagnostics; no reported oracle is deployable routing.",
        "",
        "## Frozen criteria",
        "",
    ]
    for family, criteria in decision["criteria"].items():
        lines.append(f"### {family}")
        lines.append("")
        for name, passed in criteria.items():
            lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
        lines.append("")
    raw = decision["raw_criterion_values"]
    lines.extend(
        [
            "## Identity diagnostics",
            "",
            f"- Maximum primary path frequency: `{raw['maximum_primary_oracle_frequency']:.4f}`",
            f"- Exact accuracy tie rate: `{raw['exact_accuracy_tie_rate']:.4f}`",
            "- Assignment permutation and wrong-task assignment are mathematically redundant "
            "here because every split has exactly two held-out tasks; both are reported.",
            "",
            "## Protocol consequence",
            "",
        ]
    )
    if decision["outcome"] == "FAIL":
        lines.append(
            "At least one shared requirement failed. H1 learned routing terminates and M4/M5 "
            "are not authorized. The historical canonical Gate 1 decision remains FAIL."
        )
    elif decision["outcome"] == "PASS_SOFT":
        lines.append(
            "All shared requirements passed but at least one hard-identity requirement failed. "
            "Only the preregistered soft/selective M4 branch is authorized."
        )
    else:
        lines.append(
            "All shared and hard-identity requirements passed. The preregistered hard-routing "
            "M4 branch is authorized."
        )
    return "\n".join(lines) + "\n"


def _termination_report(decision: dict[str, Any]) -> str:
    failed = [
        name
        for family in decision["criteria"].values()
        for name, passed in family.items()
        if not passed
    ]
    return "\n".join(
        [
            "# H1 termination report",
            "",
            "## Decision",
            "",
            "Gate 1R produced **FAIL** under the frozen confirmatory rule. H1 learned routing "
            "is terminated; M4 and M5 are not run.",
            "",
            "## Failed frozen criteria",
            "",
            *[f"- `{name}`" for name in failed],
            "",
            "## Evidence boundary",
            "",
            "This fresh-task result does not alter, overwrite, or reinterpret the protected "
            "canonical Gate 1 FAIL. M3A remains exploratory only. All Gate 1R oracle and "
            "intervention identities that consume query labels are analysis-only.",
            "",
            "## Final scope",
            "",
            "The repository preserves the negative H1 result and stops before learned-router "
            "development, as preregistered.",
        ]
    ) + "\n"


def _preflight(project_root: Path, config_path: Path) -> tuple[str, bool]:
    if sha256_file(config_path) != FROZEN_CONFIG_SHA256:
        raise RuntimeError("frozen Gate 1R configuration hash mismatch")
    runtime_sha = _git(project_root, "rev-parse", "HEAD")
    tag_commit = _git(project_root, "rev-list", "-n", "1", PREREGISTRATION_TAG)
    if tag_commit != PREREGISTRATION_COMMIT:
        raise RuntimeError("preregistration tag does not resolve to the frozen commit")
    clean = not _git(project_root, "status", "--porcelain")
    if not clean:
        raise RuntimeError("formal Gate 1R requires a clean runtime checkout")
    existing = [relative for relative in REQUIRED_OUTPUTS if (project_root / relative).exists()]
    if existing:
        raise RuntimeError(f"Gate 1R outputs already exist: {existing}")
    return runtime_sha, clean


def _validate_decision_schema(decision: dict[str, Any], schema: dict[str, Any]) -> None:
    missing = set(schema["required"]) - set(decision)
    if missing:
        raise ValueError(f"decision lacks schema-required fields: {sorted(missing)}")
    if decision["schema_version"] != 1:
        raise ValueError("decision schema_version must be 1")
    if decision["outcome"] not in {"PASS_HARD", "PASS_SOFT", "FAIL"}:
        raise ValueError("decision outcome is invalid")
    if decision["preregistration_tag"] != PREREGISTRATION_TAG:
        raise ValueError("decision preregistration tag is invalid")
    if decision["working_tree_clean"] is not True:
        raise ValueError("decision does not record a clean initial tree")
    if decision["task_split_hashes"] != {
        "6": "e2aca00b68c72b67eb6f9e9b0b0c53210908312204c357e505919f129c80031b",
        "36": "9a8fb6bb52f30fdd02e16519d3013b250b7e1dcca1093a346bb8407b40125051",
    }:
        raise ValueError("decision split hashes are invalid")
    for field in ("preregistration_commit_sha", "runtime_git_sha"):
        value = decision[field]
        if not isinstance(value, str) or len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"decision {field} is not a lowercase Git SHA")
    config_hash = decision["config_hash"]
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError("decision config hash is invalid")
    if not isinstance(decision["test_count"], int) or decision["test_count"] < 1:
        raise ValueError("decision test count is invalid")
    if decision["elapsed_time"] < 0 or decision["gpu_hours"] < 0:
        raise ValueError("decision timing is invalid")
    for family in ("shared_requirements", "pass_hard_additional_requirements"):
        values = decision["criteria"].get(family)
        if not isinstance(values, dict) or not all(
            isinstance(value, bool) for value in values.values()
        ):
            raise ValueError(f"decision criterion family {family} is invalid")
    host_gpu = decision["host_gpu"]
    if not all(host_gpu.get(field) for field in ("host", "device", "gpu_name")):
        raise ValueError("decision host/GPU provenance is incomplete")


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
    runtime_sha, clean = _preflight(project_root, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["model_and_training"]
    episode_config = config["episode_protocol"]
    device = torch.device(arguments.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal Gate 1R requires an available CUDA GPU")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    backbone = build_backbone(
        str(model_config["backbone"]), pretrained=bool(model_config["pretrained"])
    )
    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    epsilon_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    split_hashes: dict[str, str] = {}
    run_directory = arguments.run_root / arguments.experiment_id
    for frozen_split in config["fresh_task_splits"]:
        split_seed = int(frozen_split["seed"])
        split = _load_split(
            project_root / "configs" / "task_splits" / f"medmnist_seed{split_seed}.json"
        )
        if split.split_hash != frozen_split["split_hash"]:
            raise RuntimeError(f"split {split_seed} hash mismatch")
        if list(split.meta_test) != list(frozen_split["meta_test"]):
            raise RuntimeError(f"split {split_seed} meta-test tasks mismatch")
        split_hashes[str(split_seed)] = split.split_hash
        train_features = _features(
            split.meta_train,
            "train",
            data_root=arguments.data_root,
            cache_root=arguments.cache_root,
            backbone=backbone,
            device=device,
            model_config=model_config,
            code_version=runtime_sha,
            max_per_class=int(model_config["max_train_samples_per_class"]),
        )
        evaluation_features = _features(
            split.meta_test,
            "test",
            data_root=arguments.data_root,
            cache_root=arguments.cache_root,
            backbone=backbone,
            device=device,
            model_config=model_config,
            code_version=runtime_sha,
            max_per_class=int(model_config["max_evaluation_samples_per_class"]),
        )
        for train_seed in episode_config["train_seeds"]:
            bank_started = time.perf_counter()
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
            experts, records = train_expert_bank(
                train_features,
                split,
                training_config,
                run_directory / f"split_{split_seed}" / f"train_seed_{train_seed}",
                device=device,
            )
            elapsed_bank = time.perf_counter() - bank_started
            resources = count_resources(backbone, list(experts.values()))
            resource_rows.append(
                {
                    "experiment_id": arguments.experiment_id,
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    **asdict(resources),
                    "wall_seconds_train_and_load": elapsed_bank,
                    "gpu_hours_train_and_load": elapsed_bank / 3600,
                    "max_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "device_name": torch.cuda.get_device_name(device),
                    "checkpoint_count": len(records),
                }
            )
            expert_name_set = "|".join(sorted(experts))
            router_names = tuple(
                name
                for name in sorted(experts)
                if name == "shared" or name.startswith("source_")
            )
            router_name_set = "|".join(router_names)
            for task_name, feature_set in evaluation_features.items():
                for shots in episode_config["shots"]:
                    queries = feasible_query_count(
                        feature_set.labels,
                        shots=int(shots),
                        desired_queries=int(episode_config["query_per_class"]),
                    )
                    for resample in range(int(episode_config["support_resamples"])):
                        rows, audit = evaluate_confirmatory_episode(
                            experts,
                            feature_set.features,
                            feature_set.labels,
                            shots=int(shots),
                            queries_per_class=queries,
                            seed=split_seed * 100_000 + int(train_seed) * 1_000 + int(shots),
                            repetition=resample,
                            temperature=float(model_config["temperature"]),
                            device=device,
                            accuracy_tie_tolerance=float(
                                config["oracle_identity"]["accuracy_tie_tolerance"]
                            ),
                            epsilon_accuracy=float(config["oracle_identity"]["epsilon_accuracy"]),
                            random_repeats=int(episode_config["random_repeats"]),
                        )
                        common = {
                            "experiment_id": arguments.experiment_id,
                            "config_hash": FROZEN_CONFIG_SHA256,
                            "runtime_git_sha": runtime_sha,
                            "split_seed": split_seed,
                            "split_hash": split.split_hash,
                            "train_seed": int(train_seed),
                            "task": task_name,
                            "shots": int(shots),
                            "support_resample": resample,
                            "queries_per_class": queries,
                            "episode_hash": audit["episode_hash"],
                            "expert_name_set": expert_name_set,
                            "router_candidate_name_set": router_name_set,
                            "query_labels_analysis_only": True,
                            "top1_top2_accuracy_margin": audit[
                                "top1_top2_accuracy_margin"
                            ],
                            "exact_accuracy_tie": audit["exact_accuracy_tie"],
                            "epsilon_optimal_size": audit["epsilon_optimal_size"],
                        }
                        all_rows.extend([{**common, **row} for row in rows])
                        epsilon_rows.append(
                            {
                                **{key: common[key] for key in EPISODE_KEYS},
                                "primary_oracle": audit["primary_oracle"],
                                "top1_expert": audit["top1_expert"],
                                "top2_expert": audit["top2_expert"],
                                "top1_top2_accuracy_margin": audit[
                                    "top1_top2_accuracy_margin"
                                ],
                                "exact_accuracy_tie": audit["exact_accuracy_tie"],
                                "epsilon_accuracy": config["oracle_identity"][
                                    "epsilon_accuracy"
                                ],
                                "epsilon_optimal_size": audit["epsilon_optimal_size"],
                                "epsilon_optimal_experts": "|".join(
                                    audit["epsilon_optimal_experts"]
                                ),
                            }
                        )
    frame, intervention_metadata = add_confirmatory_interventions(pd.DataFrame(all_rows))
    bootstrap_repeats = int(episode_config["bootstrap_repeats"])
    effects = intervention_effects(frame, repeats=bootstrap_repeats)
    stability = ranking_stability(frame)
    loo = leave_one_out(frame, repeats=bootstrap_repeats)
    result = decide_gate1r(
        frame, effects, stability, config, repeats=bootstrap_repeats
    )
    primary = frame[frame["method"] == "router_candidate_oracle"]
    elapsed = time.perf_counter() - started
    decision: dict[str, Any] = {
        "schema_version": 1,
        **result,
        "experiment_id": arguments.experiment_id,
        "thresholds": {
            "shared_requirements": config["shared_requirements"],
            "pass_hard_additional_requirements": config[
                "pass_hard_additional_requirements"
            ],
        },
        "candidate_sets": config["candidate_sets"],
        "identity_rules": config["oracle_identity"],
        "preregistration_commit_sha": PREREGISTRATION_COMMIT,
        "preregistration_tag": PREREGISTRATION_TAG,
        "runtime_git_sha": runtime_sha,
        "working_tree_clean": clean,
        "task_split_hashes": split_hashes,
        "config_hash": FROZEN_CONFIG_SHA256,
        "host_gpu": {
            "host": platform.node(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "elapsed_time": elapsed,
        "gpu_hours": elapsed / 3600,
        "test_count": arguments.test_count,
    }
    decision["raw_criterion_values"].update(
        {
            "paired_reporting_levels": _paired_reporting(
                frame, repeats=bootstrap_repeats
            ),
            "best_path_entropy": _entropy(primary["selected_expert"]),
            "specialist_only_frequencies": frame[
                frame["method"] == "specialist_only_oracle"
            ]["selected_expert"].value_counts(normalize=True).sort_index().to_dict(),
            "intervention_metadata": intervention_metadata,
            "assignment_permutation_equals_wrong_task_assignment": bool(
                effects[effects["method"] == "task_assignment_permutation"]
                ["mean_accuracy_drop"].reset_index(drop=True).equals(
                    effects[effects["method"] == "wrong_task_expert_assignment"]
                    ["mean_accuracy_drop"].reset_index(drop=True)
                )
            ),
        }
    )
    schema = json.loads(
        (project_root / "schemas" / "gate1r_decision.schema.json").read_text(
            encoding="utf-8"
        )
    )
    _validate_decision_schema(decision, schema)
    decision["decision_schema_validation"] = "VALID"
    results_root = project_root / "results"
    reports_root = project_root / "reports"
    fixed_names = sorted({name for names in frame["expert_name_set"] for name in names.split("|")})
    matrix = (
        frame[frame["method"].isin(fixed_names)]
        .groupby(["split_seed", "train_seed", "task", "method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            query_nll_mean=("loss", "mean"),
            episode_count=("episode_hash", "nunique"),
        )
    )
    _atomic_frame(frame, results_root / "gate1r_episode_metrics.parquet", parquet=True)
    _atomic_frame(matrix, results_root / "gate1r_expert_task_matrix.csv")
    _atomic_frame(effects, results_root / "gate1r_intervention_effects.csv")
    _atomic_frame(stability, results_root / "gate1r_ranking_stability.csv")
    _atomic_frame(pd.DataFrame(epsilon_rows), results_root / "gate1r_epsilon_optimal_sets.csv")
    _atomic_frame(loo, results_root / "gate1r_leave_one_expert_out.csv")
    _atomic_frame(pd.DataFrame(resource_rows), results_root / "gate1r_resource_usage.csv")
    atomic_write_json(reports_root / "gate1r_decision.json", decision)
    atomic_write_text(
        reports_root / "gate1r_report.md",
        _report(decision, episode_count=int(primary["episode_hash"].nunique())),
    )
    if decision["outcome"] == "FAIL":
        atomic_write_text(
            reports_root / "h1_termination_report.md", _termination_report(decision)
        )
    missing = [relative for relative in REQUIRED_OUTPUTS if not (project_root / relative).exists()]
    if missing:
        raise RuntimeError(f"Gate 1R failed to produce outputs: {missing}")
    atomic_write_json(
        run_directory / "completion.json",
        {
            "status": "SUCCEEDED",
            "outcome": decision["outcome"],
            "runtime_git_sha": runtime_sha,
            "elapsed_time": elapsed,
            "required_outputs": list(REQUIRED_OUTPUTS),
        },
    )
    print(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "outcome": decision["outcome"],
                "experiment_id": arguments.experiment_id,
                "runtime_git_sha": runtime_sha,
                "elapsed_time": elapsed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
