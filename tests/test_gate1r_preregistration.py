from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path

import torch

from tamoe.analysis.oracle import deterministic_accuracy_oracle, epsilon_optimal_experts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def test_preregistration_validator_accepts_frozen_protocol(tmp_path: Path) -> None:
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    shutil.copytree(PROJECT_ROOT / "schemas", tmp_path / "schemas")
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "validate_gate1r_preregistration.py"))
    result = module["validate_preregistration"](tmp_path)
    assert result["status"] == "VALID"
    assert result["fresh_split_seeds"] == [6, 36]
    assert result["canonical_decision"] == "FAIL"
    assert result["gate1r_results_present"] is False


def test_fresh_splits_and_candidate_sets_are_exact() -> None:
    config = _json("configs/m3_gate1r_confirmatory.json")
    assert [item["seed"] for item in config["fresh_task_splits"]] == [6, 36]
    assert config["fresh_task_splits"][0]["meta_test"] == ["breastmnist", "octmnist"]
    assert config["fresh_task_splits"][1]["meta_test"] == ["bloodmnist", "pathmnist"]
    assert config["candidate_sets"]["router_candidate_set"] == ["shared", "source_*"]
    assert config["candidate_sets"]["baseline_only_set"] == ["single"]


def test_confirmatory_grid_and_outcomes_are_frozen() -> None:
    config = _json("configs/m3_gate1r_confirmatory.json")
    episode = config["episode_protocol"]
    assert episode["train_seeds"] == [2, 3, 4]
    assert episode["shots"] == [1, 5, 16]
    assert episode["support_resamples"] == 10
    assert episode["bootstrap_repeats"] == 10000
    assert config["outcome_rule"]["allowed_outcomes"] == ["PASS_HARD", "PASS_SOFT", "FAIL"]


def test_primary_oracle_uses_accuracy_then_nll_then_name() -> None:
    within_tolerance = deterministic_accuracy_oracle(
        ["source_z", "source_b", "source_a"],
        torch.tensor([0.8, 0.8 - 5e-13, 0.8 - 5e-13], dtype=torch.float64),
        torch.tensor([0.5, 0.4, 0.4], dtype=torch.float64),
        accuracy_tie_tolerance=1e-12,
    )
    assert within_tolerance.expert_name == "source_a"
    outside_tolerance = deterministic_accuracy_oracle(
        ["higher_accuracy", "lower_nll"],
        torch.tensor([0.8, 0.8 - 2e-12], dtype=torch.float64),
        torch.tensor([1.0, 0.1], dtype=torch.float64),
        accuracy_tie_tolerance=1e-12,
    )
    assert outside_tolerance.expert_name == "higher_accuracy"


def test_epsilon_optimal_set_is_lexicographic_and_inclusive() -> None:
    result = epsilon_optimal_experts(
        ["source_c", "source_a", "shared"],
        torch.tensor([0.79, 0.80, 0.795], dtype=torch.float64),
        epsilon_accuracy=0.01,
    )
    assert result == ("shared", "source_a", "source_c")


def test_canonical_artifact_hashes_are_unchanged() -> None:
    diagnostics = _json("reports/gate1_posthoc_diagnostics.json")
    paths = {
        "gate1_episode_metrics.parquet": PROJECT_ROOT / "results" / "gate1_episode_metrics.parquet",
        "expert_task_matrix.csv": PROJECT_ROOT / "results" / "expert_task_matrix.csv",
        "gate1_decision.json": PROJECT_ROOT / "reports" / "gate1_decision.json",
    }
    for name, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == diagnostics["source_sha256"][name]


def test_decision_schema_requires_confirmatory_provenance() -> None:
    schema = _json("schemas/gate1r_decision.schema.json")
    required = set(schema["required"])
    assert {
        "outcome",
        "preregistration_commit_sha",
        "preregistration_tag",
        "runtime_git_sha",
        "working_tree_clean",
        "task_split_hashes",
        "config_hash",
        "host_gpu",
        "elapsed_time",
        "gpu_hours",
        "test_count",
    }.issubset(required)
