"""Validate frozen Gate 1R protocol invariants without reading fresh results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def validate_preregistration(project_root: Path) -> dict[str, Any]:
    config = _load(project_root / "configs" / "m3_gate1r_confirmatory.json")
    canonical = _load(project_root / "configs" / "m2_pilot.json")
    schema = _load(project_root / "schemas" / "gate1r_preregistration.schema.json")
    decision_schema = _load(project_root / "schemas" / "gate1r_decision.schema.json")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("preregistration schema must use JSON Schema draft 2020-12")
    if decision_schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("decision schema must use JSON Schema draft 2020-12")
    if config["schema_version"] != 1 or config["protocol_status"] != "preregistered":
        raise ValueError("unexpected protocol schema or status")
    if config["canonical_protection"]["canonical_decision"] != "FAIL":
        raise ValueError("canonical Gate 1 decision protection is invalid")
    if config["canonical_protection"]["canonical_task_split_seeds_excluded"] != [0, 1]:
        raise ValueError("canonical task splits must be explicitly excluded")
    expected_tasks = {6: ["breastmnist", "octmnist"], 36: ["bloodmnist", "pathmnist"]}
    configured_splits = {int(item["seed"]): item for item in config["fresh_task_splits"]}
    if set(configured_splits) != {6, 36}:
        raise ValueError("only fresh task splits 6 and 36 are allowed")
    for seed, expected_meta_test in expected_tasks.items():
        split = _load(project_root / "configs" / "task_splits" / f"medmnist_seed{seed}.json")
        configured = configured_splits[seed]
        if split["meta_test"] != expected_meta_test or configured["meta_test"] != expected_meta_test:
            raise ValueError(f"fresh meta-test tasks differ for split {seed}")
        if configured["split_hash"] != split["split_hash"]:
            raise ValueError(f"split hash differs for split {seed}")
    frozen_mapping = {
        "backbone": "backbone",
        "pretrained": "pretrained",
        "adapter_rank": "adapter_rank",
        "training_steps": "training_steps",
        "training_shots": "training_shots",
        "training_queries_per_class": "training_queries_per_class",
        "training_n_way": "training_n_way",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "temperature": "temperature",
        "feature_batch_size": "feature_batch_size",
        "max_train_samples_per_class": "max_train_samples_per_class",
        "max_evaluation_samples_per_class": "max_evaluation_samples_per_class",
    }
    model = config["model_and_training"]
    for preregistered_name, canonical_name in frozen_mapping.items():
        if model[preregistered_name] != canonical[canonical_name]:
            raise ValueError(f"canonical mechanism changed: {preregistered_name}")
    episode = config["episode_protocol"]
    if episode["train_seeds"] != [2, 3, 4] or episode["shots"] != [1, 5, 16]:
        raise ValueError("confirmatory seeds or shots differ")
    if episode["support_resamples"] != 10 or episode["bootstrap_repeats"] != 10000:
        raise ValueError("confirmatory resampling counts differ")
    if episode["query_per_class"] != canonical["query_per_class"]:
        raise ValueError("query-per-class definition changed")
    if episode["random_repeats"] < canonical["random_router_repeats"]:
        raise ValueError("random repeats cannot be below canonical")
    candidates = config["candidate_sets"]
    if candidates["router_candidate_set"] != ["shared", "source_*"]:
        raise ValueError("primary router candidate set changed")
    if candidates["baseline_only_set"] != ["single"]:
        raise ValueError("single must remain baseline-only")
    if candidates["router_candidate_oracle"] != "shared_plus_all_source_experts":
        raise ValueError("primary router oracle scope changed")
    identity = config["oracle_identity"]
    if identity["accuracy_tie_tolerance"] != 1e-12 or identity["epsilon_accuracy"] != 0.01:
        raise ValueError("oracle identity thresholds changed")
    if identity["primary_rule"] != [
        "maximum_accuracy",
        "minimum_query_nll_within_accuracy_tolerance",
        "lexicographic_expert_name",
    ]:
        raise ValueError("oracle identity rule changed")
    if config["endpoints"]["random_expected_definition"] != (
        "exact_episode_mean_accuracy_over_router_candidate_set"
    ):
        raise ValueError("random expected must be exact, not simulated")
    statistics = config["statistics"]
    if statistics["hierarchical_resampling_order"] != [
        "split",
        "task_within_split",
        "train_seed_within_task",
        "support_episode_within_train_seed",
    ]:
        raise ValueError("hierarchical resampling order changed")
    if statistics["bootstrap_repeats"] != 10000 or statistics["confidence_level"] != 0.95:
        raise ValueError("confirmatory inference settings changed")
    expected_core = [
        "forced_worst",
        "random_expected",
        "repeated_random",
        "task_modal_swap",
        "task_assignment_permutation",
        "wrong_task_expert_assignment",
    ]
    if config["interventions"]["core"] != expected_core:
        raise ValueError("core intervention set changed")
    expected_shared = {
        "oracle_headroom": {
            "each_split_delta_oracle_shared_mean_gt": 0.01,
            "each_split_delta_oracle_single_mean_gt": 0.01,
            "each_split_hierarchical_ci_low_gt": 0.0,
            "minimum_positive_task_count_delta_oracle_shared": 3,
            "minimum_positive_task_count_delta_oracle_single": 3,
            "fresh_task_count": 4,
        },
        "no_global_path_dominance": {
            "maximum_primary_oracle_best_frequency_lt": 0.5,
            "minimum_paths_with_frequency_gte": 3,
            "path_frequency_threshold": 0.05,
            "all_tasks_same_modal_path_allowed": False,
        },
        "core_intervention_evidence": {
            "forced_worst_minimum_passing_tasks": 3,
            "random_expected_minimum_passing_tasks": 2,
            "assignment_minimum_passing_tasks": 2,
            "passing_task_mean_accuracy_drop_gt": 0.01,
            "passing_task_hierarchical_ci_low_gt": 0.0,
            "assignment_passes_if_either": [
                "task_assignment_permutation",
                "wrong_task_expert_assignment",
            ],
            "task_dependent_pattern_minimum_range_gt": 0.005,
            "task_dependent_pattern_applies_to": [
                "task_assignment_permutation",
                "wrong_task_expert_assignment",
            ],
        },
    }
    if config["shared_requirements"] != expected_shared:
        raise ValueError("shared Gate 1R requirements changed")
    expected_hard = {
        "minimum_episode_fraction_margin_gt_epsilon": 0.5,
        "margin_epsilon": 0.01,
        "minimum_tasks_modal_consistent_two_of_three_train_seeds": 3,
        "fresh_task_count": 4,
        "minimum_median_within_task_spearman": 0.3,
        "maximum_exact_accuracy_tie_rate_lt": 0.2,
    }
    if config["pass_hard_additional_requirements"] != expected_hard:
        raise ValueError("PASS_HARD requirements changed")
    outcomes = config["outcome_rule"]
    if outcomes["allowed_outcomes"] != ["PASS_HARD", "PASS_SOFT", "FAIL"]:
        raise ValueError("Gate 1R outcome set changed")
    if config["preregistration_tag"] != "gate1r-preregistered-v1":
        raise ValueError("unexpected preregistration tag")
    expected_outputs = [
        "reports/gate1r_report.md",
        "reports/gate1r_decision.json",
        "results/gate1r_episode_metrics.parquet",
        "results/gate1r_expert_task_matrix.csv",
        "results/gate1r_intervention_effects.csv",
        "results/gate1r_ranking_stability.csv",
        "results/gate1r_epsilon_optimal_sets.csv",
        "results/gate1r_leave_one_expert_out.csv",
        "results/gate1r_resource_usage.csv",
    ]
    if config["required_result_files"] != expected_outputs:
        raise ValueError("required Gate 1R output family changed")
    forbidden = {
        "reports/gate1r_report.md",
        "reports/gate1r_decision.json",
        "results/gate1r_episode_metrics.parquet",
    }
    present = {str(path.relative_to(project_root)).replace("\\", "/") for path in project_root.rglob("gate1r*")}
    leaked_results = sorted(forbidden & present)
    if leaked_results:
        raise ValueError(f"fresh Gate 1R results exist before preregistration: {leaked_results}")
    return {
        "status": "VALID",
        "protocol_id": config["protocol_id"],
        "fresh_split_seeds": sorted(configured_splits),
        "fresh_meta_test_tasks": expected_tasks,
        "canonical_decision": "FAIL",
        "gate1r_results_present": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    print(json.dumps(validate_preregistration(arguments.project_root.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
