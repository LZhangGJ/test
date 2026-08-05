from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from tamoe.analysis.gate1r import decide_gate1r, evaluate_confirmatory_episode, hierarchical_ci
from tamoe.experts.adapters import ResidualAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _identity_adapter() -> ResidualAdapter:
    adapter = ResidualAdapter(embedding_dim=2, rank=1)
    with torch.no_grad():
        adapter.down.weight.zero_()
        adapter.up.weight.zero_()
    return adapter.eval()


def test_confirmatory_episode_excludes_single_from_primary_and_random_expected() -> None:
    experts = {
        "shared": _identity_adapter(),
        "single": _identity_adapter(),
        "source_a": _identity_adapter(),
        "source_b": _identity_adapter(),
    }
    features = torch.tensor(
        [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5, dtype=torch.float32
    )
    labels = torch.tensor([0] * 5 + [1] * 5)
    rows, audit = evaluate_confirmatory_episode(
        experts,
        features,
        labels,
        shots=1,
        queries_per_class=2,
        seed=7,
        repetition=0,
        temperature=0.1,
        device=torch.device("cpu"),
        accuracy_tie_tolerance=1e-12,
        epsilon_accuracy=0.01,
        random_repeats=2,
    )
    assert audit["candidate_names"] == ("shared", "source_a", "source_b")
    metrics = {row["method"]: row for row in rows}
    expected = sum(metrics[name]["accuracy"] for name in audit["candidate_names"]) / 3
    assert metrics["random_expected"]["accuracy"] == expected
    assert metrics["router_candidate_oracle"]["selected_expert"] != "single"


def test_hierarchical_ci_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "task": ["a", "a", "b", "b"],
            "train_seed": [1, 2, 1, 2],
            "value": [0.0, 0.1, 0.2, 0.3],
        }
    )
    first = hierarchical_ci(frame, "value", ["task", "train_seed"], repeats=200, seed=9)
    second = hierarchical_ci(frame, "value", ["task", "train_seed"], repeats=200, seed=9)
    assert first == second


def test_gate1r_fails_when_one_path_globally_dominates() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "m3_gate1r_confirmatory.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for split_seed, tasks in ((6, ("a", "b")), (36, ("c", "d"))):
        for task in tasks:
            for train_seed in (2, 3, 4):
                for support_resample in range(3):
                    common = {
                        "split_seed": split_seed,
                        "train_seed": train_seed,
                        "task": task,
                        "shots": 1,
                        "support_resample": support_resample,
                        "episode_hash": f"{split_seed}:{task}:{train_seed}:{support_resample}",
                        "macro_f1": 0.0,
                        "loss": 1.0,
                    }
                    for method, accuracy in {
                        "router_candidate_oracle": 0.9,
                        "shared": 0.7,
                        "single": 0.7,
                        "random_expected": 0.6,
                    }.items():
                        rows.append(
                            {
                                **common,
                                "method": method,
                                "accuracy": accuracy,
                                "selected_expert": "shared",
                                "top1_top2_accuracy_margin": 0.2,
                                "exact_accuracy_tie": False,
                            }
                        )
    effects = pd.DataFrame(
        [
            {
                "method": method,
                "split_seed": split_seed,
                "task": task,
                "mean_accuracy_drop": 0.2 + 0.01 * index,
                "accuracy_drop_ci_low": 0.1,
                "accuracy_drop_ci_high": 0.3,
            }
            for index, method in enumerate(
                [
                    "forced_worst",
                    "random_expected",
                    "task_assignment_permutation",
                    "wrong_task_expert_assignment",
                ]
            )
            for split_seed, tasks in ((6, ("a", "b")), (36, ("c", "d")))
            for task in tasks
        ]
    )
    stability = pd.DataFrame(
        {
            "modal_consistent_two_of_three": [True] * 4,
            "median_pairwise_spearman_shot_support": [0.8] * 4,
        }
    )
    decision = decide_gate1r(
        pd.DataFrame(rows), effects, stability, config, repeats=100
    )
    assert decision["criteria"]["shared_requirements"]["A2_maximum_best_frequency"] is False
    assert decision["outcome"] == "FAIL"
