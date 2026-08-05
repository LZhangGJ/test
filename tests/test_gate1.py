from __future__ import annotations

import pandas as pd
import torch

from tamoe.analysis.gate1 import (
    Gate1Thresholds,
    bootstrap_mean_ci,
    decide_gate1,
    feasible_query_count,
    macro_f1,
)


def test_macro_f1_is_one_for_perfect_prediction() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    assert macro_f1(labels, labels) == 1.0


def test_bootstrap_mean_ci_is_deterministic() -> None:
    values = torch.tensor([0.1, 0.2, 0.3]).numpy()
    assert bootstrap_mean_ci(values, repeats=100, seed=7) == bootstrap_mean_ci(
        values, repeats=100, seed=7
    )


def test_feasible_query_count_preserves_disjoint_sampling() -> None:
    labels = torch.tensor([0] * 23 + [1] * 40)
    assert feasible_query_count(labels, shots=5, desired_queries=20) == 18
    assert feasible_query_count(labels, shots=16, desired_queries=20) == 7


def test_gate_fails_global_dominance() -> None:
    rows = []
    for split_seed in (0, 1):
        for task in ("a", "b"):
            for repetition in range(4):
                common = {
                    "split_seed": split_seed, "train_seed": 0, "task": task, "shots": 1,
                    "support_resample": repetition, "episode_hash": f"{split_seed}-{task}-{repetition}",
                    "macro_f1": 0.0, "loss": 1.0,
                }
                for method, accuracy in {
                    "shared": 0.5, "single": 0.5, "episode_oracle": 0.8,
                    "forced_worst": 0.2, "random_00": 0.4, "swap": 0.3,
                    "mask_most_used": 0.3, "permuted_task_assignment": 0.3,
                }.items():
                    rows.append({**common, "method": method, "accuracy": accuracy,
                                 "selected_expert": "always_best"})
    decision = decide_gate1(
        pd.DataFrame(rows), bootstrap_repeats=100,
        thresholds=Gate1Thresholds(minimum_accuracy_gap=0.0),
    )
    assert decision["criteria"]["no_global_expert_dominance"] is False
    assert decision["status"] == "FAIL"
