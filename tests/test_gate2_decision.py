from __future__ import annotations

import json
import runpy
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_gate2_decision_passes_only_when_all_required_controls_are_worse() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_gate2_soft.py"))
    config = json.loads(
        (PROJECT_ROOT / "configs" / "m4_gate2_soft.json").read_text(encoding="utf-8")
    )
    config["bootstrap_repeats"] = 100
    methods = {
        "support_conditioned_soft_mixture": 0.8,
        "random_expected": 0.6,
        "query_only_weighting": 0.6,
        "support_query_shuffle": 0.6,
        "support_label_removal": 0.6,
        "wrong_support_task": 0.6,
        "support_prototype_weighting": 0.7,
        "shared": 0.5,
        "shared_fallback": 0.7,
        "capacity_matched_single": 0.5,
        "compute_matched_support_top1_diagnostic": 0.7,
        "router_candidate_oracle": 0.9,
    }
    rows = []
    for split_seed, tasks in ((6, ("a", "b")), (36, ("c", "d"))):
        for task in tasks:
            for train_seed in (2, 3, 4):
                for support_resample in range(2):
                    common = {
                        "split_seed": split_seed,
                        "train_seed": train_seed,
                        "task": task,
                        "shots": 1,
                        "support_resample": support_resample,
                        "episode_hash": f"{split_seed}:{task}:{train_seed}:{support_resample}",
                        "loss": 1.0,
                        "route_entropy": 0.5,
                    }
                    rows.extend(
                        {**common, "method": method, "accuracy": accuracy}
                        for method, accuracy in methods.items()
                    )
    decision, comparison, ablation = module["_decision"](pd.DataFrame(rows), config)
    assert decision["outcome"] == "PASS"
    assert decision["raw_values"]["oracle_gap_recovery"] == 0.75
    assert not comparison.empty
    assert set(ablation["comparison"]) == {
        "support_query_shuffle",
        "support_label_removal",
        "wrong_support_task",
    }
