from __future__ import annotations

import pandas as pd

from tamoe.analysis.posthoc import bootstrap_ci, fixed_expert_rows


def test_fixed_expert_rows_supports_split_specific_banks() -> None:
    frame = pd.DataFrame(
        [
            {"method": "shared", "expert_name_set": "shared|source_a"},
            {"method": "source_a", "expert_name_set": "shared|source_a"},
            {"method": "oracle", "expert_name_set": "shared|source_a"},
            {"method": "source_b", "expert_name_set": "shared|source_b"},
        ]
    )
    assert fixed_expert_rows(frame)["method"].tolist() == ["shared", "source_a", "source_b"]


def test_posthoc_bootstrap_is_deterministic() -> None:
    first = bootstrap_ci([0.0, 0.1, 0.2], repeats=1000, seed=9)
    second = bootstrap_ci([0.0, 0.1, 0.2], repeats=1000, seed=9)
    assert first == second
