import pytest
import torch

from tamoe.analysis.oracle import episode_oracle


def test_episode_oracle_selects_maximum_metric() -> None:
    result = episode_oracle(torch.tensor([0.4, 0.8, 0.6]))
    assert result.expert_index == 1
    assert result.metric == pytest.approx(0.8)
