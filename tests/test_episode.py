import torch

from tamoe.episodes.synthetic import make_synthetic_episode


def test_synthetic_episode_is_deterministic_and_disjoint() -> None:
    kwargs = {
        "num_classes": 3,
        "samples_per_class": 6,
        "shots": 2,
        "queries_per_class": 2,
        "seed": 11,
    }
    first = make_synthetic_episode(**kwargs)
    second = make_synthetic_episode(**kwargs)
    assert torch.equal(first.support_indices, second.support_indices)
    assert torch.equal(first.query_indices, second.query_indices)
    assert set(first.support_indices.tolist()).isdisjoint(second.query_indices.tolist())
