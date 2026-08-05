from dataclasses import fields

import torch

from tamoe.episodes.sampler import (
    RouterInput,
    assert_router_input_is_metadata_free,
    materialize_episode,
    sample_episode_indices,
)


def test_episode_sampling_is_repeatable_disjoint_and_variable_way() -> None:
    class_to_indices = {label: tuple(range(label * 20, (label + 1) * 20)) for label in range(7)}
    first = sample_episode_indices(
        class_to_indices, shots=2, queries_per_class=3, n_way=5, seed=17, repetition=4
    )
    second = sample_episode_indices(
        class_to_indices, shots=2, queries_per_class=3, n_way=5, seed=17, repetition=4
    )
    assert first == second
    assert len(first.class_ids) == 5
    assert set(first.support_indices).isdisjoint(first.query_indices)
    assert set(first.support_labels) == set(range(5))


def test_support_resampling_changes_indices_but_not_protocol() -> None:
    mapping = {label: tuple(range(label * 30, (label + 1) * 30)) for label in range(3)}
    first = sample_episode_indices(mapping, shots=5, queries_per_class=5, seed=9, repetition=0)
    second = sample_episode_indices(mapping, shots=5, queries_per_class=5, seed=9, repetition=1)
    assert first.support_indices != second.support_indices
    assert first.class_ids == second.class_ids


def test_materialized_router_view_excludes_task_identity_and_query_labels() -> None:
    dataset = [(torch.full((3, 8, 8), float(index)), index // 10) for index in range(30)]
    episode = sample_episode_indices(
        {label: tuple(range(label * 10, (label + 1) * 10)) for label in range(3)},
        shots=2,
        queries_per_class=2,
        seed=3,
    )
    materialized = materialize_episode(dataset, episode)
    assert_router_input_is_metadata_free(materialized.router_input)
    assert {field.name for field in fields(RouterInput)} == {
        "support_images",
        "support_labels",
        "query_images",
    }
    assert materialized.query_labels.shape == (6,)
