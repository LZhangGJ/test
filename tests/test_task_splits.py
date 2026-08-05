from tamoe.data.medmnist_tasks import MEDMNIST_TASKS
from tamoe.data.task_splits import generate_task_split, validate_split_suite


def test_group_split_is_deterministic_disjoint_and_group_safe() -> None:
    first = generate_task_split(6)
    second = generate_task_split(6)
    assert first == second
    first.validate()
    organ_locations = [
        set(("organamnist", "organcmnist", "organsmnist")) & set(partition)
        for partition in (first.meta_train, first.meta_validation, first.meta_test)
    ]
    assert sorted(map(len, organ_locations)) == [0, 0, 3]


def test_preregistered_seed_suite_covers_every_task_as_meta_test() -> None:
    splits = tuple(generate_task_split(seed) for seed in (0, 1, 6, 36))
    validate_split_suite(splits)
    observed = set().union(*(set(split.meta_test) for split in splits))
    assert observed == {task.key for task in MEDMNIST_TASKS}
