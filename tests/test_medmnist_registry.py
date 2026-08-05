from tamoe.data.medmnist_tasks import MEDMNIST_TASKS, ObjectiveType, grouped_tasks


def test_initial_registry_contains_only_compatible_tasks() -> None:
    keys = {task.key for task in MEDMNIST_TASKS}
    assert len(keys) == 10
    assert "chestmnist" not in keys
    assert "retinamnist" not in keys
    assert all(
        task.objective in {ObjectiveType.BINARY_CLASS, ObjectiveType.MULTI_CLASS}
        for task in MEDMNIST_TASKS
    )


def test_organ_views_share_one_group() -> None:
    groups = grouped_tasks()
    assert groups["organmnist"] == ("organamnist", "organcmnist", "organsmnist")
    assert len(groups) == 8
