from tamoe.data.task_splits import generate_task_split
from tamoe.experts.bank import build_expert_definitions


def test_expert_bank_has_shared_single_and_one_expert_per_training_group() -> None:
    split = generate_task_split(0)
    definitions = build_expert_definitions(split)
    assert definitions[0].name == "shared"
    assert definitions[1].name == "single"
    source = definitions[2:]
    assert len(source) == len(split.meta_train_groups)
    assert {task for item in source for task in item.source_tasks} == set(split.meta_train)
    organ = [item for item in source if item.name == "source_organmnist"]
    if organ:
        assert set(organ[0].source_tasks) == {
            "organamnist",
            "organcmnist",
            "organsmnist",
        }
