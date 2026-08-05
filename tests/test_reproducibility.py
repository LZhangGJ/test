import random

from tamoe.utils.reproducibility import seed_everything


def test_python_seed_is_repeatable() -> None:
    first_state = seed_everything(23)
    first = [random.random() for _ in range(3)]
    second_state = seed_everything(23)
    second = [random.random() for _ in range(3)]
    assert first == second
    assert first_state.seed == second_state.seed == 23
