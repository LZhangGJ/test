from pathlib import Path

import pytest

from tamoe.config import SmokeConfig


def test_config_hash_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"seed": 7}', encoding="utf-8")
    first = SmokeConfig.from_json(path)
    second = SmokeConfig.from_json(path)
    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == 64


def test_config_rejects_overlapping_sample_budget() -> None:
    config = SmokeConfig(samples_per_class=2, shots=1, queries_per_class=2)
    with pytest.raises(ValueError, match="shots"):
        config.validate()
