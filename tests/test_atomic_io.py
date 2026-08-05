import json
from pathlib import Path

from tamoe.utils.atomic_io import atomic_write_json, atomic_write_text


def test_atomic_writers_replace_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    atomic_write_text(path, "old")
    atomic_write_json(path, {"status": "SUCCEEDED"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "SUCCEEDED"}
    assert not list(path.parent.glob("*.tmp"))
