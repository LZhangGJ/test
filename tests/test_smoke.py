import json
from pathlib import Path

from tamoe.config import SmokeConfig
from tamoe.smoke import run_smoke


def test_end_to_end_cpu_smoke(tmp_path: Path) -> None:
    run_directory = run_smoke(
        SmokeConfig(seed=5), output_root=tmp_path / "runs", project_root=tmp_path
    )
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "SUCCEEDED"
    assert summary["support_query_disjoint"] is True
    assert summary["optimizer_step_completed"] is True
    assert (run_directory / "COMPLETED.json").exists()
