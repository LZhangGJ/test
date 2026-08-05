"""Generate committed, group-aware MedMNIST task split definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tamoe.data.task_splits import generate_task_split, validate_split_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    splits = tuple(
        generate_task_split(
            int(seed),
            validation_group_count=int(config["validation_group_count"]),
            test_group_count=int(config["test_group_count"]),
        )
        for seed in config["task_split_seeds"]
    )
    validate_split_suite(splits)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for split in splits:
        split.write(arguments.output_directory / f"medmnist_seed{split.seed}.json")
    print(json.dumps({"status": "SUCCEEDED", "split_hashes": [s.split_hash for s in splits]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
