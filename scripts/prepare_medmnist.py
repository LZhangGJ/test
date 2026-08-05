"""Download MedMNIST files separately from model training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tamoe.data.medmnist_dataset import prepare_task
from tamoe.data.medmnist_tasks import MEDMNIST_TASKS
from tamoe.utils.atomic_io import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(task.key for task in MEDMNIST_TASKS))
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    prepared = []
    for task in (item.strip() for item in arguments.tasks.split(",") if item.strip()):
        prepared.extend(prepare_task(task, arguments.root, size=arguments.size))
    atomic_write_json(arguments.manifest, [asdict(item) for item in prepared])
    print(json.dumps({"status": "SUCCEEDED", "records": len(prepared)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
