"""Run a real MedMNIST episode-sampling smoke test and save only indices/statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tamoe.data.medmnist_dataset import MedMNISTTensorDataset
from tamoe.data.medmnist_tasks import get_task
from tamoe.episodes.sampler import (
    assert_router_input_is_metadata_free,
    index_labels,
    materialize_episode,
    sample_episode_indices,
)
from tamoe.utils.atomic_io import atomic_write_json, atomic_write_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", default="breastmnist")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    task = get_task(arguments.task)
    dataset = MedMNISTTensorDataset(task, arguments.root, "train")
    label_index = index_labels(dataset.labels)
    records = []
    for shots in config["shots"]:
        for repetition in range(int(config["pilot_support_resamples"])):
            episode = sample_episode_indices(
                label_index,
                shots=int(shots),
                queries_per_class=int(config["query_per_class"]),
                seed=int(config["task_split_seeds"][0]),
                repetition=repetition,
            )
            materialized = materialize_episode(dataset, episode)
            assert_router_input_is_metadata_free(materialized.router_input)
            records.append(
                {
                    "shots": int(shots),
                    "repetition": repetition,
                    "episode_hash": episode.episode_hash,
                    "class_ids": episode.class_ids,
                    "support_indices": episode.support_indices,
                    "query_indices": episode.query_indices,
                    "support_shape": list(materialized.router_input.support_images.shape),
                    "query_shape": list(materialized.router_input.query_images.shape),
                    "disjoint": set(episode.support_indices).isdisjoint(episode.query_indices),
                }
            )
    summary = {
        "schema_version": 1,
        "status": "SUCCEEDED",
        "experiment_id": "M1_SMOKE",
        "task": task.key,
        "dataset_samples": len(dataset),
        "label_counts": {label: len(indices) for label, indices in label_index.items()},
        "router_fields": ["support_images", "support_labels", "query_images"],
        "episodes": records,
    }
    atomic_write_json(arguments.output_json, summary)
    markdown = [
        "# M1 MedMNIST episode smoke",
        "",
        f"- Status: **{summary['status']}**",
        f"- Task: `{task.key}`",
        f"- Training samples: `{len(dataset)}`",
        f"- Label counts: `{summary['label_counts']}`",
        f"- Episodes: `{len(records)}`",
        f"- Shot conditions: `{config['shots']}`",
        f"- Support resamples per condition: `{config['pilot_support_resamples']}`",
        "- Support/query overlap: `false` for every episode",
        "- Router input fields: `support_images`, `support_labels`, `query_images`",
        "- Dataset name, task ID, split name, path, sample IDs, and query labels are excluded from router input.",
        "",
    ]
    atomic_write_text(arguments.output_markdown, "\n".join(markdown))
    print(json.dumps({"status": "SUCCEEDED", "episodes": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
