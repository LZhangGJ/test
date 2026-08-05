"""Run the frozen support-risk routing pilot on exact Gate 2 expert banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional

from tamoe.analysis.gate1 import feasible_query_count
from tamoe.analysis.gate1r import EPISODE_KEYS
from tamoe.data.feature_cache import FeatureSet, extract_or_load_features
from tamoe.data.medmnist_dataset import MedMNISTTensorDataset, sha256_file
from tamoe.data.medmnist_tasks import get_task
from tamoe.data.task_splits import TaskSplit
from tamoe.experts.bank import ExpertDefinition, build_expert_definitions
from tamoe.experts.training import load_expert_checkpoint
from tamoe.models.backbones import build_backbone
from tamoe.routing.support_risk import (
    SupportRiskEpisode,
    evaluate_risk_ablation,
    evaluate_support_risk_episode,
)
from tamoe.utils.atomic_io import atomic_write_json, atomic_write_text

GATE2_DECISION_SHA256 = "78f0c8e4e0bb2a2c8e756bc9c0b1eec0150c258ff5b92125b2fb1e6ab91c898f"
PROTECTED_HASHES = {
    "results/gate1_episode_metrics.parquet": "1ddf550772924d684f9f66bfcc96ff1ee4d4f0df7a9dadfc1d60f92669a8bc59",
    "results/expert_task_matrix.csv": "5f729762cec976f19bbbaad4e32533f144467c603e7c7e06453bfc0d5a8ec190",
    "reports/gate1_decision.json": "9f55f5caea46247cd92f6da268795e20d94c363134c5a1e29bf810936465db16",
}
RISK_METHODS = (
    "naive_support_loss_top1",
    "leave_one_out_support_risk_top1",
    "leave_one_out_support_risk_soft_mixture",
    "shrinkage_support_risk_mixture",
    "shrinkage_support_risk_with_shared_fallback",
)


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={project_root.as_posix()}", "-C", str(project_root),
         *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def _load_split(path: Path) -> TaskSplit:
    split = TaskSplit(**json.loads(path.read_text(encoding="utf-8")))
    split.validate()
    return split


def _features(
    tasks: tuple[str, ...] | list[str],
    split_name: str,
    *,
    data_root: Path,
    cache_root: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    model_config: dict[str, Any],
    code_version: str,
    max_per_class: int,
) -> dict[str, FeatureSet]:
    return {
        task: extract_or_load_features(
            MedMNISTTensorDataset(get_task(task), data_root, split_name),
            backbone,
            cache_root,
            device=device,
            batch_size=int(model_config["feature_batch_size"]),
            code_version=code_version,
            max_samples_per_class=max_per_class,
            seed=0,
            num_workers=0,
        )
        for task in tasks
    }


def _load_bank(
    bank_root: Path,
    split: TaskSplit,
    train_seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], tuple[ExpertDefinition, ...], dict[str, str]]:
    bank_directory = bank_root / f"split_{split.seed}" / f"train_seed_{train_seed}"
    manifest = json.loads(
        (bank_directory / "expert_bank_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["split_hash"] != split.split_hash or manifest["split_seed"] != split.seed:
        raise RuntimeError("Gate 2 expert-bank manifest does not match task split")
    definitions = build_expert_definitions(split)
    experts: dict[str, torch.nn.Module] = {}
    hashes: dict[str, str] = {}
    for definition in definitions:
        checkpoint = bank_directory / definition.name / "checkpoint.pt"
        expert, config = load_expert_checkpoint(checkpoint, device=device)
        if config.seed != train_seed:
            raise RuntimeError(f"checkpoint seed mismatch: {checkpoint}")
        experts[definition.name] = expert.eval()
        hashes[f"split_{split.seed}/seed_{train_seed}/{definition.name}"] = sha256_file(checkpoint)
    if len(experts) != 7:
        raise RuntimeError("exact Gate 2 bank must contain seven experts")
    return experts, definitions, hashes


def _references(
    experts: dict[str, torch.nn.Module],
    definitions: tuple[ExpertDefinition, ...],
    features: dict[str, FeatureSet],
    device: torch.device,
) -> dict[str, Tensor]:
    by_name = {definition.name: definition for definition in definitions}
    result: dict[str, Tensor] = {}
    with torch.inference_mode():
        for name in sorted(experts):
            if name == "single":
                continue
            total: Tensor | None = None
            count = 0
            for task in by_name[name].source_tasks:
                values = features[task].features
                for start in range(0, len(values), 2048):
                    adapted = experts[name](values[start : start + 2048].to(device))
                    chunk = functional.normalize(adapted, dim=-1).sum(dim=0).cpu()
                    total = chunk if total is None else total + chunk
                    count += len(adapted)
            if total is None or count == 0:
                raise RuntimeError(f"cannot construct expert reference for {name}")
            result[name] = total / count
    return result


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def _balanced_hierarchical_ci(
    frame: pd.DataFrame,
    value: str,
    levels: list[str],
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    if frame.empty or frame[value].isna().any() or frame.duplicated(levels).any():
        raise ValueError("balanced hierarchical bootstrap requires one complete value per leaf")

    def nested_values(group: pd.DataFrame, remaining: list[str]) -> np.ndarray:
        if not remaining:
            if len(group) != 1:
                raise ValueError("balanced hierarchical bootstrap requires one value per leaf")
            return np.asarray(float(group[value].iloc[0]), dtype=np.float64)
        children = [
            nested_values(child, remaining[1:])
            for _, child in group.groupby(remaining[0], sort=True)
        ]
        if not children or len({child.shape for child in children}) != 1:
            raise ValueError("balanced hierarchical bootstrap requires equal child shapes")
        return np.stack(children)

    values = nested_values(frame, levels)
    generator = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    batch_size = 500
    reduction_axes = tuple(range(1, values.ndim + 1))
    for start in range(0, repeats, batch_size):
        size = min(batch_size, repeats - start)
        indices: list[np.ndarray] = []
        for axis, dimension in enumerate(values.shape):
            sampled = generator.integers(
                0,
                dimension,
                size=(size, *values.shape[: axis + 1]),
            )
            indices.append(
                sampled.reshape((*sampled.shape, *([1] * (values.ndim - axis - 1))))
            )
        estimates.append(values[tuple(indices)].mean(axis=reduction_axes))
    bootstrap = np.concatenate(estimates)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "n": int(values.size),
    }


def _paired_frame(frame: pd.DataFrame, method: str, baseline: str) -> pd.DataFrame:
    selected = frame[frame["method"] == method]
    control = frame[frame["method"] == baseline]
    metrics = ["accuracy", "macro_f1", "loss", "ece", "brier"]
    joined = selected[EPISODE_KEYS + metrics].merge(
        control[EPISODE_KEYS + metrics],
        on=EPISODE_KEYS,
        suffixes=("_method", "_baseline"),
        validate="one_to_one",
    )
    joined["accuracy_delta"] = joined["accuracy_method"] - joined["accuracy_baseline"]
    joined["nll_improvement"] = joined["loss_baseline"] - joined["loss_method"]
    for metric in metrics:
        joined[f"{metric}_delta"] = (
            joined[f"{metric}_method"] - joined[f"{metric}_baseline"]
        )
    joined["support_episode"] = (
        joined["shots"].astype(str) + ":" + joined["support_resample"].astype(str)
    )
    return joined


def _comparison(
    frame: pd.DataFrame,
    method: str,
    baseline: str,
    *,
    repeats: int,
) -> dict[str, Any]:
    paired = _paired_frame(frame, method, baseline)
    aggregate_levels = ["split_seed", "task", "train_seed", "support_episode"]
    metric_deltas = {
        output_name: _balanced_hierarchical_ci(
            paired,
            f"{column}_delta",
            aggregate_levels,
            repeats=repeats,
            seed=_seed(f"support-risk:{method}:{baseline}:aggregate:{column}"),
        )
        for output_name, column in (
            ("accuracy", "accuracy"),
            ("macro_f1", "macro_f1"),
            ("nll", "loss"),
            ("ece", "ece"),
            ("brier", "brier"),
        )
    }
    aggregate = metric_deltas["accuracy"]
    split = {
        str(int(split_seed)): _balanced_hierarchical_ci(
            group,
            "accuracy_delta",
            ["task", "train_seed", "support_episode"],
            repeats=repeats,
            seed=_seed(f"support-risk:{method}:{baseline}:split:{split_seed}"),
        )
        for split_seed, group in paired.groupby("split_seed", sort=True)
    }
    task = {
        f"{split_seed}:{task_name}": _balanced_hierarchical_ci(
            group,
            "accuracy_delta",
            ["train_seed", "support_episode"],
            repeats=repeats,
            seed=_seed(f"support-risk:{method}:{baseline}:task:{split_seed}:{task_name}"),
        )
        for (split_seed, task_name), group in paired.groupby(["split_seed", "task"], sort=True)
    }
    return {
        "method": method,
        "baseline": baseline,
        "aggregate": aggregate,
        "split": split,
        "task": task,
        "paired_metric_deltas": metric_deltas,
        "mean_nll_improvement": float(paired["nll_improvement"].mean()),
    }


def _aggregate_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return (
        frame.groupby("method", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            nll=("loss", "mean"),
            ece=("ece", "mean"),
            brier=("brier", "mean"),
            episode_count=("episode_hash", "nunique"),
        )
        .sort_values("accuracy", ascending=False)
        .to_dict(orient="records")
    )


def _decide(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    repeats = int(config["bootstrap_repeats"])
    comparisons = [
        _comparison(frame, method, "capacity_matched_single", repeats=repeats)
        for method in RISK_METHODS
    ]
    primary = config["primary_method"]
    primary_comparison = next(item for item in comparisons if item["method"] == primary)
    ablations = [
        _comparison(frame, primary, control, repeats=repeats)
        for control in ("support_shuffle", "support_label_removal", "wrong_task_support")
    ]
    aggregates = {row["method"]: row for row in _aggregate_metrics(frame)}
    primary_accuracy = aggregates[primary]["accuracy"]
    single_accuracy = aggregates["capacity_matched_single"]["accuracy"]
    shared_accuracy = aggregates["shared"]["accuracy"]
    oracle_accuracy = aggregates["oracle_analysis_only"]["accuracy"]
    denominator = oracle_accuracy - single_accuracy
    recovery = (primary_accuracy - single_accuracy) / denominator if denominator > 1e-12 else None
    thresholds = config["go_criteria"]
    passing_ablations = [
        item["baseline"]
        for item in ablations
        if item["aggregate"]["mean"]
        >= thresholds["support_ablation_accuracy_drop_gte"]
        and item["aggregate"]["ci_low"]
        > thresholds["support_ablation_hierarchical_ci_low_gt"]
    ]
    criteria = {
        "accuracy_improvement_over_capacity_matched_single": primary_accuracy - single_accuracy
        >= thresholds["accuracy_improvement_over_capacity_matched_single_gte"],
        "paired_aggregate_ci_lower_bound": primary_comparison["aggregate"]["ci_low"]
        > thresholds["paired_aggregate_hierarchical_ci_low_gt"],
        "oracle_gap_recovery": recovery is not None
        and recovery >= thresholds["oracle_gap_recovery_gte"],
        "positive_task_count": sum(
            item["mean"] > 0 for item in primary_comparison["task"].values()
        )
        >= thresholds["minimum_positive_task_count"],
        "positive_both_fresh_splits": sum(
            item["mean"] > 0 for item in primary_comparison["split"].values()
        )
        >= thresholds["positive_split_count_required"],
        "support_ablation_evidence": len(passing_ablations)
        >= thresholds["minimum_passing_support_ablations"],
        "nll_no_worse_than_shared": aggregates[primary]["nll"] <= aggregates["shared"]["nll"],
    }
    raw = {
        "primary_method": primary,
        "primary_accuracy": primary_accuracy,
        "capacity_matched_single_accuracy": single_accuracy,
        "shared_accuracy": shared_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "accuracy_improvement_over_capacity_matched_single": primary_accuracy - single_accuracy,
        "oracle_gap_recovery": recovery,
        "positive_task_count": sum(
            item["mean"] > 0 for item in primary_comparison["task"].values()
        ),
        "positive_split_count": sum(
            item["mean"] > 0 for item in primary_comparison["split"].values()
        ),
        "passing_support_ablations": passing_ablations,
        "primary_vs_capacity_matched_single": primary_comparison,
        "risk_method_comparisons": comparisons,
        "support_ablation_comparisons": ablations,
        "aggregate_method_metrics": list(aggregates.values()),
    }
    return {
        "outcome": "GO_FIXED_BANK_ROUTING" if all(criteria.values())
        else "STOP_FIXED_BANK_ROUTING",
        "criteria": criteria,
        "raw_values": raw,
    }, comparisons, ablations


def _plot(comparisons: list[dict[str, Any]], output: Path) -> None:
    matplotlib_cache = Path(tempfile.gettempdir()) / "tamoe-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = {
        "naive_support_loss_top1": "Naive support loss top-1",
        "leave_one_out_support_risk_top1": "LOO support risk top-1",
        "leave_one_out_support_risk_soft_mixture": "LOO support risk soft mixture",
        "shrinkage_support_risk_mixture": "Shrinkage risk mixture",
        "shrinkage_support_risk_with_shared_fallback": "Shrinkage risk + shared fallback",
    }
    ordered = list(reversed(RISK_METHODS))
    records = {item["method"]: item["aggregate"] for item in comparisons}
    means = np.array([records[method]["mean"] for method in ordered])
    lows = np.array([records[method]["ci_low"] for method in ordered])
    highs = np.array([records[method]["ci_high"] for method in ordered])
    y = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(10, 5.8), facecolor="#FAFAFA")
    axis.set_facecolor("#FAFAFA")
    for index, method in enumerate(ordered):
        primary = method == "shrinkage_support_risk_with_shared_fallback"
        color = "#2563EB" if primary else "#64748B"
        axis.errorbar(
            means[index],
            y[index],
            xerr=[
                [max(0.0, means[index] - lows[index])],
                [max(0.0, highs[index] - means[index])],
            ],
            fmt="o",
            markersize=8,
            markerfacecolor=color if primary else "#FAFAFA",
            markeredgecolor=color,
            ecolor=color,
            elinewidth=2,
            capsize=4,
            zorder=3,
        )
        axis.annotate(
            f"{means[index]:+.4f}",
            xy=(highs[index], y[index]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#1F2937",
        )
    axis.axvline(0, color="#334155", linewidth=1.2, linestyle="--", label="No difference")
    axis.axvline(0.005, color="#B45309", linewidth=1.2, linestyle=":", label="Go threshold")
    axis.set_yticks(y, [labels[method] for method in ordered])
    axis.set_xlabel("Paired accuracy delta vs capacity-matched single")
    fig.suptitle(
        "Support-risk routing paired accuracy deltas",
        x=0.125,
        y=0.97,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.125,
        0.925,
        "Hierarchical 95% intervals; identical 5-shot and 10-shot episodes (N=480)",
        fontsize=10,
        color="#475569",
    )
    axis.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#94A3B8")
    axis.margins(x=0.08)
    handles, legend_labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.93, 0.015),
        ncol=2,
    )
    fig.subplots_adjust(left=0.31, right=0.93, top=0.84, bottom=0.18)
    # Locked research blossom: compact five-petal mark at the header's top-right.
    center_x, center_y = 0.965, 0.94
    for angle in np.linspace(0, 2 * np.pi, 5, endpoint=False):
        fig.text(
            center_x + 0.012 * np.cos(angle),
            center_y + 0.018 * np.sin(angle),
            "•",
            color="#2563EB",
            fontsize=11,
            ha="center",
            va="center",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=output.parent, suffix=".png")
    os.close(descriptor)
    temporary = Path(name)
    try:
        fig.savefig(temporary, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        os.replace(temporary, output)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def _report(decision: dict[str, Any]) -> str:
    raw = decision["raw_values"]
    metrics = sorted(
        raw["aggregate_method_metrics"], key=lambda item: item["accuracy"], reverse=True
    )
    lines = [
        "# Support-risk routing pilot",
        "",
        f"- Frozen outcome: **{decision['outcome']}**",
        f"- Experiment: `{decision['experiment_id']}`",
        "- This is a new falsifiable hypothesis, not a reinterpretation or rescue of Gate 2.",
        "- Exact Gate 2 checkpoints are loaded read-only; no router, encoder, or uncertainty head is trained.",
        "- Primary method was frozen before results: `shrinkage_support_risk_with_shared_fallback`.",
        "",
        "## Frozen Go criteria",
        "",
    ]
    lines.extend(
        f"- `{name}`: **{'PASS' if passed else 'FAIL'}**"
        for name, passed in decision["criteria"].items()
    )
    lines.extend(
        [
            "",
            "## Core values",
            "",
            f"- Primary accuracy: `{raw['primary_accuracy']:.4f}`",
            f"- Capacity-matched single accuracy: `{raw['capacity_matched_single_accuracy']:.4f}`",
            "- Accuracy improvement: "
            f"`{raw['accuracy_improvement_over_capacity_matched_single']:+.4f}`",
            f"- Oracle-gap recovery: `{raw['oracle_gap_recovery']:.4f}`"
            if raw["oracle_gap_recovery"] is not None
            else "- Oracle-gap recovery: not interpretable.",
            f"- Positive tasks: `{raw['positive_task_count']}/4`",
            f"- Positive splits: `{raw['positive_split_count']}/2`",
            f"- Passing support ablations: `{raw['passing_support_ablations']}`",
            "",
            "## Aggregate metrics",
            "",
            "| Method | Accuracy | Macro-F1 | NLL | ECE | Brier |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['method']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['nll']:.4f} | {row['ece']:.4f} | {row['brier']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired hierarchical 95% intervals vs capacity-matched single",
            "",
            "Deltas are method minus control; negative NLL, ECE, and Brier deltas are better.",
            "",
            "| Method | Accuracy delta | Macro-F1 delta | NLL delta | ECE delta | Brier delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    def interval(value: dict[str, Any]) -> str:
        return f"{value['mean']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]"

    for comparison in raw["risk_method_comparisons"]:
        paired = comparison["paired_metric_deltas"]
        lines.append(
            f"| {comparison['method']} | {interval(paired['accuracy'])} | "
            f"{interval(paired['macro_f1'])} | {interval(paired['nll'])} | "
            f"{interval(paired['ece'])} | {interval(paired['brier'])} |"
        )
    primary = raw["primary_vs_capacity_matched_single"]
    lines.extend(
        [
            "",
            "## Primary accuracy deltas by split and task",
            "",
            "| Scope | Mean delta | Paired hierarchical 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for split_seed, value in primary["split"].items():
        lines.append(
            f"| split {split_seed} | {value['mean']:+.4f} | "
            f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}] |"
        )
    for task_name, value in primary["task"].items():
        lines.append(
            f"| task {task_name} | {value['mean']:+.4f} | "
            f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Support ablation accuracy drops",
            "",
            "Positive values mean the frozen primary outperformed the ablated support control.",
            "",
            "| Ablation | Mean drop | Paired hierarchical 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for comparison in raw["support_ablation_comparisons"]:
        value = comparison["aggregate"]
        lines.append(
            f"| {comparison['baseline']} | {value['mean']:+.4f} | "
            f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}] |"
        )
    lines.extend(["", "## Decision consequence", ""])
    if decision["outcome"] == "STOP_FIXED_BANK_ROUTING":
        lines.append(
            "At least one required criterion failed. Fixed-bank support-risk routing stops; no model "
            "complexity is added after this result. The terminated H1 project remains unchanged."
        )
    else:
        lines.append(
            "Every frozen criterion passed. The result supports a separately scoped fixed-bank "
            "support-risk routing direction without reopening the terminated H1 gates."
        )
    return "\n".join(lines) + "\n"


def _sanitize(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-count", type=int, required=True)
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    config_path = arguments.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    runtime_sha = _git(project_root, "rev-parse", "HEAD")
    if _git(project_root, "status", "--porcelain"):
        raise RuntimeError("formal support-risk pilot requires a clean checkout")
    if sha256_file(project_root / "reports" / "gate2_decision.json") != GATE2_DECISION_SHA256:
        raise RuntimeError("protected Gate 2 decision hash changed")
    for relative, expected in PROTECTED_HASHES.items():
        if sha256_file(project_root / relative) != expected:
            raise RuntimeError(f"protected artifact changed: {relative}")
    for relative in config["required_outputs"]:
        if (project_root / relative).exists():
            raise RuntimeError(f"pilot output already exists: {relative}")
    gate2_config = json.loads(
        (project_root / "configs" / "m4_gate2_soft.json").read_text(encoding="utf-8")
    )
    if config["support_resamples"] != gate2_config["support_resamples"]:
        raise RuntimeError("support resamples differ from Gate 2")
    if config["task_split_seeds"] != gate2_config["fresh_task_split_seeds"]:
        raise RuntimeError("task splits differ from Gate 2")
    if config["train_seeds"] != gate2_config["train_seeds"]:
        raise RuntimeError("train seeds differ from Gate 2")
    model_config = json.loads(
        (project_root / "configs" / "m3_gate1r_confirmatory.json").read_text(encoding="utf-8")
    )["model_and_training"]
    device = torch.device(arguments.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal support-risk pilot requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    backbone = build_backbone(model_config["backbone"], pretrained=model_config["pretrained"])
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    bundles: list[tuple[dict[str, Any], SupportRiskEpisode]] = []
    checkpoint_hashes: dict[str, str] = {}
    for split_seed in config["task_split_seeds"]:
        split = _load_split(
            project_root / "configs" / "task_splits" / f"medmnist_seed{split_seed}.json"
        )
        train_features = _features(
            split.meta_train,
            "train",
            data_root=arguments.data_root,
            cache_root=arguments.cache_root,
            backbone=backbone,
            device=device,
            model_config=model_config,
            code_version=config["reuse"]["feature_cache_code_version"],
            max_per_class=int(model_config["max_train_samples_per_class"]),
        )
        evaluation_features = _features(
            split.meta_test,
            "test",
            data_root=arguments.data_root,
            cache_root=arguments.cache_root,
            backbone=backbone,
            device=device,
            model_config=model_config,
            code_version=config["reuse"]["feature_cache_code_version"],
            max_per_class=int(model_config["max_evaluation_samples_per_class"]),
        )
        for train_seed in config["train_seeds"]:
            experts, definitions, hashes = _load_bank(
                arguments.bank_root, split, int(train_seed), device
            )
            checkpoint_hashes.update(hashes)
            references = _references(experts, definitions, train_features, device)
            for task, feature_set in evaluation_features.items():
                for shots in config["shots"]:
                    queries = feasible_query_count(
                        feature_set.labels,
                        shots=int(shots),
                        desired_queries=int(config["query_per_class"]),
                    )
                    for resample in range(int(config["support_resamples"])):
                        episode_started = time.perf_counter()
                        episode_rows, bundle = evaluate_support_risk_episode(
                            experts,
                            references,
                            feature_set.features,
                            feature_set.labels,
                            shots=int(shots),
                            queries_per_class=queries,
                            seed=int(split_seed) * 100_000 + int(train_seed) * 1_000 + int(shots),
                            repetition=resample,
                            prototype_temperature=float(model_config["temperature"]),
                            original_route_temperature=float(gate2_config["routing"]["route_temperature"]),
                            risk_temperature=float(config["risk_estimator"]["risk_temperature"]),
                            prior_strength=float(
                                config["risk_estimator"]["shrinkage_prior_strength"]
                            ),
                            variance_scale=float(
                                config["risk_estimator"]["shrinkage_variance_scale"]
                            ),
                            shared_fallback_weight=float(
                                config["risk_estimator"]["shared_fallback_weight"]
                            ),
                            device=device,
                        )
                        common = {
                            "split_seed": int(split_seed),
                            "split_hash": split.split_hash,
                            "train_seed": int(train_seed),
                            "task": str(task),
                            "shots": int(shots),
                            "support_resample": resample,
                            "episode_hash": bundle.episode_hash,
                            "queries_per_class": queries,
                            "candidate_name_set": "|".join(bundle.candidate_names),
                            "task_id_visible_to_router": False,
                            "episode_wall_seconds": time.perf_counter() - episode_started,
                        }
                        rows.extend([{**common, **row} for row in episode_rows])
                        risk_rows.extend(
                            [{**common, **record} for record in bundle.risk_statistics]
                        )
                        bundles.append((common, bundle))
    lookup = {
        (
            common["split_seed"], common["train_seed"], common["task"],
            common["shots"], common["support_resample"],
        ): bundle
        for common, bundle in bundles
    }
    tasks_by_split = {
        split: sorted({key[2] for key in lookup if key[0] == split})
        for split in config["task_split_seeds"]
    }
    fallback_weight = float(config["risk_estimator"]["shared_fallback_weight"])
    for common, recipient in bundles:
        tasks = tasks_by_split[int(common["split_seed"])]
        other_task = tasks[(tasks.index(str(common["task"])) + 1) % len(tasks)]
        wrong_key = (
            common["split_seed"], common["train_seed"], other_task,
            common["shots"], common["support_resample"],
        )
        shuffle_key = (
            common["split_seed"], common["train_seed"], other_task,
            common["shots"],
            (int(common["support_resample"]) + 1) % int(config["support_resamples"]),
        )
        for method, donor in (
            ("wrong_task_support", lookup[wrong_key]),
            ("support_shuffle", lookup[shuffle_key]),
        ):
            values = evaluate_risk_ablation(
                recipient,
                donor.shrinkage_weights,
                shared_fallback_weight=fallback_weight,
            )
            rows.append(
                {
                    **common,
                    "method": method,
                    "selected_expert": recipient.candidate_names[
                        int(donor.shrinkage_weights.argmax().item())
                    ],
                    "router_input": "mismatched_labeled_support",
                    **values,
                }
            )
    frame = pd.DataFrame(rows)
    decision_core, comparisons, ablations = _decide(frame, config)
    elapsed = time.perf_counter() - started
    decision = {
        "schema_version": 1,
        **decision_core,
        "thresholds": config["go_criteria"],
        "experiment_id": arguments.experiment_id,
        "config_hash": config_hash,
        "runtime_git_sha": runtime_sha,
        "gate2_decision_sha256": GATE2_DECISION_SHA256,
        "checkpoint_hashes": checkpoint_hashes,
        "protected_artifact_hashes": PROTECTED_HASHES,
        "host_gpu": {
            "host": platform.node(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "elapsed_time": elapsed,
        "gpu_hours": elapsed / 3600,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "test_count": arguments.test_count,
        "schema_validation": "VALID",
    }
    schema = json.loads(
        (project_root / "schemas" / "support_risk_pilot_decision.schema.json").read_text(
            encoding="utf-8"
        )
    )
    if set(schema["required"]) - set(decision):
        raise RuntimeError("support-risk decision lacks schema-required fields")
    results = {
        "schema_version": 1,
        "experiment_id": arguments.experiment_id,
        "config_hash": config_hash,
        "runtime_git_sha": runtime_sha,
        "episode_metrics": frame.to_dict(orient="records"),
        "support_risk_statistics": risk_rows,
        "aggregate_method_metrics": _aggregate_metrics(frame),
        "paired_hierarchical_comparisons": comparisons,
        "support_ablation_deltas": ablations,
        "resource_usage": {
            "elapsed_time": elapsed,
            "gpu_hours": elapsed / 3600,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "checkpoint_count": len(checkpoint_hashes),
            "learned_router_parameters": 0,
            "hyperparameter_trials": int(config["risk_estimator"]["hyperparameter_trials"]),
        },
    }
    report_path = project_root / "reports" / "support_risk_pilot_report.md"
    decision_path = project_root / "reports" / "support_risk_pilot_decision.json"
    results_path = project_root / "results" / "support_risk_pilot_results.json"
    figure_path = project_root / "figures" / "support_risk_paired_deltas.png"
    atomic_write_json(decision_path, _sanitize(decision))
    atomic_write_json(results_path, _sanitize(results))
    atomic_write_text(report_path, _report(decision))
    _plot(comparisons, figure_path)
    for relative in config["required_outputs"]:
        if not (project_root / relative).exists():
            raise RuntimeError(f"required support-risk output missing: {relative}")
    run_directory = arguments.run_root / arguments.experiment_id
    atomic_write_json(
        run_directory / "completion.json",
        {
            "status": "SUCCEEDED",
            "outcome": decision["outcome"],
            "runtime_git_sha": runtime_sha,
            "elapsed_time": elapsed,
        },
    )
    print(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "outcome": decision["outcome"],
                "experiment_id": arguments.experiment_id,
                "elapsed_time": elapsed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
