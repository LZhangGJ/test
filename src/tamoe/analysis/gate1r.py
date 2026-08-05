"""Confirmatory Gate 1R evaluation and frozen three-way decision logic."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional

from tamoe.analysis.gate1 import macro_f1, oracle_convex_mixture, sample_oracle_metrics
from tamoe.analysis.oracle import deterministic_accuracy_oracle, epsilon_optimal_experts
from tamoe.episodes.sampler import index_labels, sample_episode_indices
from tamoe.experts.adapters import ResidualAdapter
from tamoe.models.episodic_head import prototype_logits

EPISODE_KEYS = [
    "split_seed", "train_seed", "task", "shots", "support_resample", "episode_hash"
]


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def _metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    prediction = logits.argmax(dim=-1)
    return {
        "accuracy": float((prediction == labels).float().mean().item()),
        "macro_f1": macro_f1(prediction, labels),
        "loss": float(functional.cross_entropy(logits, labels).item()),
    }


def _selected_row(
    name: str, selected: str, metrics: dict[str, dict[str, float]]
) -> dict[str, Any]:
    return {"method": name, "selected_expert": selected, **metrics[selected]}


def evaluate_confirmatory_episode(
    experts: dict[str, ResidualAdapter],
    features: Tensor,
    labels: Tensor,
    *,
    shots: int,
    queries_per_class: int,
    seed: int,
    repetition: int,
    temperature: float,
    device: torch.device,
    accuracy_tie_tolerance: float,
    epsilon_accuracy: float,
    random_repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode = sample_episode_indices(
        index_labels(labels.tolist()), shots=shots, queries_per_class=queries_per_class,
        seed=seed, repetition=repetition,
    )
    support_index = torch.tensor(episode.support_indices, dtype=torch.long)
    query_index = torch.tensor(episode.query_indices, dtype=torch.long)
    support = features[support_index].to(device)
    query = features[query_index].to(device)
    support_labels = torch.tensor(episode.support_labels, dtype=torch.long, device=device)
    query_labels = torch.tensor(episode.query_labels, dtype=torch.long, device=device)
    names = sorted(experts)
    logits_by_name: dict[str, Tensor] = {}
    metrics: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for name in names:
            logits = prototype_logits(
                experts[name](support), support_labels, experts[name](query),
                temperature=temperature,
            )
            logits_by_name[name] = logits
            metrics[name] = _metrics(logits, query_labels)
    candidates = tuple(name for name in names if name == "shared" or name.startswith("source_"))
    specialists = tuple(name for name in candidates if name.startswith("source_"))
    if "single" not in names or "shared" not in names or not specialists:
        raise ValueError("Gate 1R bank lacks shared, single, or source experts")

    def oracle(scope: tuple[str, ...]) -> str:
        result = deterministic_accuracy_oracle(
            scope,
            torch.tensor([metrics[name]["accuracy"] for name in scope], dtype=torch.float64),
            torch.tensor([metrics[name]["loss"] for name in scope], dtype=torch.float64),
            accuracy_tie_tolerance=accuracy_tie_tolerance,
        )
        return result.expert_name

    primary = oracle(candidates)
    full = oracle(tuple(names))
    specialist = oracle(specialists)
    minimum_nll = min(candidates, key=lambda name: (metrics[name]["loss"], name))
    accuracy_values = torch.tensor([metrics[name]["accuracy"] for name in candidates])
    maximum = float(accuracy_values.max())
    tied = [
        name for name in candidates
        if maximum - metrics[name]["accuracy"] <= accuracy_tie_tolerance
    ]
    sorted_accuracy = sorted(
        candidates, key=lambda name: (-metrics[name]["accuracy"], metrics[name]["loss"], name)
    )
    epsilon_set = epsilon_optimal_experts(
        candidates, accuracy_values.double(), epsilon_accuracy=epsilon_accuracy
    )
    rows = [
        {"method": name, "selected_expert": name, **metrics[name]} for name in names
    ]
    rows.extend(
        [
            _selected_row("router_candidate_oracle", primary, metrics),
            _selected_row("full_analysis_oracle", full, metrics),
            _selected_row("specialist_only_oracle", specialist, metrics),
            _selected_row("minimum_nll_oracle", minimum_nll, metrics),
        ]
    )
    candidate_logits = torch.stack([logits_by_name[name] for name in candidates])
    mixture_metrics, mixture_weights = oracle_convex_mixture(candidate_logits, query_labels)
    rows.append({"method": "oracle_mixture", "selected_expert": "convex", **mixture_metrics})
    rows.append(
        {"method": "sample_oracle", "selected_expert": "per_sample",
         **sample_oracle_metrics(candidate_logits, query_labels)}
    )
    exact_mean = {
        metric: float(np.mean([metrics[name][metric] for name in candidates]))
        for metric in ("accuracy", "macro_f1", "loss")
    }
    rows.append({"method": "random_expected", "selected_expert": "exact_mean", **exact_mean})
    worst = min(
        candidates, key=lambda name: (metrics[name]["accuracy"], -metrics[name]["loss"], name)
    )
    rows.append(_selected_row("forced_worst", worst, metrics))
    generator = random.Random(f"gate1r:{seed}:{repetition}:{shots}")
    for repeat in range(random_repeats):
        selected = candidates[generator.randrange(len(candidates))]
        rows.append(_selected_row(f"repeated_random_{repeat:02d}", selected, metrics))
    audit = {
        "episode_hash": episode.episode_hash,
        "candidate_names": candidates,
        "specialist_names": specialists,
        "epsilon_optimal_experts": epsilon_set,
        "epsilon_optimal_size": len(epsilon_set),
        "top1_expert": sorted_accuracy[0],
        "top2_expert": sorted_accuracy[1],
        "top1_top2_accuracy_margin": metrics[sorted_accuracy[0]]["accuracy"]
        - metrics[sorted_accuracy[1]]["accuracy"],
        "exact_accuracy_tie": len(tied) > 1,
        "primary_oracle": primary,
        "mixture_weights": dict(zip(candidates, mixture_weights, strict=True)),
    }
    return rows, audit


def _fixed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.apply(
        lambda row: str(row["method"]) in str(row["expert_name_set"]).split("|"), axis=1
    )].copy()


def add_confirmatory_interventions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    fixed = _fixed(frame)
    primary = frame[frame["method"] == "router_candidate_oracle"]
    global_modal = sorted(
        primary["selected_expert"].value_counts().items(), key=lambda item: (-item[1], item[0])
    )[0][0]
    task_modal = {
        key: sorted(values.value_counts().items(), key=lambda item: (-item[1], item[0]))[0][0]
        for key, values in primary.groupby(["split_seed", "task"])["selected_expert"]
    }
    tasks_by_split = {
        int(split): sorted(values.unique())
        for split, values in primary.groupby("split_seed")["task"]
    }
    additions: list[dict[str, Any]] = []
    excluded = {"method", "selected_expert", "accuracy", "macro_f1", "loss"}
    for key, group in fixed.groupby(EPISODE_KEYS, sort=False):
        base = dict(zip(EPISODE_KEYS, key, strict=True))
        split_seed, _train_seed, task, _shots, _resample, _hash = key
        candidates = str(group["router_candidate_name_set"].iloc[0]).split("|")
        records = {str(row.method): row for row in group.itertuples(index=False)}
        common = {
            column: getattr(next(iter(records.values())), column)
            for column in frame.columns if column not in {*EPISODE_KEYS, *excluded}
        }

        def append(
            method: str,
            selected: str,
            *,
            episode_records: dict[str, Any] = records,
            episode_base: dict[str, Any] = base,
            episode_common: dict[str, Any] = common,
        ) -> None:
            record = episode_records[selected]
            additions.append({
                **episode_base, **episode_common, "method": method,
                "selected_expert": selected,
                "accuracy": record.accuracy, "macro_f1": record.macro_f1, "loss": record.loss,
            })

        modal = task_modal[(int(split_seed), str(task))]
        modal_index = candidates.index(modal)
        append("task_modal_swap", candidates[(modal_index + 1) % len(candidates)])
        other_tasks = tasks_by_split[int(split_seed)]
        other_task = other_tasks[(other_tasks.index(str(task)) + 1) % len(other_tasks)]
        wrong = task_modal[(int(split_seed), other_task)]
        append("task_assignment_permutation", wrong)
        append("wrong_task_expert_assignment", wrong)
        remaining_global = [name for name in candidates if name != global_modal]
        if global_modal in candidates:
            best = min(
                remaining_global,
                key=lambda name: (-records[name].accuracy, records[name].loss, name),
            )
            append("global_most_used_mask", best)
            primary_name = frame[
                (frame["method"] == "router_candidate_oracle")
                & np.logical_and.reduce([frame[col] == value for col, value in base.items()])
            ]["selected_expert"].iloc[0]
            if primary_name == global_modal:
                append("conditional_global_mask", best)
        remaining_modal = [name for name in candidates if name != modal]
        primary_name = frame[
            (frame["method"] == "router_candidate_oracle")
            & np.logical_and.reduce([frame[col] == value for col, value in base.items()])
        ]["selected_expert"].iloc[0]
        if primary_name == modal:
            best = min(
                remaining_modal,
                key=lambda name: (-records[name].accuracy, records[name].loss, name),
            )
            append("task_modal_conditional_mask", best)
    metadata = {
        "global_most_used_path": global_modal,
        "task_modal_paths": {f"{key[0]}:{key[1]}": value for key, value in task_modal.items()},
    }
    return pd.concat([frame, pd.DataFrame(additions)], ignore_index=True), metadata


def _nested_tree(frame: pd.DataFrame, value: str, levels: list[str]) -> Any:
    if not levels:
        return frame[value].to_numpy(dtype=np.float64)
    return {
        key: _nested_tree(group, value, levels[1:])
        for key, group in frame.groupby(levels[0], sort=True)
    }


def hierarchical_ci(
    frame: pd.DataFrame, value: str, levels: list[str], *, repeats: int, seed: int
) -> dict[str, float | int]:
    tree = _nested_tree(frame, value, levels)
    generator = np.random.default_rng(seed)

    def draw(node: Any) -> float:
        if isinstance(node, np.ndarray):
            return float(generator.choice(node, size=len(node), replace=True).mean())
        keys = list(node)
        sampled = generator.choice(keys, size=len(keys), replace=True)
        return float(np.mean([draw(node[key]) for key in sampled]))

    estimates = np.fromiter((draw(tree) for _ in range(repeats)), dtype=np.float64, count=repeats)
    values = frame[value].to_numpy(dtype=np.float64)
    return {
        "mean": float(values.mean()), "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)), "n": int(len(values)),
    }


def _with_support_episode(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_support_episode"] = (
        result["shots"].astype(str) + ":" + result["support_resample"].astype(str)
    )
    return result


def _paired(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    primary = frame[frame["method"] == "router_candidate_oracle"][EPISODE_KEYS + ["accuracy"]]
    other = frame[frame["method"] == baseline][EPISODE_KEYS + ["accuracy"]]
    joined = primary.merge(other, on=EPISODE_KEYS, suffixes=("_oracle", "_baseline"), validate="one_to_one")
    joined["delta"] = joined["accuracy_oracle"] - joined["accuracy_baseline"]
    return joined


def intervention_effects(frame: pd.DataFrame, *, repeats: int) -> pd.DataFrame:
    repeated = (
        frame[frame["method"].str.startswith("repeated_random_")]
        .groupby(EPISODE_KEYS, as_index=False)[["accuracy", "macro_f1", "loss"]].mean()
        .assign(method="repeated_random_mean")
    )
    analysis = pd.concat([frame, repeated], ignore_index=True)
    methods = [
        "forced_worst", "random_expected", "repeated_random_mean", "task_modal_swap",
        "task_assignment_permutation", "wrong_task_expert_assignment",
        "global_most_used_mask", "conditional_global_mask", "task_modal_conditional_mask",
    ]
    primary = analysis[analysis["method"] == "router_candidate_oracle"]
    rows: list[dict[str, Any]] = []
    for method in methods:
        selected = analysis[analysis["method"] == method]
        joined = primary[EPISODE_KEYS + ["accuracy", "loss"]].merge(
            selected[EPISODE_KEYS + ["accuracy", "loss"]], on=EPISODE_KEYS,
            suffixes=("_oracle", "_selected"), how="inner", validate="one_to_one",
        )
        joined["accuracy_drop"] = joined["accuracy_oracle"] - joined["accuracy_selected"]
        joined["loss_increase"] = joined["loss_selected"] - joined["loss_oracle"]
        joined = _with_support_episode(joined)
        for (split_seed, task), group in joined.groupby(["split_seed", "task"], sort=True):
            acc = hierarchical_ci(
                group, "accuracy_drop", ["train_seed", "_support_episode"], repeats=repeats,
                seed=_seed(f"gate1r:int:{method}:{split_seed}:{task}:acc"),
            )
            nll = hierarchical_ci(
                group, "loss_increase", ["train_seed", "_support_episode"], repeats=repeats,
                seed=_seed(f"gate1r:int:{method}:{split_seed}:{task}:nll"),
            )
            rows.append({
                "method": method, "split_seed": int(split_seed), "task": str(task),
                "mean_accuracy_drop": acc["mean"], "accuracy_drop_ci_low": acc["ci_low"],
                "accuracy_drop_ci_high": acc["ci_high"], "mean_nll_increase": nll["mean"],
                "nll_increase_ci_low": nll["ci_low"], "nll_increase_ci_high": nll["ci_high"],
                "episode_count": acc["n"], "analysis_only_query_label_derived": True,
            })
    return pd.DataFrame(rows)


def ranking_stability(frame: pd.DataFrame) -> pd.DataFrame:
    fixed = _fixed(frame)
    candidates = fixed[fixed.apply(
        lambda row: str(row["method"]) in str(row["router_candidate_name_set"]).split("|"), axis=1
    )]
    primary = frame[frame["method"] == "router_candidate_oracle"]
    rows: list[dict[str, Any]] = []
    for (split_seed, task), group in candidates.groupby(["split_seed", "task"], sort=True):
        task_primary = primary[(primary["split_seed"] == split_seed) & (primary["task"] == task)]
        seed_modals = []
        for _seed_value, seed_rows in task_primary.groupby("train_seed"):
            seed_modals.append(sorted(
                seed_rows["selected_expert"].value_counts().items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0])
        vectors = []
        for (_shots, _resample), condition in group.groupby(["shots", "support_resample"]):
            vectors.append(condition.groupby("method")["accuracy"].mean().rank(ascending=False))
        correlations = [
            float(a.corr(b, method="spearman"))
            for a, b in itertools.combinations(vectors, 2)
        ]
        correlations = [value for value in correlations if math.isfinite(value)]
        if not correlations:
            correlations = [0.0]
        rows.append({
            "split_seed": int(split_seed), "task": str(task),
            "train_seed_modal_paths": "|".join(seed_modals),
            "modal_consistent_two_of_three": max(seed_modals.count(name) for name in set(seed_modals)) >= 2,
            "median_pairwise_spearman_shot_support": float(np.median(correlations)),
            "mean_pairwise_spearman_shot_support": float(np.mean(correlations)),
            "minimum_pairwise_spearman_shot_support": float(min(correlations)),
            "condition_count": len(vectors),
        })
    return pd.DataFrame(rows)


def leave_one_out(frame: pd.DataFrame, *, repeats: int) -> pd.DataFrame:
    fixed = _fixed(frame)
    details = []
    for key, group in fixed.groupby(EPISODE_KEYS, sort=False):
        candidates = str(group["router_candidate_name_set"].iloc[0]).split("|")
        records = group.set_index("method")

        def best(scope: list[str], *, episode_records: pd.DataFrame = records) -> str:
            return deterministic_accuracy_oracle(
                scope,
                torch.tensor(
                    [episode_records.loc[name, "accuracy"] for name in scope],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [episode_records.loc[name, "loss"] for name in scope],
                    dtype=torch.float64,
                ),
            ).expert_name

        full = best(candidates)
        for masked in candidates:
            remaining = [name for name in candidates if name != masked]
            chosen = best(remaining)
            details.append({
                **dict(zip(EPISODE_KEYS, key, strict=True)), "scope": "router_candidate",
                "masked_expert": masked, "accuracy_drop": float(
                    records.loc[full, "accuracy"] - records.loc[chosen, "accuracy"]
                ), "nll_increase": float(records.loc[chosen, "loss"] - records.loc[full, "loss"]),
            })
        specialists = [name for name in candidates if name.startswith("source_")]
        full_specialist = best(specialists)
        for masked in specialists:
            remaining = [name for name in specialists if name != masked]
            chosen = best(remaining)
            details.append({
                **dict(zip(EPISODE_KEYS, key, strict=True)), "scope": "specialist_only",
                "masked_expert": masked, "accuracy_drop": float(
                    records.loc[full_specialist, "accuracy"] - records.loc[chosen, "accuracy"]
                ), "nll_increase": float(
                    records.loc[chosen, "loss"] - records.loc[full_specialist, "loss"]
                ),
            })
    detail = _with_support_episode(pd.DataFrame(details))
    rows = []
    for (scope, split_seed, task, masked), group in detail.groupby(
        ["scope", "split_seed", "task", "masked_expert"], sort=True
    ):
        acc = hierarchical_ci(
            group, "accuracy_drop", ["train_seed", "_support_episode"], repeats=repeats,
            seed=_seed(f"gate1r:loo:{scope}:{split_seed}:{task}:{masked}:acc"),
        )
        rows.append({
            "scope": scope, "split_seed": int(split_seed), "task": str(task),
            "masked_expert": masked, "mean_accuracy_drop": acc["mean"],
            "accuracy_drop_ci_low": acc["ci_low"], "accuracy_drop_ci_high": acc["ci_high"],
            "mean_nll_increase": float(group["nll_increase"].mean()), "episode_count": acc["n"],
            "analysis_only_query_label_derived": True,
        })
    return pd.DataFrame(rows)


def decide_gate1r(
    frame: pd.DataFrame, effects: pd.DataFrame, stability: pd.DataFrame,
    config: dict[str, Any], *, repeats: int,
) -> dict[str, Any]:
    thresholds = config["shared_requirements"]
    split_stats: dict[str, Any] = {}
    task_stats: dict[str, Any] = {}
    for baseline in ("shared", "single", "random_expected"):
        paired = _with_support_episode(_paired(frame, baseline))
        split_stats[baseline] = {}
        for split_seed, group in paired.groupby("split_seed"):
            split_stats[baseline][str(split_seed)] = hierarchical_ci(
                group, "delta", ["task", "train_seed", "_support_episode"], repeats=repeats,
                seed=_seed(f"gate1r:split:{baseline}:{split_seed}"),
            )
        task_stats[baseline] = {
            f"{split}:{task}": float(group["delta"].mean())
            for (split, task), group in paired.groupby(["split_seed", "task"])
        }
    h = thresholds["oracle_headroom"]
    shared_bool = {
        "A1_each_split_oracle_shared_mean": all(
            value["mean"] > h["each_split_delta_oracle_shared_mean_gt"]
            for value in split_stats["shared"].values()
        ),
        "A1_each_split_oracle_single_mean": all(
            value["mean"] > h["each_split_delta_oracle_single_mean_gt"]
            for value in split_stats["single"].values()
        ),
        "A1_each_split_hierarchical_ci": all(
            value["ci_low"] > h["each_split_hierarchical_ci_low_gt"]
            for baseline in ("shared", "single") for value in split_stats[baseline].values()
        ),
        "A1_positive_tasks_oracle_shared": sum(
            value > 0 for value in task_stats["shared"].values()
        ) >= h["minimum_positive_task_count_delta_oracle_shared"],
        "A1_positive_tasks_oracle_single": sum(
            value > 0 for value in task_stats["single"].values()
        ) >= h["minimum_positive_task_count_delta_oracle_single"],
    }
    primary = frame[frame["method"] == "router_candidate_oracle"]
    frequencies = primary["selected_expert"].value_counts(normalize=True)
    task_modals = primary.groupby(["split_seed", "task"])["selected_expert"].agg(
        lambda values: sorted(values.value_counts().items(), key=lambda item: (-item[1], item[0]))[0][0]
    )
    d = thresholds["no_global_path_dominance"]
    shared_bool.update({
        "A2_maximum_best_frequency": float(frequencies.max())
        < d["maximum_primary_oracle_best_frequency_lt"],
        "A2_minimum_frequent_paths": int((frequencies >= d["path_frequency_threshold"]).sum())
        >= d["minimum_paths_with_frequency_gte"],
        "A2_task_modal_diversity": len(set(task_modals)) > 1,
    })
    core = thresholds["core_intervention_evidence"]

    def passing_tasks(method: str) -> int:
        rows = effects[effects["method"] == method]
        return int(((rows["mean_accuracy_drop"] > core["passing_task_mean_accuracy_drop_gt"])
                    & (rows["accuracy_drop_ci_low"]
                       > core["passing_task_hierarchical_ci_low_gt"])).sum())

    assignment_counts = {
        method: passing_tasks(method) for method in core["assignment_passes_if_either"]
    }
    pattern_ranges = {
        method: float(rows["mean_accuracy_drop"].max() - rows["mean_accuracy_drop"].min())
        for method, rows in effects[
            effects["method"].isin(core["task_dependent_pattern_applies_to"])
        ].groupby("method")
    }
    shared_bool.update({
        "A3_forced_worst": passing_tasks("forced_worst")
        >= core["forced_worst_minimum_passing_tasks"],
        "A3_random_expected": passing_tasks("random_expected")
        >= core["random_expected_minimum_passing_tasks"],
        "A3_assignment": max(assignment_counts.values())
        >= core["assignment_minimum_passing_tasks"],
        "A3_task_dependent_pattern": max(pattern_ranges.values())
        > core["task_dependent_pattern_minimum_range_gt"],
    })
    epsilon = config["pass_hard_additional_requirements"]["margin_epsilon"]
    episode_audit = primary[[*EPISODE_KEYS, "top1_top2_accuracy_margin", "exact_accuracy_tie"]]
    hard_raw = {
        "episode_fraction_margin_gt_epsilon": float(
            (episode_audit["top1_top2_accuracy_margin"] > epsilon).mean()
        ),
        "tasks_modal_consistent_two_of_three": int(stability["modal_consistent_two_of_three"].sum()),
        "median_within_task_spearman": float(
            stability["median_pairwise_spearman_shot_support"].median()
        ),
        "exact_accuracy_tie_rate": float(episode_audit["exact_accuracy_tie"].mean()),
    }
    hard_thresholds = config["pass_hard_additional_requirements"]
    hard_bool = {
        "B1_episode_margin": hard_raw["episode_fraction_margin_gt_epsilon"]
        >= hard_thresholds["minimum_episode_fraction_margin_gt_epsilon"],
        "B2_train_seed_modal_consistency": hard_raw["tasks_modal_consistent_two_of_three"]
        >= hard_thresholds["minimum_tasks_modal_consistent_two_of_three_train_seeds"],
        "B3_ranking_spearman": hard_raw["median_within_task_spearman"]
        >= hard_thresholds["minimum_median_within_task_spearman"],
        "B4_exact_tie_rate": hard_raw["exact_accuracy_tie_rate"]
        < hard_thresholds["maximum_exact_accuracy_tie_rate_lt"],
    }
    shared_pass = all(shared_bool.values())
    hard_pass = all(hard_bool.values())
    outcome = "PASS_HARD" if shared_pass and hard_pass else "PASS_SOFT" if shared_pass else "FAIL"
    return {
        "outcome": outcome,
        "criteria": {
            "shared_requirements": shared_bool,
            "pass_hard_additional_requirements": hard_bool,
        },
        "raw_criterion_values": {
            "split_paired_statistics": split_stats,
            "task_mean_deltas": task_stats,
            "primary_oracle_frequencies": {str(k): float(v) for k, v in frequencies.items()},
            "maximum_primary_oracle_frequency": float(frequencies.max()),
            "frequent_path_count": int((frequencies >= d["path_frequency_threshold"]).sum()),
            "task_modal_paths": {f"{k[0]}:{k[1]}": v for k, v in task_modals.items()},
            "intervention_passing_task_counts": {
                "forced_worst": passing_tasks("forced_worst"),
                "random_expected": passing_tasks("random_expected"), **assignment_counts,
            },
            "assignment_task_ranges": pattern_ranges,
            **hard_raw,
        },
    }
