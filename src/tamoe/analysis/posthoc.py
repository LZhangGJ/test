"""Read-only post-hoc methodology audit for the immutable canonical Gate 1 run."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPISODE_KEYS = [
    "split_seed",
    "train_seed",
    "task",
    "shots",
    "support_resample",
    "episode_hash",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_expert_rows(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame.apply(
        lambda row: str(row["method"]) in str(row["expert_name_set"]).split("|"), axis=1
    )
    return frame[mask].copy()


def validate_canonical_inputs(
    frame: pd.DataFrame, matrix: pd.DataFrame, decision: dict[str, Any]
) -> dict[str, Any]:
    if decision.get("status") != "FAIL":
        raise ValueError("M3A requires the immutable canonical Gate 1 FAIL decision")
    required = {
        *EPISODE_KEYS,
        "method",
        "selected_expert",
        "expert_name_set",
        "accuracy",
        "macro_f1",
        "loss",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"canonical episode table lacks columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("canonical episode table contains null audit fields")
    if frame.duplicated([*EPISODE_KEYS, "method"]).any():
        raise ValueError("canonical episode table contains duplicate episode-method rows")
    oracle = frame[frame["method"] == "episode_oracle"]
    if oracle["episode_hash"].nunique() != len(oracle):
        raise ValueError("canonical oracle rows are not unique by episode")
    fixed = fixed_expert_rows(frame)
    recomputed = (
        fixed.groupby(["split_seed", "train_seed", "task", "method"], as_index=False)
        .agg(accuracy_mean=("accuracy", "mean"), loss_mean=("loss", "mean"))
        .sort_values(["split_seed", "train_seed", "task", "method"])
        .reset_index(drop=True)
    )
    expected = (
        matrix[["split_seed", "train_seed", "task", "method", "accuracy_mean", "loss_mean"]]
        .sort_values(["split_seed", "train_seed", "task", "method"])
        .reset_index(drop=True)
    )
    keys_equal = recomputed.iloc[:, :4].equals(expected.iloc[:, :4])
    values_equal = np.allclose(
        recomputed[["accuracy_mean", "loss_mean"]],
        expected[["accuracy_mean", "loss_mean"]],
        rtol=0,
        atol=1e-12,
    )
    if not keys_equal or not values_equal:
        raise ValueError("expert-task matrix does not match canonical episode metrics")
    return {
        "decision_status": "FAIL",
        "episode_count": int(len(oracle)),
        "row_count": int(len(frame)),
        "matrix_consistent": True,
        "schema_valid": True,
    }


def bootstrap_ci(values: Iterable[float], *, repeats: int, seed: int) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise ValueError("cannot bootstrap an empty sample")
    generator = np.random.default_rng(seed)
    sampled = generator.choice(array, size=(repeats, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "n": int(len(array)),
    }


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def episode_rankings(fixed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in fixed.groupby(EPISODE_KEYS, sort=False):
        accuracy_ranked = group.sort_values(
            ["accuracy", "loss", "method"], ascending=[False, True, True]
        )
        loss_ranked = group.sort_values(["loss", "accuracy", "method"], ascending=[True, False, True])
        first_accuracy, second_accuracy = accuracy_ranked.iloc[:2].itertuples(index=False)
        first_loss, second_loss = loss_ranked.iloc[:2].itertuples(index=False)
        rows.append(
            {
                **dict(zip(EPISODE_KEYS, key, strict=True)),
                "top1_expert_accuracy_rank": first_accuracy.method,
                "top2_expert_accuracy_rank": second_accuracy.method,
                "top1_accuracy": float(first_accuracy.accuracy),
                "top2_accuracy": float(second_accuracy.accuracy),
                "accuracy_margin": float(first_accuracy.accuracy - second_accuracy.accuracy),
                "best_loss_expert": first_loss.method,
                "second_best_loss_expert": second_loss.method,
                "best_loss": float(first_loss.loss),
                "second_best_loss": float(second_loss.loss),
                "loss_margin": float(second_loss.loss - first_loss.loss),
                "accuracy_top_tie_count": int((group["accuracy"] == group["accuracy"].max()).sum()),
            }
        )
    return pd.DataFrame(rows)


def pairwise_expert_matrix(
    fixed: pd.DataFrame, *, bootstrap_repeats: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split_seed, task), group in fixed.groupby(["split_seed", "task"], sort=True):
        experts = sorted(group["method"].unique())
        indexed = group.set_index([*EPISODE_KEYS, "method"])
        for expert_a, expert_b in itertools.combinations(experts, 2):
            a = indexed.xs(expert_a, level="method")
            b = indexed.xs(expert_b, level="method")
            joined = a[["accuracy", "loss"]].join(
                b[["accuracy", "loss"]], lsuffix="_a", rsuffix="_b", how="inner"
            )
            accuracy_delta = joined["accuracy_a"] - joined["accuracy_b"]
            loss_delta = joined["loss_b"] - joined["loss_a"]
            accuracy_ci = bootstrap_ci(
                accuracy_delta,
                repeats=bootstrap_repeats,
                seed=_stable_seed(f"pairwise:{split_seed}:{task}:{expert_a}:{expert_b}"),
            )
            rows.append(
                {
                    "split_seed": int(split_seed),
                    "task": str(task),
                    "expert_a": expert_a,
                    "expert_b": expert_b,
                    "episode_count": len(joined),
                    "accuracy_a_win_rate": float((accuracy_delta > 0).mean()),
                    "accuracy_tie_rate": float((accuracy_delta == 0).mean()),
                    "accuracy_b_win_rate": float((accuracy_delta < 0).mean()),
                    "mean_accuracy_a_minus_b": accuracy_ci["mean"],
                    "accuracy_delta_ci_low": accuracy_ci["ci_low"],
                    "accuracy_delta_ci_high": accuracy_ci["ci_high"],
                    "loss_a_win_rate": float((loss_delta > 0).mean()),
                    "loss_tie_rate": float((loss_delta == 0).mean()),
                    "loss_b_win_rate": float((loss_delta < 0).mean()),
                    "mean_loss_b_minus_a": float(loss_delta.mean()),
                    "analysis_only_query_label_derived": True,
                }
            )
    return pd.DataFrame(rows)


def _best_remaining(group: pd.DataFrame, masked_expert: str) -> pd.Series:
    remaining = group[group["method"] != masked_expert]
    if remaining.empty:
        raise ValueError("cannot mask the only expert")
    return remaining.sort_values(
        ["accuracy", "loss", "method"], ascending=[False, True, True]
    ).iloc[0]


def leave_one_expert_out(
    fixed: pd.DataFrame, oracle: pd.DataFrame, *, bootstrap_repeats: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    oracle_lookup = oracle.set_index(EPISODE_KEYS)
    detail: list[dict[str, Any]] = []
    for key, group in fixed.groupby(EPISODE_KEYS, sort=False):
        canonical = oracle_lookup.loc[key]
        full = group.sort_values(
            ["accuracy", "loss", "method"], ascending=[False, True, True]
        ).iloc[0]
        for expert in sorted(group["method"]):
            remaining = _best_remaining(group, expert)
            detail.append(
                {
                    **dict(zip(EPISODE_KEYS, key, strict=True)),
                    "masked_expert": expert,
                    "canonical_oracle_expert": canonical["selected_expert"],
                    "full_oracle_expert_loss_tiebreak": full["method"],
                    "remaining_oracle_expert": remaining["method"],
                    "accuracy_drop": float(full["accuracy"] - remaining["accuracy"]),
                    "loss_increase": float(remaining["loss"] - full["loss"]),
                    "masked_was_canonical_oracle": bool(canonical["selected_expert"] == expert),
                    "masked_was_loss_tiebreak_oracle": bool(full["method"] == expert),
                }
            )
    detail_frame = pd.DataFrame(detail)
    summaries: list[dict[str, Any]] = []
    for (split_seed, task, expert), group in detail_frame.groupby(
        ["split_seed", "task", "masked_expert"], sort=True
    ):
        accuracy_ci = bootstrap_ci(
            group["accuracy_drop"], repeats=bootstrap_repeats,
            seed=_stable_seed(f"loo:{split_seed}:{task}:{expert}:accuracy"),
        )
        loss_ci = bootstrap_ci(
            group["loss_increase"], repeats=bootstrap_repeats,
            seed=_stable_seed(f"loo:{split_seed}:{task}:{expert}:loss"),
        )
        summaries.append(
            {
                "mask_type": "leave_one_expert_out_unconditional",
                "split_seed": int(split_seed),
                "task": str(task),
                "masked_expert": expert,
                "episode_count": len(group),
                "conditional_episode_count": int(group["masked_was_canonical_oracle"].sum()),
                "affected_accuracy_episode_count": int((group["accuracy_drop"] > 0).sum()),
                "mean_accuracy_drop": accuracy_ci["mean"],
                "accuracy_drop_ci_low": accuracy_ci["ci_low"],
                "accuracy_drop_ci_high": accuracy_ci["ci_high"],
                "mean_loss_increase": loss_ci["mean"],
                "loss_increase_ci_low": loss_ci["ci_low"],
                "loss_increase_ci_high": loss_ci["ci_high"],
                "analysis_only_query_label_derived": True,
            }
        )

    global_expert = str(oracle["selected_expert"].value_counts().index[0])
    task_modal = (
        oracle.groupby(["split_seed", "task"])["selected_expert"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )

    def conditional_summary(mask_type: str, masked_by_task: dict[tuple[int, str], str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for (split_seed, task), masked_expert in masked_by_task.items():
            group = detail_frame[
                (detail_frame["split_seed"] == split_seed)
                & (detail_frame["task"] == task)
                & (detail_frame["masked_expert"] == masked_expert)
                & detail_frame["masked_was_canonical_oracle"]
            ]
            if group.empty:
                continue
            accuracy_ci = bootstrap_ci(
                group["accuracy_drop"], repeats=bootstrap_repeats,
                seed=_stable_seed(f"{mask_type}:{split_seed}:{task}:accuracy"),
            )
            loss_ci = bootstrap_ci(
                group["loss_increase"], repeats=bootstrap_repeats,
                seed=_stable_seed(f"{mask_type}:{split_seed}:{task}:loss"),
            )
            output.append(
                {
                    "mask_type": mask_type,
                    "split_seed": int(split_seed),
                    "task": str(task),
                    "masked_expert": masked_expert,
                    "episode_count": len(group),
                    "conditional_episode_count": len(group),
                    "affected_accuracy_episode_count": int((group["accuracy_drop"] > 0).sum()),
                    "mean_accuracy_drop": accuracy_ci["mean"],
                    "accuracy_drop_ci_low": accuracy_ci["ci_low"],
                    "accuracy_drop_ci_high": accuracy_ci["ci_high"],
                    "mean_loss_increase": loss_ci["mean"],
                    "loss_increase_ci_low": loss_ci["ci_low"],
                    "loss_increase_ci_high": loss_ci["ci_high"],
                    "analysis_only_query_label_derived": True,
                }
            )
        return output

    task_keys = sorted(task_modal)
    summaries.extend(
        conditional_summary(
            "global_most_used_conditional",
            {key: global_expert for key in task_keys},
        )
    )
    summaries.extend(conditional_summary("task_modal_conditional", task_modal))
    semantics = {
        "global_most_used_expert": global_expert,
        "task_modal_experts": {f"{key[0]}:{key[1]}": value for key, value in task_modal.items()},
        "conditional_mask_semantics": (
            "Restrict to episodes where the masked expert was the stored canonical episode oracle; "
            "then use query-label accuracy, loss, and method name as descending/ascending/ascending "
            "tie-breaks to choose the best remaining expert."
        ),
    }
    return pd.DataFrame(summaries), semantics


def random_mean_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[frame["method"].str.startswith("random_")]
        .groupby(EPISODE_KEYS, as_index=False)[["accuracy", "macro_f1", "loss"]]
        .mean()
        .assign(method="random_mean")
    )


def paired_effects(frame: pd.DataFrame, *, bootstrap_repeats: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    random_mean = random_mean_rows(frame)
    selected = pd.concat(
        [
            frame[frame["method"].isin(["episode_oracle", "shared", "single", "oracle_mixture"])],
            random_mean,
        ],
        ignore_index=True,
    )
    wide_accuracy = selected.pivot(index=EPISODE_KEYS, columns="method", values="accuracy")
    wide_loss = selected.pivot(index=EPISODE_KEYS, columns="method", values="loss")
    comparisons = ["shared", "single", "random_mean"]
    rows: list[dict[str, Any]] = []
    for (split_seed, task), indices in wide_accuracy.groupby(level=["split_seed", "task"]).groups.items():
        accuracy_group = wide_accuracy.loc[indices]
        loss_group = wide_loss.loc[indices]
        for baseline in comparisons:
            accuracy_delta = accuracy_group["episode_oracle"] - accuracy_group[baseline]
            loss_delta = loss_group[baseline] - loss_group["episode_oracle"]
            accuracy_ci = bootstrap_ci(
                accuracy_delta, repeats=bootstrap_repeats,
                seed=_stable_seed(f"task-effect:{split_seed}:{task}:{baseline}:accuracy"),
            )
            loss_ci = bootstrap_ci(
                loss_delta, repeats=bootstrap_repeats,
                seed=_stable_seed(f"task-effect:{split_seed}:{task}:{baseline}:loss"),
            )
            rows.append(
                {
                    "split_seed": int(split_seed),
                    "task": str(task),
                    "baseline": baseline,
                    "mean_accuracy_advantage": accuracy_ci["mean"],
                    "accuracy_ci_low": accuracy_ci["ci_low"],
                    "accuracy_ci_high": accuracy_ci["ci_high"],
                    "mean_loss_advantage": loss_ci["mean"],
                    "loss_ci_low": loss_ci["ci_low"],
                    "loss_ci_high": loss_ci["ci_high"],
                    "episode_count": accuracy_ci["n"],
                }
            )
    mixture_accuracy = wide_accuracy["oracle_mixture"] - wide_accuracy["episode_oracle"]
    mixture_loss = wide_loss["episode_oracle"] - wide_loss["oracle_mixture"]
    mixture = {
        "mixture_minus_hard_oracle_accuracy": bootstrap_ci(
            mixture_accuracy, repeats=bootstrap_repeats, seed=_stable_seed("mixture:accuracy")
        ),
        "hard_oracle_minus_mixture_loss": bootstrap_ci(
            mixture_loss, repeats=bootstrap_repeats, seed=_stable_seed("mixture:loss")
        ),
        "mixture_accuracy_win_rate": float((mixture_accuracy > 0).mean()),
        "mixture_accuracy_tie_rate": float((mixture_accuracy == 0).mean()),
    }
    return pd.DataFrame(rows), mixture


def _cluster_bootstrap(
    values: pd.Series, cluster_labels: pd.Series, *, repeats: int, seed: int
) -> dict[str, float | int]:
    clusters = {
        label: values[cluster_labels == label].to_numpy(dtype=np.float64)
        for label in sorted(cluster_labels.unique())
    }
    names = list(clusters)
    generator = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = generator.choice(names, size=len(names), replace=True)
        estimates[index] = np.concatenate([clusters[name] for name in sampled]).mean()
    array = values.to_numpy(dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "cluster_count": len(names),
        "episode_count": len(array),
    }


def _hierarchical_bootstrap(
    values: pd.Series, *, repeats: int, seed: int
) -> dict[str, float | int]:
    table = values.rename("delta").reset_index()
    structure: dict[int, dict[str, dict[int, np.ndarray]]] = {}
    for (split_seed, task, train_seed), group in table.groupby(
        ["split_seed", "task", "train_seed"]
    ):
        structure.setdefault(int(split_seed), {}).setdefault(str(task), {})[int(train_seed)] = (
            group["delta"].to_numpy(dtype=np.float64)
        )
    splits = sorted(structure)
    generator = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled_values: list[np.ndarray] = []
        for split_seed in generator.choice(splits, size=len(splits), replace=True):
            tasks = sorted(structure[int(split_seed)])
            for task in generator.choice(tasks, size=len(tasks), replace=True):
                seeds = sorted(structure[int(split_seed)][str(task)])
                for train_seed in generator.choice(seeds, size=len(seeds), replace=True):
                    episodes = structure[int(split_seed)][str(task)][int(train_seed)]
                    sampled_values.append(
                        generator.choice(episodes, size=len(episodes), replace=True)
                    )
        estimates[repeat] = np.concatenate(sampled_values).mean()
    array = values.to_numpy(dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "split_count": len(splits),
        "task_cluster_count": int(table.groupby(["split_seed", "task"]).ngroups),
        "train_seed_cluster_count": int(
            table.groupby(["split_seed", "task", "train_seed"]).ngroups
        ),
        "episode_count": len(array),
    }


def multilevel_bootstrap(frame: pd.DataFrame, *, repeats: int) -> dict[str, Any]:
    random_mean = random_mean_rows(frame)
    selected = pd.concat(
        [frame[frame["method"].isin(["episode_oracle", "shared", "single"])], random_mean],
        ignore_index=True,
    )
    wide = selected.pivot(index=EPISODE_KEYS, columns="method", values="accuracy")
    output: dict[str, Any] = {}
    index_frame = wide.index.to_frame(index=False)
    for baseline in ("shared", "single", "random_mean"):
        delta = wide["episode_oracle"] - wide[baseline]
        labels = {
            "task_cluster": index_frame["split_seed"].astype(str)
            + ":"
            + index_frame["task"].astype(str),
            "split_cluster": index_frame["split_seed"].astype(str),
            "train_seed_cluster": index_frame["split_seed"].astype(str)
            + ":"
            + index_frame["task"].astype(str)
            + ":"
            + index_frame["train_seed"].astype(str),
        }
        delta = pd.Series(delta.to_numpy(), index=wide.index)
        output[baseline] = {
            "episode_bootstrap": bootstrap_ci(
                delta, repeats=repeats, seed=_stable_seed(f"multilevel:{baseline}:episode")
            ),
            **{
                name: _cluster_bootstrap(
                    delta.reset_index(drop=True), label.reset_index(drop=True), repeats=repeats,
                    seed=_stable_seed(f"multilevel:{baseline}:{name}"),
                )
                for name, label in labels.items()
            },
            "hierarchical_bootstrap": _hierarchical_bootstrap(
                delta, repeats=repeats, seed=_stable_seed(f"multilevel:{baseline}:hierarchical")
            ),
        }
    return output


def ranking_stability(fixed: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = ["train_seed", "shots", "support_resample"]
    for (split_seed, task), task_rows in fixed.groupby(["split_seed", "task"], sort=True):
        task_rankings = rankings[
            (rankings["split_seed"] == split_seed) & (rankings["task"] == task)
        ]
        audit_modal = str(task_rankings["top1_expert_accuracy_rank"].value_counts().index[0])
        for dimension in dimensions:
            rank_vectors: list[pd.Series] = []
            level_best: list[str] = []
            for _, level_rows in task_rows.groupby(dimension, sort=True):
                aggregate = level_rows.groupby("method").agg(
                    accuracy=("accuracy", "mean"), loss=("loss", "mean")
                )
                ordered = aggregate.sort_values(
                    ["accuracy", "loss"], ascending=[False, True]
                )
                level_best.append(str(ordered.index[0]))
                rank_vectors.append(aggregate["accuracy"].rank(ascending=False, method="average"))
            correlations = [
                float(left.corr(right, method="spearman"))
                for left, right in itertools.combinations(rank_vectors, 2)
            ]
            correlations = [value for value in correlations if math.isfinite(value)]
            rows.append(
                {
                    "split_seed": int(split_seed),
                    "task": str(task),
                    "condition_dimension": dimension,
                    "level_count": len(level_best),
                    "audit_episode_modal_expert_loss_tiebreak": audit_modal,
                    "level_modal_expert": pd.Series(level_best).value_counts().index[0],
                    "level_modal_consistency": float(pd.Series(level_best).value_counts(normalize=True).iloc[0]),
                    "distinct_level_best_experts": len(set(level_best)),
                    "mean_pairwise_spearman_rank": float(np.mean(correlations)) if correlations else 1.0,
                    "minimum_pairwise_spearman_rank": min(correlations) if correlations else 1.0,
                    "mean_top1_top2_accuracy_margin": float(task_rankings["accuracy_margin"].mean()),
                    "median_top1_top2_accuracy_margin": float(task_rankings["accuracy_margin"].median()),
                    "near_tie_rate_accuracy_margin_le_0_01": float(
                        (task_rankings["accuracy_margin"] <= 0.01).mean()
                    ),
                    "mean_best_second_loss_margin": float(task_rankings["loss_margin"].mean()),
                    "accuracy_top_tie_rate": float((task_rankings["accuracy_top_tie_count"] > 1).mean()),
                }
            )
    return pd.DataFrame(rows)
