"""Generate the M3A read-only audit without changing canonical Gate 1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tamoe.analysis.posthoc import (
    EPISODE_KEYS,
    bootstrap_ci,
    episode_rankings,
    fixed_expert_rows,
    leave_one_expert_out,
    multilevel_bootstrap,
    paired_effects,
    pairwise_expert_matrix,
    ranking_stability,
    sha256_file,
    validate_canonical_inputs,
)
from tamoe.utils.atomic_io import atomic_write_json, atomic_write_text


def _markdown(diagnostics: dict[str, Any]) -> str:
    margins = diagnostics["top1_top2_margins"]
    mixture = diagnostics["hard_vs_soft_mixture"]
    tie_rate = margins["accuracy_margin_le_0_01_rate"]
    hard_soft = (
        "Hard query-label oracle selection remains the accuracy ceiling, while the convex oracle "
        "mixture has materially lower query NLL. Given frequent accuracy near ties and unstable "
        "expert ranks, any later feasibility work should prioritize support-conditioned soft "
        "mixtures for robustness, but must separately verify accuracy. Both quantities here are "
        "analysis-only query-label ceilings, not deployable routers."
        if tie_rate >= 0.25
        else "Margins are usually non-trivial; hard routing remains feasible, while the oracle "
        "mixture is only an analysis-only query-label ceiling."
    )
    lines = [
        "# M3A canonical Gate 1 methodology audit",
        "",
        "## Protection and scope",
        "",
        f"- Canonical experiment: `{diagnostics['canonical_experiment_id']}`",
        "- Canonical decision remains **FAIL** and is not recomputed or reinterpreted as PASS.",
        "- No model was trained and no fresh task split was evaluated.",
        f"- Canonical schema validation: `{diagnostics['validation']}`",
        f"- Source SHA-256: `{diagnostics['source_sha256']}`",
        "",
        "## Oracle margins and near ties",
        "",
        f"- Mean top-1 minus top-2 accuracy margin: `{margins['accuracy_margin']['mean']:.4f}`.",
        f"- Accuracy margin <= 0.01: `{tie_rate:.2%}` of episodes.",
        f"- Exact top-accuracy tie: `{margins['exact_top_accuracy_tie_rate']:.2%}` of episodes.",
        f"- Mean best-versus-second loss margin: `{margins['loss_margin']['mean']:.4f}`.",
        "- Canonical expert identity uses stored episode-oracle selections. For methodology audit "
        "rankings, equal accuracy is resolved by lower query loss and then expert name.",
        f"- Stored canonical identity differs from that explicit loss tie-break on "
        f"`{diagnostics['tie_break_audit']['identity_mismatch_count']}`/"
        f"`{diagnostics['tie_break_audit']['episode_count']}` episodes. Oracle accuracy is unchanged, "
        "but identity-frequency, conditional-mask, and assignment diagnostics can be tie-sensitive.",
        "",
        "## Mask and leave-one-out semantics",
        "",
        f"- The global highest-frequency canonical oracle expert is "
        f"`{diagnostics['mask_semantics']['global_most_used_expert']}`.",
        "- `global_most_used_conditional` deletes that expert only on episodes where it was the "
        "stored canonical episode oracle, then selects the best remaining expert using query-label "
        "accuracy (lower query loss and name break ties).",
        "- This estimates the global expert's marginal indispensability and expert-bank redundancy; "
        "it is not, by itself, a sufficient test of task specificity.",
        "- Leave-one-out, global/task-modal conditional masks, canonical `swap`, "
        "`mask_most_used`, and `permuted_task_assignment` are analysis-only, query-label-derived "
        "diagnostics.",
        f"- Global conditional mask summary: `{diagnostics['conditional_mask_summary']['global']}`.",
        f"- Task-modal conditional mask summary: "
        f"`{diagnostics['conditional_mask_summary']['task_modal']}`.",
        f"- Unconditional leave-one-out summary: `{diagnostics['leave_one_out_summary']}`.",
        "",
        "## Pairwise and task-level effects",
        "",
        f"- Pairwise summary: `{diagnostics['pairwise_summary']}`.",
        f"- Oracle comparison summary: `{diagnostics['task_effect_summary']}`.",
        "",
        "## Bootstrap hierarchy",
        "",
        "Episode bootstrap, task-cluster, split-cluster, train-seed-cluster, and nested "
        "split/task/train-seed/episode bootstrap intervals are reported separately in the JSON. "
        "The two-split interval is necessarily coarse and must not be treated as a large-sample CI.",
        "",
        "## Functional diversity versus near-tie selection",
        "",
        f"- Oracle-shared hierarchical mean/CI: "
        f"`{diagnostics['multilevel_bootstrap']['shared']['hierarchical_bootstrap']}`.",
        f"- Oracle-single hierarchical mean/CI: "
        f"`{diagnostics['multilevel_bootstrap']['single']['hierarchical_bootstrap']}`.",
        "- Positive aggregate headroom coexists with frequent near ties. The bank therefore shows "
        "functional diversity at aggregate level, while many individual oracle identities are "
        "replaceable or tie-sensitive.",
        f"- Rank-stability summary: `{diagnostics['ranking_stability_summary']}`.",
        "",
        "## Hard routing versus soft mixture",
        "",
        f"- Mixture minus hard-oracle accuracy: "
        f"`{mixture['mixture_minus_hard_oracle_accuracy']}`.",
        f"- Hard-oracle minus mixture loss: `{mixture['hard_oracle_minus_mixture_loss']}`.",
        f"- Diagnosis: {hard_soft}",
        "",
        "## Canonical decision",
        "",
        "M3A is explanatory only. It does not alter the original Gate 1 FAIL, does not authorize "
        "M4, and does not use split 6 or 36.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    arguments = parser.parse_args()
    if arguments.bootstrap_repeats < 1_000:
        raise ValueError("M3A bootstrap_repeats must be at least 1000")
    source_hashes = {
        "gate1_episode_metrics.parquet": sha256_file(arguments.metrics),
        "expert_task_matrix.csv": sha256_file(arguments.matrix),
        "gate1_decision.json": sha256_file(arguments.decision),
    }
    frame = pd.read_parquet(arguments.metrics)
    matrix = pd.read_csv(arguments.matrix)
    decision = json.loads(arguments.decision.read_text(encoding="utf-8"))
    validation = validate_canonical_inputs(frame, matrix, decision)
    fixed = fixed_expert_rows(frame)
    oracle = frame[frame["method"] == "episode_oracle"].copy()
    rankings = episode_rankings(fixed)
    pairwise = pairwise_expert_matrix(fixed, bootstrap_repeats=arguments.bootstrap_repeats)
    leave_one_out, mask_semantics = leave_one_expert_out(
        fixed, oracle, bootstrap_repeats=arguments.bootstrap_repeats
    )
    task_effects, mixture = paired_effects(frame, bootstrap_repeats=arguments.bootstrap_repeats)
    stability = ranking_stability(fixed, rankings)
    multilevel = multilevel_bootstrap(frame, repeats=arguments.bootstrap_repeats)
    margin_accuracy = bootstrap_ci(
        rankings["accuracy_margin"], repeats=arguments.bootstrap_repeats, seed=2026080501
    )
    margin_loss = bootstrap_ci(
        rankings["loss_margin"], repeats=arguments.bootstrap_repeats, seed=2026080502
    )
    oracle_identity = oracle[[*EPISODE_KEYS, "selected_expert"]].merge(
        rankings[[*EPISODE_KEYS, "top1_expert_accuracy_rank", "accuracy_top_tie_count"]],
        on=EPISODE_KEYS,
        validate="one_to_one",
    )
    identity_mismatch = (
        oracle_identity["selected_expert"] != oracle_identity["top1_expert_accuracy_rank"]
    )
    unconditional_loo = leave_one_out[
        leave_one_out["mask_type"] == "leave_one_expert_out_unconditional"
    ]
    global_masks = leave_one_out[
        leave_one_out["mask_type"] == "global_most_used_conditional"
    ]
    task_masks = leave_one_out[
        leave_one_out["mask_type"] == "task_modal_conditional"
    ]

    def weighted_mask_summary(rows: pd.DataFrame) -> dict[str, Any]:
        weights = rows["episode_count"].to_numpy(dtype=float)
        return {
            "task_split_count": int(len(rows)),
            "conditional_episode_count": int(rows["episode_count"].sum()),
            "weighted_mean_accuracy_drop": float(
                np.average(rows["mean_accuracy_drop"], weights=weights)
            ),
            "task_split_ci_low_above_zero_count": int((rows["accuracy_drop_ci_low"] > 0).sum()),
        }

    pairwise_summary = {
        "task_split_pair_count": int(len(pairwise)),
        "mean_accuracy_tie_rate": float(pairwise["accuracy_tie_rate"].mean()),
        "accuracy_ci_excludes_zero_count": int(
            ((pairwise["accuracy_delta_ci_low"] > 0) | (pairwise["accuracy_delta_ci_high"] < 0)).sum()
        ),
    }
    task_effect_summary = {
        baseline: {
            "positive_mean_task_count": int((rows["mean_accuracy_advantage"] > 0).sum()),
            "ci_low_above_zero_task_count": int((rows["accuracy_ci_low"] > 0).sum()),
            "task_count": int(len(rows)),
            "minimum_mean_accuracy_advantage": float(rows["mean_accuracy_advantage"].min()),
            "maximum_mean_accuracy_advantage": float(rows["mean_accuracy_advantage"].max()),
        }
        for baseline, rows in task_effects.groupby("baseline")
    }
    diagnostics = {
        "schema_version": 1,
        "analysis": "M3A_posthoc_methodology_audit",
        "canonical_experiment_id": decision["metadata"]["experiment_id"],
        "canonical_decision_original": decision["status"],
        "canonical_decision_unchanged": True,
        "m4_authorized": False,
        "fresh_split_results_used": False,
        "bootstrap_repeats": arguments.bootstrap_repeats,
        "source_sha256": source_hashes,
        "validation": validation,
        "top1_top2_margins": {
            "accuracy_margin": margin_accuracy,
            "loss_margin": margin_loss,
            "accuracy_margin_le_0_01_rate": float((rankings["accuracy_margin"] <= 0.01).mean()),
            "exact_top_accuracy_tie_rate": float((rankings["accuracy_top_tie_count"] > 1).mean()),
        },
        "tie_break_audit": {
            "episode_count": int(len(oracle_identity)),
            "identity_mismatch_count": int(identity_mismatch.sum()),
            "identity_mismatch_rate": float(identity_mismatch.mean()),
            "mismatch_with_accuracy_tie_count": int(
                (identity_mismatch & (oracle_identity["accuracy_top_tie_count"] > 1)).sum()
            ),
            "metric_impact": "none_on_episode_oracle_accuracy",
            "identity_diagnostic_impact": "yes",
        },
        "mask_semantics": mask_semantics,
        "conditional_mask_summary": {
            "global": weighted_mask_summary(global_masks),
            "task_modal": weighted_mask_summary(task_masks),
        },
        "leave_one_out_summary": {
            "task_split_expert_count": int(len(unconditional_loo)),
            "maximum_mean_accuracy_drop": float(unconditional_loo["mean_accuracy_drop"].max()),
            "median_mean_accuracy_drop": float(unconditional_loo["mean_accuracy_drop"].median()),
            "ci_low_above_zero_count": int((unconditional_loo["accuracy_drop_ci_low"] > 0).sum()),
        },
        "pairwise_summary": pairwise_summary,
        "task_level_effects": task_effects.to_dict(orient="records"),
        "task_effect_summary": task_effect_summary,
        "multilevel_bootstrap": multilevel,
        "hard_vs_soft_mixture": mixture,
        "ranking_stability_summary": {
            "mean_pairwise_spearman": float(stability["mean_pairwise_spearman_rank"].mean()),
            "minimum_pairwise_spearman": float(stability["minimum_pairwise_spearman_rank"].min()),
            "mean_level_modal_consistency": float(stability["level_modal_consistency"].mean()),
        },
        "analysis_only_query_label_derived": [
            "episode_oracle",
            "oracle_mixture",
            "sample_oracle",
            "forced_worst",
            "swap",
            "mask_most_used",
            "permuted_task_assignment",
            "leave_one_expert_out",
            "global_most_used_conditional",
            "task_modal_conditional",
        ],
    }
    arguments.results_root.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(arguments.results_root / "gate1_pairwise_expert_matrix.csv", index=False)
    leave_one_out.to_csv(
        arguments.results_root / "gate1_leave_one_expert_out.csv", index=False
    )
    stability.to_csv(arguments.results_root / "gate1_ranking_stability.csv", index=False)
    atomic_write_json(arguments.reports_root / "gate1_posthoc_diagnostics.json", diagnostics)
    atomic_write_text(
        arguments.reports_root / "gate1_methodology_audit.md", _markdown(diagnostics)
    )
    final_hashes = {
        "gate1_episode_metrics.parquet": sha256_file(arguments.metrics),
        "expert_task_matrix.csv": sha256_file(arguments.matrix),
        "gate1_decision.json": sha256_file(arguments.decision),
    }
    if final_hashes != source_hashes:
        raise RuntimeError("a canonical input changed during the read-only M3A audit")
    print(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "canonical_decision": decision["status"],
                "canonical_inputs_unchanged": True,
                "m4_authorized": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
