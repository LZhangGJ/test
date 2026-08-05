from __future__ import annotations

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest
import torch

from tamoe.analysis.gate1r import hierarchical_ci
from tamoe.experts.adapters import ResidualAdapter
from tamoe.models.episodic_head import prototype_logits
from tamoe.routing.support_risk import (
    _risk_statistics,
    calibration_metrics,
    evaluate_support_risk_episode,
    shrink_cross_validated_risks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_support_risk_pilot.py"))


def _identity_adapter() -> ResidualAdapter:
    adapter = ResidualAdapter(embedding_dim=2, rank=1)
    with torch.no_grad():
        adapter.down.weight.zero_()
        adapter.up.weight.zero_()
    return adapter.eval()


def test_calibration_and_brier_are_zero_for_perfect_probabilities() -> None:
    labels = torch.tensor([0, 1])
    log_probabilities = torch.tensor([[1.0, 1e-12], [1e-12, 1.0]]).log()
    metrics = calibration_metrics(log_probabilities, labels)
    assert metrics["ece"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["brier"] == pytest.approx(0.0, abs=1e-7)


def test_variance_aware_shrinkage_moves_noisy_risk_toward_pool() -> None:
    risks = torch.tensor([0.0, 2.0])
    shrunk = shrink_cross_validated_risks(
        risks,
        torch.tensor([0.0, 100.0]),
        20,
        prior_strength=5.0,
        variance_scale=0.25,
    )
    pooled = risks.mean()
    assert abs(float(shrunk[1] - pooled)) < abs(float(risks[1] - pooled))
    assert torch.isfinite(shrunk).all()


def test_vectorized_leave_one_out_risk_matches_explicit_sample_removal() -> None:
    generator = torch.Generator().manual_seed(17)
    support = torch.randn(9, 4, generator=generator)
    labels = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
    actual = _risk_statistics(support, labels, temperature=0.2)
    losses = []
    correct = []
    for index in range(len(support)):
        keep = torch.arange(len(support)) != index
        logits = prototype_logits(
            support[keep], labels[keep], support[index : index + 1], temperature=0.2
        )
        losses.append(torch.nn.functional.cross_entropy(logits, labels[index : index + 1]))
        correct.append(float(logits.argmax(dim=-1).item() == labels[index].item()))
    expected_losses = torch.stack(losses)
    assert actual["loo_cross_entropy"] == pytest.approx(float(expected_losses.mean()), abs=1e-6)
    assert actual["loo_loss_variance"] == pytest.approx(
        float(expected_losses.var(unbiased=False)), abs=1e-6
    )
    assert actual["loo_accuracy"] == pytest.approx(sum(correct) / len(correct), abs=1e-7)


def test_support_risk_episode_uses_exact_episode_and_reports_every_holdout() -> None:
    experts = {
        "shared": _identity_adapter(),
        "single": _identity_adapter(),
        "source_a": _identity_adapter(),
        "source_b": _identity_adapter(),
    }
    references = {
        "shared": torch.tensor([0.5, 0.5]),
        "source_a": torch.tensor([1.0, 0.0]),
        "source_b": torch.tensor([0.0, 1.0]),
    }
    features = torch.tensor([[1.0, 0.0]] * 8 + [[0.0, 1.0]] * 8)
    labels = torch.tensor([0] * 8 + [1] * 8)
    rows, episode = evaluate_support_risk_episode(
        experts,
        references,
        features,
        labels,
        shots=5,
        queries_per_class=2,
        seed=7,
        repetition=1,
        prototype_temperature=0.1,
        original_route_temperature=0.25,
        risk_temperature=0.25,
        prior_strength=5.0,
        variance_scale=0.25,
        shared_fallback_weight=0.5,
        device=torch.device("cpu"),
    )
    methods = {row["method"] for row in rows}
    assert {
        "naive_support_loss_top1",
        "leave_one_out_support_risk_top1",
        "leave_one_out_support_risk_soft_mixture",
        "shrinkage_support_risk_mixture",
        "shrinkage_support_risk_with_shared_fallback",
        "original_support_prototype",
        "original_support_soft_mixture",
        "support_label_removal",
    } <= methods
    assert episode.candidate_names == ("shared", "source_a", "source_b")
    assert len(episode.risk_statistics) == 3
    assert all(record["support_sample_count"] == 10 for record in episode.risk_statistics)
    assert all("shrinkage_cross_entropy_risk" in record for record in episode.risk_statistics)
    assert torch.isclose(episode.shrinkage_weights.sum(), torch.tensor(1.0))
    assert all(0 <= row["accuracy"] <= 1 for row in rows)


def test_balanced_hierarchical_wrapper_reuses_canonical_bootstrap() -> None:
    frame = pd.DataFrame(
        [
            {"split": split, "task": task, "episode": episode, "value": split + task + episode}
            for split in range(2)
            for task in range(2)
            for episode in range(3)
        ]
    )
    expected = hierarchical_ci(
        frame, "value", ["split", "task", "episode"], repeats=100, seed=9
    )
    actual = RUNNER["_balanced_hierarchical_ci"](
        frame, "value", ["split", "task", "episode"], repeats=100, seed=9
    )
    assert actual == expected
    with pytest.raises(ValueError, match="balanced hierarchical"):
        RUNNER["_balanced_hierarchical_ci"](
            frame.iloc[:-1], "value", ["split", "task", "episode"], repeats=10, seed=9
        )


def _decision_rows() -> list[dict[str, object]]:
    accuracies = {
        "naive_support_loss_top1": 0.74,
        "leave_one_out_support_risk_top1": 0.75,
        "leave_one_out_support_risk_soft_mixture": 0.77,
        "shrinkage_support_risk_mixture": 0.78,
        "shrinkage_support_risk_with_shared_fallback": 0.80,
        "capacity_matched_single": 0.60,
        "shared": 0.55,
        "oracle_analysis_only": 0.90,
        "support_shuffle": 0.70,
        "support_label_removal": 0.70,
        "wrong_task_support": 0.70,
    }
    rows: list[dict[str, object]] = []
    for split_seed, tasks in ((6, ("a", "b")), (36, ("c", "d"))):
        for task in tasks:
            for train_seed in (2, 3, 4):
                for shots in (5, 10):
                    for support_resample in range(2):
                        for method, accuracy in accuracies.items():
                            loss = 0.7 if method == "shrinkage_support_risk_with_shared_fallback" else 1.0
                            rows.append(
                                {
                                    "split_seed": split_seed,
                                    "train_seed": train_seed,
                                    "task": task,
                                    "shots": shots,
                                    "support_resample": support_resample,
                                    "episode_hash": (
                                        f"{split_seed}:{task}:{train_seed}:{shots}:"
                                        f"{support_resample}"
                                    ),
                                    "method": method,
                                    "accuracy": accuracy,
                                    "macro_f1": accuracy - 0.01,
                                    "loss": loss,
                                    "ece": 0.1,
                                    "brier": 0.2,
                                }
                            )
    return rows


def test_frozen_decision_go_and_stop_paths() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "support_risk_pilot.json").read_text(encoding="utf-8")
    )
    config["bootstrap_repeats"] = 100
    frame = pd.DataFrame(_decision_rows())
    decision, comparisons, ablations = RUNNER["_decide"](frame, config)
    assert decision["outcome"] == "GO_FIXED_BANK_ROUTING"
    assert decision["raw_values"]["oracle_gap_recovery"] == pytest.approx(2 / 3)
    assert len(comparisons) == 5
    assert len(ablations) == 3
    assert set(comparisons[0]["paired_metric_deltas"]) == {
        "accuracy",
        "macro_f1",
        "nll",
        "ece",
        "brier",
    }

    failed = frame.copy()
    selected = failed["method"] == "shrinkage_support_risk_with_shared_fallback"
    failed.loc[selected, "loss"] = 1.1
    stopped, _, _ = RUNNER["_decide"](failed, config)
    assert stopped["outcome"] == "STOP_FIXED_BANK_ROUTING"
    assert not stopped["criteria"]["nll_no_worse_than_shared"]


def test_pilot_configuration_is_frozen_to_requested_episode_axes() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "support_risk_pilot.json").read_text(encoding="utf-8")
    )
    gate2 = json.loads(
        (PROJECT_ROOT / "configs" / "m4_gate2_soft.json").read_text(encoding="utf-8")
    )
    assert config["shots"] == [5, 10]
    assert config["task_split_seeds"] == gate2["fresh_task_split_seeds"]
    assert config["train_seeds"] == gate2["train_seeds"]
    assert config["support_resamples"] == gate2["support_resamples"]
    assert config["risk_estimator"]["learned_parameters"] == 0
    assert config["risk_estimator"]["hyperparameter_trials"] == 1
