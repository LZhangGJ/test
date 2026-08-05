from pathlib import Path

import torch
from torch import nn

from tamoe.data.feature_cache import FeatureSet
from tamoe.experts.adapters import ResidualAdapter
from tamoe.experts.training import ExpertTrainConfig, load_expert_checkpoint, train_expert
from tamoe.metrics.resources import count_resources
from tamoe.models.episodic_head import prototype_logits


def test_adapter_shapes_and_resource_accounting_are_capacity_explicit() -> None:
    backbone = nn.Linear(8, 8)
    backbone.requires_grad_(False)
    experts = [ResidualAdapter(8, 2) for _ in range(3)]
    embeddings = torch.randn(5, 8)
    assert experts[0](embeddings).shape == embeddings.shape
    resources = count_resources(backbone, experts)
    assert resources.expert_bank_size == 3
    assert resources.trainable_parameters == sum(
        parameter.numel() for expert in experts for parameter in expert.parameters()
    )
    assert resources.activated_parameters_per_query < resources.total_parameters


def test_prototype_head_supports_variable_label_spaces() -> None:
    support = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = prototype_logits(support, labels, query, temperature=0.2)
    assert logits.shape == (2, 2)
    assert logits.argmax(dim=-1).tolist() == [0, 1]


def test_expert_training_checkpoint_resumes_exact_config(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(7)
    labels = torch.arange(3).repeat_interleave(20)
    centers = torch.nn.functional.one_hot(labels, num_classes=3).float()
    features = torch.cat((centers, torch.randn((60, 5), generator=generator) * 0.05), dim=1)
    feature_set = FeatureSet(
        features=features,
        labels=labels,
        sample_indices=torch.arange(60),
        cache_key="synthetic-cache",
    )
    config = ExpertTrainConfig(
        embedding_dim=8,
        rank=2,
        steps=3,
        shots=2,
        queries_per_class=2,
        n_way=3,
        learning_rate=1e-2,
        weight_decay=0.0,
        temperature=0.2,
        seed=11,
        schedule="balanced_round_robin",
    )
    checkpoint = tmp_path / "expert.pt"
    _, first = train_expert(
        {"task": feature_set},
        config,
        checkpoint,
        tmp_path / "train.jsonl",
        device=torch.device("cpu"),
    )
    _, resumed = train_expert(
        {"task": feature_set},
        config,
        checkpoint,
        tmp_path / "resume.jsonl",
        device=torch.device("cpu"),
    )
    assert first.start_step == 0
    assert resumed.start_step == config.steps
    assert resumed.resumed is True
    assert resumed.final_loss == first.final_loss
    loaded, loaded_config = load_expert_checkpoint(checkpoint, device=torch.device("cpu"))
    assert loaded_config == config
    assert loaded(features[:2]).shape == (2, 8)
