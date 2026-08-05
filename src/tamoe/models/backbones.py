"""Configurable frozen visual backbones."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class BackboneInfo:
    name: str
    weights: str
    embedding_dim: int
    preprocess_version: str


class FrozenBackbone(nn.Module):
    def __init__(self, encoder: nn.Module, info: BackboneInfo) -> None:
        super().__init__()
        self.encoder = encoder
        self.info = info
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> FrozenBackbone:
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            return self.encoder(images)


def build_backbone(name: str, *, pretrained: bool = True) -> FrozenBackbone:
    normalized = name.lower()
    if normalized == "tiny":
        from tamoe.models.tiny import TinyBackbone

        model = TinyBackbone(embedding_dim=32)
        return FrozenBackbone(
            model,
            BackboneInfo(
                name="tiny",
                weights="random_seeded",
                embedding_dim=32,
                preprocess_version="tensor_0_1_v1",
            ),
        )
    if normalized != "resnet18":
        raise ValueError(f"unsupported backbone {name!r}; allowed: tiny, resnet18")
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    embedding_dim = model.fc.in_features
    model.fc = nn.Identity()
    return FrozenBackbone(
        model,
        BackboneInfo(
            name="resnet18",
            weights="IMAGENET1K_V1" if pretrained else "random_seeded",
            embedding_dim=embedding_dim,
            preprocess_version="rgb_0_1_imagenet_norm_v1",
        ),
    )


def normalize_for_backbone(images: Tensor, backbone_name: str) -> Tensor:
    if backbone_name.lower() == "tiny":
        return images
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (images - mean) / std
