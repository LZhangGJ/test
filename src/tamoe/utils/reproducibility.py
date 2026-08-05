"""Deterministic random seed management."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedState:
    seed: int
    torch_available: bool
    cuda_seeded: bool


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import torch
    except ImportError:
        return SeedState(seed=seed, torch_available=False, cuda_seeded=False)

    torch.manual_seed(seed)
    cuda_seeded = torch.cuda.is_available()
    if cuda_seeded:
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return SeedState(seed=seed, torch_available=True, cuda_seeded=cuda_seeded)
