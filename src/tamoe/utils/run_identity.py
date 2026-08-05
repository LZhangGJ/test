"""Stable configuration hashes and collision-resistant run IDs."""

from __future__ import annotations

import re
from datetime import UTC, datetime

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(value: str) -> str:
    cleaned = _SAFE_COMPONENT.sub("-", value.strip()).strip("-._")
    if not cleaned:
        raise ValueError("run ID component cannot be empty")
    return cleaned


def make_run_id(study: str, experiment_name: str, config_hash: str) -> str:
    if len(config_hash) < 8:
        raise ValueError("config_hash must contain at least eight characters")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{_safe(study)}_{_safe(experiment_name)}_{config_hash[:12]}_{timestamp}"
