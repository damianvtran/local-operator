"""Adapter registry. One place that knows which harnesses exist and how to build them."""

from __future__ import annotations

from pathlib import Path

from local_operator.harness_import.sources.base import SourceAdapter
from local_operator.harness_import.sources.claude import ClaudeAdapter
from local_operator.harness_import.sources.codex import CodexAdapter
from local_operator.harness_import.sources.omp import OmpAdapter


def build_adapters(config: Path) -> dict[str, SourceAdapter]:
    """Codex needs the config root for its private SQLite snapshot; others do not."""
    adapters: list[SourceAdapter] = [
        ClaudeAdapter(),
        CodexAdapter(config),
        OmpAdapter(),
    ]
    return {adapter.source: adapter for adapter in adapters}
