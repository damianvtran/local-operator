"""Offline, read-only import of Claude Code, Codex and omp conversations/memory.

The package is deliberately a LEAF on every foreign store: it opens source files
read-only (SQLite through a private snapshot), never writes into a foreign root,
never calls the network, and maps everything through one canonical model before
any target row is built. Surfaces (desktop routes, CLI, later TUI/agent tool)
call :class:`local_operator.harness_import.service.HarnessImportService`; none
of them re-implements parsing or compaction mapping.
"""

from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.types import (
    ImportPreview,
    ImportProgress,
    ImportRequest,
    ImportResult,
    SourceInfo,
    SourceKind,
    SourceSession,
)

__all__ = [
    "HarnessImportService",
    "ImportPreview",
    "ImportProgress",
    "ImportRequest",
    "ImportResult",
    "SourceInfo",
    "SourceKind",
    "SourceSession",
]
