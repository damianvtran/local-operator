"""Diagnostic, aggregated token-consumption analytics.

Every provider call in every session contributes to one shared, on-disk
ledger so the harness has an accurate, universal view of where tokens go —
which provider, which model, and (by estimate) which part of the context:
the packaged system prompt, the operator's custom instructions and agent/team
profile prompts, the tool inventory, tool schemas, the environment block,
loaded knowledge/skills, the conversation, and tool results — plus the
authoritative output split (thinking vs generation) and cache rates.

Design contract (why this package looks the way it does):

- **Never on the critical path.** The provider stream is fully consumed
  before anything is recorded, so recording adds zero latency to the
  response the user is waiting on. The one piece of work that must run on the
  event loop — snapshotting the request's component *character lengths*
  before the transcript mutates — is O(blocks + tools + messages) string
  length reads and benchmarks at well under a millisecond even on a
  340k-token context. Tokenisation, apportionment, and the SQLite write all
  happen on a background daemon thread.

- **Best-effort, bounded, non-blocking.** ``record`` does a single
  ``queue.put_nowait`` and returns; a full queue DROPS the sample rather than
  blocking a session. The writer thread batches inserts so N concurrent calls
  cost one transaction, not N fsyncs.

- **Parallel-safe by construction.** Several ``lop`` sessions run at once (one
  per terminal). The store is one SQLite database in WAL mode with a busy
  timeout, so cross-process writes are atomic and serialised by the engine —
  the same discipline ``providers/usage_cache.py`` and ``auth.db`` use.

- **A pure accelerator, never a dependency.** Every disk and thread operation
  is exception-safe: a read-only home directory, a locked file, or a full
  disk degrades to "no analytics", never to a broken session.
"""

from __future__ import annotations

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    COMPONENT_LABELS,
    CallSnapshot,
    UsageAggregate,
    UsagePeriod,
    apportion_components,
    price_snapshot,
    snapshot_component_chars,
    split_system_prompt,
)
from local_operator.analytics.recorder import (
    AnalyticsRecorder,
    get_recorder,
    record_call,
    reset_recorder_for_test,
)
from local_operator.analytics.store import AnalyticsStore, default_db_path

__all__ = [
    "COMPONENT_KEYS",
    "COMPONENT_LABELS",
    "CallSnapshot",
    "UsageAggregate",
    "UsagePeriod",
    "apportion_components",
    "price_snapshot",
    "snapshot_component_chars",
    "split_system_prompt",
    "AnalyticsRecorder",
    "get_recorder",
    "record_call",
    "reset_recorder_for_test",
    "AnalyticsStore",
    "default_db_path",
]
