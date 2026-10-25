"""The subagent roster sidecar's name, schema version and reader: a stdlib-only leaf.

WHY THIS MODULE EXISTS. The roster sidecar (``subagent-roster.v1.json``) is
WRITTEN by the live owner (``session.session.Session._persist_subagent_roster``)
but READ by several cold, read-only paths: the desktop facade's cold restore
(``attached.AttachedSession._restore_cold_subagents`` / ``_durable_roster``), the
desktop child-trajectory roster read, and delete receipts in ``cleanup``. Those
readers used to import the reader and filename from ``session.session``, and
importing that module executes ``harness.guard_area`` → ``tools.builtin`` and
~90 further modules (the whole live tool stack). Measured on a 32k-row journal
(2026-10-25, CORE-L3C-ENTER-ATTRIBUTION): that one lazy import was the dominant
term of a cache-hit desktop session enter — a read-only open paid for a tool
registry it never constructs.

CONSTRAINTS.
- Stdlib only. ``tests/unit/server/test_desktop_cold_enter_fastpath.py`` pins it:
  ``test_the_roster_leaf_imports_nothing_from_the_package`` (this module loads no
  other ``local_operator`` module) and
  ``test_cold_desktop_enter_never_imports_the_live_session_or_tool_stack`` (a real
  cold desktop enter, roster restore included, never loads ``session.session`` or
  ``local_operator.tools``). Adding a heavy import here re-opens the regression.
- ``session.session`` re-exports every name below under its historical spelling
  (``SUBAGENT_ROSTER_SIDECAR``, ``SUBAGENT_ROSTER_CUSTOM_TYPE``,
  ``_SUBAGENT_ROSTER_VERSION``, ``_read_roster_sidecar``), so the writer and
  every existing caller/test keep one source of truth instead of a restated copy.
- The reader's acceptance rules are the writer's contract and are unchanged by
  the move: version must match, ``generation`` a non-negative non-bool int,
  ``jobs`` and ``records`` lists; anything else (absent, half-written, foreign
  version) reads as ``None`` so callers fall back to the legacy transcript row.
"""

from __future__ import annotations

import json
from typing import Any

#: Transcript custom-entry type of the legacy (pre-sidecar) roster snapshot.
#: See ``session.session`` for the full history; defined here so cold readers
#: can name the fallback row without importing the live session module.
SUBAGENT_ROSTER_CUSTOM_TYPE = "subagent_roster"

#: The roster's own store, beside ``transcript.jsonl`` in the session directory.
SUBAGENT_ROSTER_SIDECAR = "subagent-roster.v1.json"

#: Schema version the writer stamps and the reader requires.
SUBAGENT_ROSTER_VERSION = 1


def read_roster_sidecar(path: Any) -> dict[str, Any] | None:
    """Read a supported sidecar, or ``None`` so the caller falls back to the transcript."""
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("version") != SUBAGENT_ROSTER_VERSION:
            return None
        generation = payload.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            return None
        if not isinstance(payload.get("jobs"), list) or not isinstance(
            payload.get("records"), list
        ):
            return None
        return payload
    except (OSError, ValueError, TypeError):
        return None
