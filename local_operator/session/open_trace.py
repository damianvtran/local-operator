"""Env-gated JSONL trace for the open-frame read path.

WHY AN ENV-GATED TRACE RATHER THAN A LOG LINE. Opening a peer conversation
spans THREE processes — the desktop daemon, this device's relay and the owner's
relay — and "where did the milliseconds go" is a question only rows from more
than one of them can answer, correlated by time. A log line forces a choice of
level the operator must enable globally; this trace is one file per process,
turned on with ``LOP_OPEN_TRACE=<path>`` in exactly the process under study
(a rig cell, or a live machine with the env set for one daemon). Disabled
tracing performs no JSON encoding or file I/O: the writer returns on a
module-level ``None`` check. Call sites still read span-start clocks, so the
disabled overhead is small, not literally zero.

ROWS ARE APPEND-ONLY JSONL under a lock: ``at`` (epoch ms), ``stage``, an
optional ``ms`` span the caller just finished timing, and whatever fields the
site knows. The writer NEVER raises — a trace that can break the read it
observes is worse than no trace — and the module is inert when the variable is
unset.

THE MARKS SHIP with the change they instrumented (all guarded by
:func:`enabled`), so the numbers on the PR stay reproducible on any machine:
set the variable, run the same drive, read the JSONL rows back.
"""

from __future__ import annotations

import json
import os
import threading
import time

#: Reads the variable ONCE at import: the mark sites guard on :func:`enabled`,
#: and a per-call ``getenv`` would be a string lookup on the read path the trace
#: exists to measure. A process that wants a trace sets the variable before it
#: starts — there is deliberately no setter, so no second opinion about whether
#: tracing is on can grow beside this one.
_PATH = os.environ.get("LOP_OPEN_TRACE") or None
_LOCK = threading.Lock()


def enabled() -> bool:
    """Whether this process traces. Gates every mark site."""
    return _PATH is not None


def mark(stage: str, ms: float | None = None, **fields: object) -> None:
    """Append one trace row. Never raises, never mutates ``fields``."""
    if _PATH is None:
        return
    row: dict[str, object] = {"at": time.time() * 1000.0, "stage": stage}
    if ms is not None:
        row["ms"] = round(ms, 3)
    row.update(fields)
    try:
        line = json.dumps(row, separators=(",", ":"), default=str)
        with _LOCK, open(_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:  # noqa: BLE001 — see the module docstring: observability
        pass  # must never fail the read it observes.
