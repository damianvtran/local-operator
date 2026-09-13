"""How a persisted subagent row resolves on the way back in.

A roster row is a record of what a child's state WAS when the roster was last
written — and on every restore path that matters, the process that wrote it is
gone. A row handed back verbatim therefore paints a spinner for a child that
cannot be working and counts phantom activity on the band, which is worse than
the empty panel it replaces: an empty panel is obviously incomplete, while
phantom activity is confidently wrong and invites a cancel that finds nothing.

This module owns the ONE resolution policy, because TWO writers restore the same
roster and they must agree:

* the OWNER path — ``Session._load_subagent_roster`` rehydrating
  ``AsyncJobManager`` from the sidecar, which is what a successor runtime
  publishes as the session's live roster; and
* the COLD VIEWER path — ``AttachedSession._restore_cold_subagents`` painting a
  session nothing is running (design §4, D3).

They were written separately once, and the cost was measured: the viewer's
resolved rows (a settled ``completed``, an ``interrupted`` carrying its cause)
were visible for under a second before the successor's own restore replaced them
with the raw persisted statuses — a blanket ``interrupted`` with the cause
dropped — and the dock went back to being unable to answer "was this work lost?"
(UX review round 1, U2). One function, two callers, no second opinion.

The resolution order, because each step is stronger evidence than the next:

1. **The record settled it.** ``records[].outcome`` names how the child ended,
   and that is a fact, not an inference — a row restored as a bare
   ``interrupted`` beside a record reading ``completed`` was the reported
   defect.
2. **The record did not, so the child's own transcript decides** — see
   :func:`child_tail_state`. The reason goes on the row so a panel can say WHY
   rather than only that it stopped.
3. **No record and no transcript** → today's ``interrupted``, now naming
   ``owner-lost`` rather than carrying no cause at all.

Anything already terminal is untouched: ``completed``/``failed``/``cancelled``
are facts the last runtime settled and a later process must not relitigate.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from local_operator.session.transcript import (
    ENTRY_MESSAGE,
    TRANSCRIPT_FILENAME,
    TranscriptEntry,
)

#: Outcomes a roster RECORD can carry that already settle a child's state. Any
#: other value (``None``, or a token a newer runtime invented) means "the
#: record did not settle it" and the row falls through to the child's own
#: transcript.
_SETTLED_RECORD_OUTCOMES = frozenset({"completed", "failed", "error", "interrupted"})

#: How much of a child transcript's TAIL the restore reads to decide whether
#: the child was still working. Bounded because rule 2 runs per unsettled child
#: and a 50 MB journal must not be parsed to answer one question; the shape
#: examined is structural (what the LAST row is), so a sliced tail is enough.
#: The design's cost note bounds this the same way on both call sites: only
#: records whose ``outcome`` is unset are read at all.
CHILD_TAIL_BYTES = 256 * 1024


def record_field(record: Any, name: str, default: Any = None) -> Any:
    """One field off a roster record, which is a raw sidecar DICT.

    The sidecar stores ``records`` as plain JSON objects (``SubagentComms.snapshot``
    output), while ``jobs`` come back as ``JobState``/``AsyncJob`` models — so the
    one reader that consults both cannot assume either shape. Kept as a named
    helper rather than a ``getattr``/``[]`` dance at each call site so the
    dict-vs-model distinction is stated once.
    """
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def child_tail_state(session_dir: Any) -> str:
    """``"mid-turn"`` / ``"finished"`` / ``"unknown"`` for one child's journal.

    Structural, never semantic — the only question is what the child's LAST
    message row is:

    * a ``tool`` result, or an assistant message whose ``tool_calls`` have no
      rows after them → the child was CUT OFF MID-TURN;
    * an assistant message with no pending tool call → the child actually
      FINISHED and only its parent's record was lost.

    Anything else, or anything unreadable, answers ``"unknown"`` so the row
    keeps today's ``interrupted`` spelling instead of inventing an outcome. A
    bounded tail read is deliberate: the journal is append-mostly, so the last
    256 KiB is the part that carries the answer.
    """
    if not session_dir:
        return "unknown"
    try:
        path = Path(session_dir) / TRANSCRIPT_FILENAME
        if not path.exists():
            return "unknown"
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - CHILD_TAIL_BYTES)
            handle.seek(start)
            blob = handle.read()
        lines = blob.split(b"\n")
        if start > 0:
            # The first line is the tail of a row whose head is before the
            # window; it cannot be parsed as a whole row.
            lines = lines[1:]
        last_payload: dict[str, Any] | None = None
        for raw in lines:
            if not raw.strip():
                continue
            entry = TranscriptEntry.from_json(raw.decode("utf-8", errors="replace"))
            if entry is not None and entry.type == ENTRY_MESSAGE:
                last_payload = entry.payload
    except (OSError, ValueError):
        return "unknown"
    if last_payload is None:
        return "unknown"
    role = str(last_payload.get("role") or "")
    if role == "tool":
        return "mid-turn"
    if role == "assistant":
        return "mid-turn" if last_payload.get("tool_calls") else "finished"
    return "unknown"


def restored_job_row(job: Any, record: Any | None) -> Any:
    """One non-terminal job row, resolved against the record and the child.

    ``job`` may be either shape the two callers hold (a frontend ``JobState`` or
    a manager ``AsyncJob``); both are pydantic models with ``status``,
    ``restored`` and ``cut_off_cause``, so ``model_copy`` is the one update
    mechanism that works for both without a per-caller branch.
    """
    record_outcome = ""
    if record is not None:
        record_outcome = str(record_field(record, "outcome") or "")
    if record_outcome in _SETTLED_RECORD_OUTCOMES:
        return job.model_copy(
            update={"status": record_outcome, "restored": True, "cut_off_cause": ""}
        )
    tail = child_tail_state(record_field(record, "session_dir") if record is not None else None)
    if tail == "finished":
        # The child produced a settled answer; only its parent's record of that
        # was lost. Reporting the child as cut off would be a lie the panel then
        # offers to resume.
        return job.model_copy(update={"status": "completed", "restored": True, "cut_off_cause": ""})
    # ``mid-turn`` and ``unknown`` land in the same place: the child has no
    # settled outcome of its own and cannot be running any more, so it stopped
    # under the process that owned it. One return rather than two identical
    # ones, so the fallthrough reads as intended (review round 1, NIT-1).
    return job.model_copy(
        update={"status": "interrupted", "restored": True, "cut_off_cause": "owner-lost"}
    )


def roster_records(payload: Mapping[str, Any] | None) -> Sequence[Any]:
    """The sidecar's ``records`` list, or an empty sequence when it has none.

    Typed here rather than inline so the ``Any`` coming out of a raw JSON dict
    is narrowed once: the caller passes it straight into
    :func:`resolve_restored_rows`, whose ``records`` parameter is a ``Sequence``.
    """
    records = (payload or {}).get("records")
    return records if isinstance(records, list) else ()


def resolve_restored_rows(jobs: Sequence[Any], records: Sequence[Any] = ()) -> list[Any]:
    """Roster rows as they must appear with the process that ran them gone.

    The rule is stated in the module docstring. ``records`` is the roster
    sidecar's ``records`` list, whose ``outcome`` and ``session_dir`` are what
    make a restored row diagnosable rather than a blanket ``interrupted``. It is
    optional so an older sidecar (or a caller that has none) keeps today's
    behaviour exactly.

    A ``running`` row that was PARKED (``queued``) never started, so it has no
    transcript to show or resume and is DROPPED — an ``interrupted`` row would
    invite a resume that finds nothing. ``AsyncJobManager.restore`` applies the
    same rule for its own table; dropping here as well keeps the two callers in
    step for the rows the manager never sees (a cold viewer builds no manager).
    """
    by_id: dict[str, Any] = {}
    for record in records:
        job_id = str(record_field(record, "job_id") or "")
        if job_id:
            by_id[job_id] = record
    rows: list[Any] = []
    for job in jobs:
        status = str(getattr(job, "status", "") or "")
        if status == "running":
            if bool(getattr(job, "queued", False)):
                continue
            rows.append(restored_job_row(job, by_id.get(str(getattr(job, "id", "") or ""))))
            continue
        rows.append(job.model_copy(update={"restored": True}))
    return rows
