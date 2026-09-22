"""A turn a session owes, when the runtime that would have run it was leaving.

``<config_dir>/wakes/spooled/<session_id>.json`` answers the one question nothing
could answer before it existed: *a session spooled a message that asked for a
turn, then its runtime left — who is going to start the next one?*

**The defect, in the operator's own measurement.** On 2026-09-21 a session
retired for a newer build with messages sitting in its spool and no successor:
nothing raised a runtime, the mail waited, and the sender's receipt —
``inbox.SPOOL_RECEIPT_WAKE``, "held for the next runtime — it runs it" — was a
promise no process owned. The spool is only ever drained BY a runtime that
exists (``process._drain_inbox_into``), and the one always-on process whose job
is "make a runtime exist for this session" is the wake supervisor, which fires
from :mod:`local_operator.wakes.store`'s index — schedules only. A row spooled
by a *draining* runtime is not a schedule, so it had no owner at all: the
draining runtime could not spawn its own successor (it holds the transcript
lease until it exits), and the next engage might be hours away or never.

**Why a file of its own rather than a key in the index.** The index is a
PROJECTION the owning session rewrites from its transcript on every open
(``store``'s docstring: "derived, never authoritative"). An obligation stored
there would be erased by the next rewrite of a session that has no schedules —
silently, and in the direction of losing the work. This file is written by the
party that OWES it and read by the party that discharges it, and neither is the
session's schedule state.

**Why under ``wakes/``.** Same reason the index lives there: a process that must
not load the harness (the supervisor, and the TUI's picker) has to find the
obligations without opening thousands of session directories. A flat directory
holds one file per session that owes a turn — empty on a healthy machine — so a
pass costs one listing plus one small read per obligation.

**Import-light, stdlib-only**, pinned by ``tests/unit/test_import_graph.py`` for
the same reason ``store`` and ``deliveries`` are: this module sits on the
supervisor's resident set.

**The spool is the source of truth, this is the index of where to look.**
:func:`spool_owes_turn` reads the session's own ``inbox.jsonl`` and decides
whether a turn is still owed by it; a record whose spool no longer owes one is
dropped by the next pass (:func:`_reconcile_spooled` in the supervisor) rather than
kept as a standing obligation. The predicate is duplicated from
``session.runtime.inbox`` in the minimal form the supervisor can afford — two
field names and one token, spelled here because the alternative is importing
``local_operator.session`` into the supervisor's process, which is the one thing
its docstring forbids.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Subdirectory of the wakes dir holding one ``<session_id>.json`` per session
#: that owes a turn. A SUBdirectory for the reason ``deliveries`` gives: an index
#: scan lists ``wakes/`` and skips anything that is not a ``.json`` file, so an
#: obligation can never be read as a schedule.
SPOOLED_DIRNAME = "spooled"

#: Bumped only on an incompatible change to the entry shape, exactly as
#: ``store.INDEX_SCHEMA`` is.
SPOOLED_SCHEMA = 1

#: The spool file inside a session directory, and the two row fields that decide
#: whether a row OWES a turn. Spelled here rather than imported from
#: ``session.runtime.inbox`` for the import-weight reason the module docstring
#: gives; ``inbox.InboxLine`` is the authority on the writer's side, and a row
#: whose fields disagree with these reads as "owes nothing", which fails towards
#: not starting a runtime nobody asked for.
INBOX_NAME = "inbox.jsonl"
WAKE_FIELD = "wake"
SOURCE_FIELD = "source"
SOURCE_USER = "user"

#: Backoff before the second attempt, and its multiplier. Mirrors
#: ``deliveries``: the first retry is quick because the operator is waiting on
#: the turn, and the walk backs off so one session that cannot be raised does not
#: occupy an engagement slot on every pass.
RETRY_BASE_S = 15.0
RETRY_FACTOR = 2.0

#: The ceiling on one walk's delay, so a session that has failed for hours is
#: still retried hourly rather than never — a cold start can fail on a transient
#: (a credential, a lock) that a later pass will not see.
RETRY_CAP_S = 3600.0

#: Where the attempt walk stops getting faster, and the only bound on it. There
#: is deliberately NO attempt cap: a capped walk gave up permanently, which in
#: review round 1 (R1-1, R1-2) turned out to be defect (B) returning through the
#: bookkeeping — a handover whose owner outlived the cap (a slow clean exit, or
#: the wedged owner this whole change is about) or a row spooled after the walk
#: ended was left with no successor at all, forever. The walk therefore keeps
#: going at this ceiling, and the properties that were doing the cap's real work
#: are the ones that remain:
#:
#: * CHURN IS BOUNDED BY THE BACKOFF: attempts are 15 s, 30 s, 60 s, …, 1 h apart,
#:   and an attempt is only COUNTED when a raise was actually tried (a failed
#:   engage, or a wedged owner that holds the lease without answering) — see
#:   ``supervisor._SPOOLED_ATTEMPT_REASONS``. A session whose own runtime is
#:   still alive is not an attempt: nothing was tried, and the handover has not
#:   even begun.
#: * LATENCY HAS A CEILING BESIDE THE SLICE, and it is the supervisor's own sleep
#:   (review round 5, R5-4): when it is awake the record is acted on within
#:   ``SLICE_S`` (10 s), but a supervisor already asleep behind a scheduled wake
#:   wakes at most ``MAX_SLEEP_S`` (1 h) later. Stated because the spool path's
#:   own bound is the slice, and a reader who takes that as the whole bound would
#:   be wrong by three orders of magnitude in the one case where nothing else is
#:   due. (The writer also raises the supervisor itself when it records an
#:   obligation — see ``serving._spool_for_successor`` for why that call has to be
#:   there at all.)
#: * IMMORTALITY IS THE PRICE, and it is the honest one: a session that still
#:   owes a turn is still work, so the resident supervisor stays up to do it (one
#:   boot per hour past the ceiling, not a loop). What stops it is the spool
#:   emptying — the authority — not a counter.


def spooled_dir(config_dir: Path) -> Path:
    """The directory holding one obligation file per session."""
    return Path(config_dir) / "wakes" / SPOOLED_DIRNAME


def spooled_path(config_dir: Path, session_id: str) -> Path:
    """Where ``session_id``'s obligation lives. The id is never sanitised: it is
    a session directory name the caller read off disk, exactly as ``store`` and
    ``deliveries`` treat theirs."""
    return spooled_dir(config_dir) / f"{session_id}.json"


def _now_ms() -> int:
    return int(time.time() * 1000)


def read_spooled(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Every readable obligation, keyed by session id. Never raises.

    One entry may be unreadable (a torn write, a hand-edited file, a half-written
    temp file): that costs one session's turn, never the whole sweep, so a bad
    record is skipped and the caller carries on with the rest.
    """
    out: dict[str, dict[str, Any]] = {}
    directory = spooled_dir(config_dir)
    try:
        candidates = sorted(directory.glob("*.json"))
    except OSError:
        return out
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("schema") != SPOOLED_SCHEMA:
            continue
        session_id = data.get("session_id") or path.stem
        if not isinstance(session_id, str) or not session_id:
            continue
        out[session_id] = data
    return out


def read_spooled_turn(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """One session's obligation, or ``None``."""
    try:
        data = json.loads(spooled_path(config_dir, session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SPOOLED_SCHEMA:
        return None
    return data


def note_spooled_turn(
    config_dir: Path,
    session_id: str,
    *,
    cwd: str = "",
    rows: int = 1,
    now_ms: int | None = None,
) -> bool:
    """Record that ``session_id`` is owed a turn. Returns whether it landed.

    Called by the spool writer (``serving._spool_for_successor``) after a row
    that ASKS for a turn has been appended — a peer's ``send --wake`` or the
    owner's own prompt — and it is the whole of the promise the receipt makes.

    RE-NOTING IS NOT A NEW OBLIGATION, and ROUND 1 CHANGED WHAT THAT MEANS. A
    second row spooled while the first is still owed must not lose the count of
    what is waiting, so an existing record keeps its own ``noted_at_ms`` and its
    ``attempts`` — but it must not simply inherit the previous walk's wait
    either: a row that arrives while the record is an hour into a backed-off walk
    is FRESH WORK, and the receipt its sender is about to be handed promises a
    runtime ("held for the next runtime — it runs it"). So a re-note keeps the
    recorded attempts and pulls the next attempt no later than
    :data:`RETRY_BASE_S` from now (:func:`next_attempt_at_ms`), which is the
    ordering that makes the promise true without letting a stream of notes
    restart an already-short wait.
    """
    moment = _now_ms() if now_ms is None else now_ms
    if not str(session_id or ""):
        # A nameless obligation is unwritable AND unreadable: the supervisor keys
        # the spool it consults by this id, so a record without one would answer
        # "nobody owes anything" while sitting in the directory forever. Refused
        # rather than written, which leaves the spool row exactly as deliverable
        # as it was (an ordinary engage still reads it).
        return False
    existing = read_spooled_turn(config_dir, session_id) or {}
    record: dict[str, Any] = {
        "schema": SPOOLED_SCHEMA,
        "session_id": session_id,
        "cwd": cwd or str(existing.get("cwd") or ""),
        "noted_at_ms": existing.get("noted_at_ms") or moment,
        "updated_at_ms": moment,
        "rows": int(existing.get("rows") or 0) + max(1, int(rows)),
        "attempts": _attempts_of(existing),
    }
    # ONLY A RE-NOTE CARRIES A WAIT. A brand-new obligation has no
    # ``next_attempt_ms`` at all, which reads as "due now" (see
    # :func:`next_attempt_at_ms`) — the wait being shortened here belongs to the
    # walk a PREVIOUS row started, and inventing one for the first row would
    # delay the raise it is asking for by :data:`RETRY_BASE_S` for no reason.
    recorded_next = existing.get("next_attempt_ms")
    if isinstance(recorded_next, int) and not isinstance(recorded_next, bool):
        record["next_attempt_ms"] = min(recorded_next, moment + int(RETRY_BASE_S * 1000))
    if existing.get("last_error"):
        record["last_error"] = existing["last_error"]
    return _write(config_dir, session_id, record)


def clear_spooled_turn(
    config_dir: Path, session_id: str, *, expected_updated_at_ms: int | None = None
) -> bool:
    """Drop the obligation. Returns whether a record was removed.

    Called by whoever DISCHARGES it: the successor's own spool drain (the rows
    are delivered, so nothing is owed) and the supervisor's reconciliation (the
    spool no longer holds a row that asks for a turn). Best-effort, and never a
    reason to fail a delivery — a stale record costs one wasted engage, and the
    reconciliation on the next pass removes it.

    ``expected_updated_at_ms`` IS THE COMPARE-AND-DELETE GUARD (review round 1,
    R1-4). Both callers judge the record stale from a read that happened
    elsewhere — one in another PROCESS (the writer is the draining runtime, the
    clearer is its successor) — so an unconditional unlink can delete a record
    that a row landing in that window has just re-armed, which is defect (B)
    returning silently. A caller that holds the value it judged passes it, and a
    record whose ``updated_at_ms`` has moved since (a re-note, a failed attempt)
    is left alone. The window between this read and the unlink is a few
    microseconds rather than the caller's own read-to-unlink gap; the residual is
    stated in :func:`local_operator.wakes.supervisor._reconcile_spooled`, and it
    fails towards keeping a record that a later pass will drop rather than
    dropping one that is still owed.
    """
    if expected_updated_at_ms is not None:
        current = read_spooled_turn(config_dir, session_id)
        if current is None:
            return False
        if current.get("updated_at_ms") != expected_updated_at_ms:
            return False
    try:
        spooled_path(config_dir, session_id).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("could not clear the owed turn for %s", session_id, exc_info=True)
        return False


def note_attempt(config_dir: Path, session_id: str, *, error: str = "") -> None:
    """Record a failed attempt to raise this session, with its backoff.

    The supervisor's own bookkeeping, and the same shape ``deliveries`` uses for
    a fire it could not deliver: the record is a claim that the turn is still
    owed, so a failure must not erase it — it must make the next attempt later.
    """
    record = read_spooled_turn(config_dir, session_id)
    if record is None:
        return
    attempts = _attempts_of(record) + 1
    record["attempts"] = attempts
    record["last_attempt_ms"] = _now_ms()
    record["next_attempt_ms"] = _now_ms() + int(backoff_s(attempts) * 1000)
    if error:
        record["last_error"] = str(error)[:400]
    _write(config_dir, session_id, record)


def _attempts_of(record: Mapping[str, Any]) -> int:
    """``attempts`` as an int, or ``0`` for anything a foreign writer left there."""
    raw = record.get("attempts")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    return 0


def backoff_s(attempts: int) -> float:
    """Seconds to wait after ``attempts`` consecutive failures.

    Public because the supervisor's log line quotes it and the tests drive it
    directly, exactly as ``deliveries.backoff_s`` is used.
    """
    exponent = max(0, int(attempts) - 1)
    return min(RETRY_CAP_S, RETRY_BASE_S * (RETRY_FACTOR**exponent))


def next_attempt_at_ms(record: Mapping[str, Any]) -> int:
    """When this obligation may be acted on again. Never raises, never "never".

    DEFENSIVE ON PURPOSE, AND NOT FOR DECORATION. The record is written by several
    processes and read on the supervisor's hot path (``_due_sessions``,
    ``_has_fireable_wakes``, both of which run inside ``sweep``/``serve``), so a
    hand-edited, truncated or future-schema field must cost ONE session's turn
    rather than raise ``ValueError`` out of the whole pass — the contract
    ``_due_sessions`` states for the index and its sibling readers already honour
    (review round 1, R1-5, which caught this one raising).

    An unreadable schedule reads as DUE NOW, which is the direction that cannot
    lose work: the walk's own backoff bounds what that costs.
    """
    recorded = record.get("next_attempt_ms")
    if isinstance(recorded, int) and not isinstance(recorded, bool):
        return recorded
    return 0


def spool_has_owner_row(session_dir: Path) -> bool:
    """Whether this session's spool holds the OWNER's own words, which always run.

    The narrower question ``drop_owed_turn`` needs (see its docstring): a
    ``SOURCE_USER`` row is dischargeable by a raise — the successor's boot drain
    runs it — so a record that arrived for one must never be dropped on the
    grounds that the PEER rows beside it cannot be delivered yet.

    Same read and same never-raises contract as :func:`spool_owes_turn`, which it
    deliberately duplicates in one direction rather than parameterising: a
    predicate that answers two questions through a flag is one refactor away from
    answering the wrong one at one of its two call sites.
    """
    path = Path(session_dir) / INBOX_NAME
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and str(row.get(SOURCE_FIELD) or "") == SOURCE_USER:
            return True
    return False


def spool_owes_turn(session_dir: Path) -> bool:
    """Whether this session's spool still holds a row that asks for a turn.

    The read that keeps an obligation from outliving its cause, and it is
    deliberately generous in only one direction: a row is turn-asking when
    ``wake`` is true or ``source`` is the owner's own prompt
    (``inbox.SOURCE_USER``). A quiet note does NOT owe a turn — that is the
    design's own contract for it ("``wake=False`` means 'read this on your next
    turn', not 'start one now'", ``peer_send.deliver_peer_message``) — and a
    recall marker is explicitly not a message.

    Never raises: an unreadable spool is not the same fact as an empty one, and
    the caller treats both as "nothing owed" — the cost of the wrong reading here
    is one engage for a session that had nothing to run, against the cost of the
    loop it would otherwise keep alive.
    """
    path = Path(session_dir) / INBOX_NAME
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get(WAKE_FIELD):
            return True
        if str(row.get(SOURCE_FIELD) or "") == SOURCE_USER:
            return True
    return False


def _write(config_dir: Path, session_id: str, record: Mapping[str, Any]) -> bool:
    """Stage and replace one record. Never raises.

    Atomic (temp file plus ``os.replace``) like ``store`` and ``deliveries``: the
    supervisor may read this directory at any moment, and a torn file would be
    read as no obligation at all — the failure direction that loses work.
    """
    path = spooled_path(config_dir, session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{session_id}.",
            suffix=".json",
            delete=False,
        )
        with handle:
            json.dump(record, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
        return True
    except OSError:
        # Best-effort by contract with the caller: a spool write that landed must
        # not be turned into a failure because its index entry could not. The row
        # is still in the spool and an ordinary engage still delivers it.
        logger.warning("could not record the owed turn for %s", session_id, exc_info=True)
        return False
