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

import asyncio
import json
import uuid
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, ParamSpec, TypeVar

from local_operator.wakes.lock import WakeWriteLock, WakeLockBusy, WakeLockUnavailable

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
    _validate_session_id(session_id)
    return spooled_dir(config_dir) / f"{session_id}.json"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _validate_session_id(session_id: str) -> None:
    if not session_id or session_id in (".", "..") or Path(session_id).name != session_id:
        raise ValueError("invalid spool session id")


class SpoolWriteError(OSError):
    """The obligation was not durably settled; never receipt it as held success."""


def _session_lock(config_dir: Path, session_id: str, *, read: bool = False) -> WakeWriteLock:
    # Persistent session-specific inodes survive record deletion/recreation (ABA).
    # Mixed older, unlocked writers are outside this cooperative protocol.
    _validate_session_id(session_id)
    directory = spooled_dir(config_dir)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if read:
        return WakeWriteLock(directory, name=f".{session_id}.lock", timeout_s=0.0)
    return WakeWriteLock(directory, name=f".{session_id}.lock")


def _raw_read(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    try:
        data = json.loads(spooled_path(config_dir, session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == SPOOLED_SCHEMA else None


def _generation(value: object) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def read_spooled_turn(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """Publish a CAS token only from a fresh read under the transaction mutex.

    A busy/migration-failed record remains readable, but cannot authorize deletion.
    In particular a marker written before an inbox append must not publish a token.
    """
    try:
        with _session_lock(config_dir, session_id, read=True):
            record = _raw_read(config_dir, session_id)
            if record is not None and not _generation(record.get("generation")):
                record["generation"] = uuid.uuid4().hex
                if not _write(config_dir, session_id, record):
                    record["generation"] = None
            return record
    except (WakeLockBusy, WakeLockUnavailable, OSError):
        record = _raw_read(config_dir, session_id)
        if record is not None:
            record["generation"] = None
        return record


def read_spooled(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Retain readable busy records, withholding only their deletion authority."""
    out: dict[str, dict[str, Any]] = {}
    try:
        paths = sorted(spooled_dir(config_dir).glob("*.json"))
    except OSError:
        return out
    for path in paths:
        record = read_spooled_turn(config_dir, path.stem)
        if record is not None:
            out[path.stem] = record
    return out


P = ParamSpec("P")
T = TypeVar("T")


async def await_spool_write(function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Join submitted storage even on cancellation before relinquishing ownership.

    Shield alone is insufficient: a cancelled await does not stop its OS thread.
    Repeated cancellation is remembered, never allowed to orphan a submitted write.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                result = task.result()
                break
    if cancelled:
        raise asyncio.CancelledError
    return result


def _note_locked(
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
    existing = _raw_read(config_dir, session_id) or {}
    record: dict[str, Any] = {
        "schema": SPOOLED_SCHEMA,
        "generation": uuid.uuid4().hex,
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
    if not _write(config_dir, session_id, record):
        raise SpoolWriteError("could not persist spool obligation")
    return True


def _raise_reader(config_dir: Path, session_id: str) -> None:
    """Activation is deliberately outside the per-session transaction mutex."""
    try:
        from local_operator.wakes.install import ensure_supervisor_installed

        ensure_supervisor_installed(Path(config_dir))
    except Exception:  # noqa: BLE001 — the raiser is best-effort, the record is not
        logger.debug("could not raise the wake supervisor for %s", session_id, exc_info=True)


def note_spooled_turn(config_dir: Path, session_id: str, *, cwd: str = "", rows: int = 1,
                      now_ms: int | None = None) -> bool:
    """Persist a fresh opaque revision off-loop at awaited async call sites."""
    try:
        with _session_lock(config_dir, session_id):
            noted = _note_locked(config_dir, session_id, cwd=cwd, rows=rows, now_ms=now_ms)
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        raise SpoolWriteError("spool obligation lock unavailable") from exc
    if noted:
        _raise_reader(config_dir, session_id)
    return noted


_OMITTED = object()


def clear_spooled_turn(config_dir: Path, session_id: str, *,
                       expected_generation: object = _OMITTED,
                       expected_updated_at_ms: object = _OMITTED) -> bool:
    """Explicit omission is unconditional; old timestamp-only CAS fails closed."""
    if expected_updated_at_ms is not _OMITTED and expected_generation is _OMITTED:
        return False
    return _clear_guarded(config_dir, session_id, expected_generation=expected_generation)


def _clear_guarded(config_dir: Path, session_id: str, *, expected_generation: object,
                   session_dir: Path | None = None, owner_only: bool = False) -> bool:
    # Predicate and generation share the admission mutex: an earlier no-row
    # decision must not authorize deletion of a later, fully committed admission.
    try:
        with _session_lock(config_dir, session_id, read=True):
            current = _raw_read(config_dir, session_id)
            if current is None:
                return False
            if expected_generation is not _OMITTED:
                if not _generation(expected_generation) or current.get("generation") != expected_generation:
                    return False
            if session_dir is not None:
                owed = spool_has_owner_row(session_dir) if owner_only else spool_owes_turn(session_dir)
                if owed:
                    return False
            spooled_path(config_dir, session_id).unlink()
            return True
    except (WakeLockBusy, WakeLockUnavailable, OSError):
        return False


def clear_if_no_owed_turn(config_dir: Path, session_id: str, session_dir: Path, *,
                         expected_generation: object) -> bool:
    """Normal settlement and supervisor reconciliation use the owed-row authority."""
    return _clear_guarded(config_dir, session_id, expected_generation=expected_generation,
                          session_dir=session_dir)


def clear_if_no_owner_row(config_dir: Path, session_id: str, session_dir: Path, *,
                         expected_generation: object) -> bool:
    """Peer deferral deliberately preserves the stronger owner-row distinction."""
    return _clear_guarded(config_dir, session_id, expected_generation=expected_generation,
                          session_dir=session_dir, owner_only=True)


def note_attempt(config_dir: Path, session_id: str, *, error: str = "") -> None:
    """Backoff persistence completes before the engagement owner releases its key."""
    try:
        with _session_lock(config_dir, session_id):
            record = _raw_read(config_dir, session_id)
            if record is None:
                return
            attempts = _attempts_of(record) + 1
            record.update(generation=uuid.uuid4().hex, attempts=attempts,
                          last_attempt_ms=_now_ms(),
                          next_attempt_ms=_now_ms() + int(backoff_s(attempts) * 1000))
            if error:
                record["last_error"] = str(error)[:400]
            if not _write(config_dir, session_id, record):
                raise SpoolWriteError("could not persist spool backoff")
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        raise SpoolWriteError("spool attempt lock unavailable") from exc


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


def spooled_wake_fires(session_dir: Path) -> tuple[int, float | None]:
    """How many spooled wake FIRES await delivery, and the oldest write.

    The count ``lop wake status`` renders. A runtime that drains for a replaced
    build SPOOLS the wakes that fire meanwhile (``Session.retire_wakes_to_inbox``)
    and nothing reports them until a successor boots and drains — the
    2026-10-01/02 incident read as a live, freshly-fired session on every
    surface while ~14 h of fires sat here. This is the read that makes them
    visible; :func:`spool_owes_turn` beside it answers the coarser question the
    supervisor needs ("does anything here owe a turn?").

    A ROW COUNTS WHEN ``wake`` IS TRUE AND ITS ``source`` IS NOT THE OWNER, and
    both exclusions are load-bearing rather than fussy (review round 1, R2):

    * ``serving._spool_for_successor`` spools the owner's own prompt and steer
      with ``wake=True`` on the committed-arm paths so the successor RUNS
      them — but a user row's receipt governs, and counting one here would
      render the operator's own queued message as a fire;
    * a coalesced row stands for several firings (``wake_fires`` — see
      ``session.runtime.inbox.coalesce_wake_rows``), so it counts as its own
      ``wake_fires`` when that is a positive int, and as ONE otherwise — zero
      would hide the row, and ``True`` is an ``int`` in Python and must not
      read as a count (review round 1, Q1).

    Same read and same never-raises contract as the predicates above: the
    oldest ``written_at`` (epoch seconds) is tracked among the COUNTED rows —
    ``None`` when no counted row carries a usable one. A missing file is zero
    fires; a torn or unparseable line is skipped, because the writer may be
    appending while this reads and a status surface must render the rest of
    the picture rather than die on the line being written.
    """
    path = Path(session_dir) / INBOX_NAME
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return (0, None)
    fires = 0
    oldest: float | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or not row.get(WAKE_FIELD):
            continue
        if str(row.get(SOURCE_FIELD) or "") == SOURCE_USER:
            continue
        fires += _row_fires(row.get("wake_fires"))
        written = row.get("written_at")
        if isinstance(written, (int, float)) and not isinstance(written, bool):
            value = float(written)
            if oldest is None or value < oldest:
                oldest = value
    return (fires, oldest)


def _row_fires(raw: Any) -> int:
    """``wake_fires`` off a persisted row, for the status count.

    Only a POSITIVE int counts as N; everything else reads as one, the
    uncoalesced default. Deliberately not ``inbox._fire_count``'s
    int-coercing form: that runs on rows a runtime will deliver and may take a
    numeric string, while this runs on raw JSON for a status line and stays
    strictly typed — and ``isinstance(True, int)`` is true in Python, so the
    bool check is what keeps a flag from reading as a count.
    """
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        return 1
    return raw


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
