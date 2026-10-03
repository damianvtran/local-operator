"""Arm and cancel a session's monitors from OUTSIDE the session (contract §12).

The monitors family's external writers: :func:`arm_monitor` (the desktop arm
route) and :func:`cancel_monitor` (``lop monitor cancel`` and the desktop
cancel route), modeled on :mod:`local_operator.wakes.arm` — the same
three-part invariant and the same write order, because the failure modes are
identical in kind:

- **The base is the TRANSCRIPT, never the index.** An append REPLACES the
  session's monitor list (the ``monitor_schedules`` custom entry), so a stale
  base — the index lags, and is absent for a session that has not been opened —
  would write back a list that resurrects a cancelled watch or drops a live
  one.
- **Transcript first, index second.** Only the transcript append may fail this
  request; the index is derived and rebuilt on the next open, so its write is
  best-effort and reported in the outcome rather than raised (the wake
  writer's asymmetry, verbatim).
- **Post-write verification.** Last-writer-wins on a full-list snapshot is
  safe by construction only while nothing else writes, and a live session's
  own ``_persist_monitor_schedules`` is not a party to the lock. After the
  append this re-reads the latest entry and checks it is OURS: a mismatch
  means a writer that does not take the lock landed on top, and the answer is
  one retry from the new base and then a conflict refusal rather than a
  success report that lost the user's cancel.

``lop monitor cancel`` and the desktop monitor routes are the callers. There
is deliberately NO request-id settling or rollback-journal machinery here —
the wake module carries those for a retry contract (the desktop route's
``request_id``) that monitor writes answer structurally instead: a retried
cancel's worst case is an honest ``no monitor with id`` refusal about a watch
that is already gone, which is the state the user asked for, and a retried
ARM is idempotent by the dedupe identity itself (an identical spec is
answered with the existing row, never a second write).

The owner guard mirrors ``wakes.arm._refuse_if_owned``: a write behind a live
owner is undone by that owner's next persist — its in-memory list was loaded
before this cancel — so the writer refuses with a sentence that names who owns
the conversation, rather than reporting a cancel that will resurrect. The full
reasoning for the three owner states lives in the wake module; this is the
same decision tree for the monitors family.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from local_operator.monitors.spec import MONITOR_SCHEDULES_CUSTOM_TYPE, MonitorSpec
from local_operator.session.transcript import read_latest_custom_entry
from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable, WakeWriteLock

logger = logging.getLogger(__name__)

#: The per-session mutex file, beside the transcript's other sidecars. Dotted
#: and DISTINCT from the wake writer's ``.wake-write.lock``: the two families'
#: read-merge-append sequences replace different snapshots, so the mutual
#: exclusion that matters is per-family (the lock module's own narrowing
#: argument). The class is shared rather than copied so the platform handling
#: cannot drift (the ``aida.state`` precedent for ``name=``).
MONITOR_LOCK_NAME = ".monitor-write.lock"

#: Statuses carried HERE rather than mapped by each caller, so the CLI (which
#: prints the sentence and exits 1) and any future route cannot disagree about
#: what a refusal means.
STATUS_SESSION_NOT_FOUND = 404
STATUS_MONITOR_NOT_FOUND = 404
STATUS_CONFLICT = 409
#: A malformed request (a bad duration, a bad regex) reads as a 422; everything
#: well-formed but unsupportable (the interval floor, a past ``until``, a
#: target that is not read-only) is a 409 conflict with what can be watched —
#: the wake writer's split, verbatim.
STATUS_INVALID = 422
#: 503 for the owner states and a contended lock: every one means "nothing was
#: written, and retrying is what fixes it".
STATUS_WRITE_BUSY = 503
STATUS_OWNER_BUSY = 503

NOTHING_WRITTEN = "Nothing was written."

#: The refusal for an owner that EXISTS but is not answering; one sentence,
#: shared by the two places that can meet the state (the wake twin's rule).
WEDGED_MESSAGE = (
    "This conversation's runtime is not responding. Retry in a moment, or stop that conversation."
)

#: The refusal for an owner that is ALIVE and ANSWERING. One sentence for both
#: places that can meet the state — the desktop route resolves the owner before
#: the lock, the writer's own guard re-meets it under it — because a refusal
#: rendered two different ways depending on which check caught it is the split
#: the wake twin was already bitten by. "Change them", not "cancel them": the
#: desktop route arms through this writer too.
OWNER_ANSWERED_MESSAGE = (
    "This conversation is open in a running session, which owns its monitors. "
    + NOTHING_WRITTEN
    + " Retry in a moment, or change them from that session."
)


class MonitorWriteError(Exception):
    """A refusal the caller can render verbatim (the ``WakeWriteError`` twin).

    ``message`` is always the user-facing sentence; ``status``/``code`` are for
    callers that map refusals onto an API.
    """

    def __init__(self, message: str, *, status: int, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class MonitorWriteOutcome:
    """What a successful write did, for the caller's receipt.

    ``index_written`` is "the derived index now reflects this change, as far as
    this process can tell": false only when the index write itself failed — an
    empty remainder REMOVES the entry (which is also what releases the cleanup
    reap guard, §11.5), and ``index_path`` is then empty. The transcript still
    won and the next open heals the index, which is why that is a reported
    field rather than an error.
    """

    session_id: str
    monitor_id: str
    name: str
    remaining: int
    index_written: bool = False
    index_path: str = ""
    #: Arm only: the first check's instant (§5.4's fresh-counters rule), or
    #: ``None`` on a cancel.
    next_due_at: int | None = None
    #: Arm only: an identical spec was ALREADY armed, so the row was left
    #: alone (nothing was appended) and ``monitor_id`` names the existing row.
    already_armed: bool = False
    #: Arm only: the identical spec existed DISABLED; this arm reset its
    #: counters (its id and snapshot baseline survive), which is §11.3's
    #: "re-arm to reactivate".
    reactivated: bool = False


async def cancel_monitor(
    config_dir: Path,
    session_id: str,
    monitor_id: str,
) -> MonitorWriteOutcome:
    """Drop one monitor from ``session_id``'s standing watches.

    Refuses a session that does not exist rather than inventing one (the wake
    writer's rule: an index row with no transcript fires into nothing), and
    refuses to write behind a live owner (:func:`_refuse_if_owned`).
    """
    session_dir = Path(config_dir) / "sessions" / session_id
    if not await asyncio.to_thread(session_dir.is_dir):
        raise MonitorWriteError(
            f"no session {session_id!r}", status=STATUS_SESSION_NOT_FOUND, code="session_not_found"
        )

    # One writer at a time, per session, per family. The lock spans the read,
    # the guard, the append, the verification and the index write, and a
    # timed-out acquire is a REFUSAL rather than "carry on unlocked": the
    # failure that would follow is the lost cancel, and a retryable 503 is the
    # honest answer to contention.
    lock = WakeWriteLock(session_dir, name=MONITOR_LOCK_NAME)
    try:
        await asyncio.to_thread(lock.acquire)
    except WakeLockBusy as busy:
        raise MonitorWriteError(
            str(busy), status=STATUS_WRITE_BUSY, code="monitor_write_busy"
        ) from None
    except WakeLockUnavailable as unusable:
        raise MonitorWriteError(
            f"Cannot cancel a monitor for this conversation: {unusable}. "
            "Restore write permission on its directory, or reopen the conversation "
            "from a writable location, and retry.",
            status=STATUS_CONFLICT,
            code="monitor_write_unavailable",
        ) from None
    try:
        remaining, name = await _cancel_locked(config_dir, session_dir, session_id, monitor_id)
        previous = await asyncio.to_thread(_read_index_entry, config_dir, session_id)
        index_written = False
        index_path = ""
        try:
            written = await asyncio.to_thread(
                _write_index,
                config_dir,
                session_id,
                _resolved_cwd(session_dir, previous),
                remaining,
                previous,
            )
            index_written = True
            index_path = str(written) if written is not None else ""
        except Exception:  # noqa: BLE001 — the transcript already has it (store.py's contract)
            logger.warning(
                "could not update the monitor index entry for %s", session_id, exc_info=True
            )
    finally:
        await asyncio.to_thread(lock.release)

    # The cancelled monitor's derived state files are cache-like (§10.3): the
    # row is gone from the transcript and the index, and a later re-arm of the
    # same spec rebuilds both. Best-effort, off the loop, like every write
    # here.
    await asyncio.to_thread(_remove_state, config_dir, session_id, monitor_id)

    return MonitorWriteOutcome(
        session_id=session_id,
        monitor_id=monitor_id,
        name=name,
        remaining=len(remaining),
        index_written=index_written,
        index_path=index_path,
    )


async def arm_monitor(
    config_dir: Path,
    session_id: str,
    request: Mapping[str, Any],
    *,
    cwd: str | None = None,
    now_ms: int | None = None,
) -> MonitorWriteOutcome:
    """Add one monitor to ``session_id``'s standing watches. ``request`` is the
    same create mapping the agent's ``monitor`` tool takes, validated by the
    same function (``build_monitor_spec``), so a watch armed from the desktop
    and one armed by the model are validated and allocated identically.

    Refuses a session that does not exist rather than inventing one (the wake
    writer's rule), and refuses to write behind a live owner
    (:func:`_refuse_if_owned`): there is deliberately no routed command ladder
    for monitors, so a live conversation's own ``monitor`` tool is the only
    writer that may touch its list while it runs — an append from here would be
    deleted by that session's next persist while this writer had reported 200,
    and until then nothing would tick the watch either.
    """
    session_dir = Path(config_dir) / "sessions" / session_id
    if not await asyncio.to_thread(session_dir.is_dir):
        raise MonitorWriteError(
            f"no session {session_id!r}", status=STATUS_SESSION_NOT_FOUND, code="session_not_found"
        )
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)

    # One writer at a time, per session, per family — the cancel twin's lock,
    # held across the read, the guards, the append, the verification and the
    # index write, because the append REPLACES the whole list and two writers
    # that interleave their reads lose or duplicate a row.
    lock = WakeWriteLock(session_dir, name=MONITOR_LOCK_NAME)
    try:
        await asyncio.to_thread(lock.acquire)
    except WakeLockBusy as busy:
        raise MonitorWriteError(
            str(busy), status=STATUS_WRITE_BUSY, code="monitor_write_busy"
        ) from None
    except WakeLockUnavailable as unusable:
        raise MonitorWriteError(
            f"Cannot arm a monitor for this conversation: {unusable}. "
            "Restore write permission on its directory, or reopen the conversation "
            "from a writable location, and retry.",
            status=STATUS_CONFLICT,
            code="monitor_write_unavailable",
        ) from None
    try:
        # The index is read for the TWO things this write needs and nothing
        # else: the cwd the checks run in (``_resolved_cwd`` — the transcript
        # carries no such answer) and the unknown keys the rewrite must
        # preserve (``stopped_at``). It is still NOT the base for the list:
        # that is ``_read_rows``' job, for the reasons its docstring gives.
        previous = await asyncio.to_thread(_read_index_entry, config_dir, session_id)
        resolved_cwd = cwd or _resolved_cwd(session_dir, previous)
        result = await _arm_locked(
            config_dir, session_dir, session_id, request, cwd=resolved_cwd, now=now
        )
        index_written = False
        index_path = ""
        try:
            index_rows = await asyncio.to_thread(
                _compose_index_rows, config_dir, session_id, result.rows
            )
            written = await asyncio.to_thread(
                _write_index, config_dir, session_id, resolved_cwd, index_rows, previous
            )
            index_written = True
            index_path = str(written) if written is not None else ""
        except Exception:  # noqa: BLE001 — the transcript already has it (store.py's contract)
            logger.warning(
                "could not update the monitor index entry for %s", session_id, exc_info=True
            )
    finally:
        await asyncio.to_thread(lock.release)

    return MonitorWriteOutcome(
        session_id=session_id,
        monitor_id=result.monitor_id,
        name=result.name,
        remaining=len(result.rows),
        index_written=index_written,
        index_path=index_path,
        next_due_at=result.next_due_at,
        already_armed=result.already_armed,
        reactivated=result.reactivated,
    )


async def _cancel_locked(
    config_dir: Path,
    session_dir: Path,
    session_id: str,
    monitor_id: str,
) -> tuple[list[MonitorSpec], str]:
    """read -> mutate -> guard -> append -> verify -> guard, under the lock.

    Returns the remaining list and the cancelled row's name. The loop runs at
    most twice; a mismatch with a non-party writer rebases once and then
    refuses (the wake ``_mutate_locked`` shape, without that module's
    request-id settling — see the module docstring). One settle IS carried
    over from the wake semantics: a rebase that finds the monitor already gone
    reports success, because absence is exactly the state the request asked
    for — refusing there would say "no monitor" about a row this very call
    removed.
    """
    remaining: list[MonitorSpec] = []
    name = ""
    seen = False
    for attempt in (0, 1):
        rows, next_seq = await asyncio.to_thread(_read_rows, session_dir)
        match = next((spec for spec in rows if spec.id == monitor_id), None)
        if match is None:
            if attempt and seen:
                # THIS REQUEST'S CHANGE IS ALREADY IN THIS BASE: our earlier
                # append (or a peer that saw it) removed the row, and the
                # newest snapshot is a list without it, which is the outcome
                # the caller asked for.
                logger.info("monitor cancel for %s is already in the base it read", session_id)
                return rows, name
            known = ", ".join(spec.id for spec in rows) or "none"
            raise MonitorWriteError(
                f"No monitor with id '{monitor_id}' on this conversation (known: {known}).",
                status=STATUS_MONITOR_NOT_FOUND,
                code="monitor_not_found",
            )
        seen = True
        name = match.name
        remaining = [spec for spec in rows if spec.id != monitor_id]
        await _refuse_if_owned(config_dir, session_id)
        entry_id = await _append(session_dir, remaining, next_seq)
        try:
            # THE POST-APPEND GUARD (the wake twin's review wound): a runtime
            # can claim the session inside the append and republish its stale
            # list on its next persist, deleting this cancel from the
            # transcript AND the index. Asking again here makes that a refusal
            # instead of a silent loss, and the append is undone before the
            # refusal so the retry it asks for is honest.
            await _refuse_if_owned(config_dir, session_id)
        except MonitorWriteError:
            await _append(session_dir, rows, next_seq)
            raise
        if await asyncio.to_thread(_latest_entry_id, session_dir) == entry_id:
            return remaining, name
        if attempt:
            # A writer that does not take the lock (a live session's own
            # persist) landed on top of both attempts; reconciling beats
            # reporting a cancel that may not be in effect.
            raise MonitorWriteError(
                "The conversation changed while your change was being applied. "
                "Reconcile its monitor list before retrying.",
                status=STATUS_CONFLICT,
                code="monitor_write_conflict",
            )
    raise MonitorWriteError(  # pragma: no cover - the loop always returns or raises
        "The monitor list could not be settled.",
        status=STATUS_CONFLICT,
        code="monitor_write_conflict",
    )


@dataclass(frozen=True)
class _ArmResult:
    """What one arm produced, before the index write: the receipt's facts and
    the list the index should carry (the read list for a no-op arm, the
    appended one for a create)."""

    rows: list[MonitorSpec]
    monitor_id: str
    name: str
    next_due_at: int | None
    already_armed: bool = False
    reactivated: bool = False


async def _arm_locked(
    config_dir: Path,
    session_dir: Path,
    session_id: str,
    request: Mapping[str, Any],
    *,
    cwd: str,
    now: int,
) -> _ArmResult:
    """read -> validate -> dedupe -> cap -> guard -> append -> verify -> guard.

    The same order as ``MonitorScheduler.create`` (shape first, so a bad
    request is a sentence and never a half-armed row; dedupe second, because an
    identical spec is an answer rather than an error; then the cap; then the
    storm guard), and the same append discipline as ``_cancel_locked`` — the
    loop runs at most twice; a mismatch with a non-party writer rebases once
    and then refuses.

    TWO settles differ from cancel's, both because this is an ADD: a rebase
    that finds the request's own spec already in the base reports SUCCESS (the
    dedupe identity is also the retry's idempotency key — that request's write,
    or a peer's identical one, is in force), and a base whose matching spec is
    DISABLED is reactivated (counters reset) exactly as the in-session flow
    does, rather than duplicated beside.
    """
    from local_operator.monitors.readonly import external_monitor_verdict
    from local_operator.monitors.settings import read_monitor_settings
    from local_operator.monitors.spec import (
        allocate_monitor_id,
        build_monitor_spec,
        spec_identity,
    )

    settings = read_monitor_settings()
    for attempt in (0, 1):
        rows, stored_next_seq = await asyncio.to_thread(_read_rows, session_dir)
        monitor_id, bumped_next_seq = allocate_monitor_id(
            [spec.id for spec in rows], high_water=stored_next_seq
        )
        outcome = build_monitor_spec(
            dict(request),
            monitor_id=monitor_id,
            now_ms=now,
            settings=settings,
            cwd=cwd,
            validate=external_monitor_verdict,
        )
        if "error" in outcome:
            raise _refusal(outcome)
        spec = outcome["spec"]
        identity = spec_identity(spec.tool, spec.arguments)
        match = next(
            (row for row in rows if spec_identity(row.tool, row.arguments) == identity), None
        )
        if match is not None:
            counters = (
                await asyncio.to_thread(_read_counters, config_dir, session_id, match.id) or {}
            )
            if counters.get("disabled"):
                # RE-ARM TO REACTIVATE (§11.3): reset the failures and keep the
                # id and the snapshot baseline. The transcript is NOT re-appended
                # because the list did not move — the counters file and the index
                # are what did.
                fresh = _fresh_first_counters(match.id, now)
                await asyncio.to_thread(_write_counters, config_dir, session_id, match.id, fresh)
                return _ArmResult(
                    rows=rows,
                    monitor_id=match.id,
                    name=match.name,
                    next_due_at=_due_of(fresh),
                    reactivated=True,
                )
            # THE DUPLICATE ANSWER: the arm the caller asked for is already in
            # force. Nothing is appended — a second row polling one call is the
            # duplication the identity exists to settle (§11.4).
            return _ArmResult(
                rows=rows,
                monitor_id=match.id,
                name=match.name,
                next_due_at=_due_of(counters),
                already_armed=True,
            )
        if len(rows) >= settings.max_monitors:
            raise MonitorWriteError(
                f"monitor limit reached ({settings.max_monitors} per session) — "
                "cancel one first (monitor list).",
                status=STATUS_CONFLICT,
                code="monitor_refused",
            )
        same_name = sum(1 for row in rows if row.name == spec.name)
        if same_name >= 2:
            raise MonitorWriteError(
                f"three monitors named '{spec.name}' is a storm — cancel one or "
                "use a distinct name.",
                status=STATUS_CONFLICT,
                code="monitor_refused",
            )
        await _refuse_if_owned(config_dir, session_id)
        entry_id = await _append(session_dir, [*rows, spec], bumped_next_seq)
        try:
            # THE POST-APPEND GUARD (the cancel twin's), for the same wound: a
            # runtime can claim the session inside the append and republish its
            # stale list on its next persist, deleting this arm from the
            # transcript AND the index. Asking again makes that a refusal
            # instead of a silent loss, and the append is undone first so the
            # retry it asks for is honest.
            await _refuse_if_owned(config_dir, session_id)
        except MonitorWriteError:
            await _append(session_dir, rows, stored_next_seq)
            raise
        if await asyncio.to_thread(_latest_entry_id, session_dir) == entry_id:
            fresh = _fresh_first_counters(spec.id, now)
            await asyncio.to_thread(_write_counters, config_dir, session_id, spec.id, fresh)
            return _ArmResult(
                rows=[*rows, spec],
                monitor_id=spec.id,
                name=spec.name,
                next_due_at=_due_of(fresh),
            )
        if attempt:
            # A writer that does not take the lock landed on top of both
            # attempts; reconciling beats reporting an arm that may not be in
            # effect. (The rebase above catches the ordinary case — the spec is
            # in the base — so this is the genuinely unsettled one.)
            raise MonitorWriteError(
                "The conversation changed while your change was being applied. "
                "Reconcile its monitor list before retrying.",
                status=STATUS_CONFLICT,
                code="monitor_write_conflict",
            )
    raise MonitorWriteError(  # pragma: no cover - the loop always returns or raises
        "The monitor list could not be settled.",
        status=STATUS_CONFLICT,
        code="monitor_write_conflict",
    )


async def _refuse_if_owned(config_dir: Path, session_id: str) -> None:
    """Refuse to append to a transcript a runtime process owns.

    The ``wakes.arm._refuse_if_owned`` decision tree for the monitors family,
    in the same order (wedged first because it is also a live pid): a live
    owner's in-memory list was loaded before this cancel, so its next persist
    would overwrite the row back and the monitor would fire again while the
    CLI had reported it cancelled.
    """
    from local_operator.mobile.attach_client import (
        dialable_record_exists,
        find_runtime_record,
    )
    from local_operator.wakes.supervisor import wedged_runtime

    wedged = await asyncio.to_thread(wedged_runtime, Path(config_dir), session_id)
    if wedged is not None:
        raise MonitorWriteError(
            WEDGED_MESSAGE, status=STATUS_OWNER_BUSY, code="monitor_owner_wedged"
        )
    # Off the loop: this reads ``.session.pid`` and, for the zombie proof,
    # forks a ``ps`` — the same probe the attach paths use, so this guard and
    # the process that would take the session agree about who is holding it.
    _record, owner = await asyncio.to_thread(find_runtime_record, Path(config_dir), session_id)
    if owner is None:
        return
    dialable = await asyncio.to_thread(dialable_record_exists, Path(config_dir), owner)
    if dialable is None:
        # The registry could not be read, so "no usable record" and "a record
        # I cannot see" are indistinguishable from here; the refusal names the
        # unread registry and asks for the retry that would resolve it.
        raise MonitorWriteError(
            f"This conversation is held by another process (pid {owner}) and the runtime "
            "registry could not be read to say whether it is answering, so the monitors "
            "were left alone. Retry in a moment.",
            status=STATUS_OWNER_BUSY,
            code="monitor_owner_present",
        )
    if dialable:
        raise MonitorWriteError(
            OWNER_ANSWERED_MESSAGE, status=STATUS_OWNER_BUSY, code="monitor_owner_present"
        )
    raise MonitorWriteError(
        "This conversation's monitor file is claimed by process "
        f"{owner}, which does not answer as a runtime — either a conversation open "
        "in an older build, or a stale owner marker left by a pid the system has "
        "since reused. " + NOTHING_WRITTEN + " If nothing has this conversation open, "
        f"the marker at {Path(config_dir) / 'sessions' / session_id / '.session.pid'} is stale "
        "and can be removed; otherwise change its monitors from that session.",
        status=STATUS_OWNER_BUSY,
        code="monitor_owner_present",
    )


# ---------------------------------------------------------------------------
# Filesystem steps (each run off the event loop by its caller)
# ---------------------------------------------------------------------------


def _read_rows(session_dir: Path) -> tuple[list[MonitorSpec], Any]:
    """The session's monitors and its persisted ``next_seq``, from the
    TRANSCRIPT's latest snapshot.

    The index is deliberately not consulted even as a fallback (the wake
    reader's rule): it lags, it is absent for a session that has never been
    opened, and the append below REPLACES the list — so a stale base is how a
    live watch is resurrected by an unrelated cancel.

    Read through :func:`read_latest_custom_entry`, not ``Transcript``: the
    latter parses the whole journal to answer one question about one row, and
    this function runs twice per attempt (the wake module measured 8.84 s and
    885 MB of traced peak against 17 ms for the bounded reader on the
    operator's 262 MB transcript).

    ``next_seq`` rides the entry (the persisted high-water mark): the cancel
    must write it back UNCHANGED, because an append that dropped it would let
    a later arm reissue a cancelled id.
    """
    entry = read_latest_custom_entry(session_dir, MONITOR_SCHEDULES_CUSTOM_TYPE)
    if entry is None:
        return [], None
    details = entry.payload.get("details")
    if not isinstance(details, Mapping):
        return [], None
    raw_rows = details.get("monitors")
    if raw_rows is None:
        return [], details.get("next_seq")
    if not isinstance(raw_rows, list):
        # An entry that EXISTS and is not a list is corrupt, not empty, and
        # guessing there would mean writing someone's armed watches away.
        raise MonitorWriteError(
            "The session's stored monitor list is unreadable; reopen the session and retry.",
            status=STATUS_CONFLICT,
            code="monitor_store_corrupt",
        )
    rows: list[MonitorSpec] = []
    for raw in raw_rows:
        try:
            rows.append(MonitorSpec.model_validate(raw))
        except Exception:  # noqa: BLE001 — one bad row costs one row, never the list
            logger.warning("dropping malformed persisted monitor spec: %r", raw)
    return rows, details.get("next_seq")


def _latest_entry_id(session_dir: Path) -> str | None:
    # The same bounded reader as ``_read_rows``, for the same reason: this asks
    # one row's id, and a whole-journal parse is what the bounded reader
    # exists to avoid.
    entry = read_latest_custom_entry(session_dir, MONITOR_SCHEDULES_CUSTOM_TYPE)
    return entry.id if entry is not None else None


async def _append(session_dir: Path, rows: list[MonitorSpec], next_seq: Any) -> str:
    """Append the new full-list snapshot and return its entry id.

    The ONLY step allowed to fail the write. ``next_seq`` is re-emitted when
    the read produced one, so a cancel can never lower the high-water mark
    (omitted only when the entry never carried it, which only pre-``next_seq``
    transcripts do).
    """
    payload: dict[str, Any] = {"monitors": [spec.model_dump() for spec in rows]}
    if next_seq is not None:
        payload["next_seq"] = next_seq
    transcript = await asyncio.to_thread(_transcript, session_dir)
    entry = await transcript.append_custom(MONITOR_SCHEDULES_CUSTOM_TYPE, payload)
    return entry.id


def _transcript(session_dir: Path):
    from local_operator.session.transcript import Transcript

    # ``defer_materialise``: a read must not create a session directory, and
    # the append path recreates a vanished one itself. This module never
    # invents a session — the caller refuses one that is not there.
    return Transcript(session_dir, defer_materialise=True)


def _read_index_entry(config_dir: Path, session_id: str) -> dict[str, Any]:
    from local_operator.monitors.store import read_entry

    try:
        return read_entry(Path(config_dir), session_id) or {}
    except Exception:  # noqa: BLE001 — an unreadable entry preserves nothing
        logger.warning("could not read the monitor index entry for %s", session_id, exc_info=True)
        return {}


def _write_index(
    config_dir: Path,
    session_id: str,
    cwd: str,
    rows: Sequence[Any],
    preserve: Mapping[str, Any],
):
    from local_operator.monitors.store import write_entry

    # ``preserve`` carries the keys this module does not own — ``stopped_at``
    # (a stopped session's monitors stay parked through this write; dropping it
    # would un-park the whole session as a side effect of cancelling one
    # watch). An empty remainder REMOVES the entry, which is also what releases
    # the cleanup reap guard (§11.5). ``rows`` may be specs (cancel: the
    # remainder is the list) or dicts (arm: the spec plus its counters/health —
    # ``_compose_index_rows`` — because the arm receipt and the listing must
    # show the new row's due instant); ``write_entry`` accepts either.
    return write_entry(
        Path(config_dir),
        session_id,
        cwd=cwd,
        monitors=rows,
        preserve=preserve,
    )


def _refusal(outcome: Mapping[str, Any]) -> MonitorWriteError:
    """A ``build_monitor_spec`` error as a refusal. Malformed (an unparseable
    duration, a bad regex) is the caller's input to fix and reads as a 422;
    everything else — the interval floor, a past time, a target that is not
    read-only — is a conflict with what can be watched and reads as a 409."""
    malformed = bool(outcome.get("malformed"))
    return MonitorWriteError(
        str(outcome.get("error") or "the monitor could not be armed"),
        status=STATUS_INVALID if malformed else STATUS_CONFLICT,
        code="monitor_invalid" if malformed else "monitor_refused",
    )


def _fresh_first_counters(monitor_id: str, now: int) -> dict[str, Any]:
    """A new (or reactivated) monitor's counters, its first check scheduled in
    the scheduler's own window (§5.4), so an externally armed watch starts on
    the same clock one armed from inside would."""
    from local_operator.monitors.scheduler import (
        FIRST_CHECK_MAX_MS,
        FIRST_CHECK_MIN_MS,
        fresh_counters,
    )

    due = now + int(random.uniform(float(FIRST_CHECK_MIN_MS), float(FIRST_CHECK_MAX_MS)))
    return fresh_counters(monitor_id, due)


def _due_of(counters: Mapping[str, Any]) -> int | None:
    """A tolerant ``next_due_at`` read — the counters file is another
    process's output, so a junk value reads as absent rather than travelling
    onto the receipt as a number nobody reported."""
    due = counters.get("next_due_at")
    if isinstance(due, bool) or not isinstance(due, int):
        return None
    return due


def _read_counters(config_dir: Path, session_id: str, monitor_id: str) -> dict[str, Any] | None:
    from local_operator.monitors import state as monitor_state

    try:
        return monitor_state.read_counters(Path(config_dir), session_id, monitor_id)
    except Exception:  # noqa: BLE001 — absent and unreadable are the same state here
        logger.debug("could not read monitor counters for %s", monitor_id, exc_info=True)
        return None


def _write_counters(
    config_dir: Path, session_id: str, monitor_id: str, counters: Mapping[str, Any]
) -> None:
    """Best-effort counters write (the scheduler's own ``_write_counters``
    swallows and warns for the same reason: the row is already armed, and a
    failed derived write must not fail the arm)."""
    try:
        from local_operator.monitors import state as monitor_state

        monitor_state.write_counters(Path(config_dir), session_id, monitor_id, counters)
    except Exception:  # noqa: BLE001 — derived state; the first tick rewrites it
        logger.warning("monitor counters write failed for %s", monitor_id, exc_info=True)


def _compose_index_rows(
    config_dir: Path, session_id: str, specs: Sequence[MonitorSpec]
) -> list[dict[str, Any]]:
    """The index rows for a list: the spec plus its counters/health — the exact
    key set ``MonitorScheduler.index_rows`` composes from memory, rebuilt here
    from the counters files so an external arm does not drop the new row's due
    instant (the receipt's own field) or a sibling's health out of the derived
    index. Absent counters read as the fresh defaults the next session open
    would rebuild anyway; the index is best-effort between change events (§10.2).
    """
    rows: list[dict[str, Any]] = []
    for spec in specs:
        counters = _read_counters(config_dir, session_id, spec.id) or {}
        rows.append(
            {
                **spec.model_dump(),
                "next_due_at": _due_of(counters),
                "last_check_at": counters.get("last_check_at", 0),
                "checks": counters.get("checks", 0),
                "deliveries": counters.get("deliveries", 0),
                "consecutive_failures": counters.get("consecutive_failures", 0),
                "disabled": bool(counters.get("disabled")),
                "disabled_reason": counters.get("disabled_reason", ""),
                # The health fields (§D6). This composer and
                # ``MonitorScheduler.index_rows`` must keep an IDENTICAL key
                # set — the index is one file written by both — so a field
                # added to one belongs in the other in the same change.
                "unavailable_since": counters.get("unavailable_since", 0),
                "last_error": counters.get("last_error", ""),
            }
        )
    return rows


def _resolved_cwd(session_dir: Path, previous: Mapping[str, Any]) -> str:
    """Which directory the checks run in.

    The index entry's own value first — it is what the session reads on its next
    open and must not be rewritten to something else by an unrelated cancel —
    then the desktop marker a created session carries, then the session
    directory as a last resort (the wake writer's three steps, verbatim)."""
    import json

    from local_operator.session.retention import DESKTOP_MARKER_NAME

    existing = previous.get("cwd")
    if isinstance(existing, str) and existing:
        return existing
    try:
        marker = json.loads((session_dir / DESKTOP_MARKER_NAME).read_text(encoding="utf-8"))
        cwd = marker.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    except (OSError, ValueError):
        pass
    return str(session_dir)


def _remove_state(config_dir: Path, session_id: str, monitor_id: str) -> None:
    try:
        from local_operator.monitors import state as monitor_state

        monitor_state.remove_monitor_state(Path(config_dir), session_id, monitor_id)
    except Exception:  # noqa: BLE001 — derived files; the next arm rebuilds them
        logger.debug("could not remove monitor state for %s", monitor_id, exc_info=True)
