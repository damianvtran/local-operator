"""Cancel a session's monitors from OUTSIDE the session (contract §12, CLI row).

The monitors family's external writer, modeled on
:mod:`local_operator.wakes.arm` — the same three-part invariant and the same
write order, because the failure modes are identical in kind:

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

``lop monitor cancel`` is the caller that exists today; the desktop command
routes (the UI slice) can reuse this writer when they arrive. There is
deliberately NO request-id settling or rollback-journal machinery here — the
wake module carries those for a retry contract (the desktop route's
``request_id``) that monitor cancels do not yet have; a retried cancel is a
fresh request whose worst case is an honest ``no monitor with id`` refusal
about a watch that is already gone, which is the state the user asked for.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
    """What a successful cancel did, for the caller's receipt.

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
            "This conversation is open in a running session, which owns its monitors. "
            + NOTHING_WRITTEN
            + " Retry in a moment, or cancel them from that session.",
            status=STATUS_OWNER_BUSY,
            code="monitor_owner_present",
        )
    raise MonitorWriteError(
        "This conversation's monitor file is claimed by process "
        f"{owner}, which does not answer as a runtime — either a conversation open "
        "in an older build, or a stale owner marker left by a pid the system has "
        "since reused. " + NOTHING_WRITTEN + " If nothing has this conversation open, "
        f"the marker at {Path(config_dir) / 'sessions' / session_id / '.session.pid'} is stale "
        "and can be removed; otherwise cancel its monitors from that session.",
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
    rows: list[MonitorSpec],
    preserve: Mapping[str, Any],
):
    from local_operator.monitors.store import write_entry

    # ``preserve`` carries the keys this module does not own — ``stopped_at``
    # (a stopped session's monitors stay parked through this write; dropping it
    # would un-park the whole session as a side effect of cancelling one
    # watch). An empty remainder REMOVES the entry, which is also what releases
    # the cleanup reap guard (§11.5).
    return write_entry(
        Path(config_dir),
        session_id,
        cwd=cwd,
        monitors=rows,
        preserve=preserve,
    )


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
