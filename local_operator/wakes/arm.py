"""Arm, cancel and edit a session's wakes from OUTSIDE the session.

Three writers existed for schedule state and this module is the second one,
unified. ``Session._persist_wake_schedules`` is the one writer while a runtime
owns a session; this is the writer for a session with NO runtime, where there
is nobody to ask. ``lop wake create`` was already such a writer (its own
transcript append, its own index write, its own validation) and the desktop
routes added two more (arm and edit) — so the reason this module exists is not
tidiness. Each copy had drifted, and the drift was silent:

- the CLI allocated ``w{len(existing) + 1}``, so a cancel-then-arm REUSED a
  live id and a 17th wake was written as ``w17`` past the 16 cap;
- the CLI's index write passed no ``preserve``, so arming a wake on a STOPPED
  session un-parked it (dropping ``stopped_at``) and wiped the lateness
  telemetry the session records;
- and every copy re-derived the transcript-vs-index rule for itself.

**The invariant this module implements is three-part** (see
``wakes/store.py``'s docstring for the full statement): the transcript's
latest ``wake_schedules`` entry is the truth; the index is a derived
projection that a reader must never patch; and a write must leave the two
agreeing, or leave a marker saying it could not.

Consequences that shape every function here:

- **The base is the TRANSCRIPT, never the index.** An append REPLACES the
  session's schedule list, so reading a stale base — the index lags, and is
  absent for a session that has not been opened — would silently cancel live
  reminders. The CLI already did this and said why; it is the rule now.
- **Transcript first, index second, install hook third** — the same order as
  ``_persist_wake_schedules``, and the same asymmetry: only the transcript
  append may fail the request, because an index entry with no transcript is a
  reminder the supervisor faithfully fires into nothing.
- **Post-write verification.** Last-writer-wins on a full-list snapshot is
  safe *by construction* only while nothing else writes; the TUI and a
  resumed session can both append to the same transcript. So after the append
  this re-reads the latest entry and checks it is OURS — a mismatch means
  another writer landed on top of us, and the honest answer is to retry once
  from the new base and then refuse with a conflict rather than report a
  success that lost the user's wake.

**Not the place for either live-session policy, but it is the place for the
guard.** A live runtime's schedules are in memory and republished wholesale on
its next persist, so an append here would be overwritten from the transcript
AND from the index while the wake never fired (the supervisor skips a session
with a live record). The route resolves the owner first and asks it; when there
is nobody to ask this module writes — and immediately before every append it
re-asks whether an owner has appeared, refusing rather than writing behind one
(:func:`_refuse_if_owned`). That re-ask is the last line of defence, not the
first: the route already refused a live or wedged owner before calling here.

The other half of writing correctly from outside is serialisation. Two writers
whose reads interleave merge onto the same base, so one row is lost or written
twice; :mod:`local_operator.wakes.lock` is the per-session mutex that makes the
read-merge-append atomic between external writers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from local_operator.harness.wake import (
    WAKE_SCHEDULES_CUSTOM_TYPE,
    WakeSchedule,
    build_wake_edit,
    build_wake_schedule,
)
from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable, WakeWriteLock

logger = logging.getLogger(__name__)

#: What the caller of :func:`arm_wake` and friends is told to do about an
#: error. The status is carried HERE rather than mapped by each caller so the
#: CLI (which prints the sentence and exits 1) and the desktop route (which
#: raises that status) cannot disagree about what an error means.
STATUS_SESSION_NOT_FOUND = 404
STATUS_WAKE_NOT_FOUND = 404
STATUS_INVALID = 422
STATUS_CONFLICT = 409
#: 503 for both owner states and for a contended lock: every one of them means
#: "nothing was written, and retrying is what fixes it", which is what makes
#: them different from the 409s above (a conflict with the state itself).
STATUS_OWNER_BUSY = 503
STATUS_WRITE_BUSY = 503

#: The refusal for an owner that EXISTS but is not answering. One sentence, used
#: by the route's own pre-flight refusal and by the guard at the append below,
#: so a user cannot get two different descriptions of the same dead end.
WEDGED_MESSAGE = (
    "This conversation's runtime is not responding. Retry in a moment, or stop that conversation."
)


class WakeWriteError(Exception):
    """A refusal the caller can render verbatim.

    ``message`` is always the user-facing sentence — for a validation failure
    it is the SAME text ``build_wake_schedule`` returns to the agent's tool, so
    one wording exists for "you cannot arm a 17th wake" across every surface.
    """

    def __init__(self, message: str, *, status: int, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class WakeWriteOutcome:
    """What a successful write did, for the caller's receipt.

    ``index_written`` is "the derived index now reflects this change, as far as
    this process can tell": true when the write did not raise, which includes
    a cancel that emptied the list and REMOVED the file (``index_path`` is then
    empty). It is False only when the index write itself failed — the
    transcript still won and the next open heals it, which is why that is a
    reported field rather than an error.
    """

    session_id: str
    wake_id: str
    next_due_at: int | None
    schedules: list[WakeSchedule]
    index_written: bool = False
    index_path: str = ""
    supervisor: str = ""


#: (wake_id, full new list, the row's due instant) — what a mutation produces
#: from the current base. Raising :class:`WakeWriteError` is how it refuses.
Mutation = Callable[[list[WakeSchedule], int], "tuple[str, list[WakeSchedule], int | None]"]


async def arm_wake(
    config_dir: Path,
    session_id: str,
    request: Mapping[str, Any],
    *,
    cwd: str | None = None,
    now_ms: int | None = None,
) -> WakeWriteOutcome:
    """Add one schedule to ``session_id``'s wakes. ``request`` is the same
    ``message``/``in``/``at``/``every``/``until``/``limit`` mapping the agent's
    ``wake`` tool takes, validated by the same function."""

    def mutate(
        existing: list[WakeSchedule], now: int
    ) -> tuple[str, list[WakeSchedule], int | None]:
        built = build_wake_schedule(dict(request), existing, now)
        if "error" in built:
            raise _refusal(built)
        schedule = built["schedule"]
        return schedule.id, [*existing, schedule], schedule.next_due_at

    return await _apply(config_dir, session_id, mutate, cwd=cwd, now_ms=now_ms)


async def cancel_wake(
    config_dir: Path,
    session_id: str,
    wake_id: str,
    *,
    cwd: str | None = None,
    now_ms: int | None = None,
) -> WakeWriteOutcome:
    """Drop one schedule. An empty remainder REMOVES the index entry rather
    than writing an empty one, which is also what releases the cleanup reap
    guard (``session/cleanup.py``) that a wake-carrying session holds."""

    def mutate(
        existing: list[WakeSchedule], now: int
    ) -> tuple[str, list[WakeSchedule], int | None]:
        if not any(schedule.id == wake_id for schedule in existing):
            raise _unknown_wake(wake_id, existing)
        remaining = [schedule for schedule in existing if schedule.id != wake_id]
        return wake_id, remaining, None

    return await _apply(config_dir, session_id, mutate, cwd=cwd, now_ms=now_ms)


async def edit_wake(
    config_dir: Path,
    session_id: str,
    wake_id: str,
    request: Mapping[str, Any],
    *,
    cwd: str | None = None,
    now_ms: int | None = None,
) -> WakeWriteOutcome:
    """Change an existing schedule's message and/or bounds.

    A key ABSENT from ``request`` keeps the row's current value; a key present
    with value ``None`` clears it. Callers that model the body as a pydantic
    request get that for free from ``exclude_unset=True`` — the distinction is
    the difference between "leave the limit" and "remove the limit", so a
    plain dump is not equivalent.
    """

    def mutate(
        existing: list[WakeSchedule], now: int
    ) -> tuple[str, list[WakeSchedule], int | None]:
        if not any(schedule.id == wake_id for schedule in existing):
            raise _unknown_wake(wake_id, existing)
        built = build_wake_edit(dict(request), existing, wake_id, now)
        if "error" in built:
            raise _refusal(built)
        row = built["schedule"]
        # IN PLACE: a row that moved to the end of the list on every edit would
        # reorder the page under the user for a message-only change.
        updated = [row if schedule.id == wake_id else schedule for schedule in existing]
        return row.id, updated, row.next_due_at

    return await _apply(config_dir, session_id, mutate, cwd=cwd, now_ms=now_ms)


# ---------------------------------------------------------------------------
# The shared write sequence
# ---------------------------------------------------------------------------


async def _apply(
    config_dir: Path,
    session_id: str,
    mutate: Mutation,
    *,
    cwd: str | None,
    now_ms: int | None,
) -> WakeWriteOutcome:
    session_dir = Path(config_dir) / "sessions" / session_id
    if not await asyncio.to_thread(session_dir.is_dir):
        # Mirroring the CLI's refusal, for the same reason: a wake id in the
        # index with no transcript fires into nothing. The desktop route
        # creates the session before calling this, so a miss here is a real
        # miss rather than the cold-start case.
        raise WakeWriteError(
            f"no session {session_id!r}", status=STATUS_SESSION_NOT_FOUND, code="session_not_found"
        )
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)

    # ONE WRITER AT A TIME, and the lock spans the awaits below rather than just
    # the append: the base is read, merged and replaced, so two writers that
    # interleave their reads lose or duplicate a row (measured on the route: six
    # rows from five arms, all answered 200). It is taken on a worker thread and
    # released on one, so the event loop stays free, and a timed-out acquire is a
    # REFUSAL rather than "carry on unlocked": the failure that would follow is
    # the duplicate row, and a retryable 503 is the honest answer to contention.
    # See :mod:`local_operator.wakes.lock` for why this is not the session lease
    # and why the owning runtime is deliberately not a party to it.
    lock = WakeWriteLock(session_dir)
    try:
        await asyncio.to_thread(lock.acquire)
    except WakeLockBusy as busy:
        raise WakeWriteError(str(busy), status=STATUS_WRITE_BUSY, code="wake_write_busy") from None
    except WakeLockUnavailable as unusable:
        # The lock file itself could not be created — a read-only session
        # directory, or one that vanished between the existence check above and
        # the open. Nothing was written, and re-running cannot help until the
        # directory's mode changes, so this is a 409 that names the fix rather
        # than the 500 an untranslated OSError used to answer with (review
        # round 2, R7; QA Q5 — the base answered 200 here, so this sentence is
        # what a user gets instead of a silent success).
        raise WakeWriteError(
            f"Cannot schedule a wake for this conversation: {unusable}. "
            "Restore write permission on its directory, or reopen the conversation "
            "from a writable location, and retry.",
            status=STATUS_CONFLICT,
            code="wake_write_unavailable",
        ) from None
    try:
        rows, wake_id, due = await _mutate_locked(config_dir, session_dir, session_id, mutate, now)
        # The INDEX write is inside the lock too, and that is not incidental: it
        # is written from the list this call produced, so a writer that released
        # before writing it could overwrite a successor's newer entry with its
        # own older rows.
        previous = await asyncio.to_thread(_read_index_entry, config_dir, session_id)
        index_written = False
        index_path = ""
        try:
            written = await asyncio.to_thread(
                _write_index,
                config_dir,
                session_id,
                cwd or _resolved_cwd(session_dir, previous),
                rows,
                previous,
            )
            index_written = True
            index_path = str(written) if written is not None else ""
        except Exception:  # noqa: BLE001 — the transcript already has it (store.py's contract)
            logger.warning(
                "could not update the wake index entry for %s", session_id, exc_info=True
            )
    finally:
        await asyncio.to_thread(lock.release)

    supervisor = ""
    if rows:
        # Only when something is armed, and OUTSIDE the lock: installing the
        # supervisor shells out to launchd, and holding a per-session write lock
        # across a subprocess would queue every other writer behind it for no
        # reason — the hook touches no schedule state.
        #
        # A cancel that emptied the list leaves the supervisor to retire on its
        # own (it re-reads the index every slice and exits when nothing is
        # fireable), and installing one for an empty index would fight that
        # retirement.
        supervisor = await asyncio.to_thread(_ensure_supervisor, config_dir)

    return WakeWriteOutcome(
        session_id=session_id,
        wake_id=wake_id,
        next_due_at=due,
        schedules=rows,
        index_written=index_written,
        index_path=index_path,
        supervisor=supervisor,
    )


async def _mutate_locked(
    config_dir: Path,
    session_dir: Path,
    session_id: str,
    mutate: Mutation,
    now: int,
) -> tuple[list[WakeSchedule], str, int | None]:
    """read -> mutate -> guard -> append -> verify, under the caller's lock.

    Returns the full new list, the row's id and its due instant.
    """
    rows: list[WakeSchedule] = []
    wake_id = ""
    due: int | None = None
    for attempt in (0, 1):
        existing = await asyncio.to_thread(_read_rows, session_dir)
        wake_id, rows, due = mutate(existing, now)
        await _refuse_if_owned(config_dir, session_id)
        entry = await _append(session_dir, rows)
        if await asyncio.to_thread(_latest_entry_id, session_dir) == entry:
            break
        # STILL REACHED, but no longer by a peer using this module: the lock
        # above serialises every external writer, so a mismatch now means a
        # writer that does not take it (a live session's own persist) landed on
        # top of us. Rebasing on ITS snapshot is what makes the retry safe — the
        # mutation is re-applied to the newer base, so the row is added once
        # against that base rather than appended again on top of our own write.
        if attempt:
            raise WakeWriteError(
                "The session changed while your change was being applied — retry.",
                status=STATUS_CONFLICT,
                code="wake_write_conflict",
            )
        logger.info("wake write for %s was overtaken by another writer; retrying", session_id)
    return rows, wake_id, due


async def _refuse_if_owned(config_dir: Path, session_id: str) -> None:
    """Refuse to append to a transcript a runtime process owns.

    THE LOST-REMINDER GUARD, and it is here rather than only in the route
    because the route's decision is a moment old by the time this runs. An owner
    that appears in between — the desktop run pane's ``/watch`` heartbeat warming
    the visible session, a second app opening the conversation, the supervisor
    engaging one — loads its schedule list BEFORE this append and republishes
    that stale list on its next persist (a fire, a retirement, the agent's own
    ``wake`` tool), which deletes the row from both the transcript and the index
    while the supervisor skips any session with a live record. Measured: a 200
    with ``index_written: true``, then an empty transcript snapshot and a removed
    index entry once the owner persisted.

    THE WINDOW THIS LEAVES, stated rather than implied: the check is a moment
    before the append, so an owner that starts between the two still wins. The
    window is now the append itself (sub-millisecond on an ordinary transcript;
    seconds on a very large one) rather than the route's whole read-merge-write,
    and it cannot be closed from this side: the runtime is not a party to the
    lock above (see ``wakes/lock.py`` for why adding it would not save the row —
    the owner's in-memory list is stale regardless of ordering) and taking the
    session lease would refuse every attach and paint this server's pid into
    ``.session.pid`` for the duration of a write.

    OWNED MEANS ANY LIVE PROCESS, NOT ONLY A DIALABLE ONE, and that distinction
    is the whole of review round 2's R6. Asking ``_has_live_runtime`` asks "does a
    *dialable discovery record* exist", which is the SUPERVISOR's question (it
    delivers into a record). An owner that exists as a live process holding the
    session but publishes no usable record — a mixed-version rollout, a
    registrant that failed to start, the rebind window where the record still
    names the previous session — is owned all the same, and the tree already says
    so twice: ``session_lease.acquire_session_lease`` refuses a new claim while a
    live legacy pid mirror exists ("It is still authoritative when live or
    uncertain"), and this route's own ``_rollback_created`` asks the wider
    question to decide a directory is not its own. Predicating on the narrower
    one appended the row behind such an owner and answered 200 with
    ``index_written: true`` — the lost-reminder leak this guard exists to
    prevent, one layer down. So the test is the owner PID
    (``find_runtime_record(...)[1]``, which is ``None`` only when nothing holds
    the session) and the record is not part of it.

    BOTH OWNER STATES ARE REFUSED, and the order below is load-bearing: a WEDGED
    owner (alive, heartbeat stale, lease held so no engage can succeed) is asked
    FIRST because it is also a live pid — the owner question first would shadow
    the wedged answer with the generic one and lose the sentence that names the
    next step.
    """
    from local_operator.mobile.attach_client import find_runtime_record
    from local_operator.wakes.supervisor import wedged_runtime

    wedged = await asyncio.to_thread(wedged_runtime, Path(config_dir), session_id)
    if wedged is not None:
        raise WakeWriteError(WEDGED_MESSAGE, status=STATUS_OWNER_BUSY, code="wake_owner_wedged")
    # Off the loop: this reads ``.session.pid`` and, for the zombie proof, forks a
    # ``ps`` — the same probe the attach paths use, so this guard and the process
    # that would take the session agree about who is holding it.
    _record, owner = await asyncio.to_thread(find_runtime_record, Path(config_dir), session_id)
    if owner is not None:
        raise WakeWriteError(
            "This conversation is open in a running session, which owns its schedules. "
            "Retry in a moment, or change them from that session.",
            status=STATUS_OWNER_BUSY,
            code="wake_owner_present",
        )


def _refusal(outcome: Mapping[str, Any]) -> WakeWriteError:
    """A ``build_wake_schedule`` error as a refusal. Malformed (an unparseable
    duration, a bound on a one-shot) is the caller's input to fix and reads as
    a 422; everything else — the cap, a past time — is a conflict with the
    session's existing state and reads as a 409."""
    malformed = bool(outcome.get("malformed"))
    return WakeWriteError(
        str(outcome.get("error") or "the wake could not be scheduled"),
        status=STATUS_INVALID if malformed else STATUS_CONFLICT,
        code="wake_invalid" if malformed else "wake_refused",
    )


def _unknown_wake(wake_id: str, existing: list[WakeSchedule]) -> WakeWriteError:
    known = ", ".join(schedule.id for schedule in existing) or "none"
    return WakeWriteError(
        f"No wake schedule with id '{wake_id}' (known: {known}).",
        status=STATUS_WAKE_NOT_FOUND,
        code="wake_not_found",
    )


# ---------------------------------------------------------------------------
# Filesystem steps (each run off the event loop by its caller)
# ---------------------------------------------------------------------------


def _transcript(session_dir: Path):
    from local_operator.session.transcript import Transcript

    # ``defer_materialise``: a read must not create a session directory, and
    # the append path recreates a vanished one itself. This module never
    # invents a session — only the create+arm route does, and it does so
    # before calling here.
    return Transcript(session_dir, defer_materialise=True)


def _read_rows(session_dir: Path) -> list[WakeSchedule]:
    """The session's schedules, from the TRANSCRIPT's latest snapshot.

    The index is deliberately not consulted even as a fallback: it lags, it is
    absent for a session that has never been opened, and the append below
    REPLACES the list — so a stale base is how a live reminder gets cancelled
    by an unrelated edit.
    """
    entry = _transcript(session_dir).latest_custom_entry(WAKE_SCHEDULES_CUSTOM_TYPE)
    if entry is None:
        return []
    details = entry.payload.get("details")
    if not isinstance(details, Mapping):
        return []
    raw_rows = details.get("schedules")
    if raw_rows is None:
        return []
    if not isinstance(raw_rows, list):
        # An entry that EXISTS and is not a list is corrupt, not empty, and
        # guessing there would mean treating someone's armed reminders as
        # "none" and then writing that emptiness back.
        raise WakeWriteError(
            "The session's stored wake schedules are unreadable; reopen the session and retry.",
            status=STATUS_CONFLICT,
            code="wake_store_corrupt",
        )
    rows: list[WakeSchedule] = []
    for raw in raw_rows:
        try:
            rows.append(WakeSchedule.model_validate(raw))
        except Exception:  # noqa: BLE001 — one bad row costs one row, never the list
            logger.warning("dropping malformed persisted wake schedule: %r", raw)
    return rows


def _latest_entry_id(session_dir: Path) -> str | None:
    entry = _transcript(session_dir).latest_custom_entry(WAKE_SCHEDULES_CUSTOM_TYPE)
    return entry.id if entry is not None else None


async def _append(session_dir: Path, rows: list[WakeSchedule]) -> str:
    """Append the new full-list snapshot and return its entry id.

    The ONLY step allowed to fail the write: everything after it is derived
    from what this put on disk.

    The transcript is CONSTRUCTED on the worker thread as well as appended from
    the loop. ``Transcript.__init__`` reads and parses the whole journal
    synchronously, which on a 103 MB transcript is seconds of event loop and
    the reason every other step here is already threaded (review round 1, R3).
    """
    transcript = await asyncio.to_thread(_transcript, session_dir)
    entry = await transcript.append_custom(
        WAKE_SCHEDULES_CUSTOM_TYPE,
        {"schedules": [schedule.model_dump() for schedule in rows]},
    )
    return entry.id


def _read_index_entry(config_dir: Path, session_id: str) -> dict[str, Any]:
    from local_operator.wakes.store import read_entry

    try:
        return read_entry(Path(config_dir), session_id) or {}
    except Exception:  # noqa: BLE001 — an unreadable entry preserves nothing
        logger.warning("could not read the wake index entry for %s", session_id, exc_info=True)
        return {}


def _write_index(
    config_dir: Path,
    session_id: str,
    cwd: str,
    rows: list[WakeSchedule],
    preserve: Mapping[str, Any],
):
    from local_operator.wakes.store import write_entry

    # ``preserve`` carries the keys this module does not own — ``stopped_at``
    # (a stopped session's wakes stay parked through an arm; dropping it would
    # un-park the whole session as a side effect of scheduling something) and
    # the session's own ``last_fired_at``/``last_attempt_at`` lateness stamps.
    return write_entry(
        Path(config_dir),
        session_id,
        cwd=cwd,
        schedules=rows,
        preserve=preserve,
    )


def _resolved_cwd(session_dir: Path, previous: Mapping[str, Any]) -> str:
    """Which directory the supervisor should start a runtime in.

    The index entry's own value first — it is what the supervisor reads today
    and must not be rewritten to something else by an unrelated arm — then the
    desktop marker a created session carries, then the session directory as a
    last resort (what the CLI did before this module existed)."""
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


def _ensure_supervisor(config_dir: Path) -> str:
    try:
        from local_operator.wakes.install import ensure_supervisor_installed

        return ensure_supervisor_installed(Path(config_dir)).reason
    except Exception:  # noqa: BLE001 — best-effort, exactly as the persist path treats it
        logger.debug("wake supervisor install failed", exc_info=True)
        return ""
