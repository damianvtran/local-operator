"""The queued / deferred move: a durable source-owned record and its driver.

WHY THIS EXISTS. Today a move against a busy session is refused and the user
retries (``--wait N``) — the design's stance is "busy moves are refused, not
drained" because a preemptive drain can cut a turn in flight
(``docs/design/mesh-session-mobility.md`` §6.4). This module is the other half
that stance always implied: an EXPLICIT request (``--queue``, the UI's "Move at
the next safe point") is accepted, recorded durably on the SOURCE, and the move
proceeds by itself at the next safe point — a turn boundary, with the attached
clients announced first (design note §5.4; the note is the contract, read it
before changing any transition below).

THE RECORD. ``<config>/network/queue/move-<session_id>.json`` — one small JSON
file per queued move, beside the mesh store's other durable objects (same 0600
+ atomic-write conventions, ``network/store.py``). It is owned by the SOURCE
device and it must survive the requesting window closing, so its writers are
the source RELAY's transitions and ``--cancel-queued`` (possibly another
process) — never the requester's memory.

STORE DISCIPLINE (design note §2.3 F3, frozen). Atomic write is not
concurrency control: every read-modify-write takes a per-record cross-process
lock (``flock``, through ``wakes.lock.WakeWriteLock`` — the existing
bundled-platform implementation) and writes temp-file + ``os.replace`` at
0600. Terminal states are write-once: ``cancelled``/``failed``/``resumed``
each win exactly once and no later transition may overwrite them, which is
what makes "cancel vs a relay transition" race-safe without either side
re-reading a stale copy.

THE PHASES (wire vocabulary extends the existing move phases additively):
``queued → finishing → paused → copying → resumed``.
``queued`` is the accepted state; ``finishing`` is "waiting for the current
writer to reach a safe point" (a turn in flight, a parked gate, a lease holder
that is not a runtime — the design refuses to drain turns, so the wait is the
point); ``paused`` is the point of no return — the runtime has latched its
admission refusal and is announcing ``move_pending`` to attached clients while
the attach window runs (the runtime CLAIMS this phase atomically with the
cancel check, see :func:`claim_pause`); ``copying`` is the handoff running
(the destination pulls, the source journal commits); ``resumed`` is the
commit — the conversation lives at the destination. ``failed`` and
``cancelled`` are the other terminals; a failed record carries the refusal
sentence verbatim (the ``_source_refused`` discipline).

WHO WRITES WHAT. ``queued`` — the relay's ``_source_prepare`` at enqueue.
``finishing`` — the driver when it learns a writer is blocking, or that the
source is cold and it is about to proceed. ``paused`` — the RUNTIME, through
:func:`claim_pause`, because the claim must be atomic with the cancel check in
the same lock (a runtime that latched and then found a cancel won needs the
latch released; claiming first removes that unwind path for every case but a
microscopic race, which is why ``end_retire`` carries it). ``copying``/
``resumed``/``failed`` — the driver, from the source's own durable progress
(the journal and the tombstone — never a guess). ``cancelled`` —
``--cancel-queued``, under the same lock, and only while the record has not
passed ``paused``.

RE-ARMING. If the runtime dies while queued the record survives — it is a file,
not runtime memory — and the driver re-delivers the intent to the next runtime
that engages (:func:`reconcile_on_start` restarts drivers for records that were
live when the relay died, and :func:`drive` re-delivers whenever a runtime
reappears). A record whose source is cold AND had its writer vanish proceeds
once no writer is present: there is no turn to reach a boundary.

WHAT THIS MODULE DOES NOT DO. It does not copy bytes, retire runtimes, or
touch the journal — the copy is the existing pull (`net_session_move`), driven
by the destination, and this module's driver only DELIVERS the intent to the
runtime and SENDS THE INVITE that starts that pull once the source is at its
safe point. ``move_pending``/``move_committed`` frames and the attach window
live in the runtime (``session/runtime/server.py``); the CLI verbs live in
``cli.py``; the wire phase ``cancel_queue`` is handled in ``mobility.py``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from local_operator.network.relay import RelayServer

logger = logging.getLogger(__name__)

#: Subdirectory of ``<config>/network/`` holding one record per queued move.
QUEUE_DIRNAME = "queue"

#: The queue phase vocabulary (design note §5.4, additive to the move phases).
QUEUE_PHASE_QUEUED = "queued"
QUEUE_PHASE_FINISHING = "finishing"
QUEUE_PHASE_PAUSED = "paused"
QUEUE_PHASE_COPYING = "copying"
QUEUE_PHASE_RESUMED = "resumed"
QUEUE_PHASE_FAILED = "failed"
QUEUE_PHASE_CANCELLED = "cancelled"

#: The ordered phases a live (non-failed, non-cancelled) move passes through.
QUEUE_PHASES: tuple[str, ...] = (
    QUEUE_PHASE_QUEUED,
    QUEUE_PHASE_FINISHING,
    QUEUE_PHASE_PAUSED,
    QUEUE_PHASE_COPYING,
    QUEUE_PHASE_RESUMED,
)

#: Terminal states. Write-once: the first terminal transition wins and nothing
#: may leave a terminal state (the §2.3 F3 discipline).
QUEUE_TERMINAL_PHASES: frozenset[str] = frozenset(
    {QUEUE_PHASE_RESUMED, QUEUE_PHASE_FAILED, QUEUE_PHASE_CANCELLED}
)

#: How long a terminal record is kept for surfaces to read its outcome. Pruned
#: by the reconciler; a queued move's own receipt may be read for a day.
QUEUE_TERMINAL_TTL_S = 24 * 3600.0

#: The driver's loop cadence while a runtime holds the intent: settle, re-read
#: the record, settle again. The visible effect is the "finishing → paused"
#: latency a surface observes; the cost is a file read per tick.
DRIVER_POLL_S = 1.0

#: The driver's slower cadence for the two waits where nothing polls the runtime
#: — waiting out a lease holder, and the backoff between failed delivery
#: attempts to a runtime that answered nothing.
DRIVER_WAIT_S = 2.0

#: How long the driver watches the destination's pull for a commit before the
#: record folds to ``failed`` with an honest sentence. Sized off the recall's
#: own budget (``mobility.KEEP_COPY_WAIT_S``, 300 s) plus the confirmation
#: window, because the copy it is watching runs on the destination.
COPY_WATCH_S = 420.0


def queue_dir(root: Path | str) -> Path:
    return Path(root) / "network" / QUEUE_DIRNAME


def record_path(root: Path | str, session_id: str) -> Path:
    return queue_dir(root) / f"move-{session_id}.json"


def lock_for(root: Path | str, session_id: str):
    """The per-record cross-process lock (never held across a blocking call).

    ``wakes.lock.WakeWriteLock`` is the tree's bundled-platform flock; the
    queue gets its own file so the wake writer's mutex and this one can never
    interleave. The busy sentence talks about the move queue because that is
    what a person pressing cancel is looking at.

    THE DIRECTORY IS ENSURED HERE rather than by each writer, and the lock is
    the right owner of that step: ``flock`` opens a file in this directory, so
    a first-ever record (or a ``--cancel-queued`` arriving before anything was
    ever queued) would otherwise fail with ENOENT instead of taking the lock.
    ``0700`` matches ``network/store.py``'s directory convention — the record's
    own mode is the 0600 the staged write gives it.
    """
    from local_operator.wakes.lock import WakeWriteLock

    directory = queue_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:  # noqa: PERF203 — a directory on a no-chmod fs still works
        pass
    return WakeWriteLock(
        directory,
        name=f"move-{session_id}.lock",
        busy_sentence=(
            "Another process is updating this conversation's queued move. Retry in a "
            "moment; a later retry will succeed."
        ),
    )


def read_record(root: Path | str, session_id: str) -> dict[str, Any] | None:
    """One record, or ``None`` when absent or unreadable.

    Unreadable is treated exactly like absent — the record is a coordination
    file, not a journal, and a reader that raised would take down a driver for
    one bad byte. The writers keep it whole (staged + ``os.replace``).
    """
    path = record_path(root, session_id)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("move queue: unreadable record %s; treating as absent", path)
        return None
    if not isinstance(data, dict) or data.get("version") != 1:
        logger.warning("move queue: skipping record %s with unknown version", path)
        return None
    return data


def _write_record_raw(root: Path | str, session_id: str, record: dict[str, Any]) -> Path:
    """Staged 0600 write + ``os.replace``. The caller holds the record lock."""
    path = record_path(root, session_id)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, separators=(",", ":"), sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _record_or_none(root: Path | str, session_id: str) -> dict[str, Any] | None:
    return read_record(root, session_id)


def is_terminal(record: Mapping[str, Any]) -> bool:
    return str(record.get("phase") or "") in QUEUE_TERMINAL_PHASES


def _stamp(record: dict[str, Any], phase: str, *, detail: str = "", **extra: Any) -> dict[str, Any]:
    """Advance ``record`` to ``phase`` in memory: phase, history, timestamp."""
    history = list(record.get("phases") or [])
    history.append({"phase": phase, "at": time.time()})
    updated = dict(record)
    updated.update({"phase": phase, "phases": history, "updated_at": time.time(), **extra})
    if detail:
        updated["detail"] = detail
    return updated


def enqueue(
    root: Path | str,
    session_id: str,
    *,
    to_device: str,
    to_name: str,
    request_id: str,
    requested_by: str,
    engage_on_arrival: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Create (or return the existing) queue record for ``session_id``.

    Returns ``(record, created)``. Idempotent for the SAME ``request_id`` (a
    retried request refreshes nothing and reports the live record); a DIFFERENT
    request while one is live is refused by the caller, which is why the
    existing record is returned for it to inspect. A record whose previous move
    already reached a terminal state is replaced by the new request — cancel
    (or failure) ends the intent, and re-queuing after one is ordinary.
    """
    with lock_for(root, session_id):
        existing = _record_or_none(root, session_id)
        if existing is not None and not is_terminal(existing):
            return existing, False
        record = {
            "version": 1,
            "session_id": session_id,
            "to_device": to_device,
            "to_name": to_name,
            "request_id": request_id,
            "requested_by": requested_by,
            "mode": "move",
            "engage_on_arrival": bool(engage_on_arrival),
            "phase": QUEUE_PHASE_QUEUED,
            "phases": [{"phase": QUEUE_PHASE_QUEUED, "at": time.time()}],
            "detail": "",
            "queued_at": time.time(),
            "updated_at": time.time(),
        }
        _write_record_raw(root, session_id, record)
        return record, True


def transition(
    root: Path | str,
    session_id: str,
    phase: str,
    *,
    detail: str = "",
    writer: str = "relay",
    **extra: Any,
) -> dict[str, Any] | None:
    """Advance the record to ``phase``, under the lock, monotonically.

    Returns the new record, or ``None`` when there was nothing to advance:
    a missing record, an unknown phase, a phase already at or past ``phase``,
    or — the important one — a terminal state that has already won. Terminal is
    write-once, so ``cancel`` racing a commit resolves to whichever took the
    lock first, and neither side ever overwrites the other's decision.

    ``extra`` fields (e.g. the failure ``code``) are merged into the record
    only when the transition happens.
    """
    if phase not in QUEUE_PHASES and phase not in QUEUE_TERMINAL_PHASES:
        raise ValueError(f"unknown move-queue phase {phase!r}")
    with lock_for(root, session_id):
        record = _record_or_none(root, session_id)
        if record is None:
            return None
        current = str(record.get("phase") or "")
        if current in QUEUE_TERMINAL_PHASES:
            return None
        if current == phase:
            return record
        if phase in QUEUE_PHASES:
            # Monotone within the live sequence: never go backwards. A jump
            # forward (queued → copying for a cold source) is allowed — the
            # skipped phases are genuinely not applicable, and the history
            # stays honest because it names only what happened.
            order = {name: index for index, name in enumerate(QUEUE_PHASES)}
            if order.get(phase, -1) <= order.get(current, -1):
                return None
        updated = _stamp(record, phase, detail=detail, writer=writer, **extra)
        _write_record_raw(root, session_id, updated)
        return updated


def claim_pause(root: Path | str, session_id: str) -> tuple[dict[str, Any] | None, str]:
    """The runtime's atomic "I am pausing now" — the cancel point of no return.

    Called by the session runtime when it has reached its safe point and
    latched its admission refusal, immediately before it announces
    ``move_pending``. Under the same lock a ``--cancel-queued`` takes:

    * record live (``queued``/``finishing``) → write ``paused``; return
      ``(record, "claimed")`` — from here a cancel REFUSES ("already started"),
    * record terminal → ``(record, "terminal")`` — the runtime must RELEASE its
      latch and stay alive; the caller does that (``end_retire``),
    * record missing → ``(None, "absent")`` — same release.

    The atomicity is the whole point: without it a cancel landing between the
    runtime's check and its announcement would be told "nothing changed" while
    the runtime retired anyway.
    """
    with lock_for(root, session_id):
        record = _record_or_none(root, session_id)
        if record is None:
            return None, "absent"
        current = str(record.get("phase") or "")
        if current in QUEUE_TERMINAL_PHASES:
            return record, "terminal"
        if current == QUEUE_PHASE_PAUSED:
            return record, "already"
        if current not in (QUEUE_PHASE_QUEUED, QUEUE_PHASE_FINISHING):
            # copying/resumed: the move is ahead of this runtime's knowledge
            # (a stale re-delivery to a re-engaged runtime); treat like terminal.
            return record, "terminal"
        updated = _stamp(record, QUEUE_PHASE_PAUSED, writer="runtime")
        _write_record_raw(root, session_id, updated)
        return updated, "claimed"


def cancel(root: Path | str, session_id: str) -> tuple[dict[str, Any] | None, str]:
    """``--cancel-queued``: stop a queued move that has not started.

    Returns ``(record, outcome)`` where outcome is one of:

    * ``"cancelled"`` — the record is now terminal ``cancelled``; nothing moved,
    * ``"too_late"`` — the record had already reached ``paused``/``copying``
      (the runtime owns the outcome now); the record returned is the live one
      and its ``phase`` is what the refusal names,
    * ``"already"`` — the record was already terminal (``resumed``/``failed``/
      ``cancelled``); there is nothing to cancel and the phase says why,
    * ``"absent"`` — there is no record for this session here.
    """
    with lock_for(root, session_id):
        record = _record_or_none(root, session_id)
        if record is None:
            return None, "absent"
        current = str(record.get("phase") or "")
        if current in QUEUE_TERMINAL_PHASES:
            # A second cancel press, or a cancel after a failure: NOT "too
            # late" (which claims the move started) — the phase word says it.
            return record, "already"
        if current in (QUEUE_PHASE_PAUSED, QUEUE_PHASE_COPYING):
            return record, "too_late"
        updated = _stamp(record, QUEUE_PHASE_CANCELLED, writer="cancel")
        _write_record_raw(root, session_id, updated)
        return updated, "cancelled"


def payload(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The wire/receipt projection of a record (a copy; never the live dict)."""
    if not isinstance(record, Mapping):
        return None
    return {
        "phase": str(record.get("phase") or ""),
        "phases": list(record.get("phases") or []),
        "to_device": str(record.get("to_device") or ""),
        "to_name": str(record.get("to_name") or ""),
        "detail": str(record.get("detail") or ""),
        "code": str(record.get("code") or ""),
        "request_id": str(record.get("request_id") or ""),
        "updated_at": float(record.get("updated_at") or 0.0),
    }


def sweep_terminal(root: Path | str, *, max_age_s: float = QUEUE_TERMINAL_TTL_S) -> list[str]:
    """Remove terminal records older than ``max_age_s``. Returns their ids."""
    directory = queue_dir(root)
    removed: list[str] = []
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return removed
    except OSError:
        logger.warning("move queue: cannot list %s", directory)
        return removed
    now = time.time()
    for name in names:
        if not name.startswith("move-") or not name.endswith(".json"):
            continue
        session_id = name[len("move-") : -len(".json")]
        record = read_record(root, session_id)
        if record is None or not is_terminal(record):
            continue
        updated = float(record.get("updated_at") or 0.0)
        if now - updated < max_age_s:
            continue
        try:
            record_path(root, session_id).unlink()
            removed.append(session_id)
        except OSError:
            logger.debug("move queue: could not prune %s", session_id, exc_info=True)
    return removed


# ---------------------------------------------------------------------------
# The driver: deliver the intent, wait for the safe point, drive the copy
# ---------------------------------------------------------------------------


def reconcile_on_start(server: "RelayServer") -> list[str]:
    """Restart drivers for records that were live when the relay died (§5.4).

    Also prunes expired terminal records, so the queue directory does not grow
    forever. Runs on the relay's start-hook thread; each driver is its own
    daemon thread, so a queued move never makes relay start-up wait on a peer.
    """
    root = server.root
    swept = sweep_terminal(root)
    started: list[str] = []
    directory = queue_dir(root)
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return started
    except OSError:
        logger.warning("move queue: cannot list %s", directory)
        return started
    for name in names:
        if not name.startswith("move-") or not name.endswith(".json"):
            continue
        session_id = name[len("move-") : -len(".json")]
        record = read_record(root, session_id)
        if record is None or is_terminal(record):
            continue
        start_driver(server, session_id)
        started.append(session_id)
    if started or swept:
        logger.info("move queue: start-up reconcile started=%s swept=%s", started, swept)
    return started


#: One driver per (root, session). A second call for a live session is a no-op
#: — the driver is re-entrant by session, which is what makes reconcile-on-start
#: and enqueue racing safe.
_DRIVERS: dict[tuple[str, str], threading.Thread] = {}
_DRIVERS_LOCK = threading.Lock()


def start_driver(server: "RelayServer", session_id: str) -> bool:
    """Start (or no-op) the driver thread for one queued move."""
    key = (str(server.root), session_id)
    with _DRIVERS_LOCK:
        existing = _DRIVERS.get(key)
        if existing is not None and existing.is_alive():
            return False
        thread = threading.Thread(
            target=_driver_main,
            args=(server, session_id),
            name=f"mesh-queue-{session_id[:8]}",
            daemon=True,
        )
        _DRIVERS[key] = thread
        thread.start()
        return True


def _driver_main(server: "RelayServer", session_id: str) -> None:
    try:
        drive(server, session_id)
    except Exception:  # noqa: BLE001 — a driver must never take the relay down
        logger.warning("move queue: driver for %s failed", session_id, exc_info=True)
    finally:
        with _DRIVERS_LOCK:
            _DRIVERS.pop((str(server.root), session_id), None)


def _find_runtime(root: Path, session_id: str) -> Any:
    """The live runtime's discovery record, or ``None`` (never raises).

    An unreadable registry is not evidence of a runtime — but it is also not
    evidence of NONE, and the caller's two consuming paths differ exactly
    there; the registry read is best-effort and its failure reads as "no
    record", which the lease probe then covers (fail closed).
    """
    try:
        from local_operator.mobile.attach_client import find_runtime_record

        record, _pid = find_runtime_record(Path(root), session_id)
        return record
    except Exception:  # noqa: BLE001 — see the docstring
        logger.debug("move queue: runtime record read failed for %s", session_id, exc_info=True)
        return None


def _lease_holder_present(root: Path, session_id: str) -> bool:
    """Whether a live process holds this session's lease without a record.

    The "viewed"/wedge case the mover already refuses on: ``find_runtime_record``
    returns no record but a pid when ``lop exec``, a headless REPL or a
    booting runtime holds the lease. The queue waits for it, exactly as the
    design waits for a turn: it is a writer, and a copy under one is the thing
    the protocol exists to prevent. Delegates to ``mobility._lease_refusal``
    rather than re-reading the lease, so "held" means the same thing here as
    it does to the move that refuses on it — including the fail-closed
    ``uncertain`` verdict.
    """
    from local_operator.network import mobility

    try:
        return bool(mobility._lease_refusal(Path(root), session_id, None))  # noqa: SLF001
    except Exception:  # noqa: BLE001 — an unreadable lease reads as held (fail closed)
        logger.debug("move queue: lease probe failed for %s", session_id, exc_info=True)
        return True


def drive(server: "RelayServer", session_id: str) -> str:
    """The driver body. Returns the terminal phase it left the record in.

    THE DRIVER DOES NOT POLL THE RUNTIME. Delivering ``queue_move`` is the
    whole of its involvement with the runtime: from then on the runtime owns
    the boundary watch, and every transition the driver has to see arrives in
    the RECORD — the runtime claims ``paused`` itself (:func:`claim_pause`),
    and a cancel writes ``cancelled``. So after one successful delivery per
    runtime process the driver only reads files and waits: for the runtime's
    discovery record to disappear (the retire happened; the copy may start),
    or for the record to end under it.

    Re-delivery happens exactly when it should: a new runtime process (a
    different pid) gets the intent again — that is the "re-arms on the next
    engage" contract — and a source with NO runtime and NO lease holder has
    nothing to wait for. A dead runtime is not a writer, and holding the move
    hostage to somebody opening the session is not what "move at the next safe
    point" meant.
    """
    root = server.root
    delivered_pid: int | None = None
    last_attempt = 0.0

    while True:
        record = read_record(root, session_id)
        if record is None:
            return ""
        phase = str(record.get("phase") or "")
        if is_terminal(record):
            return phase
        if phase == QUEUE_PHASE_COPYING:
            return _drive_copy(server, session_id)
        if phase not in (QUEUE_PHASE_QUEUED, QUEUE_PHASE_FINISHING, QUEUE_PHASE_PAUSED):
            return phase

        runtime = _find_runtime(root, session_id)
        if runtime is not None:
            pid = int(getattr(runtime, "pid", -1) or -1)
            if delivered_pid != pid and time.monotonic() - last_attempt >= DRIVER_WAIT_S:
                if not _supports_queue(runtime):
                    # FAIL CLOSED, the fence's own shape: an owner too old to
                    # advertise the queued-move capability never installed the
                    # boundary watch, so accepting the queue would be a
                    # promise nobody keeps. A move is worth a /reload.
                    transition(
                        root,
                        session_id,
                        QUEUE_PHASE_FAILED,
                        detail=(
                            "this conversation's runtime is too old to queue a move "
                            "safely; reload it first, then queue the move again"
                        ),
                        code="not_implemented",
                        writer="relay",
                    )
                    return QUEUE_PHASE_FAILED
                last_attempt = time.monotonic()
                outcome = _deliver_intent(server, session_id)
                if outcome == "cancelled":
                    return ""
                if outcome == "delivered":
                    delivered_pid = pid
                    # The intent is installed: from here the record's own word
                    # for the state is "waiting for the safe point".
                    transition(root, session_id, QUEUE_PHASE_FINISHING, writer="relay")
            time.sleep(DRIVER_POLL_S)
            continue

        delivered_pid = None
        if _lease_holder_present(root, session_id):
            # A lease without a record is still a writer (``lop exec``, a
            # headless REPL, a runtime mid-boot). The queue waits for it — the
            # design's "refused, not drained" at its most literal.
            transition(root, session_id, QUEUE_PHASE_FINISHING, writer="relay")
            time.sleep(DRIVER_WAIT_S)
            continue

        # NO WRITER: the safe point. ``paused`` is only ever written by a
        # runtime that actually paused (:func:`claim_pause`); a source that
        # went cold without one goes straight to the copy, and the phase
        # history says exactly that (a skipped ``paused`` is the honest record
        # of a session nobody had to quiesce).
        transition(root, session_id, QUEUE_PHASE_COPYING, writer="relay")
        return _drive_copy(server, session_id)


def _deliver_intent(server: "RelayServer", session_id: str) -> str:
    """One delivery attempt to the session's live runtime. ONE-SHOT connection.

    A fresh attach connection per attempt, held for the one request and closed
    (the ``_retire_local_runtime._ask`` shape), rather than a long-lived
    client: the client's reader task is bound to the loop that created it, so
    a connection REUSED across ``asyncio.run`` calls would send requests no
    reader could answer. The cadence that pays for this is the driver's — one
    connect per runtime PROCESS, not per poll.

    Returns ``"delivered"`` (the runtime accepted the intent and now owns the
    boundary watch), ``"cancelled"`` (the record ended underneath the attempt
    — the caller stops), or ``"unreachable"`` (dial or request failed; the
    caller retries, and a runtime that never answers is handled by the
    runtime record eventually disappearing).
    """
    import asyncio

    from local_operator.mobile.attach_client import AttachClient

    record = _find_runtime(server.root, session_id)
    if record is None:
        return "unreachable"
    # The record's own target fields ride along: the runtime's ``move_pending``
    # announce names where the conversation is going, and this is the sender
    # that holds the record saying it.
    queued = read_record(server.root, session_id) or {}
    to_device = str(queued.get("to_device") or "")
    to_name = str(queued.get("to_name") or "") or to_device
    client = AttachClient(
        lambda _projection: None,
        lambda _reason: None,
        locality="remote",
        on_operator_prompt=lambda copy: logger.debug("move queue: %s", copy),
    )

    async def _ask() -> dict[str, Any]:
        await client.connect(record, session_id)
        return await client.queue_move(to_device=to_device, to_name=to_name)

    try:
        stage = asyncio.run(_ask())
    except Exception as exc:  # noqa: BLE001 — every failure here is "retry", never a fold
        logger.debug("move queue: delivery to %s failed: %s", session_id, exc)
        return "unreachable"
    finally:
        client.close()
    word = str((stage or {}).get("stage") or "")
    if word == "cancelled":
        return "cancelled"
    logger.info("move queue: %s installed the queued move (stage %s)", session_id, word or "queued")
    return "delivered"


def _supports_queue(record: Any) -> bool:
    """Whether this runtime advertises the queued-move capability (fail closed)."""
    from local_operator.session.runtime.types import QUEUED_MOVE_CAPABILITY

    caps = getattr(record, "capabilities", None) or ()
    return QUEUED_MOVE_CAPABILITY in tuple(caps)


def _drive_copy(server: "RelayServer", session_id: str) -> str:
    """Send the invite that starts the destination's pull, then watch for commit.

    The copy itself is the EXISTING pull (``net_session_move``, destination
    driven); this function contributes exactly the trigger and the watch. It
    watches this device's own durable progress — the tombstone (commit) — the
    same way ``_offload`` does, because that is the only fact the source can
    honestly report.
    """
    root = server.root
    from local_operator.network import mobility
    from local_operator.session.placement import handoff_in_flight

    record = read_record(root, session_id)
    if record is None or is_terminal(record):
        return str((record or {}).get("phase") or "")
    to_device = str(record.get("to_device") or "")
    if not to_device:
        transition(
            root,
            session_id,
            QUEUE_PHASE_FAILED,
            detail="the queued move names no destination device, so nothing was changed",
            code="not_authorised",
            writer="relay",
        )
        return QUEUE_PHASE_FAILED

    if str(record.get("phase")) != QUEUE_PHASE_COPYING:
        transition(root, session_id, QUEUE_PHASE_COPYING, writer="relay")
    record = read_record(root, session_id) or record

    # If an earlier attempt already committed (relay restarted mid-copy), the
    # tombstone is the commit: finish the record without asking anybody.
    if mobility._tombstone(root, session_id):  # noqa: SLF001
        transition(root, session_id, QUEUE_PHASE_RESUMED, writer="relay")
        return QUEUE_PHASE_RESUMED

    # A stale journal from a crashed attempt blocks a fresh invite; let the
    # existing recovery table settle it first (the instance rule keeps a LIVE
    # handoff's entry alone, which is also what we want: a handoff already in
    # flight IS this move).
    try:
        mobility.reconcile(root, server=server, only=session_id)
    except Exception:  # noqa: BLE001 — recovery is best effort; the invite decides
        logger.debug("move queue: reconcile before copy failed", exc_info=True)

    try:
        entry = handoff_in_flight(root, session_id)
    except Exception:  # noqa: BLE001
        entry = None
    if entry and str(entry.get("phase") or "") == "handing-off":
        # A handoff that is past the rollback point: watch it to the tombstone.
        return _watch_for_commit(server, session_id)
    if entry:
        # A ``prepared`` entry that reconcile left alone is written by THIS
        # relay's live instance: a handoff this queue did not start is under
        # way (some other route called ``move.prepare``), and starting a
        # second copy would be the double-writer INV-1 forbids. Fold with the
        # family's own ``in_progress`` sentence; the other handoff decides the
        # outcome and the user can queue again after it.
        transition(
            root,
            session_id,
            QUEUE_PHASE_FAILED,
            detail=(
                "a handoff of that conversation is already in progress on this device, "
                "so the queued move was not started; nothing was changed"
            ),
            code="in_progress",
            writer="relay",
        )
        return QUEUE_PHASE_FAILED

    try:
        target_device, target_name = mobility._resolve_typed_peer(  # noqa: SLF001
            server, str(record.get("to_name") or to_device)
        )
    except Exception as refusal:  # noqa: BLE001 — a Moved carries the sentence
        transition(
            root,
            session_id,
            QUEUE_PHASE_FAILED,
            detail=str(getattr(refusal, "message", refusal) or "the destination is unreachable"),
            code=str(getattr(refusal, "code", "unreachable") or "unreachable"),
            writer="relay",
        )
        return QUEUE_PHASE_FAILED
    try:
        link = mobility._link_for_move(server, target_device, target_name)  # noqa: SLF001
    except Exception as refusal:  # noqa: BLE001
        transition(
            root,
            session_id,
            QUEUE_PHASE_FAILED,
            detail=str(getattr(refusal, "message", refusal) or "the destination is unreachable"),
            code=str(getattr(refusal, "code", "unreachable") or "unreachable"),
            writer="relay",
        )
        return QUEUE_PHASE_FAILED
    try:
        accepted = mobility.LinkTransport(server, link, session_id).ask(
            {
                "op": "net_session_move",
                "phase": "invite",
                "session_id": session_id,
                "keep": False,
                "wait_s": 0.0,
                "mode": "move",
                "engage_on_arrival": bool(record.get("engage_on_arrival") or False),
                "to_device": target_device,
                # THE COMMIT'S OWN CREDENTIALS: the record's request id (its
                # transfer key) plus the resume flag, so the source's queue
                # guard admits exactly this prepare and no other (see
                # ``mobility._source_prepare``'s queue block). ``queue`` is
                # deliberately False — the queueing is DONE; this is the move.
                "request_id": str(record.get("request_id") or ""),
                "resume_queued": True,
            }
        )
    except Exception as refusal:  # noqa: BLE001 — a Moved carries the sentence
        transition(
            root,
            session_id,
            QUEUE_PHASE_FAILED,
            detail=str(getattr(refusal, "message", refusal) or "the invitation was refused"),
            code=str(getattr(refusal, "code", "refused") or "refused"),
            writer="relay",
        )
        return QUEUE_PHASE_FAILED
    if str(accepted.get("result") or "") == "refused":
        # The destination refused the invite itself (membership, an id it
        # already holds). Verbatim, like every refusal in this family.
        transition(
            root,
            session_id,
            QUEUE_PHASE_FAILED,
            detail=str(accepted.get("message") or "the destination refused the move"),
            code=str(accepted.get("code") or "refused"),
            writer="relay",
        )
        return QUEUE_PHASE_FAILED
    logger.info("move queue: invited %s to pull %s", target_device, session_id)
    return _watch_for_commit(server, session_id)


def _watch_for_commit(server: "RelayServer", session_id: str) -> str:
    """Wait for the commit (tombstone) or a reported refusal; fold on timeout.

    Refusals are scoped to the device this move was invited to
    (``from_device``), the ``_source_refused`` contract: a member that did not
    receive this move cannot end its wait.
    """
    from local_operator.network import mobility

    root = server.root
    progress = mobility.progress_for(server)
    invited = str((read_record(root, session_id) or {}).get("to_device") or "")
    deadline = time.monotonic() + COPY_WATCH_S
    while time.monotonic() < deadline:
        record = read_record(root, session_id)
        if record is None or is_terminal(record):
            return str((record or {}).get("phase") or "")
        if mobility._tombstone(root, session_id):  # noqa: SLF001
            transition(root, session_id, QUEUE_PHASE_RESUMED, writer="relay")
            logger.info("move queue: %s committed and resumed at its destination", session_id)
            return QUEUE_PHASE_RESUMED
        refusal = progress.refusal(session_id, from_device=invited)
        if refusal is not None:
            code, message = refusal
            transition(
                root,
                session_id,
                QUEUE_PHASE_FAILED,
                detail=message,
                code=code,
                writer="relay",
            )
            return QUEUE_PHASE_FAILED
        time.sleep(0.25)
    transition(
        root,
        session_id,
        QUEUE_PHASE_FAILED,
        detail=(
            "the destination did not confirm the move in time; this device still holds "
            "the conversation (its state is unchanged), and the move can be requested "
            "again"
        ),
        code="deadline_exceeded",
        writer="relay",
    )
    return QUEUE_PHASE_FAILED


def note_committed(root: Path | str, session_id: str) -> None:
    """The source's commit observed it directly: fold the record to ``resumed``.

    Called from ``mobility._source_commit`` right after the tombstone lands, so
    the record reaches its terminal state without waiting for the driver's next
    poll — and idempotently, because the driver's tombstone watch may win the
    race. Terminal is write-once; whoever is second is a no-op.
    """
    transition(root, session_id, QUEUE_PHASE_RESUMED, writer="relay")
