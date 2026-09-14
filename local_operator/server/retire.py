"""How the ``serve`` daemon leaves when a new build lands, without cutting work.

Every other participant on this host already rolls forward on its own: an idle
session runtime notices that the install on disk moved (``buildwatch``), waits
out the settle window, announces ``retiring`` to its viewers and exits, so the
viewer's next engage runs the new build (``session/runtime/process.py``). The
daemon had none of that — it kept serving the old build until a person killed
it, which is why "update the daemon" used to mean downtime.

This module gives it the same shape, with three deliberate differences, each
argued where it is implemented:

* it ANNOUNCES in its rendezvous record (``retiring_from``/``retiring_to``)
  instead of over a socket, because a daemon's readers are record readers;
* it REFUSES new session spawns once it has announced (:class:`DaemonRetiring`)
  rather than pretending it is still a place to start work;
* it NEVER restarts itself — it says which build it left for and which command
  brings it back, and leaves supervision to whoever supervises it.

**What "in flight" means here is not what it means for a runtime**, and that is
the one thing a reader must not assume: a runtime OWNS its turn, so exiting it
cancels the operator's work, while a turn on this daemon runs in a detached
child process (``session/runtime/launch.py::_spawn_runtime`` spawns
``-m local_operator.session.runtime.process``). See :func:`in_flight` for the
terms this process can actually observe, and for the term it cannot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import signal
from typing import TYPE_CHECKING, Any, Callable

from local_operator import buildwatch

if TYPE_CHECKING:
    from fastapi import FastAPI

    from local_operator.session.runtime import registry as session_registry
    from local_operator.update import BuildStamp

logger = logging.getLogger("local_operator.server.retire")

#: ``app.state`` flag: true from the instant this daemon has announced its
#: retirement until the process exits. It lives on ``app.state`` rather than on
#: the desktop pool because the pool is created LAZILY by the first request that
#: needs it (``routes/desktop_sessions.py::host``) — a flag set on an existing
#: pool would be missed by a pool that is built after the announcement, and the
#: refusal it carries is the whole point of announcing.
RETIRING_STATE_ATTR = "serve_retiring"


class DaemonRetiring(RuntimeError):
    """A retiring daemon refuses to admit new work. 503 for the client, not 500.

    Deliberately a subclass of ``RuntimeError`` so an unmapped caller still gets
    the routes' existing 503 ladder rather than a 500 with a traceback, and
    deliberately a TYPE rather than a string the routes match on: the client's
    correct response is not "retry" but "rediscover the successor through the
    record" (design §7), and that is a decision only a distinguishable refusal
    can support.
    """

    #: The machine-readable code the desktop routes put on the wire beside the
    #: sentence, so a client never has to parse English to find it.
    code = "daemon-retiring"


def retiring(app: "FastAPI") -> bool:
    """Has this daemon announced that it is leaving?

    Absent means no: an app that never published a record has no poll task to
    set the flag, and a reader must see "not retiring" rather than an error.
    """
    state: Any = getattr(app, "state", None)
    return bool(getattr(state, RETIRING_STATE_ATTR, False))


def _subscribers(value: Any) -> int:
    """``value``'s live subscriber count, or 0 when it cannot be asked.

    Duck-typed and total on purpose: the predicate below runs on a timer
    against whatever the running app actually holds, and a missing or
    differently-shaped attribute must mean "nothing observed here" rather than
    "crash the reaper". Every caller passes a real object in production — the
    tolerance is for reduced test apps.
    """
    count: Callable[[], Any] | None = getattr(value, "live_connection_count", None)
    if not callable(count):
        return 0
    try:
        return int(count())
    except Exception:  # noqa: BLE001 — a broken accessor must not pin the daemon
        logger.debug("live_connection_count failed", exc_info=True)
        return 0


def in_flight(app: "FastAPI") -> str | None:
    """The first reason this daemon must not leave yet, or ``None``.

    WHAT IS ACTUALLY IN FLIGHT FOR THIS PROCESS, verified against the code:

    1. **An open event stream (SSE).** ``EventBroker.stats()["subscribers"]``
       (``server/utils/event_broker.py``; the streams are opened by
       ``server/routes/sse.py``). These channels are per-TURN
       (``message:<record id>`` / ``job:<job id>``), not a always-on socket, so
       a non-zero count means a turn is streaming live frames to a client and
       this process is the only source of them — the broker's replay buffer dies
       with the process, so those frames cannot be recovered from the successor.
    2. **An open legacy WebSocket stream.** ``WebSocketManager``
       (``server/utils/websocket_manager.py``) carries the same frames on the
       transport that predates SSE and is still served.
    3. **The desktop plane.** ``DesktopSessions.in_flight_reason()``
       (``server/utils/desktop_sessions.py``) owns the bridge terms — an
       in-flight HTTP operation, a live watch lease and a runtime mid-spawn —
       because the pool is what knows its bridges.

    WHY NOT "A SESSION MID-TURN", the obvious fourth term: it is not a property
    of this process. Turns run in detached ``session/runtime/process.py``
    children that this daemon spawns and then merely observes (``launch.py``
    spawns, ``AttachedSession`` dials), so exiting here cannot cancel one — the
    child keeps its turn and its own residency drain decides when it stops. The
    VIEW of that turn is a term (1-3): a client watching it holds an SSE
    subscription or a desktop attach, and those keep this daemon alive. Stated
    rather than silently omitted, because "the daemon holds a turn" is exactly
    the assumption a reader would otherwise make.

    THE DRAIN IS UNBOUNDED BY DESIGN, and each term is bounded instead: a turn's
    SSE stream ends with the turn, a legacy WebSocket likewise, a desktop HTTP
    operation completes (or times out) and a watch lease lapses
    ``WATCH_TTL`` (45 s, ``desktop_sessions.py``) after the last heartbeat from
    a window that is no longer looking. Waiting is the operator's rule — nothing
    in flight may be cut — and unlike a session runtime there is no viewer here
    to re-engage a successor, so an exit under a live stream is an interruption
    nobody undoes. The release valve is the RECORD: the announcement names the
    new build, and a reader that sees it stops using this process.
    """
    state: Any = getattr(app, "state", None)
    if state is None:
        return None

    broker: Any = getattr(state, "event_broker", None)
    stats: Callable[[], dict[str, Any]] | None = getattr(broker, "stats", None)
    if callable(stats):
        try:
            streams = int(stats().get("subscribers", 0))
        except Exception:  # noqa: BLE001 — a broken probe must not pin the daemon
            logger.debug("event broker stats failed", exc_info=True)
            streams = 0
        if streams:
            return f"{streams} SSE subscription(s)"

    sockets = _subscribers(getattr(state, "websocket_manager", None))
    if sockets:
        return f"{sockets} websocket connection(s)"

    pool: Any = getattr(state, "desktop_sessions", None)
    desktop: Callable[[], Any] | None = getattr(pool, "in_flight_reason", None)
    if callable(desktop):
        try:
            reason: Any = desktop()
        except Exception:  # noqa: BLE001 — a broken probe must not pin the daemon
            logger.debug("desktop in-flight probe failed", exc_info=True)
            reason = None
        if reason:
            return reason
    return None


def announce(
    app: "FastAPI",
    publisher: "session_registry.RecordPublisher",
    *,
    retiring_from: str,
    retiring_to: str,
) -> None:
    """Latch the refusal, then publish the reason into the record.

    THE LATCH GOES FIRST, synchronously, before the record is written: from the
    instant a reader can see "retiring onto build X", this daemon must already
    be refusing to start work it is not going to finish, or the announcement
    would be a claim the process does not honour. It is a one-way latch while
    the process lives — the same shape as the desktop claim's
    (``server/desktop.py``), and for the same reason: un-announcing would make
    every reader that acted on the announcement wrong.

    The record write is ``publisher.heartbeat(**updates)``, the existing
    rewrite-whole-record path, so the staged write, the ``0600`` file and the
    fresh ``heartbeat_at`` are the shared registry's rather than a second
    spelling here. Nothing else about the record changes: it stays ``live``
    (the process is alive and answering) until the clean exit removes it.
    """
    if getattr(app, "state", None) is not None:
        setattr(app.state, RETIRING_STATE_ATTR, True)
    publisher.heartbeat(retiring_from=retiring_from, retiring_to=retiring_to)


def _request_shutdown() -> None:
    """Ask OUR OWN process to stop, through the path a real SIGTERM takes.

    ``os.kill(getpid(), SIGTERM)`` rather than ``sys.exit`` or ``os._exit``:
    uvicorn installs a SIGTERM handler while it serves, so this lands in exactly
    the clean shutdown an operator's ``kill`` produces — uvicorn stops accepting
    connections, lets in-flight responses finish, and then runs the lifespan's
    shutdown half, which removes the record under the shared ``unpublish``.
    Raising out of the poll task would unwind only that task and leave the
    daemon serving the old build forever; ``os._exit`` would skip the record
    removal and leave a file claiming a live daemon at a dead port, which is the
    artefact the record exists to prevent.

    KNOWN LIMIT, Windows: there ``SIGTERM`` is ``TerminateProcess``, so this is
    the abrupt path and the record is left for the next ``scan()`` to reap —
    which is the same self-healing path every ``SIGKILL`` on POSIX already
    takes, and the reason the reader reaps at all.
    """
    os.kill(os.getpid(), signal.SIGTERM)


async def _leave_for(
    newer: "BuildStamp",
    boot: "BuildStamp",
    app: "FastAPI",
    publisher: "session_registry.RecordPublisher",
    *,
    stop: asyncio.Event,
    exit_process: Callable[[], None],
) -> bool:
    """Announce, notice, then hand the exit to the process's own shutdown path.

    Returns True once it has asked to exit (the caller stops polling), False
    when a stop arrived during the notice — a shutdown owns its own exit, and
    racing it here would make the record's last state depend on which path won.

    **The stagger is the notice.** ``BUILD_STAGGER_S`` exists to spread a fleet
    of notices over a jittered window so nothing thunders; for this daemon it is
    doing that AND serving as the announcement's lifetime: the record carries
    ``retiring_from``/``retiring_to`` for the whole window, which is what makes
    the handover legible to a reader rather than a daemon that simply vanished.
    A jittered slice (not the full constant) keeps the spread's purpose: a host
    running several daemons should not flip them all to ``retiring`` — and then
    stop them — on one tick.
    """
    announce(
        app,
        publisher,
        retiring_from=boot.label(),
        retiring_to=newer.label(),
    )
    logger.info(
        "serve daemon: the install on disk is %s but this process loaded %s; "
        "retiring and refusing new session spawns",
        newer.label(),
        boot.label(),
    )
    record = getattr(publisher, "record", None)
    if record is not None and not getattr(record, "desktop", False):
        # UNSUPERVISED, and this is the one case a person must act on: nothing
        # re-discovers and restarts a daemon that no app started, so a silent
        # exit would strand the user with "the backend is gone" and no reason.
        # WARNING, not INFO, because the daemon's console logging runs at
        # LOG_LEVEL's default WARNING — a line nobody sees is not a notice.
        logger.warning(
            "serve daemon: nothing supervises this daemon and it will not restart "
            "itself; the new build is already on disk, so start it again with "
            "`lop serve` (or `lop update` first if the install needs finishing)",
        )
    delay = random.uniform(
        0, buildwatch.build_stagger_seconds()
    )  # noqa: S311 — jitter, not security
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
        return False  # a stop landed during the notice; its path owns the exit
    except asyncio.TimeoutError:
        pass
    exit_process()
    return True


async def retirement_poll(
    app: "FastAPI",
    publisher: "session_registry.RecordPublisher",
    *,
    stop: asyncio.Event,
    exit_process: Callable[[], None] | None = None,
) -> None:
    """Poll the install on disk; announce and leave once it has moved on.

    Started by ``server/app.py``'s lifespan beside the record publisher, and
    stopped (cancelled, with ``stop`` set) in its shutdown half. The boot stamp
    is taken HERE, first, before any sleep: a stamp read after the install
    already moved under the daemon would compare equal to the moved install and
    the handover could never be detected.

    ``buildwatch.build_changed`` is the shared rule — same stamp, an unsettled
    marker or an unreadable one is "no action", and ``LOP_BUILD_PREFIX`` (the
    e2e-only override) points the whole check at a fake install root. This loop
    adds only what is this process's own: the in-flight gate, the announcement
    and the refusal.
    """
    boot = buildwatch.boot_build()
    if boot is None:
        # No baseline, so no move can ever be PROVEN. Returning is the honest
        # shape: a watcher that polls forever comparing against nothing would
        # either never fire (what it does today) or fire on noise (worse).
        logger.debug("serve daemon: no boot build stamp; build watch disabled")
        return

    exit_process = exit_process or _request_shutdown
    last_reason: str | None = None
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=buildwatch.BUILD_CHECK_S)
            return  # stopped between checks; nothing to announce
        except asyncio.TimeoutError:
            pass
        newer = buildwatch.build_changed(boot)
        if newer is None:
            last_reason = None
            continue
        reason = in_flight(app)
        if reason is not None:
            if reason != last_reason:
                # Logged on CHANGE, not per tick: the point of the line is to
                # explain why an announced-free daemon is still serving the old
                # build, and repeating it every 5 s is how a log becomes noise
                # nobody reads.
                logger.info(
                    "serve daemon: build %s is on disk but %s is in flight; "
                    "retiring when it completes",
                    newer.label(),
                    reason,
                )
                last_reason = reason
            continue
        if await _leave_for(newer, boot, app, publisher, stop=stop, exit_process=exit_process):
            return
