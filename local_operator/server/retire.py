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
* it REFUSES new work only once the drain has emptied (:class:`DaemonRetiring`)
  — the announcement comes first and the latch comes last, see below;
* it NEVER restarts itself — it says which build it left for and which command
  brings it back, and leaves supervision to whoever supervises it.

**THE ANNOUNCEMENT COMES FIRST, AND THE LATCH COMES LAST.** They are two
separate events, and the order between them is the design:

1. a settled build change is DETECTED: ``retiring_from``/``retiring_to`` go into
   the record IMMEDIATELY and the daemon keeps serving normally. Nothing is
   latched and no request is refused, because while it is merely announced this
   process is still the only place its clients can work;
2. it keeps polling, and stays up for as long as anything holds it — see
   :func:`in_flight`, which is fail-closed as it always was;
3. only once the drain is empty does it LATCH: refuse new work with the typed
   ``503 daemon-retiring``, then exit cleanly and remove the record.

AND THE ANNOUNCEMENT IS RE-READ EVERY TICK, because it is a claim about the
install rather than a fact about this process: the poll re-asks
``buildwatch.handover_build`` while it is announced, WITHDRAWS the handover from
the record when the install turns out to be back on ``boot`` (a ``lop-update``
rolled back or superseded) or unreadable, and re-announces onto a newer build if
the install moved on again (review round 2, MINOR-2). Announcing once and acting
on it forever meant a daemon that latched, exited and removed its record leaving
its readers a ``retiring_to`` that named a build no longer on disk. The LATCH is
still one-way, and its place in the sequence is unchanged: withdrawal is a
record change, not an un-latch, and it can only happen before ``latch`` runs.

WHY NOT "DRAIN, THEN ANNOUNCE", the obvious order and this module's first shape:
because the terms that hold a daemon include a STANDING ATTACHMENT — the desktop
app's replayable ``/v1/desktop/sessions/{id}/events`` relay and the watch lease
renewed beside it are held open for as long as a conversation is on screen (see
:func:`in_flight`). Announcing only after the drain empties is CIRCULAR for the
daemon the app is attached to: the drain cannot empty until a client lets go, no
client can know to let go until the record says something, and the record says
nothing until the drain is empty. Two reviewers reproduced that on round 1
independently, by execution: 32 s of samples (six check intervals, the settle
long past) with the app's relay held showed no announcement, no record change
and no log line at a default log level.

THE DAEMON HOLDS NO AGENT TURN. Turns run in the detached
``session/runtime/process.py`` children this daemon spawns and then merely
observes; they retire on their own schedule. What this process holds is
ATTACHMENT — a replayable stream and a view lease. The announcement is what
tells an attached client to let go, so it must be written before the client has
let go; and refusing work before that has happened would break the app for an
unbounded period. Hence announce early, latch late.

WHAT A CLIENT MUST DO WITH THE ANNOUNCEMENT is a specification rather than a
hope, and it lives in ``docs/design-daemon-discovery.md`` §7: on seeing
``retiring_from``/``retiring_to`` in the record, the desktop app drops the
``/v1/desktop/sessions/{id}/events`` relay and stops the ``watch`` heartbeat,
then re-binds to the successor once it appears. Until the UI implements that
valve, an app-attached daemon announces and keeps serving — strictly better than
the silence this replaces, but NOT yet a completed update path for the
app-attached case (stated in the PR body too).

**What "in flight" means here is not what it means for a runtime**, and that is
the one thing a reader must not assume: a runtime OWNS its turn, so exiting it
cancels the operator's work, while a turn on this daemon runs in a detached
child process (``session/runtime/launch.py::_spawn_runtime`` spawns
``-m local_operator.session.runtime.process``). See :func:`in_flight` for the
terms this process can actually observe, for which of them is a STANDING
attachment rather than a turn, and for why a probe it cannot read means stay.
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

#: The sentence that rides beside :attr:`DaemonRetiring.code` on the wire. ONE
#: spelling, deliberately: every path that refuses makes the same promise, and a
#: second copy is how two refusals come to mean different things to a client.
RETIRING_MESSAGE = (
    "This backend is restarting onto a new build and is not accepting new work. "
    "Reconnect to the new backend and retry."
)

#: ``app.state`` flag: true from the instant this daemon has LATCHED — that is,
#: from the moment the drain emptied and it began refusing new work, until the
#: process exits. It lives on ``app.state`` rather than on the desktop pool
#: because the pool is created LAZILY by the first request that needs it
#: (``routes/desktop_sessions.py::host``) — a flag set on an existing pool would
#: be missed by a pool built after the latch, and the refusal it carries is the
#: whole point of latching.
#:
#: NOT set while merely announced, and that is the round-2 correction: the
#: announcement's job is to tell an attached client to let go, and a daemon that
#: refused work at that instant would be unusable for as long as the client took
#: to react.
RETIRING_STATE_ATTR = "serve_retiring"


class DaemonRetiring(RuntimeError):
    """A retiring daemon refuses to admit new work. 503 for the client, not 500.

    Deliberately a subclass of ``RuntimeError`` so an unmapped caller still gets
    the routes' existing 503 ladder rather than a 500 with a traceback, and
    deliberately a TYPE rather than a string the routes match on: the client's
    correct response is not "retry" but "rediscover the successor through the
    record" (design §7), and that is a decision only a distinguishable refusal
    can support.

    Raised only after the LATCH, so it always means "this process is leaving",
    never "this daemon is thinking about it" — see the module docstring.
    """

    #: The machine-readable code the desktop routes put on the wire beside the
    #: sentence, so a client never has to parse English to find it.
    code = "daemon-retiring"


def retiring(app: "FastAPI") -> bool:
    """Has this daemon latched — i.e. is it refusing new work?

    Absent means no: an app that never published a record has no poll task to
    set the flag, a daemon that has only announced is still serving, and a
    reader must see "not refusing" rather than an error.

    NOT the same question as "has it announced"; that one is answered by the
    record's ``retiring_from``/``retiring_to``, which is what every reader
    outside this process looks at.
    """
    state: Any = getattr(app, "state", None)
    return bool(getattr(state, RETIRING_STATE_ATTR, False))


def _subscribers(value: Any) -> int | None:
    """``value``'s live subscriber count; ``None`` when the probe could not be read.

    ABSENT and BROKEN are different answers and must never be folded together: a
    reduced app that never built a ``WebSocketManager`` has no connections to
    lose, while an accessor that RAISES is a probe that could not be asked. An
    unreadable probe means "stay" — see :func:`in_flight`, whose rule is that a
    probe this process cannot read may never be the reason it leaves.
    """
    count: Callable[[], Any] | None = getattr(value, "live_connection_count", None)
    if not callable(count):
        return 0
    try:
        return int(count())
    except Exception:  # noqa: BLE001 — unreadable, which is not the same as zero
        logger.warning(
            "serve daemon: the websocket connection count could not be read", exc_info=True
        )
        return None


def _unreadable(probe: str) -> str:
    """The in-flight reason an UNREADABLE probe produces: stay, and say which one."""
    return f"an in-flight probe that could not be read ({probe})"


def in_flight(app: "FastAPI") -> str | None:
    """The first reason this daemon must not leave yet, or ``None``.

    WHAT IS ACTUALLY IN FLIGHT FOR THIS PROCESS, verified against the code:

    1. **An open event stream (SSE).** ``EventBroker.stats()["subscribers"]``
       (``server/utils/event_broker.py``; the streams are opened by
       ``server/routes/sse.py``). These broker channels are per-TURN
       (``message:<record id>`` / ``job:<job id>``), so a non-zero count means a
       turn is streaming live frames to a client and this process is the only
       source of them — the broker's replay buffer dies with the process, so
       those frames cannot be recovered from the successor.
    2. **An open legacy WebSocket stream.** ``WebSocketManager``
       (``server/utils/websocket_manager.py``) carries the same frames on the
       transport that predates SSE and is still served.
    3. **The desktop plane.** ``DesktopSessions.in_flight_reason()``
       (``server/utils/desktop_sessions.py``) owns the bridge terms — an in-flight
       HTTP operation, a live watch lease and a runtime mid-spawn — because the
       pool is what knows its bridges.

    WHICH OF THOSE IS A STANDING ATTACHMENT, and why it decides the SHAPE of the
    retirement rather than only its timing: term 3's first two entries come from
    the desktop app's own relay. ``GET /v1/desktop/sessions/{id}/events``
    acquires the bridge BEFORE it returns response headers and releases it only
    when the stream tears down, so ``users > 0`` for as long as a conversation is
    on screen — there is no turn boundary and no TTL on that stream; the app
    holds it until the view unmounts or the app quits. The watch lease beside it
    is renewed every 15 s within a 45 s TTL, and counts whether or not the window
    is visible. Term 1 and 2 are bounded (a turn's stream ends with the turn, a
    WebSocket likewise) and do not pin an app-attached daemon on their own.

    Therefore the announcement is written BEFORE this predicate is consulted, and
    the drain is only what decides when the LATCH happens (module docstring).
    A reader that sees the announcement stops using this process, which empties
    the standing terms; the terms themselves are never cut.

    A PROBE THAT CANNOT BE READ MEANS STAY, and the direction is the point. A
    broken or unreadable probe (a raising ``stats()``, a raising accessor, a
    raising desktop probe) must not be read as "nothing in flight": the cost of
    staying is a daemon that keeps serving its old build for one more check
    (bounded, observable — it logs), while the cost of leaving is an exit under
    a live stream, which is an interruption nobody undoes. Missing probes are
    NOT failures — a reduced app that never built an ``EventBroker`` has nothing
    in flight either way, and reading that as "stay" would pin every unit-sized
    app forever.
    """
    state: Any = getattr(app, "state", None)
    if state is None:
        return None

    broker: Any = getattr(state, "event_broker", None)
    stats: Callable[[], dict[str, Any]] | None = getattr(broker, "stats", None)
    if callable(stats):
        try:
            streams = int(stats().get("subscribers", 0))
        except Exception:  # noqa: BLE001 — unreadable, and unreadable means stay
            logger.warning(
                "serve daemon: the event broker's subscriber count could not be read",
                exc_info=True,
            )
            return _unreadable("the event broker's subscriber count")
        if streams:
            return f"{streams} SSE subscription(s)"

    sockets = _subscribers(getattr(state, "websocket_manager", None))
    if sockets is None:
        return _unreadable("the websocket connection count")
    if sockets:
        return f"{sockets} websocket connection(s)"

    pool: Any = getattr(state, "desktop_sessions", None)
    desktop: Callable[[], Any] | None = getattr(pool, "in_flight_reason", None)
    if callable(desktop):
        try:
            reason: Any = desktop()
        except Exception:  # noqa: BLE001 — unreadable, and unreadable means stay
            logger.warning(
                "serve daemon: the desktop in-flight probe could not be read", exc_info=True
            )
            return _unreadable("the desktop plane")
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
    """Publish the handover into the record. THE WRITE IS THE ANNOUNCEMENT.

    NOTHING IS LATCHED HERE, and that is the round-2 correction: a daemon that
    has only announced is still the place its client works, so refusing at this
    point would break the app for as long as it takes the client to react (see
    the module docstring). The latch belongs to :func:`latch`, which the caller
    runs only once the drain has emptied.

    THE WRITE IS ALLOWED TO FAIL, and the caller must let it: ``heartbeat`` can
    raise on a read-only or full config directory (the failure class the sibling
    ``heartbeat_loop`` swallows as self-healing, ``server/registry.py``), and a
    daemon that latched on a failed write would refuse every request forever
    without ever leaving. So the caller retries the announcement on its next
    check and does not latch in the meantime — at worst the daemon keeps serving
    the old build and says so loudly, which is the behaviour this whole change
    replaces rather than a new failure mode.

    The record write is ``publisher.heartbeat(**updates)``, the existing
    rewrite-whole-record path, so the staged write, the ``0600`` file and the
    fresh ``heartbeat_at`` are the shared registry's rather than a second
    spelling here. Nothing else about the record changes: it stays ``live``
    (the process is alive and answering) until the clean exit removes it.
    """
    publisher.heartbeat(retiring_from=retiring_from, retiring_to=retiring_to)


def withdraw(publisher: "session_registry.RecordPublisher") -> None:
    """Take the handover back out of the record. THE WRITE IS THE WITHDRAWAL.

    Called when the poll re-reads the install and the move it announced is no
    longer there: the marker is back on the build this process loaded (a
    ``lop-update`` that failed and was rolled back, or was superseded by the
    running build — review round 2, MINOR-2), or the stamp cannot be read as a
    build at all (:func:`buildwatch.proves_a_move`). In both cases the daemon is
    NOT leaving: it is serving the right build, or a build nobody can identify,
    and a record that still told its readers to let go would be telling them to
    hand over to nothing.

    NOT AN UN-LATCH, and it cannot become one: this is a record field, the latch
    is :func:`latch`'s ``app.state`` flag, and the poll only reaches here before
    ``latch`` has run. A withdrawal after the refusal exists is unrepresentable
    by construction rather than by convention.

    Allowed to FAIL, exactly like :func:`announce`, and the caller must let it:
    a read-only or full record directory is the failure class the sibling
    ``heartbeat_loop`` swallows as self-healing, and the poll's answer is the same
    one it gives a failed announcement — log at WARNING, retry on the next check,
    and do NOT proceed to the latch while the record still claims a handover the
    process no longer believes in.
    """
    publisher.heartbeat(retiring_from="", retiring_to="")


def latch(app: "FastAPI") -> None:
    """Refuse new work, from here until the process exits.

    ONE-WAY while the process lives, the same shape as the desktop claim's
    (``server/desktop.py``) and for the same reason: un-announcing would make
    every reader that acted on the announcement wrong.

    Called ONLY after the announcement is readable AND the drain is empty, so
    the refusal can never arrive before the record has told a client to let go
    (see the module docstring for why that order is the design). "Readable" is
    re-read on every tick rather than remembered: the poll withdraws the
    announcement if the install on disk stops proving the move it announced, and
    the latch is reachable only on a tick where the announcement still stands.
    """
    if getattr(app, "state", None) is not None:
        setattr(app.state, RETIRING_STATE_ATTR, True)


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


def observe_poll(task: "asyncio.Task[None]") -> None:
    """Log a build watch that DIED, instead of a daemon that never retires.

    The poll is created bare in the lifespan and awaited only at teardown, so
    without this a task that raised would be silent: the daemon would keep
    serving the old build with no record change and no line explaining why —
    the same silent no-retirement this change exists to remove, arrived at from
    the other side. Attached with ``add_done_callback``, so it also covers a
    failure the poll's own ``try`` did not name.

    Deliberately does not clear the latch either: a latched daemon that failed on
    its way out has already told its readers to leave, and re-admitting work
    would make every one of them wrong. So this logs an ERROR, the announcement
    and the refusal both stand, and what can still make the promise good is
    whoever supervises the process — the next `lop serve` on the new build (or
    the app, for a daemon it owns). Silence would leave the reader with neither
    the handover nor a reason.
    """
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "serve daemon: the build watch died and this daemon will not retire on its "
            "own; restart it to pick up the new install",
            exc_info=error,
        )


async def _refusal_window(stop: asyncio.Event, exit_process: Callable[[], None]) -> None:
    """Wait out the jittered refusal window, then hand the exit to the process.

    A stop that arrives during the window ENDS this function without asking for
    an exit: a shutdown owns its own exit, and racing it here would make the
    record's last state depend on which of the two paths won (the same rule the
    runtime's ``_refresh_for`` follows for its stagger).

    WHY THERE IS A WINDOW AT ALL, now that the announcement has already lived in
    the record for as long as the drain took to empty: it gives the typed
    ``503 daemon-retiring`` a life a client can actually observe. A latch that
    exited within the same tick would answer a racing client with a closed
    socket and no explanation, where the announced record is what a client is
    meant to act on. The jittered slice of ``BUILD_STAGGER_S`` is the same
    constant the runtime side uses to spread a fleet's notices, so several
    daemons on one host do not refuse and leave on one tick.
    """
    delay = random.uniform(0, buildwatch.build_stagger_seconds())  # noqa: S311 — jitter
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
        return  # a stop landed during the window; its path owns the exit
    except asyncio.TimeoutError:
        pass
    exit_process()


async def retirement_poll(
    app: "FastAPI",
    publisher: "session_registry.RecordPublisher",
    *,
    stop: asyncio.Event,
    exit_process: Callable[[], None] | None = None,
    boot: "BuildStamp | None" = None,
) -> None:
    """Poll the install on disk; announce the move, then latch and leave.

    Started by ``server/app.py``'s lifespan beside the record publisher, and
    stopped (cancelled, with ``stop`` set) in its shutdown half. The two phases
    are the module docstring's sequence, and the loop is deliberately shaped by
    them: while ``announced`` is None this task is looking for a move and will
    not consult the drain, and once it has announced it never looks at the
    record again except to keep the drain waiting.

    ``boot`` is the build stamp this process LOADED, sampled by the caller
    BEFORE it published the record (``app.py``) — and defaulted here so a direct
    caller (a test, a driver) can sample it itself. It has to precede the record
    for a reason worth stating: a reader's first act is to wait for the record
    to appear and then act, and a baseline read AFTER the record was published
    would adopt a marker flipped in that gap as the build this process loaded —
    the change would then be invisible for the rest of the process's life, with
    no log line at all (QA round 1, Q2: 3 of 7 immediate flips were swallowed
    that way). The baseline is logged for the same reason: a watcher that can
    never fire should be diagnosable from its own output.

    ``buildwatch.build_changed`` is the shared rule — same stamp, an unsettled
    marker or an unreadable one is "no action", and ``LOP_BUILD_PREFIX`` (the
    e2e-only override) points the whole check at a fake install root. This loop
    adds only what is this process's own: the announcement, the re-read of it
    (``buildwatch.handover_build``, which is why an announcement can be WITHDRAWN
    again), the in-flight gate and the refusal.
    """
    if boot is None:
        boot = buildwatch.boot_build()
    if boot is None:
        # No baseline, so no move can ever be PROVEN. Returning is the honest
        # shape: a watcher that polls forever comparing against nothing would
        # either never fire (what it does today) or fire on noise (worse).
        logger.debug("serve daemon: no boot build stamp; build watch disabled")
        return

    logger.info("serve daemon: build watch baseline %s", boot.label())
    exit_process = exit_process or _request_shutdown
    announced: "BuildStamp | None" = None
    last_reason: str | None = None
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=buildwatch.BUILD_CHECK_S)
            return  # stopped between checks; nothing to announce
        except asyncio.TimeoutError:
            pass

        if announced is None:
            newer = buildwatch.build_changed(boot)
            if newer is None:
                continue
            try:
                announce(
                    app,
                    publisher,
                    retiring_from=boot.label(),
                    retiring_to=newer.label(),
                )
            except Exception:  # noqa: BLE001 — a failed announcement is RETRIED, never latched over
                # NOT LATCHED, and the daemon KEEPS SERVING: see `announce`.
                logger.warning(
                    "serve daemon: build %s is on disk but the handover could not be "
                    "written to %s; keeping the old build in service and retrying on the "
                    "next check",
                    newer.label(),
                    getattr(publisher, "path", "the record"),
                    exc_info=True,
                )
                continue
            announced = newer
            logger.info(
                "serve daemon: the install on disk is %s but this process loaded %s; "
                "announced in the record and still serving until nothing is in flight",
                newer.label(),
                boot.label(),
            )
            record = getattr(publisher, "record", None)
            if record is not None and not getattr(record, "desktop", False):
                # UNSUPERVISED, and this is the one case a person must act on:
                # nothing re-discovers and restarts a daemon that no app started,
                # so a silent exit would strand the user with "the backend is
                # gone" and no reason. WARNING, not INFO, because the daemon's
                # console logging runs at LOG_LEVEL's default WARNING — a line
                # nobody sees is not a notice.
                logger.warning(
                    "serve daemon: nothing supervises this daemon and it will not restart "
                    "itself; the new build is already on disk, so start it again with "
                    "`lop serve` (or `lop update` first if the install needs finishing)",
                )
            continue

        # THE PREMISE IS RE-READ, and this is the whole of round 2's MINOR-2: the
        # announcement is a claim about the INSTALL on disk, not a fact about this
        # process, so acting on it for the rest of the process's life is acting on
        # a snapshot. One read decides both ways it can stop holding — the install
        # is back on the build this process loaded (a `lop-update` that failed and
        # was rolled back, or was superseded by the running build: the process is
        # the right one after all), or the stamp no longer reads as a build at all
        # (the fail-closed direction `buildwatch.proves_a_move` states). Either
        # way this daemon is NOT leaving, and a daemon that latched, exited and
        # removed its record on a stale announcement left its readers a
        # `retiring_to` naming a build that was no longer on disk. Read per check
        # interval, which is what the detection phase above already costs per
        # tick; nothing here is a hot path.
        standing = buildwatch.handover_build(boot)
        if standing is None:
            try:
                withdraw(publisher)
            except Exception:  # noqa: BLE001 — a failed withdrawal is RETRIED, never latched over
                # Symmetrical with a failed announcement, and for the same reason:
                # the record is the channel every reader acts on, so proceeding to
                # the latch while it still claims a handover this process has
                # stopped believing would retire the daemon under a notice that
                # names the wrong build. Keep serving and try again next check.
                logger.warning(
                    "serve daemon: the handover to %s no longer holds but it could not be "
                    "withdrawn from %s; keeping the announcement and retrying on the next "
                    "check",
                    announced.label(),
                    getattr(publisher, "path", "the record"),
                    exc_info=True,
                )
                continue
            logger.info(
                "serve daemon: the handover to %s no longer holds — the install on disk is "
                "back on %s or no longer readable; withdrawn from the record and still "
                "serving on this build",
                announced.label(),
                boot.label(),
            )
            announced = None
            last_reason = None
            continue
        if standing != announced:
            # The install moved ON AGAIN while this process was announced. Nothing
            # has latched yet, so the honest record is the one naming the build
            # that is actually there now rather than the one it first noticed.
            try:
                announce(
                    app,
                    publisher,
                    retiring_from=boot.label(),
                    retiring_to=standing.label(),
                )
            except Exception:  # noqa: BLE001 — as above: retried, never latched over
                logger.warning(
                    "serve daemon: the install on disk moved on to %s but the record at %s "
                    "could not be updated; retrying on the next check",
                    standing.label(),
                    getattr(publisher, "path", "the record"),
                    exc_info=True,
                )
                continue
            logger.info(
                "serve daemon: the install on disk moved on to %s while %s was announced; "
                "the record now names %s",
                standing.label(),
                announced.label(),
                standing.label(),
            )
            announced = standing
            continue

        reason = in_flight(app)
        if reason is not None:
            if reason != last_reason:
                # Logged on CHANGE, not per tick: the point of the line is to
                # explain why an announced daemon is still serving the old
                # build, and repeating it every 5 s is how a log becomes noise
                # nobody reads.
                logger.info(
                    "serve daemon: build %s is on disk and announced; %s is still in "
                    "flight, so it keeps serving until that completes",
                    announced.label(),
                    reason,
                )
                last_reason = reason
            continue

        # THE DRAIN IS EMPTY: nothing is attached, nothing is being started, so
        # the latch can no longer cut anything. This is the only place the
        # refusal is raised, and it is one-way from here.
        latch(app)
        logger.info(
            "serve daemon: nothing is in flight; refusing new work and leaving for %s",
            announced.label(),
        )
        await _refusal_window(stop, exit_process)
        return
