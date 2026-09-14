"""Announce a changed ``serve`` build without interrupting daemon-owned work.

Production lifespan supplies no ``exit_process`` callback: marker drift only
updates ``retiring_from``/``retiring_to`` and the daemon keeps serving. These
legacy field names are new-build announcements, NOT instructions for a UI to
release SSE or watch leases. Re-read the marker on every check so a rollback or
unreadable build withdraws the announcement and a newer build retargets it.

The daemon DOES own legacy scheduled/async work. In particular,
``SchedulerService._run_tasks`` runs in this process and lifespan shutdown
cancels it. Detached desktop runtimes are not the only execution path, and
``in_flight`` is not a complete work-safety predicate. A marker also supplies no
guarantee that a successor is ready. Neither an idle-looking daemon nor a
claimed desktop daemon may therefore latch, refuse, or exit on build drift.

An explicitly injected callback retains the internal drain/latch TEST seam.
There is no production opt-in or environment switch. A future handoff protocol
must prove successor readiness and protect daemon-owned work before enabling
any exit; the callback alone does not establish that safety contract.
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
#: Production announcements never set this latch. Only the internal injected
#: callback path exercises it; announcements are not client release instructions.
RETIRING_STATE_ATTR = "serve_retiring"


class DaemonRetiring(RuntimeError):
    """A retiring daemon refuses to admit new work. 503 for the client, not 500.

    Deliberately a subclass of ``RuntimeError`` so an unmapped caller still gets
    the routes' existing 503 ladder rather than a 500 with a traceback, and
    deliberately a TYPE rather than a string the routes match on: the client's
    refusal can be distinguished from unrelated server errors. The future
    successor protocol is not implemented; design §7 specifies announcement-only
    production behavior rather than a client handoff.

    Raised only after the internal test LATCH. Production build announcements
    never raise this refusal; no production successor handoff exists yet.
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
    """An attachment reason for the internal drain tests, or ``None``.

    This is NOT a complete work-safety predicate: scheduler-owned tasks are not
    counted. Production announcement polling must never use it to authorize exit.

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

    NOTHING IS LATCHED HERE. Production continues serving indefinitely: the
    announcement instructs no client action. Only the internal callback test
    seam subsequently consults the attachment drain and exercises the latch.

    THE WRITE IS ALLOWED TO FAIL, and the caller must let it: ``heartbeat`` can
    raise on a read-only or full config directory (the failure class the sibling
    ``heartbeat_loop`` swallows as self-healing, ``server/registry.py``), and a
    daemon that latched on a failed write would refuse every request forever
    without ever leaving. So the caller retries the announcement on its next
    check and does not latch in the meantime. Production keeps serving even
    after a successful write; the failed-write gate also protects the explicit
    callback test seam.

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
    serving the loaded build, or observing an unreadable install. Clear the
    stale new-build announcement; it never instructs readers to let go.

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

    Internal callback tests call this after announcement and attachment drain.
    Production never reaches this latch: an empty drain proves neither safe
    scheduler shutdown nor a ready successor. "Readable" is
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
    """Report a failed watcher: the daemon serves but announcements are stale.

    The lifespan otherwise awaits this task only during teardown, so an early
    failure would leave record readers unaware that build detection has stopped.
    An injected test latch is not cleared by this observer.
    """
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "serve daemon: the build watch died; new-build announcements will not "
            "refresh while this daemon continues serving",
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
    """Poll and reconcile new-build announcements while continuing to serve.

    ``boot`` is sampled before record publication so a flip immediately after
    discovery cannot be mistaken for the loaded build. The shared buildwatch
    readers enforce marker settling and withdraw/retarget stale announcements.
    Only an explicitly injected internal test callback reaches the legacy drain
    and latch below; production has neither a successor-ready protocol nor a
    complete daemon-owned-work drain and must not exit on marker drift.
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
                "announced in the record; continuing to serve",
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

        # A marker proves only that another build is on disk, not that a
        # successor is ready or that scheduler-owned work can survive shutdown.
        # Production never supplies this internal test callback. Keep verifying
        # announcements above, but do not consult the incomplete drain or latch.
        if exit_process is None:
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

        # Internal injected-callback path only. The attachment probe being
        # empty is NOT proof that daemon-owned scheduled work is safe to stop.
        latch(app)
        logger.info(
            "serve daemon: nothing is in flight; refusing new work and leaving for %s",
            announced.label(),
        )
        await _refusal_window(stop, exit_process)
        return
