"""
Health check endpoint for the Local Operator API.
"""

import asyncio
import importlib.metadata
import logging
import os
import sys
import time

from fastapi import APIRouter, Request

from local_operator.server import peer_check
from local_operator.server.models.schemas import CRUDResponse, HealthCheckResponse

router = APIRouter(tags=["Health"])

#: How long one reading of the install's identity (``version``,
#: ``install_kind``) is reused by ``/health``.
#:
#: WHY A CACHE ON A LIVENESS PROBE (serve stall, 2026-11-15). The two readings
#: are a few dozen filesystem calls -- ``direct_url.json``, dist-info metadata,
#: ``realpath`` walks -- and each call releases and re-takes the GIL. While a
#: worker thread is busy building a large snapshot, every re-take waits up to
#: ``sys.getswitchinterval()`` (5 ms), so the probe's OWN reads stretched
#: ``/health`` and held the loop: a real-path sampler with two stream clients on
#: a 33 MB session attributed ~40% of the loop's busy samples to these reads.
#: Five seconds keeps the freshness the comment below asks for (an in-place
#: ``lop-update`` shows within one poll of a 10 s poller) while a probe burst
#: pays for the reading once.
INSTALL_IDENTITY_TTL_S = 5.0
_install_identity: tuple[float, str, str] | None = None

#: What ``/health`` reports when the identity read has NEVER answered in this
#: process (first probe lands on a stuck disk). Both halves are spelled
#: ``"unknown"`` on purpose: ``install_kind`` already has that value
#: (``InstallKind.UNKNOWN``) and every Python reader of it treats it as "cannot
#: tell", and a ``version`` of ``"unknown"`` is not a parseable release, so no
#: comparison can mistake it for an older or newer build.
#:
#: What the desktop app does with it (review R1-3 on #4157): its dedicated
#: sentinel is the CAPITALISED ``"Unknown"``, so this lowercase value does not
#: take that special case; it fails the app's generic readable-version gate and
#: the update check answers "unavailable" -- the user sees "Unable to determine
#: backend version." for that poll. Nothing restarts, re-execs or refuses on it
#: (drift restart compares the serve record's boot version, never ``/health``).
UNKNOWN_IDENTITY: tuple[str, str] = ("unknown", "unknown")

#: How long a ``/health`` identity read may take before probes stop waiting for
#: it and answer with the last known (or :data:`UNKNOWN_IDENTITY`) instead.
#:
#: Measured, not guessed (``local_operator/server/loop_watchdog.py``,
#: "THRESHOLDS"): the WHOLE ``/health`` round trip is a median ~8 ms on a
#: healthy loop and 183-489 ms (n=40, never past 1 s) under the GIL convoy of a
#: busy executor thread; the identity read is a part of that. Clients give up
#: on ``/health`` at 5 s. One second sits ~2x above the worst measured healthy
#: answer and leaves the probe four seconds of room under the client's
#: deadline, so only a genuinely stuck disk (the 145 s ``importlib.metadata``
#: read of 2026-11-17, see ``buildwatch.BUILD_READ_TIMEOUT_S``) falls back.
INSTALL_IDENTITY_READ_TIMEOUT_S = 1.0

logger = logging.getLogger("local_operator.server.routes.health")

#: The ONE shared identity read, joined by every probe that misses the TTL
#: while it runs (single flight). ``None`` until the first miss; a finished
#: task stays here until the next miss replaces it.
#:
#: WHY OFF THE LOOP (QA finding Q1 on #4143). On a TTL miss the identity read
#: is ``installed_version()``, ``importlib.metadata.version`` and
#: ``install_kind()`` -- synchronous disk reads. #4143 moved the build
#: watcher's identical reads off the loop after a 145 s ``read_text`` under
#: swap stalled ``serve``; ``/health`` kept doing them inline, so the same slow
#: disk could still freeze the loop, and ``/health`` is exactly what liveness
#: pollers and the desktop app hit to decide whether the daemon is alive. A
#: liveness probe must not be what makes the daemon look dead.
#:
#: WHY A SHARED TASK, NOT ``buildwatch.OffLoopBuildRead`` (review R1-1 on
#: #4157). That reader assumes ONE caller per instance (one watching task): a
#: second call while a read is in flight is answered ``READ_UNKNOWN`` at once.
#: ``/health`` has many concurrent callers, so two probes overlapping on a
#: HEALTHY disk -- a window of milliseconds -- served the second one
#: ``"unknown"`` and logged "the build read on disk did not answer", where the
#: inline read before it simply made the second probe wait. Joining one task
#: keeps the properties that reader was chosen for and drops that regression:
#:
#: * AT MOST ONE READ THREAD. A new task (and its ``to_thread``) starts only
#:   when the previous one is done, i.e. its thread has returned. A thread
#:   cannot be cancelled, so a probe burst during a stall must not stack a
#:   thread per request into the default executor.
#: * BOUNDED WAIT, ONE DEADLINE PER READ. Every caller awaits
#:   ``wait_for(shield(task))`` only until ``INSTALL_IDENTITY_READ_TIMEOUT_S``
#:   after the read STARTED (:data:`_identity_read_started`), not a fresh
#:   second each: a probe joining a read that is already past its bound answers
#:   at once, so during a 145 s stall only the probe that started the read pays
#:   the second (as with ``OffLoopBuildRead``), and no probe waits past it.
#: * ``shield``: a caller timing out or being cancelled (client disconnect)
#:   never cancels the shared read -- otherwise ``done()`` would turn true
#:   while the thread is still stuck and the next probe would start a second.
#: * NEVER A 500. A read that raises, or does not answer in time, serves the
#:   last known identity, else :data:`UNKNOWN_IDENTITY`.
#: * A LATE ANSWER IS KEPT. The task itself writes the cache when the thread
#:   returns, even if every waiter has given up; unlike the build watcher
#:   (which re-reads every tick and must not act on a stale answer), a late
#:   identity is still the best "last known" this route can serve.
#: * ONE WARNING PER STREAK, as ``OffLoopBuildRead`` logs it: the first
#:   unanswered probe of a streak logs WARNING (console logging runs at
#:   WARNING), the rest DEBUG, and the read that answers again logs one INFO.
#:   A waiter that the read answers in time logs nothing, so concurrent probes
#:   on a healthy disk are silent.
_identity_read: "asyncio.Task[tuple[str, str]] | None" = None
#: ``loop.time()`` when :data:`_identity_read` started; the shared deadline is
#: this plus ``INSTALL_IDENTITY_READ_TIMEOUT_S``. The loop clock rather than
#: ``time.monotonic`` so the TTL's clock (which tests drive by hand) cannot move
#: the read's deadline.
_identity_read_started = 0.0
#: Inside a streak of unanswered identity reads (see "ONE WARNING PER STREAK").
_identity_read_failing = False


def _read_install_identity_from_disk() -> tuple[str, str]:
    """``(version, install_kind)`` straight from disk. Blocking: call it only
    off the loop (:func:`_refresh_install_identity`), never from a coroutine."""
    # `installed_version()` rather than raw metadata: an editable install's
    # dist-info is frozen at install time, so this reported the version the
    # tree was installed AT rather than the version it now is, and the app
    # showed that stale number in Settings > Updates (QA Q3 / UX U13).
    from local_operator.update import install_kind, installed_version

    version = installed_version() or importlib.metadata.version("local-operator")
    return version, install_kind().value


async def _refresh_install_identity(asked_at: float) -> tuple[str, str]:
    """The body of the shared read: one thread hop, then fill the cache.

    ``asked_at`` is the ``time.monotonic()`` of the probe that started the
    read, taken on the loop BEFORE it, as the inline version did: the TTL
    measures from when a probe asked, not from when a thread finished. Written
    here rather than by the waiters so a late answer still lands.
    """
    global _install_identity, _identity_read_failing
    version, kind = await asyncio.to_thread(_read_install_identity_from_disk)
    _install_identity = (asked_at, version, kind)
    if _identity_read_failing:
        _identity_read_failing = False
        logger.info("/health install identity: the read is answering again")
    return version, kind


def _consume_outcome(task: "asyncio.Task[tuple[str, str]]") -> None:
    # A read every waiter abandoned may still FAIL later; retrieve that so
    # asyncio does not log "exception was never retrieved" for it.
    if not task.cancelled():
        task.exception()


def _unanswered(why: str) -> None:
    global _identity_read_failing
    if not _identity_read_failing:
        _identity_read_failing = True
        logger.warning(
            "/health install identity: the read did not answer (%s); serving the "
            "last known identity, or 'unknown' when there is none",
            why,
        )
    else:
        logger.debug("/health install identity: read still unanswered (%s)", why)


async def _read_install_identity() -> tuple[str, str]:
    """``(version, install_kind)``, re-read at most every ``INSTALL_IDENTITY_TTL_S``.

    A TTL hit does no read and no thread hop. On a miss the probe joins the
    shared read (starting it when none is running) and waits for it up to that
    read's deadline; when it does not answer in time (or raises), the last
    known identity is served -- WITHOUT refreshing its timestamp, so the next
    probe tries again -- or :data:`UNKNOWN_IDENTITY` when there has never been
    one. See :data:`_identity_read` for the why.
    """
    global _identity_read, _identity_read_started
    now = time.monotonic()
    cached = _install_identity
    if cached is not None and now - cached[0] < INSTALL_IDENTITY_TTL_S:
        return cached[1], cached[2]
    loop = asyncio.get_running_loop()
    task = _identity_read
    # A task bound to another loop (a test harness that runs one loop per
    # test) cannot be awaited here; production serves on one loop.
    if task is None or task.done() or task.get_loop() is not loop:
        task = loop.create_task(_refresh_install_identity(now))
        task.add_done_callback(_consume_outcome)
        _identity_read = task
        _identity_read_started = loop.time()
    remaining = _identity_read_started + INSTALL_IDENTITY_READ_TIMEOUT_S - loop.time()
    fallback = (cached[1], cached[2]) if cached is not None else UNKNOWN_IDENTITY
    if not task.done() and remaining <= 0:
        _unanswered(
            f"the shared read is still running past its {INSTALL_IDENTITY_READ_TIMEOUT_S:g} s bound"
        )
        return fallback
    try:
        return await asyncio.wait_for(asyncio.shield(task), max(remaining, 0.0))
    except asyncio.TimeoutError as exc:
        # Since 3.11 ``asyncio.TimeoutError`` IS the builtin, so a read that
        # itself raised TimeoutError (a network mount's own deadline) lands
        # here too; only the task tells the two apart, and the log must name
        # the one that happened (buildwatch review n1).
        if task.done() and not task.cancelled() and task.exception() is exc:
            _unanswered(f"the read raised {type(exc).__name__}: {exc}")
        else:
            _unanswered(f"no answer within {INSTALL_IDENTITY_READ_TIMEOUT_S:g} s")
        return fallback
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 -- an unreadable identity is "unknown", never a 500
        _unanswered(f"{type(exc).__name__}: {exc}")
        return fallback


@router.get(
    "/health",
    summary="Health Check",
    description="Returns the health status of the API server.",
    response_model=CRUDResponse[HealthCheckResponse],
    responses={
        200: {
            "description": "Successful response with version and instance information",
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "message": "ok",
                        "result": {
                            "version": "0.1.0",
                            "instance_id": "kR3n…",
                            "pid": 4242,
                            "prefix": "/Users/you/.local/share/uv/tools/local-operator",
                            "install_kind": "uv-tool",
                        },
                    }
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"detail": "Internal server error"}}},
        },
    },
)
async def health_check(request: Request) -> CRUDResponse[HealthCheckResponse]:
    """
    Health check endpoint.

    Retrieves the current version of the Local Operator, plus enough about WHO
    is answering for a caller to tell one daemon from another.

    A 200 here is not identification, and treating it as identification is the
    bug this route's extra fields exist to end: three daemons were live on this
    machine at once (``:1111``, ``:7341``, ``:8080``, three builds, one of them
    eleven releases behind), every one of them answered ``/health`` with 200,
    and the first answer was taken as "the backend". ``instance_id`` is minted
    per process at startup and also published in the serve rendezvous record
    (``local_operator.server.registry``), so a caller that dialled from a
    record can confirm the process that answered is the process it found;
    ``prefix``/``install_kind`` say which INSTALL is serving, which is not the
    same as the one on ``PATH``.

    Args:
        request: The incoming request, for the app state that holds
            ``instance_id`` (minted by ``lifespan``).

    Returns:
        CRUDResponse[HealthCheckResponse]: The application version and this
        process's identity.

    Raises:
        HTTPException: If there's an error retrieving version information.
    """
    version, kind = await _read_install_identity()
    from local_operator.server.reload import rolling_contract

    result = HealthCheckResponse(
        version=version,
        # `getattr` with a default rather than a direct read: a test harness
        # that builds a client over `app` WITHOUT running the lifespan (which
        # is most of the server suite) has no app state to read, and a liveness
        # probe must not be the route that breaks for it. Empty means
        # "unidentified", which is the truthful answer there — the record, the
        # other half of the handshake, does not exist in that harness either.
        instance_id=str(getattr(request.app.state, "instance_id", "")),
        # Read at most every INSTALL_IDENTITY_TTL_S rather than cached at
        # startup, deliberately: an
        # install's dist-info and `sys.prefix` can change UNDER a running
        # daemon (`lop-update` replaces the tree in place), and this route is
        # polled every 10 s by one client, not on a hot path. `pid` is this
        # process by definition: the answer and the thing answering are the
        # same process, which is the whole point of reporting it.
        pid=os.getpid(),
        prefix=sys.prefix,
        install_kind=kind,
        # Fresh serving-process capability, not copied from discovery metadata.
        rolling_reload_contract=rolling_contract(request.app),
        peer_uid_check=peer_check.current_status(),
    )

    return CRUDResponse(status=200, message="ok", result=result)
