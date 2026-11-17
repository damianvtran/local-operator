"""
Health check endpoint for the Local Operator API.
"""

import importlib.metadata
import os
import sys
import time

from fastapi import APIRouter, Request

from local_operator.buildwatch import READ_UNKNOWN, OffLoopBuildRead
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
#: (``InstallKind.UNKNOWN``) and every reader of it treats it as "cannot tell",
#: and a ``version`` of ``"unknown"`` is not a parseable release, so an update
#: banner comparing it cannot mistake it for an older or newer build.
UNKNOWN_IDENTITY: tuple[str, str] = ("unknown", "unknown")

#: How long ``/health`` waits for one off-loop identity read before answering
#: with the last known (or :data:`UNKNOWN_IDENTITY`) instead.
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

#: The ONE off-loop reader for this route (QA finding Q1 on #4143).
#:
#: WHY OFF THE LOOP. On a TTL miss the identity read is ``installed_version()``,
#: ``importlib.metadata.version`` and ``install_kind()`` -- synchronous disk
#: reads. #4143 moved the build watcher's identical reads off the loop after a
#: 145 s ``read_text`` under swap stalled ``serve``; ``/health`` kept doing them
#: inline, so the same slow disk could still freeze the loop, and ``/health`` is
#: exactly what liveness pollers and the desktop app hit to decide whether the
#: daemon is alive. A liveness probe must not be what makes the daemon look dead.
#:
#: WHY ``buildwatch.OffLoopBuildRead`` RATHER THAN A BARE ``to_thread``. It
#: already carries the three properties this needs: a bound on the wait, at
#: most ONE read in flight (a thread cannot be cancelled, so a probe burst
#: during a stall must not stack a thread per request into the default
#: executor), and "never raises" -- a failed or late read is ``READ_UNKNOWN``,
#: never a 500. Module-level, one per route, because the in-flight read and the
#: failing streak are per caller (see its docstring).
_identity_reader = OffLoopBuildRead(
    "/health install identity", timeout_s=INSTALL_IDENTITY_READ_TIMEOUT_S
)


def _read_install_identity_from_disk() -> tuple[str, str]:
    """``(version, install_kind)`` straight from disk. Blocking: call it only
    through :data:`_identity_reader`, never from a coroutine directly."""
    # `installed_version()` rather than raw metadata: an editable install's
    # dist-info is frozen at install time, so this reported the version the
    # tree was installed AT rather than the version it now is, and the app
    # showed that stale number in Settings > Updates (QA Q3 / UX U13).
    from local_operator.update import install_kind, installed_version

    version = installed_version() or importlib.metadata.version("local-operator")
    return version, install_kind().value


async def _read_install_identity() -> tuple[str, str]:
    """``(version, install_kind)``, re-read at most every ``INSTALL_IDENTITY_TTL_S``.

    A TTL hit does no read and no thread hop. On a miss the read runs off the
    loop; when it does not answer in time (or raises), the last known identity
    is served -- WITHOUT refreshing its timestamp, so the next probe tries again
    (the reader itself refuses a second read while the stuck one runs) -- or
    :data:`UNKNOWN_IDENTITY` when there has never been one.
    """
    global _install_identity
    # The clock is taken on the loop BEFORE the read, as the inline version
    # did: the TTL measures from when a probe asked, not when a thread finished.
    now = time.monotonic()
    cached = _install_identity
    if cached is not None and now - cached[0] < INSTALL_IDENTITY_TTL_S:
        return cached[1], cached[2]
    read = await _identity_reader(_read_install_identity_from_disk)
    if read is READ_UNKNOWN:
        return (cached[1], cached[2]) if cached is not None else UNKNOWN_IDENTITY
    version, kind = read
    _install_identity = (now, version, kind)
    return version, kind


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
    )

    return CRUDResponse(status=200, message="ok", result=result)
