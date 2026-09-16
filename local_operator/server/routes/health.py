"""
Health check endpoint for the Local Operator API.
"""

import importlib.metadata
import os
import sys

from fastapi import APIRouter, Request

from local_operator.server.models.schemas import CRUDResponse, HealthCheckResponse

router = APIRouter(tags=["Health"])


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
    # `installed_version()` rather than raw metadata: an editable install's
    # dist-info is frozen at install time, so this reported the version the
    # tree was installed AT rather than the version it now is, and the app
    # showed that stale number in Settings > Updates (QA Q3 / UX U13).
    from local_operator.update import install_kind, installed_version

    version = installed_version() or importlib.metadata.version("local-operator")
    result = HealthCheckResponse(
        version=version,
        # `getattr` with a default rather than a direct read: a test harness
        # that builds a client over `app` WITHOUT running the lifespan (which
        # is most of the server suite) has no app state to read, and a liveness
        # probe must not be the route that breaks for it. Empty means
        # "unidentified", which is the truthful answer there — the record, the
        # other half of the handshake, does not exist in that harness either.
        instance_id=str(getattr(request.app.state, "instance_id", "")),
        # Read per request rather than cached at startup, deliberately: an
        # install's dist-info and `sys.prefix` can change UNDER a running
        # daemon (`lop-update` replaces the tree in place), and this route is
        # polled every 10 s by one client, not on a hot path. `pid` is this
        # process by definition: the answer and the thing answering are the
        # same process, which is the whole point of reporting it.
        pid=os.getpid(),
        prefix=sys.prefix,
        install_kind=install_kind().value,
    )

    return CRUDResponse(status=200, message="ok", result=result)
