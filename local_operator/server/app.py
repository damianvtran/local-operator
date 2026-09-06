"""
FastAPI server implementation for Local Operator API.

Provides REST endpoints for interacting with the Local Operator agent
through HTTP requests instead of CLI.
"""

import os
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from importlib.metadata import version
from re import Pattern
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.routing import compile_path

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import get_env_config
from local_operator.helpers import setup_cross_platform_environment
from local_operator.jobs import JobManager
from local_operator.logger import configure_console_logging, get_logger
from local_operator.scheduler_service import SchedulerService
from local_operator.server.desktop import require_desktop
from local_operator.server.routes import (
    agents,
    auth,
    capabilities,
    chat,
    config,
    credentials,
    desktop_catalogues,
    desktop_lifecycle,
    desktop_radient,
    desktop_sessions,
    health,
    jobs,
    models,
    schedules,
    settings,
    speech,
    sse,
    static,
    transcription,
    websockets,
)
from local_operator.server.utils.event_broker import EventBroker
from local_operator.server.utils.websocket_manager import WebSocketManager
from local_operator.types import OperatorType

# NO logging configuration at import. `configure_console_logging` REPLACES the
# root logger's handlers, so calling it here made merely importing this module
# — which `generate_openapi`, the test suite's collection phase and any
# tooling that wants `app.openapi()` all do — reconfigure logging for the whole
# process. That is the same import side effect `helpers.py` was stripped of,
# and it leaked a stderr handler across the test session. The server's entry
# point is `lifespan`, which uvicorn runs on startup and nothing else runs at
# all, so the call lives there.

logger = get_logger("local_operator.server")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up application state.

    This function is called when the application starts up and shuts down.
    It initializes the credential manager, config manager, and agent registry.

    Args:
        app: The FastAPI application instance
    """
    # Console logging for the server process, at the same level (LOG_LEVEL,
    # default WARNING) and format the old import-time call used. uvicorn has
    # already configured its own named loggers by the time startup runs, and
    # this only touches the root logger, so the two do not fight.
    configure_console_logging()

    # Initialize on startup by setting up the credential and config managers
    from local_operator.paths import config_dir as resolve_config_dir

    config_dir = resolve_config_dir()
    # Honour LOCAL_OPERATOR_HOME and create it at the point of use, matching the
    # CLI session path. The literal ``~/local-operator-home`` here ignored the
    # override, so a relocated home still had a stray workspace created in the
    # real home directory.
    from local_operator.paths import ensure_agent_home_dir

    ensure_agent_home_dir()

    # Set up the subprocess environment for accessing shell commands
    setup_cross_platform_environment()

    app.state.credential_manager = CredentialManager(config_dir=config_dir)
    app.state.config_manager = ConfigManager(config_dir=config_dir)
    # Initialize AgentRegistry with a refresh interval of 3 seconds to ensure
    # changes made by child processes are quickly reflected in the parent process
    app.state.agent_registry = AgentRegistry(config_dir=config_dir, refresh_interval=3.0)
    app.state.job_manager = JobManager()
    app.state.websocket_manager = WebSocketManager()
    # The SSE fan-out. One instance per process, mirroring the websocket
    # manager: both are subscribers to the same pump, which is what keeps the
    # legacy transport byte-identical while SSE carries the richer taxonomy.
    app.state.event_broker = EventBroker()
    app.state.env_config = get_env_config()

    app.state.scheduler_service = SchedulerService(
        agent_registry=app.state.agent_registry,
        config_manager=app.state.config_manager,
        credential_manager=app.state.credential_manager,
        env_config=app.state.env_config,
        operator_type=OperatorType.SERVER,
        verbosity_level=VerbosityLevel.QUIET,
        job_manager=app.state.job_manager,
        websocket_manager=app.state.websocket_manager,
        event_broker=app.state.event_broker,
    )

    await app.state.scheduler_service.start()

    yield
    # Clean up on shutdown
    desktop_auth = getattr(app.state, "desktop_auth", None)
    if desktop_auth is not None:
        await desktop_auth.close()
        app.state.desktop_auth = None
    desktop_sessions_host = getattr(app.state, "desktop_sessions", None)
    if desktop_sessions_host is not None:
        await desktop_sessions_host.close()
        app.state.desktop_sessions = None
    # Off-record panels belong to this HTTP lifetime, not the durable session.
    app.state.desktop_asides = None
    app.state.desktop_receipts = None
    await app.state.scheduler_service.shutdown()

    app.state.credential_manager = None
    app.state.config_manager = None
    app.state.agent_registry = None
    app.state.job_manager = None
    app.state.websocket_manager = None
    app.state.event_broker.close()
    app.state.event_broker = None
    app.state.env_config = None
    app.state.scheduler_service = None


app = FastAPI(
    title="Local Operator API",
    description="REST API interface for Local Operator agent",
    version=version("local-operator"),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "Health", "description": "Health check endpoints"},
        {"name": "Chat", "description": "Chat generation endpoints"},
        {"name": "Agents", "description": "Agent management endpoints"},
        {"name": "Jobs", "description": "Job management endpoints"},
        {"name": "Configuration", "description": "Configuration management endpoints"},
        {"name": "Credentials", "description": "Credential management endpoints"},
        {"name": "Models", "description": "Model management endpoints"},
        {"name": "Schedules", "description": "Schedule management endpoints"},  # Added
        {"name": "Transcription", "description": "Audio transcription endpoints"},  # Added
        {"name": "Static", "description": "Static file hosting endpoints"},
    ],
)


#: Legacy CONTROL routes outside the agent/job families, gated on every method.
#:
#: The agent and job families are deliberately absent: they are enumerated from
#: the router by :func:`_legacy_gate_matchers` so a newly added route under them
#: is gated BY DEFAULT. This set is only the flat singleton paths, which have no
#: id segment and no family to walk.
_LEGACY_CONTROL_PATHS = frozenset(
    {
        "/v1/config",
        "/v1/config/system-prompt",
        "/v1/credentials",
        "/v1/models",
        "/v1/tools/speech",
        "/v1/transcriptions",
    }
)

#: Route families gated wholesale in managed mode. Everything the router
#: publishes under these prefixes reads or mutates the same tenant's data:
#: agent inventory and names, working-directory paths (the filesystem layout of
#: the user's machine), conversation content, system prompts, execution
#: variables, exported agent ZIPs, and job history.
#:
#: ``/v1/schedules`` is here because a schedule is not a record, it is DELAYED
#: EXECUTION. Both writes on the surface -- the agent-scoped ``POST`` and
#: ``PATCH /v1/schedules/{id}`` -- hand their ``prompt`` to
#: ``SchedulerService.add_or_update_job``, which registers ``_trigger_agent_task``
#: on APScheduler. Whatever text reaches them is later run BY THE USER'S OWN
#: AGENT, with that agent's tools and credentials and nobody watching. An
#: unauthenticated cross-origin caller reaching either one is arbitrary code
#: execution on a delay, not a defaced field, so this family cannot sit at a
#: weaker posture than the agent inventory it schedules work against.
_LEGACY_GATED_PREFIXES = ("/v1/agents", "/v1/jobs", "/v1/schedules")

#: The ONLY routes under :data:`_LEGACY_GATED_PREFIXES` left open in managed
#: mode, each with the reason it is safe. Keyed ``"METHOD /path/template"`` using
#: the router's own template, so a stale or misspelled key cannot silently widen
#: the boundary: ``test_managed_gate_covers_every_legacy_route`` fails on any
#: entry matching no live route.
#:
#: Deny-by-default is the whole point. Hand-maintained string matching missed
#: three routes in review round 1 and five more in round 2 -- including an
#: unauthenticated cross-origin ``PATCH`` that renamed an agent and persisted it
#: -- because every new route had to be REMEMBERED into the gate. A route is now
#: gated the moment it exists, and an omission fails a test instead of shipping
#: a bypass.
#:
#: Currently EMPTY, and the emptiness is load-bearing: no legacy route today has
#: a justification for staying open.
#:
#: It previously held the two agent-scoped schedules routes, excused as "paired
#: with the ungated /v1/schedules/{schedule_id}; gate the surface as one
#: change". That reasoning does not survive contact with the routing table.
#: `routes/schedules.py` has exactly ONE `@router.post`, and it is the
#: agent-scoped one -- so `POST` was never half of a symmetric pair, it was the
#: only create on the entire surface, and the symmetry argument was excusing
#: the write it should have been protecting. Review round 3 duly reproduced an
#: unauthenticated cross-origin POST persisting an active, auto-executing
#: schedule. The "one change" the note deferred to is the change that removed
#: this list: the whole family is gated by prefix above.
#:
#: Before adding a key here, check the route against the router rather than
#: against the shape of the URL: a path that LOOKS like the counterpart of
#: something ungated may be the only way in.
_LEGACY_GATE_EXCEPTIONS: dict[str, str] = {}


def _iter_routes(routes: Iterable[Any]) -> Iterator[tuple[str, frozenset[str]]]:
    """Every ``(path template, methods)`` the app publishes, nested routers included.

    FastAPI wraps each ``include_router`` in an ``_IncludedRouter`` whose own
    ``path`` is ``None``, with the real routes hanging off ``original_router``.
    A flat scan of ``app.routes`` therefore sees the four docs endpoints and
    nothing else -- which would make the gate coverage test vacuously green.
    """
    for route in routes:
        nested = getattr(route, "routes", None)
        if nested is None:
            original = getattr(route, "original_router", None)
            nested = getattr(original, "routes", None) if original is not None else None
        if nested:
            yield from _iter_routes(nested)
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path and methods:
            yield path, frozenset(methods)


def legacy_gate_routes() -> list[tuple[str, frozenset[str]]]:
    """The router's own view of the gated families, shared by the gate and its test."""
    return sorted(
        (path, methods)
        for path, methods in set(_iter_routes(app.routes))
        if path.startswith(_LEGACY_GATED_PREFIXES)
    )


@lru_cache(maxsize=1)
def _legacy_gate_matchers() -> tuple[tuple[Pattern[str], frozenset[str]], ...]:
    """Compiled ``(path regex, gated methods)`` pairs, built FROM THE ROUTER.

    Starlette's own :func:`compile_path` produces the same regex the router
    matches with, so the gate cannot drift from the routing table the way a
    hand-written ``len(parts) == 4`` check did.

    Cached because it would otherwise recompile on every request; the routing
    table is fixed once the app is constructed.
    """
    matchers: list[tuple[Pattern[str], frozenset[str]]] = []
    for path, methods in legacy_gate_routes():
        gated = {method for method in methods if f"{method} {path}" not in _LEGACY_GATE_EXCEPTIONS}
        if not gated:
            continue
        # Starlette answers HEAD from a GET route, so the gate must cover it or
        # a HEAD reads a gated response's headers without a bearer.
        if "GET" in gated:
            gated.add("HEAD")
        regex, _format, _converters = compile_path(path)
        matchers.append((regex, frozenset(gated)))
    return tuple(matchers)


def _legacy_desktop_gated(path: str, method: str) -> bool:
    """Whether this request sits behind the managed-mode desktop boundary.

    Enforced only when a desktop token is configured, so a standalone legacy
    server and every CLI/script client stay wire-compatible -- see
    :func:`managed_desktop_boundary`.
    """
    if path in _LEGACY_CONTROL_PATHS:
        return True
    if not path.startswith(_LEGACY_GATED_PREFIXES):
        return False
    return any(
        method in methods and regex.match(path) for regex, methods in _legacy_gate_matchers()
    )


@app.middleware("http")
async def managed_desktop_boundary(request: Request, call_next):
    path = request.url.path
    sensitive = path.startswith(("/v1/auth/", "/v1/settings", "/v1/mcp", "/v1/desktop/"))
    legacy_control = _legacy_desktop_gated(path, request.method)
    # Explicit desktop-token mode must close the old mutation bypass too. A
    # standalone legacy server remains wire-compatible unless opted into this
    # mode; Electron moves these calls through the same main-process proxy.
    if os.environ.get("LOCAL_OPERATOR_DESKTOP_TOKEN") and legacy_control:
        try:
            require_desktop(request)
        except HTTPException as error:
            return JSONResponse(status_code=error.status_code, content={"detail": error.detail})
    response = await call_next(request)
    if sensitive or legacy_control:
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RequestValidationError)
async def desktop_validation_error(request: Request, error: RequestValidationError):
    # Pydantic includes the rejected INPUT in its default 422 response. SecretStr
    # protects model dumps, not failures before a model was constructed.
    if request.url.path.startswith(("/v1/auth/", "/v1/settings", "/v1/mcp", "/v1/desktop/")) or (
        os.environ.get("LOCAL_OPERATOR_DESKTOP_TOKEN")
        and _legacy_desktop_gated(request.url.path, request.method)
    ):
        return JSONResponse(status_code=422, content={"detail": "The request has invalid fields."})
    return await request_validation_exception_handler(request, error)


app.include_router(capabilities.router)
app.include_router(auth.router)
app.include_router(settings.router)
app.include_router(desktop_sessions.router)
app.include_router(desktop_catalogues.router)
app.include_router(desktop_lifecycle.router)
app.include_router(desktop_radient.router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.middleware("http")
async def desktop_origin_cors(request: Request, call_next):
    """Stop echoing arbitrary origins once a desktop allowlist is configured.

    ``CORSMiddleware`` is registered with ``allow_origins=["*"]`` and
    ``allow_credentials=True``, which makes Starlette ECHO the requesting
    origin into ``Access-Control-Allow-Origin``. That turns every legacy route
    into something a drive-by page can read with ``fetch()`` while the desktop
    app holds the backend open on a predictable loopback port -- the browser
    vector behind QA's Q2, distinct from the missing bearer.

    Registered AFTER the CORS middleware ON PURPOSE: Starlette runs the most
    recently added middleware OUTERMOST, so this is the only position from
    which the header CORS just wrote is observable. A middleware declared
    above it sees no ``Access-Control-Allow-Origin`` at all (verified, not
    assumed) and would silently strip nothing.

    Scoped to the managed desktop posture. With no allowlist configured, a
    standalone server keeps its historical wildcard CORS, so CLI clients,
    scripts and existing embedders are untouched.
    """
    response = await call_next(request)
    allowed = {
        item.strip()
        for item in os.environ.get("LOCAL_OPERATOR_DESKTOP_ORIGINS", "").split(",")
        if item.strip() and item.strip() != "null"
    }
    if not allowed:
        return response
    origin = request.headers.get("origin")
    if origin is not None and origin not in allowed:
        # Removed rather than set to a placeholder: absent means "no CORS grant",
        # which is what a browser must conclude. Credentials must go with it, or
        # the pair reads as a grant to the wildcard.
        #
        # `del`, not `.pop()`: Starlette's MutableHeaders implements neither
        # `pop` nor dict's default-argument protocol, and deleting an absent
        # key is already a no-op there.
        del response.headers["access-control-allow-origin"]
        del response.headers["access-control-allow-credentials"]
    return response


# Include routers from the routes modules

# /health
app.include_router(health.router)

# /v1/chat
app.include_router(
    chat.router,
)

# /v1/agents
app.include_router(
    agents.router,
)

# /v1/jobs
app.include_router(
    jobs.router,
)

# /v1/config
app.include_router(
    config.router,
)

# /v1/credentials
app.include_router(
    credentials.router,
)

# /v1/models
app.include_router(
    models.router,
)

# /v1/static
app.include_router(
    static.router,
)

# /v1/ws
app.include_router(
    websockets.router,
)

# /v1/sse - the preferred streaming transport; /v1/ws above is the fallback
# kept for older clients.
app.include_router(
    sse.router,
)

# /v1/schedules
app.include_router(
    schedules.router,
)

# /v1/transcriptions
app.include_router(
    transcription.router,
)

# /v1/speech
app.include_router(
    speech.router,
)
