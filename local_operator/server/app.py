"""
FastAPI server implementation for Local Operator API.

Provides REST endpoints for interacting with the Local Operator agent
through HTTP requests instead of CLI.
"""

import asyncio
import secrets
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

from local_operator import buildwatch
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import get_env_config
from local_operator.helpers import setup_cross_platform_environment
from local_operator.jobs import JobManager
from local_operator.logger import configure_console_logging, get_logger
from local_operator.scheduler_service import SchedulerService
from local_operator.server import registry as serve_registry
from local_operator.server import retire as serve_retire
from local_operator.server.desktop import desktop_posture, require_desktop
from local_operator.server.routes import (
    agents,
    auth,
    capabilities,
    chat,
    config,
    credentials,
    desktop_catalogues,
    desktop_claim,
    desktop_lifecycle,
    desktop_profiles,
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

# Annotating the lifespan's record publisher (`None` on a boot that was not
# announced) needs the shared publisher's type. Zero runtime cost where it
# matters: `server/registry.py` imports this module anyway, so it is already in
# `sys.modules` by the time anything here runs.
from local_operator.session.runtime import registry as session_registry
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

    # The tokenizer's first-use cost overlaps the rest of startup. This process
    # runs sessions of its own — the stateless ``/v1/chat`` path and the
    # scheduler's agent runs — and each of them would otherwise pay ~120 ms of
    # BPE-table construction inside its first request, on the event loop this
    # daemon is also serving HTTP from.
    #
    # WARMED IN THE BACKGROUND, AND WRAPPED INCLUDING THE IMPORT. Priming the
    # cache takes ~1 s of compile; awaiting it here would move that into daemon
    # STARTUP, which is the thing an attach waits on. And a raise inside a
    # FastAPI ``lifespan`` fails startup, so the import is inside the guard
    # too, not just the call.
    try:
        from local_operator.compaction.tokens import warm_tokenizer_in_background

        warm_tokenizer_in_background()
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("tokenizer prewarm unavailable at startup", exc_info=True)

    # THE BYTECODE CACHE WARM, and the reason it belongs to the daemon rather
    # than to each runtime child. Every child this daemon spawns inherits its
    # environment, so under an interpreter that refuses bytecode writes (the
    # desktop app's spawn environment — see ``local_operator.bytecode``) each
    # child recompiles the whole import graph from source, measured at 749 ms
    # inside its first turn. Only the WRITE is refused; the READ is not.
    #
    # A CHILD SPAWNED IN THE NEXT SECOND MAY STILL COMPILE. This is
    # fire-and-forget: the population takes ~1 s, and nothing here waits for
    # it, so the guarantee is "every child that follows the population" rather
    # than "every child that follows this line". Waiting would trade an attach
    # for a daemon start, which is the wrong way round — the cache is also
    # persistent, so second and later sessions in this install's lifetime are
    # the ones that actually collect.
    try:
        from local_operator.bytecode import warm_bytecode_cache_in_background

        warm_bytecode_cache_in_background()
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("bytecode prewarm unavailable at startup", exc_info=True)

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

    # Publish this daemon's rendezvous record: the file that makes a running
    # `lop serve` findable from another process, and makes THIS install
    # identifiable (`/health` alone answered with a 200 from three different
    # builds at once on this host, which is the failure this record exists to
    # end).
    #
    # AFTER the scheduler starts, not before: a record says "a daemon you can
    # talk to", and publishing one while the scheduler is still coming up
    # would hand a reader exactly the same premature 200 this is meant to
    # remove.
    #
    # A failure here FAILS STARTUP rather than degrading, deliberately: the
    # record is this process's answer to "which install is serving", and a
    # daemon nobody can find is indistinguishable from no daemon at all — the
    # confusion this change exists to end. Nothing is being traded away for
    # that, because an unwritable config root is already fatal a few lines up:
    # the credential store, the config manager and the agent registry all write
    # into the same root.
    #
    # `instance_id` is minted here and held on app state because two readers
    # must agree on it: `/health` reports it, and a discoverer compares what
    # `/health` said against the record it dialled from. Process-scoped and
    # random, never persisted — it identifies THIS process, not the install.
    app.state.instance_id = secrets.token_urlsafe(32)
    # An announcer is what makes this process a DAEMON rather than merely an app
    # that happens to answer HTTP: `serve_command` announces the address it
    # bound, in-process for the daemon it serves and through the environment
    # only for a `--reload` child (see `server/registry.py`).
    #
    # No announcement means this app was booted by something else — a bare
    # `uvicorn local_operator.server.app:app`, a wrapper, a nested shell that
    # inherited a parent's announcement. Then NO record is published, and the
    # reason is the same one the record exists for: a record claims a daemon is
    # listening at an address, and this process has no truthful address to put
    # in one. Publishing a placeholder (an empty host, port 0) would recreate
    # exactly the artefact this module removes — a reader dials nothing, and
    # cannot tell that record from a live daemon's until the heartbeat ages out.
    announced = serve_registry.advertised_address(app)
    serve_publisher: session_registry.RecordPublisher | None = None
    serve_heartbeat: asyncio.Task[None] | None = None
    # The retirement poll and the Event that ends it: both None on a boot that
    # published no record, because a daemon with no record has no announcement
    # channel and therefore nothing to retire into (see the branch below).
    retire_task: asyncio.Task[None] | None = None
    retire_stop: asyncio.Event | None = None
    if announced is not None:
        # THE BUILD WATCH'S BASELINE IS SAMPLED HERE, BEFORE THE RECORD EXISTS —
        # and the ordering is load-bearing rather than incidental. The baseline
        # is "the build this process loaded", and the only reader that acts on
        # the announcement finds the daemon by waiting for the record to appear:
        # a baseline read AFTER the publish would adopt a marker that landed in
        # between as the build we loaded, so that update would be invisible to
        # this process for the rest of its life, with no log line at all. QA
        # round 1 reproduced exactly that (Q2: 3 of 7 flips issued immediately
        # after the record appeared were swallowed) using this repo's own
        # evidence driver. `LOP_BUILD_PREFIX` is the e2e-only override the
        # reader honours; production reads `sys.prefix`.
        boot_build = buildwatch.boot_build()
        serve_record = serve_registry.build_record(
            instance_id=app.state.instance_id, announced=announced
        )
        # The config root resolved above, passed explicitly: the publisher pins
        # the directory it publishes into for its whole life, so a heartbeat can
        # never land in a different root than the one this process was started
        # with.
        serve_publisher = serve_registry.publisher(serve_record, root=config_dir)
        app.state.serve_record = serve_record
        # The publisher goes on state TOO, not just its heartbeat Task: the
        # claim route has to refresh the record's ``desktop`` field when it
        # accepts a claim, and re-deriving a publisher there would mean
        # rebuilding the record — which re-mints ``claim_key`` and silently
        # invalidates the app's proof of ownership. Publishing through the
        # object that owns this record is the only cheap, non-destructive way
        # to rewrite it (see ``accept_claim`` and its caller).
        app.state.serve_publisher = serve_publisher
        # One task on the loop, cancelled below. The record is rewritten whole
        # by the shared `publish`, so its atomicity and 0600/0700 permissions
        # are the session registry's, not re-implemented here.
        serve_heartbeat = asyncio.create_task(serve_registry.heartbeat_loop(serve_publisher))
        app.state.serve_heartbeat = serve_heartbeat

        # Announce changed builds, but deliberately pass NO exit callback.
        # This process owns legacy scheduled/async work; shutdown below cancels
        # SchedulerService._run_tasks. A marker proves neither a safe drain nor
        # a ready successor, so production must keep serving, never latch/exit.
        # Clients must not release SSE/watch leases merely on these fields.
        # The poll lives beside its publisher because the record is its channel.
        #
        # NOT ON A `--reload` CHILD (`serve_registry.is_reload_child`): that
        # child's port belongs to uvicorn's supervisor, so a child that retired
        # would remove its record and leave the parent accepting on a socket
        # with nothing behind it — a daemon a reader cannot see and cannot
        # explain (QA round 1, Q3). A dev-mode supervisor is not a production
        # daemon and has no successor to hand the socket to.
        if not serve_registry.is_reload_child(app):
            retire_stop = asyncio.Event()
            retire_task = asyncio.create_task(
                serve_retire.retirement_poll(
                    app, serve_publisher, stop=retire_stop, boot=boot_build
                )
            )
            # Observed, not merely held: the task is cancelled at teardown and
            # nothing else ever awaits it, so a task that DIED would be silent —
            # a daemon still serving with stale build announcements and no
            # explanation of why its record stopped tracking the install.
            retire_task.add_done_callback(serve_retire.observe_poll)
            app.state.serve_retire = retire_task
            app.state.serve_retire_stop = retire_stop

    yield
    try:
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
    finally:
        # The record outlives nothing: it is removed here, LAST, and under
        # `finally:` so that no exit path can leave a file behind claiming a
        # live daemon at a port nothing listens on. The heartbeat keeps running
        # through the teardown above on purpose — the process is still alive
        # and still answering, so `live` is the truthful classification until
        # the listener actually closes; a wedged record is what a reader would
        # see if the teardown hung, which is also true.
        #
        # The removal itself is best-effort by contract (the shared
        # `unpublish` swallows a missing file): a SIGKILLed daemon leaves its
        # record for the next `scan` to reap, and an exit path must never raise
        # over a file that a reader already reaped for us.
        # Both only exist when a record was published; a boot that was not
        # announced has neither a task to cancel nor a file to remove.
        #
        # The retirement poll goes FIRST: a teardown is already an exit, and a
        # poll that announced a retirement mid-teardown would rewrite the record
        # on the way out with a handover that the shutdown then makes. `stop`
        # ends a poll parked in its notice wait; the cancel covers one parked in
        # its check sleep, where the event is not what it awaits.
        if retire_stop is not None:
            retire_stop.set()
        if retire_task is not None:
            retire_task.cancel()
            await asyncio.gather(retire_task, return_exceptions=True)
        # Clear the one-way latch, unconditionally: `app` is a module-level
        # singleton, and a lifecycle that left it set would make the NEXT boot in
        # this process refuse every desktop session (the tests' lifespan reuse is
        # the case that matters, and production's single boot is unaffected).
        app.state.serve_retiring = False
        app.state.serve_retire = None
        app.state.serve_retire_stop = None
        if serve_heartbeat is not None and serve_publisher is not None:
            serve_heartbeat.cancel()
            await asyncio.gather(serve_heartbeat, return_exceptions=True)
            serve_publisher.close()
        app.state.serve_record = None
        app.state.serve_publisher = None
        app.state.serve_heartbeat = None


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
#:
#: ``/v1/agent-name-availability`` is here because a route can be flat, read-only
#: and carry no credential of its own and still be the wrong thing to leave open:
#: it is EGRESS this machine performs on an unauthenticated caller's behalf (the
#: hub is asked whether a name is free), on an app whose ``CORSMiddleware``
#: allows every origin, so a page the operator merely visited could drive it. The
#: prefix families cannot see it — ``"/v1/agent-name-availability".startswith(
#: "/v1/agents")`` is False, the hyphen is not a segment boundary — which is why
#: ``test_managed_gate_covers_every_control_surface_route`` walks the ROUTERS
#: rather than the prefixes and fails when a route like this appears without an
#: entry here.
_LEGACY_CONTROL_PATHS = frozenset(
    {
        "/v1/agent-name-availability",
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

    Enforced only while the desktop plane is governed — by the app's env
    capability or by an accepted claim (``desktop_posture``) — so a standalone
    legacy server and every CLI/script client stay wire-compatible until
    something actually attaches, and a claim TIGHTENS the surface it finds
    rather than inheriting the standalone posture. See
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
    # Managed posture — the desktop app's env capability OR an accepted claim
    # (`server/desktop.py`). A claim that turned the routers on but left this
    # boundary standing on the environment would leave the legacy mutation
    # bypass open on exactly the daemon the app had just attached to, which is
    # the half of the bug the design's option A exists to close.
    if desktop_posture().enabled and legacy_control:
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
        desktop_posture().enabled and _legacy_desktop_gated(request.url.path, request.method)
    ):
        return JSONResponse(status_code=422, content={"detail": "The request has invalid fields."})
    return await request_validation_exception_handler(request, error)


app.include_router(capabilities.router)
# The way IN to a daemon the desktop app did not start. It carries no
# `require_desktop` dependency on purpose (that is the deadlock the design
# names): the gate is the record's 0600 key, checked inside the route.
app.include_router(desktop_claim.router)
app.include_router(auth.router)
app.include_router(settings.router)
app.include_router(desktop_sessions.router)
app.include_router(desktop_catalogues.router)
app.include_router(desktop_profiles.router)
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
    """Stop echoing arbitrary origins once the admitted allowlist is non-empty.

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

    Scoped to a NON-EMPTY admitted set -- ``desktop_posture().origins``, the
    environment's list UNIONED with whatever an accepted claim installed. With
    nothing admitted the response is returned untouched and the historical
    wildcard echo stands, which the SHIPPED app depends on: its renderer is
    loaded with ``mainWindow.loadFile(...)``, so it runs at ``file://`` and
    every request it makes carries the opaque origin ``"null"`` (a value this
    plane never admits to an allowlist). The app sets the token but never an
    origins list, so scoping the suppression to ``posture.enabled`` instead --
    what the first version of this middleware did -- stripped the grant from
    ``/health`` itself, which that renderer reads DIRECTLY as its "server
    offline" signal, and the app reported a healthy daemon as down. That false
    negative is the failure this whole program exists to remove.

    The residual is deliberate: an allowlist-less daemon -- the app-managed
    default, and a native claim that declared nothing -- still echoes, exactly
    as every daemon in this shape did before the claim work. It is not the
    tightening that protects a plane in that state; the CONTROL half is
    (``require_desktop`` on the credential/configuration families, and the
    legacy boundary), and it is untouched here. Closing the echo for an
    allowlist-less daemon needs the app to stop reading ``/health`` from the
    renderer, which the UI PR does by probing health/version from main.
    """
    response = await call_next(request)
    # The allowlist in force: the environment's, UNIONED with the origins the
    # accepted claim installed (see ``desktop_posture``). So a claim both keeps
    # this middleware from echoing arbitrary origins back to a page AND admits
    # the app that claimed the plane -- the two halves have to come from one
    # value or the app would claim its way into a CORS wall of its own making.
    # An app that declared a renderer origin in its claim body gets exactly
    # that, and a plane with nothing admitted keeps the echo its renderer reads
    # ``/health`` with.
    allowed = desktop_posture().origins
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
