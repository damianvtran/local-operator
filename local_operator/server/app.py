"""
FastAPI server implementation for Local Operator API.

Provides REST endpoints for interacting with the Local Operator agent
through HTTP requests instead of CLI.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, HTTPException
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

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


@app.middleware("http")
async def managed_desktop_boundary(request: Request, call_next):
    path = request.url.path
    sensitive = path.startswith(("/v1/auth/", "/v1/settings", "/v1/mcp", "/v1/desktop/"))
    legacy_control = path in {"/v1/config", "/v1/config/system-prompt", "/v1/credentials"}
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
        and request.url.path in {"/v1/config", "/v1/credentials", "/v1/config/system-prompt"}
    ):
        return JSONResponse(status_code=422, content={"detail": "The request has invalid fields."})
    return await request_validation_exception_handler(request, error)


app.include_router(capabilities.router)
app.include_router(auth.router)
app.include_router(settings.router)
app.include_router(desktop_sessions.router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
