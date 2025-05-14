"""
FastAPI server implementation for Local Operator API.

Provides REST endpoints for interacting with the Local Operator agent
through HTTP requests instead of CLI.
"""

from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import get_env_config
from local_operator.helpers import setup_cross_platform_environment
from local_operator.jobs import JobManager
from local_operator.logger import get_logger
from local_operator.operator import OperatorType
from local_operator.scheduler_service import SchedulerService
from local_operator.server.routes import (
    agents,
    chat,
    config,
    credentials,
    health,
    jobs,
    models,
    schedules,
    static,
    websockets,
)
from local_operator.server.utils.websocket_manager import WebSocketManager

logger = get_logger("local_operator.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up application state.

    This function is called when the application starts up and shuts down.
    It initializes the credential manager, config manager, and agent registry.

    Args:
        app: The FastAPI application instance
    """
    # Initialize on startup by setting up the credential and config managers
    config_dir = Path.home() / ".local-operator"
    agent_home_dir = Path.home() / "local-operator-home"

    # Create the agent home directory if it doesn't exist
    if not agent_home_dir.exists():
        agent_home_dir.mkdir(parents=True, exist_ok=True)

    # Set up the subprocess environment for accessing shell commands
    setup_cross_platform_environment()

    app.state.credential_manager = CredentialManager(config_dir=config_dir)
    app.state.config_manager = ConfigManager(config_dir=config_dir)
    # Initialize AgentRegistry with a refresh interval of 3 seconds to ensure
    # changes made by child processes are quickly reflected in the parent process
    app.state.agent_registry = AgentRegistry(config_dir=config_dir, refresh_interval=3.0)
    app.state.job_manager = JobManager()
    app.state.websocket_manager = WebSocketManager()
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
    )

    await app.state.scheduler_service.start()

    yield
    # Clean up on shutdown
    await app.state.scheduler_service.shutdown()

    app.state.credential_manager = None
    app.state.config_manager = None
    app.state.agent_registry = None
    app.state.job_manager = None
    app.state.websocket_manager = None
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
        {"name": "Static", "description": "Static file hosting endpoints"},
    ],
)

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

# /v1/schedules
app.include_router(
    schedules.router,
)
