"""
FastAPI server implementation for Local Operator API.

Provides REST endpoints for interacting with the Local Operator agent
through HTTP requests instead of CLI.
"""

import logging
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

from fastapi import Depends, FastAPI

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.server.routes import agents, chat, health

logger = logging.getLogger("local_operator.server")


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
    agents_dir = config_dir / "agents"
    app.state.credential_manager = CredentialManager(config_dir=config_dir)
    app.state.config_manager = ConfigManager(config_dir=config_dir)
    app.state.agent_registry = AgentRegistry(config_dir=agents_dir)
    yield
    # Clean up on shutdown
    app.state.credential_manager = None
    app.state.config_manager = None
    app.state.agent_registry = None


app = FastAPI(
    title="Local Operator API",
    description="REST API interface for Local Operator agent",
    version=version("local-operator"),
    lifespan=lifespan,
)


# Dependency functions to inject managers into route handlers
def get_credential_manager():
    """Get the credential manager from the application state."""
    return app.state.credential_manager


def get_config_manager():
    """Get the config manager from the application state."""
    return app.state.config_manager


def get_agent_registry():
    """Get the agent registry from the application state."""
    return app.state.agent_registry


# Include routers from the routes modules
app.include_router(health.router)
app.include_router(
    chat.router,
    dependencies=[
        Depends(get_credential_manager),
        Depends(get_config_manager),
        Depends(get_agent_registry),
    ],
)
app.include_router(
    agents.router,
    dependencies=[Depends(get_agent_registry)],
)
