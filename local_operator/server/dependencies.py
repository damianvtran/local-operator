from fastapi import Request

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.jobs import JobManager


# Dependency functions to inject managers into route handlers
def get_credential_manager(request: Request) -> CredentialManager:
    """Get the credential manager from the application state."""
    return request.app.state.credential_manager


def get_config_manager(request: Request) -> ConfigManager:
    """Get the config manager from the application state."""
    return request.app.state.config_manager


def get_agent_registry(request: Request) -> AgentRegistry:
    """Get the agent registry from the application state."""
    return request.app.state.agent_registry


def get_job_manager(request: Request) -> JobManager:
    """Get the job manager from the application state."""
    return request.app.state.job_manager
