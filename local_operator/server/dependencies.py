from fastapi import Depends, Request

from local_operator.agents import AgentRegistry
from local_operator.clients.radient import RadientClient
from local_operator.config import ConfigManager
from local_operator.env import EnvConfig
from local_operator.jobs import JobManager
from local_operator.providers.auth_store import AuthStore
from local_operator.scheduler_service import SchedulerService
from local_operator.server.utils.event_broker import EventBroker

# Dependency functions to inject managers into route handlers


def get_config_manager(request: Request) -> ConfigManager:
    """Get the config manager from the application state."""
    return request.app.state.config_manager


def get_agent_registry(request: Request) -> AgentRegistry:
    """Get the agent registry from the application state."""
    return request.app.state.agent_registry


def get_job_manager(request: Request) -> JobManager:
    """Get the job manager from the application state."""
    return request.app.state.job_manager


def get_event_broker(request: Request) -> EventBroker:
    """Get the SSE event broker from the application state."""
    return request.app.state.event_broker


def get_env_config(request: Request) -> EnvConfig:
    """Get the environment configuration from the application state."""
    return request.app.state.env_config


def get_scheduler_service(request: Request) -> SchedulerService:
    """Get the scheduler service from the application state."""
    return request.app.state.scheduler_service


async def get_provider_auth_store(
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
) -> AuthStore:
    # The managers are DECLARED rather than read off `app.state` inside the
    # chain below. `dependency_overrides` only substitutes what a signature
    # declares, and this dependency sits between the models routes and
    # `get_desktop_auth`, so an override supplied by a caller that mounts one
    # router in isolation was silently skipped on the way through.
    from local_operator.server.desktop import desktop_posture, require_desktop
    from local_operator.server.routes.auth import get_desktop_auth

    # A central login must not become usable through an unprotected legacy
    # endpoint merely because that endpoint historically accepted local keys.
    # ``desktop_posture`` and not the environment: a daemon that accepted a
    # claim governs its plane exactly as one the app started does, and a test
    # of the environment here is how this dependency would quietly stay shut
    # on a claimed daemon.
    if desktop_posture().enabled:
        require_desktop(request)
    # Passed explicitly: `get_desktop_auth` is invoked as a plain function here,
    # so its own `Depends(...)` defaults would arrive as `Depends` objects
    # rather than resolving themselves.
    return (await get_desktop_auth(request, manager=manager)).store


async def get_radient_client(request: Request) -> RadientClient:
    """Use canonical credential precedence without duplicating a refresh store."""
    from local_operator.providers.radient_credentials import resolve_radient_credential

    config_manager = get_config_manager(request)
    env_config = get_env_config(request)
    store = await get_provider_auth_store(request)
    api_key = await resolve_radient_credential(
        config_manager.config_dir, env_config.radient_api_base_url, store=store
    )
    return RadientClient(api_key=api_key, base_url=env_config.radient_api_base_url)
