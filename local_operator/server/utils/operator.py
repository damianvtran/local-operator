"""
Utility functions for creating and managing operators in the Local Operator API.
"""

import logging
from typing import cast

from fastapi import HTTPException

from local_operator.admin import add_admin_tools
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.executor import LocalCodeExecutor
from local_operator.model.configure import configure_model
from local_operator.operator import Operator, OperatorType
from local_operator.tools import ToolRegistry

logger = logging.getLogger("local_operator.server.utils")


def build_tool_registry(
    executor: LocalCodeExecutor, agent_registry: AgentRegistry, config_manager: ConfigManager
) -> ToolRegistry:
    """Build and initialize the tool registry with agent management tools.

    This function creates a new ToolRegistry instance and registers the core agent management tools:
    - create_agent_from_conversation: Creates a new agent from the current conversation
    - edit_agent: Modifies an existing agent's properties
    - delete_agent: Removes an agent from the registry
    - get_agent_info: Retrieves information about agents

    Args:
        executor: The LocalCodeExecutor instance containing conversation history
        agent_registry: The AgentRegistry for managing agents
        config_manager: The ConfigManager for managing configuration

    Returns:
        ToolRegistry: The initialized tool registry with all agent management tools registered
    """
    tool_registry = ToolRegistry()
    tool_registry.init_tools()
    add_admin_tools(tool_registry, executor, agent_registry, config_manager)
    return tool_registry


def create_operator(
    request_hosting: str,
    request_model: str,
    credential_manager: CredentialManager,
    config_manager: ConfigManager,
    agent_registry: AgentRegistry,
    current_agent=None,
) -> Operator:
    """Create a LocalCodeExecutor for a single chat request using the provided managers
    and the hosting/model provided in the request.

    Args:
        request_hosting: The hosting service to use
        request_model: The model name to use
        credential_manager: The credential manager for API keys
        config_manager: The configuration manager
        agent_registry: The agent registry for managing agents
        current_agent: Optional current agent to use

    Returns:
        Operator: The configured operator instance

    Raises:
        ValueError: If hosting is not set or model configuration fails
        HTTPException: If server configuration is not initialized
    """
    if credential_manager is None or config_manager is None or agent_registry is None:
        raise HTTPException(status_code=500, detail="Server configuration not initialized")
    agent_registry = cast(AgentRegistry, agent_registry)

    if not request_hosting:
        raise ValueError("Hosting is not set")

    model_configuration = configure_model(
        hosting=request_hosting,
        model_name=request_model,
        credential_manager=credential_manager,
    )

    if not model_configuration.instance:
        raise ValueError("No model instance configured")

    executor = LocalCodeExecutor(
        model_configuration=model_configuration,
        max_conversation_history=100,
        detail_conversation_length=10,
        can_prompt_user=False,
    )

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_configuration=model_configuration,
        config_manager=config_manager,
        type=OperatorType.SERVER,
        agent_registry=agent_registry,
        current_agent=current_agent,
        training_mode=False,
    )

    tool_registry = build_tool_registry(executor, agent_registry, config_manager)
    executor.set_tool_registry(tool_registry)

    return operator
