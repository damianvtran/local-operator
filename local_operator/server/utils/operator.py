"""
Utility functions for creating and managing operators in the Local Operator API.
"""

import logging
from typing import cast

from local_operator.admin import add_admin_tools
from local_operator.agents import AgentConversation, AgentRegistry
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
    persist_conversation: bool = False,
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
        persist_conversation: Whether to persist the conversation history by
            continuously updating the agent's conversation history with each new message.
            Default: False
    Returns:
        Operator: The configured operator instance

    Raises:
        ValueError: If hosting is not set or model configuration fails
    """
    agent_registry = cast(AgentRegistry, agent_registry)

    if not request_hosting:
        raise ValueError("Hosting is not set")

    agent_conversation_data = None

    if current_agent:
        agent_conversation_data = agent_registry.load_agent_conversation(current_agent.id)
    else:
        agent_conversation_data = AgentConversation(
            version="",
            conversation=[],
            execution_history=[],
        )

    model_configuration = configure_model(
        hosting=request_hosting,
        model_name=request_model,
        credential_manager=credential_manager,
    )

    if not model_configuration.instance:
        raise ValueError("No model instance configured")

    executor = LocalCodeExecutor(
        model_configuration=model_configuration,
        max_conversation_history=config_manager.get_config_value("max_conversation_history", 100),
        detail_conversation_length=config_manager.get_config_value(
            "detail_conversation_length", 35
        ),
        max_learnings_history=config_manager.get_config_value("max_learnings_history", 50),
        can_prompt_user=False,
        agent=current_agent,
    )

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_configuration=model_configuration,
        config_manager=config_manager,
        type=OperatorType.SERVER,
        agent_registry=agent_registry,
        current_agent=current_agent,
        training_mode=persist_conversation,
        auto_save_conversation=False,
    )

    tool_registry = build_tool_registry(executor, agent_registry, config_manager)
    executor.set_tool_registry(tool_registry)

    executor.load_conversation_history(agent_conversation_data.conversation)
    executor.load_execution_history(agent_conversation_data.execution_history)

    return operator
