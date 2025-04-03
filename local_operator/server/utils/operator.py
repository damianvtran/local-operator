"""
Utility functions for creating and managing operators in the Local Operator API.
"""

import logging
from typing import Optional

from local_operator.admin import add_admin_tools
from local_operator.agents import AgentRegistry
from local_operator.clients.fal import FalClient
from local_operator.clients.openrouter import OpenRouterClient
from local_operator.clients.serpapi import SerpApiClient
from local_operator.clients.tavily import TavilyClient
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.executor import LocalCodeExecutor
from local_operator.model.configure import configure_model
from local_operator.operator import Operator, OperatorType
from local_operator.tools import ToolRegistry
from local_operator.types import AgentState

logger = logging.getLogger("local_operator.server.utils")


def build_tool_registry(
    executor: LocalCodeExecutor,
    agent_registry: AgentRegistry,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
) -> ToolRegistry:
    """Build and initialize the tool registry.

    This function creates a new ToolRegistry instance, initializes default tools,
    and adds admin tools for agent management. It also sets up SERP API and Tavily
    API clients if the corresponding API keys are available.

    Args:
        executor (LocalCodeExecutor): The LocalCodeExecutor instance for conversation history.
        agent_registry (AgentRegistry): The AgentRegistry for managing agents.
        config_manager (ConfigManager): The ConfigManager for managing configuration.
        credential_manager (CredentialManager): The CredentialManager for managing credentials.

    Returns:
        ToolRegistry: The initialized tool registry with all tools registered.
    """
    tool_registry = ToolRegistry()

    serp_api_key = credential_manager.get_credential("SERP_API_KEY")
    tavily_api_key = credential_manager.get_credential("TAVILY_API_KEY")
    fal_api_key = credential_manager.get_credential("FAL_API_KEY")

    if serp_api_key:
        serp_api_client = SerpApiClient(serp_api_key)
        tool_registry.set_serp_api_client(serp_api_client)

    if tavily_api_key:
        tavily_client = TavilyClient(tavily_api_key)
        tool_registry.set_tavily_client(tavily_client)

    if fal_api_key:
        fal_client = FalClient(fal_api_key)
        tool_registry.set_fal_client(fal_client)

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
    job_id: Optional[str] = None,
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
        job_manager: The job manager for the current conversation.
        job_id: The job ID for the current conversation.
    Returns:
        Operator: The configured operator instance

    Raises:
        ValueError: If hosting is not set or model configuration fails
    """
    if not request_hosting:
        raise ValueError("Hosting is not set")

    agent_state = None

    chat_args = {}

    if current_agent:
        agent_state = agent_registry.load_agent_state(current_agent.id)

        if current_agent.temperature:
            chat_args["temperature"] = current_agent.temperature
        if current_agent.top_p:
            chat_args["top_p"] = current_agent.top_p
        if current_agent.top_k:
            chat_args["top_k"] = current_agent.top_k
        if current_agent.max_tokens:
            chat_args["max_tokens"] = current_agent.max_tokens
        if current_agent.stop:
            chat_args["stop"] = current_agent.stop
        if current_agent.frequency_penalty:
            chat_args["frequency_penalty"] = current_agent.frequency_penalty
        if current_agent.presence_penalty:
            chat_args["presence_penalty"] = current_agent.presence_penalty
        if current_agent.seed:
            chat_args["seed"] = current_agent.seed

    else:
        agent_state = AgentState(
            version="",
            conversation=[],
            execution_history=[],
            learnings=[],
            current_plan=None,
            instruction_details=None,
            agent_system_prompt=None,
        )

    model_info_client: Optional[OpenRouterClient] = None

    if request_hosting == "openrouter":
        model_info_client = OpenRouterClient(
            credential_manager.get_credential("OPENROUTER_API_KEY")
        )

    model_configuration = configure_model(
        hosting=request_hosting,
        model_name=request_model,
        credential_manager=credential_manager,
        model_info_client=model_info_client,
        **chat_args,
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
        verbosity_level=VerbosityLevel.QUIET,
        agent_registry=agent_registry,
        agent_state=agent_state,
        persist_conversation=persist_conversation,
        job_id=job_id,
    )

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_configuration=model_configuration,
        config_manager=config_manager,
        type=OperatorType.SERVER,
        agent_registry=agent_registry,
        current_agent=current_agent,
        auto_save_conversation=False,
        verbosity_level=VerbosityLevel.QUIET,
        persist_agent_conversation=persist_conversation,
    )

    tool_registry = build_tool_registry(
        executor, agent_registry, config_manager, credential_manager
    )
    executor.set_tool_registry(tool_registry)

    executor.load_agent_state(agent_state)

    return operator
