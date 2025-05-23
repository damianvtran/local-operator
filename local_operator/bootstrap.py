"""
Handles the bootstrapping and initialization of the Operator and its core components.

This module centralizes the setup logic for creating Operator instances,
configuring models, initializing executors, and building tool registries,
ensuring consistency between different entry points like the CLI and the server.
"""

from typing import Any, Optional, Union

from pydantic import SecretStr

from local_operator.admin import add_admin_tools
from local_operator.agents import AgentData, AgentRegistry, AgentState
from local_operator.clients.fal import FalClient
from local_operator.clients.openrouter import OpenRouterClient
from local_operator.clients.radient import RadientClient
from local_operator.clients.serpapi import SerpApiClient
from local_operator.clients.tavily import TavilyClient
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.executor import LocalCodeExecutor
from local_operator.logger import get_logger
from local_operator.model.configure import (
    ModelConfiguration,
    configure_model,
    validate_model,
)
from local_operator.operator import Operator, OperatorType
from local_operator.tools.general import ToolRegistry

logger = get_logger()


def build_tool_registry(
    executor: LocalCodeExecutor,
    agent_registry: AgentRegistry,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    env_config: EnvConfig,
    model_configuration: ModelConfiguration,
    scheduler_service: Optional[Any] = None,
    status_queue: Optional[Any] = None,
) -> ToolRegistry:
    """Build and initialize the tool registry.

    This function creates a new ToolRegistry instance, initializes default tools,
    and adds admin tools for agent management. It also sets up API clients
    if the corresponding API keys are available.

    Args:
        executor: The LocalCodeExecutor instance.
        agent_registry: The AgentRegistry for managing agents.
        config_manager: The ConfigManager for managing configuration.
        credential_manager: The CredentialManager for managing credentials.
        env_config: The environment configuration.

    Returns:
        The initialized tool registry with all tools registered.
    """
    tool_registry = ToolRegistry()

    serp_api_key = credential_manager.get_credential("SERP_API_KEY")
    tavily_api_key = credential_manager.get_credential("TAVILY_API_KEY")
    fal_api_key = credential_manager.get_credential("FAL_API_KEY")
    radient_api_key = credential_manager.get_credential("RADIENT_API_KEY")

    if serp_api_key:
        serp_api_client = SerpApiClient(serp_api_key)
        tool_registry.set_serp_api_client(serp_api_client)

    if tavily_api_key:
        tavily_client = TavilyClient(tavily_api_key)
        tool_registry.set_tavily_client(tavily_client)

    if fal_api_key:
        fal_client = FalClient(fal_api_key)
        tool_registry.set_fal_client(fal_client)

    if radient_api_key:
        radient_client = RadientClient(
            api_key=radient_api_key, base_url=env_config.radient_api_base_url
        )
        tool_registry.set_radient_client(radient_client)

    tool_registry.set_credential_manager(credential_manager)
    tool_registry.set_model_configuration(model_configuration)
    tool_registry.set_agent_registry(agent_registry)

    if scheduler_service:
        tool_registry.set_scheduler_service(scheduler_service)

    # Register queue and callback to bridge the boundary between the job execution
    # environment and the server environment.
    if status_queue:
        tool_registry.set_tool_execution_callback(executor.tool_execution_callback)
        tool_registry.set_status_queue(status_queue)

    tool_registry.init_tools()

    # Admin tools might depend on the executor state, ensure executor is passed
    add_admin_tools(tool_registry, executor, agent_registry, config_manager)

    return tool_registry


def initialize_operator(
    operator_type: OperatorType,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: AgentRegistry,
    env_config: EnvConfig,
    scheduler_service: Optional[Any] = None,
    status_queue: Optional[Any] = None,
    request_hosting: Optional[str] = None,
    request_model: Optional[str] = None,
    current_agent: Optional[AgentData] = None,
    persist_conversation: bool = False,
    auto_save_conversation: bool = False,
    job_id: Optional[str] = None,
    verbosity_level: VerbosityLevel = VerbosityLevel.VERBOSE,
) -> Operator:
    """
    Initializes and configures the Operator, Executor, and ToolRegistry.

    This function centralizes the setup logic for creating an Operator instance,
    handling model configuration, agent state loading, and tool registration based
    on the specified operator type (CLI or Server). The SchedulerService is
    expected to be created and managed by the caller (CLI or Server app) and
    passed here if its tools need to be registered.

    Args:
        operator_type: The type of operator being initialized (CLI or SERVER).
        config_manager: The configuration manager instance.
        credential_manager: The credential manager instance.
        agent_registry: The agent registry instance.
        env_config: The environment configuration instance.
        scheduler_service: An optional SchedulerService instance. If provided,
                           scheduling-related tools will be configured.
        request_hosting: Hosting platform override (used by server).
        request_model: Model name override (used by server).
        current_agent: The specific agent to use for this session.
        persist_conversation: Whether to persist conversation history during the session.
        auto_save_conversation: Whether to automatically save conversation at the end.
        job_id: Optional job ID for server-based operations.
        verbosity_level: The verbosity level for console output (primarily for CLI).

    Returns:
        The fully configured Operator instance.

    Raises:
        ValueError: If hosting/model configuration fails or is invalid.
    """
    hosting = request_hosting or config_manager.get_config_value("hosting")
    model_name = request_model or config_manager.get_config_value("model_name")

    if not hosting:
        raise ValueError("Hosting platform is not configured.")
    if not model_name:
        raise ValueError("Model name is not configured.")

    logger.debug(
        f"Initializing operator (Type: {operator_type.name}) with Hosting: {hosting}, "
        f"Model: {model_name}, Agent: {current_agent.name if current_agent else 'None'}"
    )

    agent_state: Optional[AgentState] = None
    chat_args = {}

    if current_agent:
        agent_state = agent_registry.load_agent_state(current_agent.id)
        logger.debug(f"Loaded state for agent: {current_agent.name} (ID: {current_agent.id})")

        # Override hosting/model from agent if set
        if current_agent.hosting:
            hosting = current_agent.hosting
            logger.debug(f"Using agent's hosting override: {hosting}")
        if current_agent.model:
            model_name = current_agent.model
            logger.debug(f"Using agent's model override: {model_name}")

        # Load chat parameters from agent
        if current_agent.temperature is not None:
            chat_args["temperature"] = current_agent.temperature
        if current_agent.top_p is not None:
            chat_args["top_p"] = current_agent.top_p
        if current_agent.top_k is not None:
            chat_args["top_k"] = current_agent.top_k
        if current_agent.max_tokens is not None:
            chat_args["max_tokens"] = current_agent.max_tokens
        if current_agent.stop:
            chat_args["stop"] = current_agent.stop
        if current_agent.frequency_penalty is not None:
            chat_args["frequency_penalty"] = current_agent.frequency_penalty
        if current_agent.presence_penalty is not None:
            chat_args["presence_penalty"] = current_agent.presence_penalty
        if current_agent.seed is not None:
            chat_args["seed"] = current_agent.seed
        logger.debug(f"Loaded chat args from agent: {chat_args}")

    else:
        # Create a default empty state if no agent is provided
        agent_state = AgentState(
            version="",
            conversation=[],
            execution_history=[],
            learnings=[],
            schedules=[],
            current_plan=None,
            instruction_details=None,
            agent_system_prompt=None,
        )
        logger.debug("No agent provided, using default empty agent state.")

    # --- Model Configuration ---
    model_info_client: Optional[Union[OpenRouterClient, RadientClient]] = None
    if hosting == "openrouter":
        api_key = credential_manager.get_credential("OPENROUTER_API_KEY")
        if api_key:
            model_info_client = OpenRouterClient(api_key)
        else:
            logger.warning("OpenRouter hosting selected but OPENROUTER_API_KEY not found.")
    elif hosting == "radient":
        api_key = credential_manager.get_credential("RADIENT_API_KEY")
        if api_key:
            model_info_client = RadientClient(api_key, env_config.radient_api_base_url)
        else:
            logger.warning("Radient hosting selected but RADIENT_API_KEY not found.")

    try:
        model_configuration: ModelConfiguration = configure_model(
            hosting=hosting,
            model_name=model_name,
            credential_manager=credential_manager,
            model_info_client=model_info_client,
            env_config=env_config,
            **chat_args,
        )
    except Exception as e:
        logger.error(f"Failed to configure model {model_name} on {hosting}: {e}", exc_info=True)
        raise ValueError(f"Failed to configure model {model_name} on {hosting}: {e}") from e

    if not model_configuration.instance:
        logger.error(f"Model instance configuration failed for {model_name} on {hosting}.")
        raise ValueError(f"No model instance configured for {hosting}/{model_name}")

    # Validate model (primarily for CLI to give early feedback)
    if operator_type == OperatorType.CLI:
        validate_model(hosting, model_name, model_configuration.api_key or SecretStr(""))
        logger.debug(f"Model {model_name} on {hosting} validated successfully.")

    executor = LocalCodeExecutor(
        model_configuration=model_configuration,
        max_conversation_history=config_manager.get_config_value("max_conversation_history", 100),
        detail_conversation_length=config_manager.get_config_value("detail_length", 15),
        max_learnings_history=config_manager.get_config_value("max_learnings_history", 50),
        can_prompt_user=(operator_type == OperatorType.CLI),
        agent=current_agent,
        verbosity_level=verbosity_level,
        agent_registry=agent_registry,
        agent_state=agent_state,
        persist_conversation=persist_conversation,
        job_id=job_id,
    )
    logger.debug(f"LocalCodeExecutor initialized. Can prompt user: {executor.can_prompt_user}")

    # --- Operator Initialization ---
    operator = Operator(
        executor=executor,  # Pass the final executor instance
        credential_manager=credential_manager,
        model_configuration=model_configuration,
        config_manager=config_manager,
        type=operator_type,
        agent_registry=agent_registry,
        current_agent=current_agent,
        auto_save_conversation=auto_save_conversation,
        verbosity_level=verbosity_level,
        persist_agent_conversation=persist_conversation,
        env_config=env_config,
    )
    logger.debug(
        f"Operator instance created. Type: {operator.type.name}, "
        f"AutoSave: {operator.auto_save_conversation}"
    )

    # --- Tool Registry Initialization ---
    # The scheduler_service instance is passed directly from the caller
    tool_registry = build_tool_registry(
        executor=executor,
        agent_registry=agent_registry,
        config_manager=config_manager,
        credential_manager=credential_manager,
        env_config=env_config,
        model_configuration=model_configuration,
        scheduler_service=scheduler_service,
        status_queue=status_queue,
    )
    executor.set_tool_registry(tool_registry)
    logger.debug("ToolRegistry built and set on executor.")

    # Load agent state into executor *after* tool registry is set
    if agent_state:
        executor.load_agent_state(agent_state)
        logger.debug("Agent state loaded into executor.")

    return operator
