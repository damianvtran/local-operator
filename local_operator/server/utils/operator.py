"""
Utility functions for creating and managing operators in the Local Operator API.
"""

import logging
from typing import Optional

from local_operator.agents import AgentData, AgentRegistry  # Import AgentData
from local_operator.bootstrap import initialize_operator  # Import the new function
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.operator import Operator, OperatorType

logger = logging.getLogger("local_operator.server.utils")

# build_tool_registry is now handled within initialize_operator in bootstrap.py


def create_operator(
    request_hosting: str,
    request_model: str,
    credential_manager: CredentialManager,
    config_manager: ConfigManager,
    agent_registry: AgentRegistry,
    env_config: EnvConfig,
    current_agent: Optional[AgentData] = None,  # Use AgentData type hint
    persist_conversation: bool = False,
    job_id: Optional[str] = None,
    scheduler_service=None,
    status_queue=None,
) -> Operator:
    """Create an Operator instance for a server request using the centralized bootstrap logic.

    Args:
        request_hosting: The hosting service requested for this specific operation.
        request_model: The model name requested for this specific operation.
        credential_manager: The credential manager instance.
        config_manager: The configuration manager instance.
        agent_registry: The agent registry instance.
        env_config: The environment configuration instance.
        current_agent: Optional agent to use for the session.
        persist_conversation: Whether to persist conversation history during the session.
        job_id: Optional job ID associated with the request.

    Returns:
        The configured Operator instance.

    Raises:
        ValueError: If operator initialization fails (e.g., invalid model/hosting).
    """
    logger.info(
        f"Creating server operator for Hosting: {request_hosting}, Model: {request_model}, "
        f"Agent: {current_agent.name if current_agent else 'None'}, Job ID: {job_id}"
    )
    try:
        operator = initialize_operator(
            operator_type=OperatorType.SERVER,
            config_manager=config_manager,
            credential_manager=credential_manager,
            agent_registry=agent_registry,
            env_config=env_config,
            request_hosting=request_hosting,
            request_model=request_model,
            current_agent=current_agent,
            persist_conversation=persist_conversation,
            auto_save_conversation=False,  # Server typically doesn't auto-save this way
            job_id=job_id,
            verbosity_level=VerbosityLevel.QUIET,  # Server usually runs quietly
            scheduler_service=scheduler_service,
            status_queue=status_queue,
        )
        logger.info("Server operator created successfully.")
        return operator
    except ValueError as e:
        logger.error(f"Failed to create server operator: {e}", exc_info=True)
        # Re-raise the error to be handled by the calling route/dependency
        raise e

    return operator
