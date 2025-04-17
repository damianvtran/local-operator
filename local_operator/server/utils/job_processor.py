"""
Utility functions for processing jobs in the Local Operator API.

This module provides functions for running jobs in separate processes,
handling the lifecycle of asynchronous jobs, and managing their execution context.
"""

import asyncio
import logging
from multiprocessing import Process
from typing import Callable, Optional

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.jobs import (
    JobContext,
    JobContextRecord,
    JobManager,
    JobResult,
    JobStatus,
)
from local_operator.server.utils.operator import create_operator
from local_operator.server.utils.websocket_manager import WebSocketManager
from local_operator.types import ConversationRecord

logger = logging.getLogger("local_operator.server.utils.job_processor")


def run_job_in_process(
    job_id: str,
    prompt: str,
    model: str,
    hosting: str,
    credential_manager: CredentialManager,
    config_manager: ConfigManager,
    agent_registry: AgentRegistry,
    env_config: EnvConfig,
    job_manager: JobManager,
    websocket_manager: Optional[WebSocketManager] = None,
    context: Optional[list[ConversationRecord]] = None,
    options: Optional[dict[str, object]] = None,
):
    """
    Run a chat job in a separate process.

    This function creates a new event loop for the process and runs the job in that context.
    It handles the entire lifecycle of the job, from updating its status to processing
    the request and storing the result.

    Args:
        job_id: The ID of the job to run
        prompt: The user prompt to process
        model: The model to use
        hosting: The hosting provider
        credential_manager: The credential manager for API keys
        config_manager: The configuration manager
        agent_registry: The agent registry for managing agents
        job_manager: The job manager for tracking job status
        context: Optional conversation context
        options: Optional model configuration options
    """
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def process_chat_job_in_context():
        try:
            # Create a new operator in this process context
            job_context = JobContext()
            with job_context:
                # Update job status to processing
                await job_manager.update_job_status(job_id, JobStatus.PROCESSING)

                # Create a new operator for this process
                process_operator = create_operator(
                    request_hosting=hosting,
                    request_model=model,
                    credential_manager=credential_manager,
                    config_manager=config_manager,
                    agent_registry=agent_registry,
                    job_id=job_id,
                    env_config=env_config,
                )

                # Initialize conversation history
                if context:
                    conversation_history = [
                        ConversationRecord(role=msg.role, content=msg.content) for msg in context
                    ]
                    process_operator.executor.initialize_conversation_history(
                        conversation_history, overwrite=True
                    )
                else:
                    try:
                        process_operator.executor.initialize_conversation_history()
                    except ValueError:
                        # Conversation history already initialized
                        pass

                # Configure model options if provided
                model_instance = process_operator.executor.model_configuration.instance
                if options:
                    # Handle temperature
                    if "temperature" in options and options["temperature"] is not None:
                        if hasattr(model_instance, "temperature"):
                            # Use setattr to avoid type checking issues
                            setattr(model_instance, "temperature", options["temperature"])

                    # Handle top_p
                    if "top_p" in options and options["top_p"] is not None:
                        if hasattr(model_instance, "top_p"):
                            # Use setattr to avoid type checking issues
                            setattr(model_instance, "top_p", options["top_p"])

                # Process the request
                _, final_response = await process_operator.handle_user_input(prompt)

                # Create result with response and context
                result = JobResult(
                    response=final_response or "",
                    context=[
                        JobContextRecord(
                            role=msg.role,
                            content=msg.content,
                            files=msg.files,
                        )
                        for msg in process_operator.executor.agent_state.conversation
                    ],
                )

                # Update job status to completed
                await job_manager.update_job_status(job_id, JobStatus.COMPLETED, result)
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {str(e)}")
            await job_manager.update_job_status(job_id, JobStatus.FAILED, {"error": str(e)})

    # Run the async function in the new event loop
    loop.run_until_complete(process_chat_job_in_context())
    loop.close()


def run_agent_job_in_process(
    job_id: str,
    prompt: str,
    model: str,
    hosting: str,
    agent_id: str,
    credential_manager: CredentialManager,
    config_manager: ConfigManager,
    agent_registry: AgentRegistry,
    env_config: EnvConfig,
    job_manager: JobManager,
    persist_conversation: bool = False,
    user_message_id: Optional[str] = None,
    websocket_manager: Optional[WebSocketManager] = None,
):
    """
    Run an agent chat job in a separate process.

    This function creates a new event loop for the process and runs the job in that context.
    It handles the entire lifecycle of the job, from updating its status to processing
    the request and storing the result.

    Args:
        job_id: The ID of the job to run
        prompt: The user prompt to process
        model: The model to use
        hosting: The hosting provider
        agent_id: The ID of the agent to use
        credential_manager: The credential manager for API keys
        config_manager: The configuration manager
        agent_registry: The agent registry for managing agents
        job_manager: The job manager for tracking job status
        persist_conversation: Whether to persist the conversation history
        user_message_id: Optional ID for the user message
    """
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def process_chat_job_in_context():
        try:
            # Create a new operator in this process context
            job_context = JobContext()
            with job_context:
                # Update job status to processing
                await job_manager.update_job_status(job_id, JobStatus.PROCESSING)

                # Retrieve the agent
                agent_obj = agent_registry.get_agent(agent_id)

                # Change to the agent's current working directory if it exists
                if (
                    agent_obj.current_working_directory
                    and agent_obj.current_working_directory != "."
                ):
                    job_context.change_directory(agent_obj.current_working_directory)

                # Create a new operator for this process
                process_operator = create_operator(
                    request_hosting=hosting,
                    request_model=model,
                    credential_manager=credential_manager,
                    config_manager=config_manager,
                    agent_registry=agent_registry,
                    current_agent=agent_obj,
                    persist_conversation=persist_conversation,
                    job_id=job_id,
                    env_config=env_config,
                )

                # Configure model options if provided
                model_instance = process_operator.executor.model_configuration.instance

                # Handle temperature
                if hasattr(agent_obj, "temperature") and agent_obj.temperature is not None:
                    if hasattr(model_instance, "temperature"):
                        # Use setattr to avoid type checking issues
                        setattr(model_instance, "temperature", agent_obj.temperature)

                # Handle top_p
                if hasattr(agent_obj, "top_p") and agent_obj.top_p is not None:
                    if hasattr(model_instance, "top_p"):
                        # Use setattr to avoid type checking issues
                        setattr(model_instance, "top_p", agent_obj.top_p)

                # Process the request
                _, final_response = await process_operator.handle_user_input(
                    prompt, user_message_id
                )

                # Create result with response and context
                result = JobResult(
                    response=final_response or "",
                    context=[
                        JobContextRecord(
                            role=msg.role,
                            content=msg.content,
                            files=msg.files,
                        )
                        for msg in process_operator.executor.agent_state.conversation
                    ],
                )

                # Update job status to completed
                await job_manager.update_job_status(job_id, JobStatus.COMPLETED, result)
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {str(e)}")
            await job_manager.update_job_status(job_id, JobStatus.FAILED, {"error": str(e)})

    # Run the async function in the new event loop
    loop.run_until_complete(process_chat_job_in_context())
    loop.close()


def create_and_start_job_process(
    job_id: str,
    process_func: Callable[..., None],
    args: tuple[object, ...],
    job_manager: JobManager,
) -> Process:
    """
    Create and start a process for a job, and register it with the job manager.

    This function creates a Process object with the given function and arguments,
    starts it, registers it with the job manager, and creates a monitoring task.

    Args:
        job_id: The ID of the job
        process_func: The function to run in the process
        args: The arguments to pass to the function
        job_manager: The job manager for tracking the process

    Returns:
        The created Process object
    """
    # Create a process for the job
    process = Process(target=process_func, args=args)

    # Register the process with the job manager before starting it
    # This avoids any potential issues with asyncio.Task objects being pickled
    job_manager.register_process(job_id, process)

    # Start the process after registration
    process.start()

    # Create a task to monitor the process
    async def monitor_process():
        # This task just exists to allow cancellation via asyncio
        pass

    # Start the monitor task
    task = asyncio.create_task(monitor_process())

    # Register the task with the job manager
    # Use a separate function to avoid capturing the task in the closure
    async def register_monitor_task():
        await job_manager.register_task(job_id, task)

    # Create a separate task for registration to avoid pickling issues
    asyncio.create_task(register_monitor_task())

    return process
