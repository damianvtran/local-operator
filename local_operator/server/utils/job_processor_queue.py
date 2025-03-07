"""
Utility functions for processing jobs in the Local Operator API with queue-based status updates.

This module provides functions for running jobs in separate processes,
handling the lifecycle of asynchronous jobs, and managing their execution context.
It uses a shared queue to communicate status updates from the child process to the parent.
"""

import asyncio
import logging
import multiprocessing
from multiprocessing import Process, Queue
from typing import Any, Callable, Optional, Tuple

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.jobs import JobContext, JobManager, JobStatus
from local_operator.server.utils.operator import create_operator
from local_operator.types import ConversationRecord

logger = logging.getLogger("local_operator.server.utils.job_processor_queue")


def run_job_in_process_with_queue(
    job_id: str,
    prompt: str,
    model: str,
    hosting: str,
    credential_manager: CredentialManager,
    config_manager: ConfigManager,
    agent_registry: AgentRegistry,
    context: Optional[list[ConversationRecord]] = None,
    options: Optional[dict[str, object]] = None,
    status_queue: Optional[Queue] = None,  # type: ignore
):
    """
    Run a chat job in a separate process, using a queue to communicate status updates.

    This function creates a new event loop for the process and runs the job in that context.
    Instead of directly updating the job status in the job manager (which would only update
    the copy in the child process), it sends status updates through a shared queue that
    can be monitored by the parent process.

    Args:
        job_id: The ID of the job to run
        prompt: The user prompt to process
        model: The model to use
        hosting: The hosting provider
        credential_manager: The credential manager for API keys
        config_manager: The configuration manager
        agent_registry: The agent registry for managing agents
        context: Optional conversation context
        options: Optional model configuration options
        status_queue: A queue to communicate status updates to the parent process
    """
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def process_chat_job_in_context():
        try:
            # Create a new operator in this process context
            job_context = JobContext()
            with job_context:
                # Send status update to the parent process
                if status_queue:
                    status_queue.put((job_id, JobStatus.PROCESSING, None))

                # Create a new operator for this process
                process_operator = create_operator(
                    request_hosting=hosting,
                    request_model=model,
                    credential_manager=credential_manager,
                    config_manager=config_manager,
                    agent_registry=agent_registry,
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
                response_json = await process_operator.handle_user_input(prompt)

                # Create result with response and context
                result = {
                    "response": response_json.response if response_json is not None else "",
                    "context": [
                        {"role": msg.role, "content": msg.content}
                        for msg in process_operator.executor.conversation_history
                    ],
                }

                # Send completed status update to the parent process
                if status_queue:
                    status_queue.put((job_id, JobStatus.COMPLETED, result))
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {str(e)}")
            if status_queue:
                status_queue.put((job_id, JobStatus.FAILED, {"error": str(e)}))

    # Run the async function in the new event loop
    loop.run_until_complete(process_chat_job_in_context())
    loop.close()


def run_agent_job_in_process_with_queue(
    job_id: str,
    prompt: str,
    model: str,
    hosting: str,
    agent_id: str,
    credential_manager: CredentialManager,
    config_manager: ConfigManager,
    agent_registry: AgentRegistry,
    persist_conversation: bool = False,
    user_message_id: Optional[str] = None,
    status_queue: Optional[Queue] = None,  # type: ignore
):
    """
    Run an agent chat job in a separate process, using a queue to communicate status updates.

    This function creates a new event loop for the process and runs the job in that context.
    Instead of directly updating the job status in the job manager (which would only update
    the copy in the child process), it sends status updates through a shared queue that
    can be monitored by the parent process.

    Args:
        job_id: The ID of the job to run
        prompt: The user prompt to process
        model: The model to use
        hosting: The hosting provider
        agent_id: The ID of the agent to use
        credential_manager: The credential manager for API keys
        config_manager: The configuration manager
        agent_registry: The agent registry for managing agents
        persist_conversation: Whether to persist the conversation history
        user_message_id: Optional ID for the user message
        status_queue: A queue to communicate status updates to the parent process
    """
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def process_chat_job_in_context():
        try:
            # Create a new operator in this process context
            job_context = JobContext()
            with job_context:
                # Send status update to the parent process
                if status_queue:
                    status_queue.put((job_id, JobStatus.PROCESSING, None))

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
                response_json = await process_operator.handle_user_input(prompt, user_message_id)

                # Create result with response and context
                result = {
                    "response": response_json.response if response_json is not None else "",
                    "context": [
                        {"role": msg.role, "content": msg.content}
                        for msg in process_operator.executor.conversation_history
                    ],
                }

                # Send completed status update to the parent process
                if status_queue:
                    status_queue.put((job_id, JobStatus.COMPLETED, result))
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {str(e)}")
            if status_queue:
                status_queue.put((job_id, JobStatus.FAILED, {"error": str(e)}))

    # Run the async function in the new event loop
    loop.run_until_complete(process_chat_job_in_context())
    loop.close()


def create_and_start_job_process_with_queue(
    job_id: str,
    process_func: Callable[..., None],
    args: tuple[object, ...],
    job_manager: JobManager,
) -> Tuple[Process, asyncio.Task[Any]]:
    """
    Create and start a process for a job, and set up a queue monitor to update the job status.

    This function creates a Process object with the given function and arguments,
    starts it, and sets up a task to monitor the status queue for updates from the child process.

    Args:
        job_id: The ID of the job
        process_func: The function to run in the process
        args: The arguments to pass to the function
        job_manager: The job manager for tracking the process

    Returns:
        A tuple containing the created Process object and the monitor task
    """
    # Create a queue for status updates
    status_queue = multiprocessing.Queue()

    # Create a process for the job, adding the status queue to the arguments
    process_args = args + (status_queue,)
    process = Process(target=process_func, args=process_args)
    process.start()

    # Register the process with the job manager
    job_manager.register_process(job_id, process)

    # Create a task to monitor the status queue
    async def monitor_status_queue():
        current_job_id = job_id  # Capture job_id in closure to avoid unbound variable issue
        try:
            while process.is_alive() or not status_queue.empty():
                if not status_queue.empty():
                    received_job_id, status, result = status_queue.get()
                    await job_manager.update_job_status(received_job_id, status, result)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            pass
        except Exception as e:
            logger.exception(f"Error monitoring status queue for job {current_job_id}: {str(e)}")

    # Start the monitor task
    monitor_task = asyncio.create_task(monitor_status_queue())

    # Register the task with the job manager
    asyncio.create_task(job_manager.register_task(job_id, monitor_task))

    return process, monitor_task
