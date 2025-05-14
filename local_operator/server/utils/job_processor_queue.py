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
from typing import TYPE_CHECKING, Callable, List, Optional  # Added TYPE_CHECKING
from uuid import UUID

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.jobs import JobContext, JobContextRecord, JobManager, JobStatus

# from local_operator.scheduler_service import SchedulerService # Moved to TYPE_CHECKING
from local_operator.server.utils.operator import create_operator
from local_operator.server.utils.websocket_manager import WebSocketManager
from local_operator.types import ConversationRecord, Schedule

if TYPE_CHECKING:
    from local_operator.scheduler_service import SchedulerService

logger = logging.getLogger("local_operator.server.utils.job_processor_queue")


def run_job_in_process_with_queue(
    job_id: str,
    prompt: str,
    attachments: List[str],
    model: str,
    hosting: str,
    credential_manager: CredentialManager,
    config_manager: ConfigManager,
    agent_registry: AgentRegistry,
    env_config: EnvConfig,
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
                    status_queue.put(("status_update", job_id, JobStatus.PROCESSING, None))

                # Create a new JobManager instance in the child process
                job_manager = JobManager()
                # Initialize the job in the child process (will be updated via queue)
                if job_id not in job_manager.jobs:
                    from local_operator.jobs import Job

                    job_manager.jobs[job_id] = Job(
                        id=job_id, prompt=prompt, model=model, hosting=hosting
                    )

                # Create a new operator for this process
                process_operator = create_operator(
                    request_hosting=hosting,
                    request_model=model,
                    credential_manager=credential_manager,
                    config_manager=config_manager,
                    agent_registry=agent_registry,
                    job_id=job_id,
                    env_config=env_config,
                    status_queue=status_queue,
                )

                # Set the status queue on the executor for execution state updates
                if status_queue and hasattr(process_operator, "executor"):
                    process_operator.executor.status_queue = status_queue

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
                _, final_response = await process_operator.handle_user_input(
                    prompt, attachments=attachments
                )

                # Create result with response and context
                result = {
                    "response": final_response or "",
                    "context": [
                        JobContextRecord(
                            role=msg.role,
                            content=msg.content,
                            files=msg.files,
                        )
                        for msg in process_operator.executor.agent_state.conversation
                    ],
                }

                # Send completed status update to the parent process
                if status_queue:
                    status_queue.put(("status_update", job_id, JobStatus.COMPLETED, result))
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {str(e)}")
            if status_queue:
                status_queue.put(("status_update", job_id, JobStatus.FAILED, {"error": str(e)}))

    # Run the async function in the new event loop
    loop.run_until_complete(process_chat_job_in_context())
    loop.close()


def run_agent_job_in_process_with_queue(
    job_id: str,
    prompt: str,
    attachments: List[str],
    model: str,
    hosting: str,
    agent_id: str,
    credential_manager: CredentialManager,
    config_manager: ConfigManager,
    agent_registry: AgentRegistry,
    env_config: EnvConfig,
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
        attachments: The attachments to process
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
                    status_queue.put(("status_update", job_id, JobStatus.PROCESSING, None))

                # Retrieve the agent
                agent_obj = agent_registry.get_agent(agent_id)

                # Change to the agent's current working directory if it exists
                if (
                    agent_obj.current_working_directory
                    and agent_obj.current_working_directory != "."
                ):
                    job_context.change_directory(agent_obj.current_working_directory)

                # Create a new JobManager instance in the child process
                job_manager = JobManager()
                # Initialize the job in the child process (will be updated via queue)
                if job_id not in job_manager.jobs:
                    from local_operator.jobs import Job

                    job_manager.jobs[job_id] = Job(
                        id=job_id, prompt=prompt, model=model, hosting=hosting, agent_id=agent_id
                    )

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
                    status_queue=status_queue,
                )

                # Set the status queue on the executor for execution state updates
                if status_queue and hasattr(process_operator, "executor"):
                    process_operator.executor.status_queue = status_queue

                # Process the request
                _, final_response = await process_operator.handle_user_input(
                    prompt, user_message_id, attachments
                )

                # Create result with response and context
                result = {
                    "response": final_response or "",
                    "context": [
                        JobContextRecord(
                            role=msg.role,
                            content=msg.content,
                            files=msg.files,
                        )
                        for msg in process_operator.executor.agent_state.conversation
                    ],
                }

                # Send completed status update to the parent process
                if status_queue:
                    status_queue.put(("status_update", job_id, JobStatus.COMPLETED, result))
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {str(e)}")
            if status_queue:
                status_queue.put(("status_update", job_id, JobStatus.FAILED, {"error": str(e)}))

    # Run the async function in the new event loop
    loop.run_until_complete(process_chat_job_in_context())
    loop.close()


def create_and_start_job_process_with_queue(
    job_id: str,
    process_func: Callable[..., None],
    args: tuple[object, ...],
    job_manager: JobManager,
    websocket_manager: WebSocketManager,
    scheduler_service: "SchedulerService",  # Changed to string literal
) -> Process:
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
        The created Process object
    """
    # Create a queue for status updates
    status_queue = multiprocessing.Queue()

    # Create a process for the job, adding the status queue to the arguments
    process_args = args + (status_queue,)
    process = Process(target=process_func, args=process_args)

    # Register the process with the job manager before starting it
    # This avoids any potential issues with asyncio.Task objects being pickled
    job_manager.register_process(job_id, process)

    # Start the process after registration
    process.start()

    # Create a task to monitor the status queue
    async def monitor_status_queue():
        current_job_id = job_id  # Capture job_id in closure to avoid unbound variable issue
        try:
            while process.is_alive() or not status_queue.empty():
                if not status_queue.empty():
                    message = status_queue.get()

                    # Handle different message types
                    if isinstance(message, tuple):
                        # Check message format based on first element
                        if len(message) >= 2 and isinstance(message[0], str):
                            msg_type = message[0]

                            if msg_type == "status_update" and len(message) == 4:
                                # Status update message: (type, job_id, status, result)
                                _, received_job_id, status, result = message
                                await job_manager.update_job_status(received_job_id, status, result)
                            elif msg_type == "execution_update" and len(message) == 3:
                                # Execution state update: (type, job_id, execution_state)
                                _, received_job_id, execution_state = message
                                await job_manager.update_job_execution_state(
                                    received_job_id, execution_state
                                )
                            elif msg_type == "message_update" and len(message) == 3:
                                # Message update: (type, job_id, message)
                                _, received_job_id, message = message

                                await websocket_manager.broadcast_update(received_job_id, message)
                            elif msg_type == "schedule_add" and len(message) == 2:
                                # Schedule add message: (type, schedule)
                                _, schedule = message

                                if schedule is not None and isinstance(schedule, Schedule):
                                    try:
                                        scheduler_service.add_or_update_job(schedule)
                                    except Exception as e:
                                        logger.error(
                                            f"Failed to add schedule via status_queue: {e}"
                                        )
                                else:
                                    logger.error(
                                        "schedule_add message did not contain a valid"
                                        f"Schedule object: {schedule}"
                                    )
                            elif msg_type == "schedule_remove" and len(message) == 2:
                                # Schedule remove message: (type, schedule_id)
                                _, schedule_id = message

                                if schedule_id is not None and (
                                    isinstance(schedule_id, UUID) or isinstance(schedule_id, str)
                                ):
                                    try:
                                        if isinstance(schedule_id, str):
                                            schedule_id_uuid = UUID(schedule_id)
                                        else:
                                            schedule_id_uuid = schedule_id
                                        scheduler_service.remove_job(schedule_id_uuid)
                                    except Exception as e:
                                        logger.error(
                                            f"Failed to remove schedule via status_queue: {e}"
                                        )
                                else:
                                    logger.error(
                                        "schedule_remove message did not contain a valid"
                                        f"schedule_id: {schedule_id}"
                                    )
                        elif len(message) == 3:
                            # Legacy format: (job_id, status, result)
                            received_job_id, status, result = message
                            await job_manager.update_job_status(received_job_id, status, result)
                        else:
                            logger.warning(f"Received message with unexpected format: {message}")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            pass
        except Exception as e:
            logger.exception(f"Error monitoring status queue for job {current_job_id}: {str(e)}")

    # Start the monitor task
    monitor_task = asyncio.create_task(monitor_status_queue())

    # Register the task with the job manager
    # Use a separate function to avoid capturing the monitor_task in the closure
    async def register_monitor_task():
        await job_manager.register_task(job_id, monitor_task)

    # Create a separate task for registration to avoid pickling issues
    asyncio.create_task(register_monitor_task())

    # Return only the process to avoid pickling the asyncio.Task
    return process
