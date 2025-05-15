"""
Job processing manager for Local Operator.

This module provides functionality to track and manage asynchronous jobs
for the Local Operator, including their status, associated agents, and timing information.
It supports running jobs in isolated contexts that can change working directories
without affecting the parent process.
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from multiprocessing import Process
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

from pydantic import BaseModel, Field, field_validator

from local_operator.types import CodeExecutionResult, ConversationRole

logger = logging.getLogger("local_operator.jobs")

T = TypeVar("T")


class JobStatus(str, Enum):
    """Enum representing the possible states of a job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobContextRecord(BaseModel):
    """Model representing a record of a job context."""

    role: ConversationRole
    content: str
    files: Optional[List[str]] = None


class JobResult(BaseModel):
    """Model representing the result of a completed job."""

    response: Optional[str] = None
    context: Optional[List[JobContextRecord]] = None
    stats: Optional[Dict[str, int]] = None
    error: Optional[str] = None


class Job(BaseModel):
    """Model representing a job in the system."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: Optional[str] = None
    status: JobStatus = Field(default=JobStatus.PENDING)
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[JobResult] = None
    task: Optional[asyncio.Task[Any]] = None
    prompt: str
    model: str
    hosting: str
    current_execution: Optional[CodeExecutionResult] = None

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @field_validator("task", mode="before")
    def validate_task(cls, v: Any) -> Optional[asyncio.Task[Any]]:
        """Validate that the task is an asyncio.Task or None."""
        if v is not None and not isinstance(v, asyncio.Task):
            raise ValueError("task must be an asyncio.Task")
        return v


class JobContext:
    """
    Context for running a job in an isolated environment.

    This class provides an isolated context for a job to run in, allowing it to
    change working directories without affecting the parent process. When used with
    a context manager (with statement), it automatically saves the original working
    directory and restores it when the context is exited, ensuring that the parent
    process's working directory remains unchanged.

    Example usage:
        ```python
        # Create a new job context
        job_context = JobContext()

        # Use the context manager to ensure the original directory is restored
        with job_context:
            # Change to the agent's working directory if needed
            if agent.current_working_directory != ".":
                job_context.change_directory(agent.current_working_directory)

            # Run operations in the agent's working directory
            # ...

        # After the with block, we're back in the original directory
        ```
    """

    def __init__(self) -> None:
        """Initialize the JobContext by saving the current working directory."""
        self.original_cwd = os.getcwd()

    def __enter__(self) -> "JobContext":
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context, restoring the original working directory."""
        os.chdir(self.original_cwd)

    def change_directory(self, path: str) -> None:
        """
        Change the working directory for this context.

        This method changes the current working directory to the specified path.
        The original working directory will still be restored when the context is exited.

        Args:
            path: The path to change to
        """
        expanded_path = os.path.expanduser(path)
        os.chdir(expanded_path)


class JobManager:
    """
    Manager for tracking and handling asynchronous jobs.

    This class provides methods to create, retrieve, update, and manage jobs
    throughout their lifecycle. Jobs can run in isolated contexts that can
    change working directories without affecting the parent process.
    """

    def __init__(self) -> None:
        """Initialize the JobManager with an empty jobs dictionary."""
        self.jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._processes: Dict[str, Process] = {}

    async def create_job(
        self,
        prompt: str,
        model: str,
        hosting: str,
        agent_id: Optional[str] = None,
        job_id: Optional[str] = None,  # Added optional job_id
    ) -> Job:
        """
        Create a new job and add it to the manager.

        Args:
            prompt: The user prompt for this job
            model: The model being used
            hosting: The hosting provider
            agent_id: Optional ID of the associated agent
            job_id: Optional ID to use for the job. If None, a new UUID is generated.

        Returns:
            The created Job object
        """
        # Use provided job_id or generate a new one
        effective_job_id = job_id if job_id else str(uuid.uuid4())
        job = Job(
            id=effective_job_id,  # Use effective_job_id
            prompt=prompt,
            model=model,
            hosting=hosting,
            agent_id=agent_id,
        )

        async with self._lock:
            self.jobs[job.id] = job

        return job

    async def get_job(self, job_id: str) -> Job:
        """
        Retrieve a job by its ID.

        Args:
            job_id: The ID of the job to retrieve

        Returns:
            The Job object

        Raises:
            KeyError: If the job with the specified ID is not found
        """
        if job_id not in self.jobs:
            raise KeyError(f'Job with ID "{job_id}" not found')
        return self.jobs[job_id]

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[Union[Dict[str, Any], JobResult]] = None,
    ) -> Job:
        """
        Update the status and optionally the result of a job.

        Args:
            job_id: The ID of the job to update
            status: The new status of the job
            result: Optional result data for the job

        Returns:
            The updated Job object

        Raises:
            KeyError: If the job with the specified ID is not found
        """
        try:
            job = await self.get_job(job_id)
        except KeyError:
            raise

        async with self._lock:
            job.status = status

            if status == JobStatus.PROCESSING and job.started_at is None:
                job.started_at = time.time()

            if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job.completed_at = time.time()

                if result:
                    if isinstance(result, dict):
                        job.result = JobResult(**result)
                    else:
                        job.result = result

        return job

    async def register_task(self, job_id: str, task: asyncio.Task[T]) -> Job:
        """
        Register an asyncio task with a job.

        Args:
            job_id: The ID of the job
            task: The asyncio task to register

        Returns:
            The updated Job object

        Raises:
            KeyError: If the job with the specified ID is not found
        """
        try:
            job = await self.get_job(job_id)
        except KeyError:
            raise

        async with self._lock:
            job.task = cast(asyncio.Task[Any], task)

        return job

    async def update_job_execution_state(
        self,
        job_id: str,
        execution_state: CodeExecutionResult,
    ) -> Job:
        """
        Update the current execution state of a job.

        Args:
            job_id: The ID of the job to update
            execution_state: The current execution state

        Returns:
            The updated Job object

        Raises:
            KeyError: If the job with the specified ID is not found
        """
        try:
            job = await self.get_job(job_id)
        except KeyError:
            raise

        async with self._lock:
            job.current_execution = execution_state

        return job

    def register_process(self, job_id: str, process: Process) -> None:
        """
        Register a multiprocessing Process with a job.

        Args:
            job_id: The ID of the job
            process: The Process to register
        """
        self._processes[job_id] = process

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        This method will terminate the process associated with the job if it exists,
        and cancel the asyncio task if it exists. It will also update the job status
        to CANCELLED.

        Args:
            job_id: The ID of the job to cancel

        Returns:
            True if the job was successfully cancelled, False otherwise

        Raises:
            KeyError: If the job with the specified ID is not found
        """
        try:
            job = await self.get_job(job_id)
        except KeyError:
            raise

        if job.status not in (JobStatus.PENDING, JobStatus.PROCESSING):
            return False

        # Cancel the asyncio task if it exists
        if job.task and not job.task.done():
            job.task.cancel()

        # Terminate the process if it exists
        if job_id in self._processes:
            process = self._processes[job_id]
            if process.is_alive():
                logger.info(f"Terminating process for job {job_id}")
                process.terminate()
                # Wait for the process to terminate
                process.join(timeout=5)
                # If the process is still alive, kill it
                if process.is_alive():
                    logger.warning(f"Process for job {job_id} did not terminate, killing it")
                    process.kill()
            # Remove the process from the dictionary
            del self._processes[job_id]

        await self.update_job_status(
            job_id, JobStatus.CANCELLED, {"error": "Job cancelled by user"}
        )
        return True

    async def list_jobs(
        self, agent_id: Optional[str] = None, status: Optional[JobStatus] = None
    ) -> List[Job]:
        """
        List jobs, optionally filtered by agent ID and/or status.

        Args:
            agent_id: Optional agent ID to filter by
            status: Optional status to filter by

        Returns:
            List of matching Job objects
        """
        result: List[Job] = []

        for job in self.jobs.values():
            if agent_id is not None and job.agent_id != agent_id:
                continue

            if status is not None and job.status != status:
                continue

            result.append(job)

        return result

    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Remove jobs older than the specified age.

        Args:
            max_age_hours: Maximum age of jobs to keep in hours

        Returns:
            Number of jobs removed
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        jobs_to_remove: List[str] = []

        for job_id, job in self.jobs.items():
            # For completed jobs, check against completion time
            if job.completed_at and (current_time - job.completed_at) > max_age_seconds:
                jobs_to_remove.append(job_id)
            # For other jobs, check against creation time
            elif (current_time - job.created_at) > max_age_seconds:
                # Cancel if still running
                if job.status in (JobStatus.PENDING, JobStatus.PROCESSING):
                    # Cancel the task if it exists
                    if job.task and not job.task.done():
                        job.task.cancel()

                    # Terminate the process if it exists
                    if job_id in self._processes:
                        process = self._processes[job_id]
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=5)
                            if process.is_alive():
                                process.kill()
                        del self._processes[job_id]

                jobs_to_remove.append(job_id)

        async with self._lock:
            for job_id in jobs_to_remove:
                del self.jobs[job_id]

        return len(jobs_to_remove)

    def get_job_summary(self, job: Job) -> Dict[str, Any]:
        """
        Create a summary dictionary of a job for API responses.

        Args:
            job: The Job object to summarize

        Returns:
            Dictionary with job summary information
        """
        summary = {
            "id": job.id,
            "agent_id": job.agent_id,
            "status": job.status.value,
            "created_at": datetime.fromtimestamp(job.created_at, tz=timezone.utc).isoformat(),
            "started_at": (
                datetime.fromtimestamp(job.started_at, tz=timezone.utc).isoformat()
                if job.started_at
                else None
            ),
            "completed_at": (
                datetime.fromtimestamp(job.completed_at, tz=timezone.utc).isoformat()
                if job.completed_at
                else None
            ),
            "result": job.result.model_dump() if job.result else None,
            "prompt": job.prompt,
            "model": job.model,
            "hosting": job.hosting,
        }

        if job.current_execution:
            summary["current_execution"] = job.current_execution.model_dump()

        return summary
