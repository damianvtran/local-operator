"""
Job processing manager for Local Operator.

This module provides functionality to track and manage asynchronous jobs
for the Local Operator, including their status, associated agents, and timing information.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("local_operator.jobs")


class JobStatus(str, Enum):
    """Enum representing the possible states of a job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobResult(BaseModel):
    """Model representing the result of a completed job."""

    response: Optional[str] = None
    context: Optional[List[Dict[str, str]]] = None
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

    class Config:
        arbitrary_types_allowed = True

    @field_validator("task", mode="before")
    def validate_task(cls, v):
        """Validate that the task is an asyncio.Task or None."""
        if v is not None and not isinstance(v, asyncio.Task):
            raise ValueError("task must be an asyncio.Task")
        return v


class JobManager:
    """
    Manager for tracking and handling asynchronous jobs.

    This class provides methods to create, retrieve, update, and manage jobs
    throughout their lifecycle.
    """

    def __init__(self):
        """Initialize the JobManager with an empty jobs dictionary."""
        self.jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create_job(
        self, prompt: str, model: str, hosting: str, agent_id: Optional[str] = None
    ) -> Job:
        """
        Create a new job and add it to the manager.

        Args:
            prompt: The user prompt for this job
            model: The model being used
            hosting: The hosting provider
            agent_id: Optional ID of the associated agent

        Returns:
            The created Job object
        """
        job = Job(prompt=prompt, model=model, hosting=hosting, agent_id=agent_id)

        async with self._lock:
            self.jobs[job.id] = job

        return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Retrieve a job by its ID.

        Args:
            job_id: The ID of the job to retrieve

        Returns:
            The Job object if found, None otherwise
        """
        return self.jobs.get(job_id)

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[Union[Dict[str, Any], JobResult]] = None,
    ) -> Optional[Job]:
        """
        Update the status and optionally the result of a job.

        Args:
            job_id: The ID of the job to update
            status: The new status of the job
            result: Optional result data for the job

        Returns:
            The updated Job object if found, None otherwise
        """
        job = await self.get_job(job_id)
        if not job:
            return None

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

    async def register_task(self, job_id: str, task: asyncio.Task[Any]) -> Optional[Job]:
        """
        Register an asyncio task with a job.

        Args:
            job_id: The ID of the job
            task: The asyncio task to register

        Returns:
            The updated Job object if found, None otherwise
        """
        job = await self.get_job(job_id)
        if not job:
            return None

        async with self._lock:
            job.task = task

        return job

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: The ID of the job to cancel

        Returns:
            True if the job was successfully cancelled, False otherwise
        """
        job = await self.get_job(job_id)
        if not job or job.status not in (JobStatus.PENDING, JobStatus.PROCESSING):
            return False

        if job.task and not job.task.done():
            job.task.cancel()

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
        result = []

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
        jobs_to_remove = []

        for job_id, job in self.jobs.items():
            # For completed jobs, check against completion time
            if job.completed_at and (current_time - job.completed_at) > max_age_seconds:
                jobs_to_remove.append(job_id)
            # For other jobs, check against creation time
            elif (current_time - job.created_at) > max_age_seconds:
                # Cancel if still running
                if job.status in (JobStatus.PENDING, JobStatus.PROCESSING) and job.task:
                    if not job.task.done():
                        job.task.cancel()
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
        return {
            "job_id": job.id,
            "agent_id": job.agent_id,
            "status": job.status,
            "created_at": datetime.fromtimestamp(job.created_at).isoformat(),
            "started_at": (
                datetime.fromtimestamp(job.started_at).isoformat() if job.started_at else None
            ),
            "completed_at": (
                datetime.fromtimestamp(job.completed_at).isoformat() if job.completed_at else None
            ),
            "result": job.result.dict() if job.result else None,
            "prompt": job.prompt,
            "model": job.model,
            "hosting": job.hosting,
        }
