import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from local_operator.jobs import JobManager, JobStatus
from local_operator.server.dependencies import get_job_manager
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Jobs"])
logger = logging.getLogger("local_operator.server.routes.jobs")


@router.get(
    "/v1/jobs/{job_id}",
    summary="Get job status",
    description="Retrieves the status and result of an asynchronous job.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Job status and result retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Job status retrieved",
                            "result": {
                                "id": "job-123456",
                                "agent_id": "test-agent",
                                "status": "completed",
                                "prompt": "Test prompt",
                                "model": "gpt-4",
                                "hosting": "openai",
                                "created_at": "2023-01-01T12:00:00Z",
                                "started_at": "2023-01-01T12:00:05Z",
                                "completed_at": "2023-01-01T12:00:15Z",
                                "result": {
                                    "response": "Test response",
                                    "context": [{"role": "user", "content": "Test prompt"}],
                                    "stats": {"total_tokens": 100},
                                },
                                "current_execution": {
                                    "id": "execution-123456",
                                    "stdout": "Hello, world!",
                                    "stderr": "",
                                    "logging": "",
                                    "message": "Code executed successfully",
                                    "code": "print('Hello, world!')",
                                    "formatted_print": "Hello, world!",
                                    "role": "assistant",
                                    "status": "success",
                                    "timestamp": "2023-01-01T12:00:05Z",
                                    "files": [],
                                    "action": "CODE",
                                    "execution_type": "action",
                                    "task_classification": "software_development",
                                },
                            },
                        }
                    }
                },
            },
            "404": {
                "description": "Job not found",
                "content": {
                    "application/json": {
                        "example": {"detail": 'Job with ID "job-123456" not found'}
                    }
                },
            },
            "500": {
                "description": "Internal Server Error",
                "content": {"application/json": {"example": {"detail": "Internal Server Error"}}},
            },
        }
    },
)
async def get_job_status(
    job_id: str = Path(..., description="The ID of the chat job to retrieve"),
    job_manager: JobManager = Depends(get_job_manager),
):
    """
    Get the status and result of an asynchronous chat job.

    Args:
        job_id: The ID of the job to check
        job_manager: The job manager instance

    Returns:
        The job status and result if available

    Raises:
        HTTPException: If the job is not found or there's an error retrieving it
    """
    try:
        job = await job_manager.get_job(job_id)

        job_summary = job_manager.get_job_summary(job)

        return CRUDResponse(
            status=200,
            message="Job status retrieved",
            result=job_summary,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f'Job with ID "{job_id}" not found')
    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception:
        logger.exception(f"Unexpected error while retrieving job {job_id}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get(
    "/v1/jobs",
    summary="List jobs",
    description="Lists all jobs, optionally filtered by agent ID and/or status.",
    openapi_extra={
        "parameters": [
            {
                "name": "agent_id",
                "in": "query",
                "description": "Filter jobs by agent ID",
                "required": False,
                "schema": {"type": "string"},
            },
            {
                "name": "status",
                "in": "query",
                "description": "Filter jobs by status",
                "required": False,
                "schema": {
                    "type": "string",
                    "enum": ["pending", "processing", "completed", "failed", "cancelled"],
                },
            },
        ],
        "responses": {
            "200": {
                "description": "List of jobs matching the filter criteria",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Jobs retrieved successfully",
                            "result": {
                                "jobs": [
                                    {
                                        "id": "job-123456",
                                        "agent_id": "test-agent",
                                        "status": "completed",
                                        "prompt": "Test prompt",
                                        "model": "gpt-4",
                                        "hosting": "openai",
                                        "created_at": "2023-01-01T12:00:00Z",
                                        "started_at": "2023-01-01T12:00:05Z",
                                        "completed_at": "2023-01-01T12:00:15Z",
                                    }
                                ],
                                "count": 1,
                            },
                        }
                    }
                },
            },
            "500": {
                "description": "Internal Server Error",
                "content": {"application/json": {"example": {"detail": "Internal Server Error"}}},
            },
        },
    },
)
async def list_jobs(
    agent_id: Optional[str] = Query(None, description="Filter jobs by agent ID"),
    status: Optional[JobStatus] = Query(None, description="Filter jobs by status"),
    job_manager: JobManager = Depends(get_job_manager),
):
    """
    List all jobs with optional filtering.

    Args:
        agent_id: Optional agent ID to filter by
        status: Optional status to filter by
        job_manager: The job manager instance

    Returns:
        List of job summaries matching the filter criteria

    Raises:
        HTTPException: If there's an error retrieving the jobs
    """
    try:
        jobs = await job_manager.list_jobs(agent_id=agent_id, status=status)
        job_summaries = [job_manager.get_job_summary(job) for job in jobs]

        return CRUDResponse(
            status=200,
            message="Jobs retrieved successfully",
            result={
                "jobs": job_summaries,
                "count": len(job_summaries),
            },
        )
    except Exception:
        logger.exception("Unexpected error while listing jobs")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.delete(
    "/v1/jobs/{job_id}",
    summary="Cancel job",
    description="Cancels a running or pending job.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Job cancelled successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Job job-123456 cancelled successfully",
                        }
                    }
                },
            },
            "400": {
                "description": "Job cannot be cancelled",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "Job job-123456 cannot be cancelled (already "
                            "completed or failed)"
                        }
                    }
                },
            },
            "404": {
                "description": "Job not found",
                "content": {
                    "application/json": {
                        "example": {"detail": 'Job with ID "job-123456" not found'}
                    }
                },
            },
            "500": {
                "description": "Internal Server Error",
                "content": {"application/json": {"example": {"detail": "Internal Server Error"}}},
            },
        }
    },
)
async def cancel_job(
    job_id: str = Path(..., description="The ID of the job to cancel"),
    job_manager: JobManager = Depends(get_job_manager),
):
    """
    Cancel a running or pending job.

    Args:
        job_id: The ID of the job to cancel
        job_manager: The job manager instance

    Returns:
        Confirmation of job cancellation

    Raises:
        HTTPException: If the job is not found or cannot be cancelled
    """
    try:
        # The get_job check is redundant as cancel_job will raise KeyError if job not found
        # We'll keep the try/except for KeyError instead
        success = await job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} cannot be cancelled (already completed or failed)",
            )

        return CRUDResponse(
            status=200,
            message=f"Job {job_id} cancelled successfully",
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f'Job with ID "{job_id}" not found')
    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception:
        logger.exception(f"Unexpected error while cancelling job {job_id}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post(
    "/v1/jobs/cleanup",
    summary="Cleanup old jobs",
    description="Removes jobs older than the specified age.",
    openapi_extra={
        "parameters": [
            {
                "name": "max_age_hours",
                "in": "query",
                "description": "Maximum age of jobs to keep in hours",
                "required": False,
                "schema": {"type": "integer", "default": 24},
            }
        ],
        "responses": {
            "200": {
                "description": "Jobs cleaned up successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Cleanup completed successfully",
                            "result": {"removed_count": 5},
                        }
                    }
                },
            },
            "500": {
                "description": "Internal Server Error",
                "content": {"application/json": {"example": {"detail": "Internal Server Error"}}},
            },
        },
    },
)
async def cleanup_jobs(
    max_age_hours: int = Query(24, description="Maximum age of jobs to keep in hours"),
    job_manager: JobManager = Depends(get_job_manager),
):
    """
    Clean up old jobs from the job manager.

    Args:
        max_age_hours: Maximum age of jobs to keep in hours
        job_manager: The job manager instance

    Returns:
        Number of jobs removed

    Raises:
        HTTPException: If there's an error during job cleanup
    """
    try:
        removed_count = await job_manager.cleanup_old_jobs(max_age_hours=max_age_hours)
        return CRUDResponse(
            status=200,
            message="Cleanup completed successfully",
            result={
                "removed_count": removed_count,
            },
        )
    except Exception:
        logger.exception("Unexpected error during job cleanup")
        raise HTTPException(status_code=500, detail="Internal Server Error")
