"""
Tests for job multiprocessing behavior.

This module contains tests to verify the behavior of job status updates
across process boundaries.
"""

import asyncio
import multiprocessing
from multiprocessing import Process, Queue
from typing import Any, Union

import pytest

from local_operator.jobs import JobManager, JobStatus


def update_job_status_in_process(
    job_id: str, job_manager: JobManager, queue: "Queue[Union[bool, str]]"
):
    """
    Update a job's status in a separate process.

    Args:
        job_id: The ID of the job to update
        job_manager: The job manager instance
        queue: A queue to communicate results back to the parent process
    """
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def update_status():
        try:
            # Update the job status to processing
            await job_manager.update_job_status(job_id, JobStatus.PROCESSING)

            # Get the job to verify the status was updated
            job = await job_manager.get_job(job_id)
            queue.put(job.status == JobStatus.PROCESSING)

            # Wait a bit to simulate processing
            await asyncio.sleep(0.1)

            # Update the job status to completed
            await job_manager.update_job_status(job_id, JobStatus.COMPLETED)

            # Get the job to verify the status was updated
            job = await job_manager.get_job(job_id)
            queue.put(job.status == JobStatus.COMPLETED)
        except Exception as e:
            queue.put(f"Error: {str(e)}")

    # Run the async function in the new event loop
    loop.run_until_complete(update_status())
    loop.close()


@pytest.mark.asyncio
async def test_job_status_update_across_processes():
    """
    Test that job status updates in a child process are not reflected in the parent process.

    This test demonstrates the issue with the current implementation where job status
    updates in a child process are not reflected in the parent process.
    """
    # Create a job manager
    job_manager = JobManager()

    # Create a job
    job = await job_manager.create_job(
        prompt="Test prompt", model="test-model", hosting="test-hosting"
    )

    # Create a queue for communication between processes
    queue = multiprocessing.Queue()

    # Create and start a process to update the job status
    process = Process(target=update_job_status_in_process, args=(job.id, job_manager, queue))
    process.start()

    # Wait for the process to complete
    process.join()

    # Get the results from the queue
    child_processing_status_updated = queue.get()
    child_completed_status_updated = queue.get()

    # Verify that the status was updated in the child process
    assert child_processing_status_updated is True
    assert child_completed_status_updated is True

    # Get the job from the parent process
    parent_job = await job_manager.get_job(job.id)

    # Verify that the status was NOT updated in the parent process
    # This is the key assertion that demonstrates the issue
    assert parent_job.status == JobStatus.PENDING


def update_job_status_with_shared_queue(
    job_id: str, status_queue: "Queue[tuple[str, JobStatus, Any]]"
):
    """
    Update a job's status using a shared queue to communicate with the parent process.

    Args:
        job_id: The ID of the job to update
        status_queue: A queue to communicate status updates to the parent process
    """
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def update_status():
        try:
            # Instead of updating the job manager directly, send status updates to the parent
            status_queue.put((job_id, JobStatus.PROCESSING, None))

            # Wait a bit to simulate processing
            await asyncio.sleep(0.1)

            # Send completed status
            result = {"response": "Test response"}
            status_queue.put((job_id, JobStatus.COMPLETED, result))
        except Exception as e:
            status_queue.put((job_id, JobStatus.FAILED, {"error": str(e)}))

    # Run the async function in the new event loop
    loop.run_until_complete(update_status())
    loop.close()


@pytest.mark.asyncio
async def test_job_status_update_with_shared_queue():
    """
    Test that job status updates can be communicated from a child process to the parent
    using a shared queue.

    This test demonstrates a solution to the issue where job status updates in a child
    process are not reflected in the parent process.
    """
    # Create a job manager
    job_manager = JobManager()

    # Create a job
    job = await job_manager.create_job(
        prompt="Test prompt", model="test-model", hosting="test-hosting"
    )

    # Create a queue for status updates
    status_queue = multiprocessing.Queue()

    # Create and start a process to update the job status
    process = Process(target=update_job_status_with_shared_queue, args=(job.id, status_queue))
    process.start()

    # Monitor the queue for status updates
    async def monitor_queue():
        while process.is_alive() or not status_queue.empty():
            if not status_queue.empty():
                job_id, status, result = status_queue.get()
                await job_manager.update_job_status(job_id, status, result)
            await asyncio.sleep(0.01)

    # Start monitoring the queue
    monitor_task = asyncio.create_task(monitor_queue())

    # Wait for the process to complete
    process.join()

    # Wait a bit for any remaining queue items to be processed
    await asyncio.sleep(0.2)

    # Cancel the monitor task
    monitor_task.cancel()

    # Get the job from the parent process
    parent_job = await job_manager.get_job(job.id)

    # Verify that the status was updated in the parent process
    assert parent_job.status == JobStatus.COMPLETED
    assert parent_job.result is not None
    assert parent_job.result.response == "Test response"
