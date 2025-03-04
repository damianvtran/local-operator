import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from local_operator.jobs import Job, JobManager, JobResult, JobStatus


@pytest.fixture
def job_manager():
    return JobManager()


@pytest.fixture
def sample_job():
    return Job(prompt="Test prompt", model="gpt-4", hosting="openai", agent_id="test-agent")


@pytest.fixture
def sample_job_result():
    return JobResult(
        response="Test response",
        context=[{"role": "user", "content": "Test prompt"}],
        stats={"total_tokens": 100},
    )


@pytest.mark.asyncio
async def test_create_job(job_manager):
    job = await job_manager.create_job(
        prompt="Test prompt", model="gpt-4", hosting="openai", agent_id="test-agent"
    )

    assert job.id in job_manager.jobs
    assert job.prompt == "Test prompt"
    assert job.model == "gpt-4"
    assert job.hosting == "openai"
    assert job.agent_id == "test-agent"
    assert job.status == JobStatus.PENDING
    assert job.created_at is not None
    assert job.started_at is None
    assert job.completed_at is None
    assert job.result is None


@pytest.mark.asyncio
async def test_get_job(job_manager, sample_job):
    job_manager.jobs[sample_job.id] = sample_job

    retrieved_job = await job_manager.get_job(sample_job.id)
    assert retrieved_job == sample_job

    with pytest.raises(KeyError, match='Job with ID "nonexistent-id" not found'):
        await job_manager.get_job("nonexistent-id")


@pytest.mark.asyncio
async def test_update_job_status(job_manager, sample_job, sample_job_result):
    job_manager.jobs[sample_job.id] = sample_job

    # Test updating to processing
    updated_job = await job_manager.update_job_status(sample_job.id, JobStatus.PROCESSING)
    assert updated_job.status == JobStatus.PROCESSING
    assert updated_job.started_at is not None
    assert updated_job.completed_at is None

    # Test updating to completed with result
    updated_job = await job_manager.update_job_status(
        sample_job.id, JobStatus.COMPLETED, sample_job_result
    )
    assert updated_job.status == JobStatus.COMPLETED
    assert updated_job.completed_at is not None
    assert updated_job.result == sample_job_result

    # Test updating nonexistent job
    with pytest.raises(KeyError, match='Job with ID "nonexistent-id" not found'):
        await job_manager.update_job_status("nonexistent-id", JobStatus.COMPLETED)

    # Test updating with dict result
    job2 = Job(prompt="Test", model="gpt-4", hosting="openai")
    job_manager.jobs[job2.id] = job2
    result_dict = {"response": "Test response", "error": None}
    updated_job = await job_manager.update_job_status(job2.id, JobStatus.COMPLETED, result_dict)
    assert updated_job.result.response == "Test response"


@pytest.mark.asyncio
async def test_register_task(job_manager, sample_job):
    job_manager.jobs[sample_job.id] = sample_job

    mock_task = AsyncMock(spec=asyncio.Task)
    updated_job = await job_manager.register_task(sample_job.id, mock_task)

    assert updated_job.task == mock_task

    # Test registering task for nonexistent job
    with pytest.raises(KeyError, match='Job with ID "nonexistent-id" not found'):
        await job_manager.register_task("nonexistent-id", mock_task)


@pytest.mark.asyncio
async def test_cancel_job(job_manager, sample_job):
    # Setup job with a mock task
    mock_task = AsyncMock(spec=asyncio.Task)
    mock_task.done.return_value = False
    sample_job.task = mock_task
    sample_job.status = JobStatus.PROCESSING
    job_manager.jobs[sample_job.id] = sample_job

    # Test successful cancellation
    result = await job_manager.cancel_job(sample_job.id)
    assert result is True
    mock_task.cancel.assert_called_once()
    assert job_manager.jobs[sample_job.id].status == JobStatus.CANCELLED

    # Test cancelling nonexistent job
    with pytest.raises(KeyError, match='Job with ID "nonexistent-id" not found'):
        await job_manager.cancel_job("nonexistent-id")

    # Test cancelling already completed job
    completed_job = Job(prompt="Test", model="gpt-4", hosting="openai", status=JobStatus.COMPLETED)
    job_manager.jobs[completed_job.id] = completed_job
    result = await job_manager.cancel_job(completed_job.id)
    assert result is False


@pytest.mark.asyncio
async def test_list_jobs(job_manager):
    # Create test jobs with different statuses and agent_ids
    job1 = Job(
        prompt="Test1",
        model="gpt-4",
        hosting="openai",
        agent_id="agent1",
        status=JobStatus.COMPLETED,
    )
    job2 = Job(
        prompt="Test2",
        model="gpt-4",
        hosting="openai",
        agent_id="agent1",
        status=JobStatus.PROCESSING,
    )
    job3 = Job(
        prompt="Test3",
        model="gpt-4",
        hosting="openai",
        agent_id="agent2",
        status=JobStatus.COMPLETED,
    )

    job_manager.jobs = {job1.id: job1, job2.id: job2, job3.id: job3}

    # Test listing all jobs
    all_jobs = await job_manager.list_jobs()
    assert len(all_jobs) == 3

    # Test filtering by agent_id
    agent1_jobs = await job_manager.list_jobs(agent_id="agent1")
    assert len(agent1_jobs) == 2
    assert all(job.agent_id == "agent1" for job in agent1_jobs)

    # Test filtering by status
    completed_jobs = await job_manager.list_jobs(status=JobStatus.COMPLETED)
    assert len(completed_jobs) == 2
    assert all(job.status == JobStatus.COMPLETED for job in completed_jobs)

    # Test filtering by both agent_id and status
    filtered_jobs = await job_manager.list_jobs(agent_id="agent1", status=JobStatus.COMPLETED)
    assert len(filtered_jobs) == 1
    assert filtered_jobs[0].agent_id == "agent1"
    assert filtered_jobs[0].status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_cleanup_old_jobs(job_manager):
    current_time = time.time()

    # Create jobs with different ages
    old_completed_job = Job(
        prompt="Old completed", model="gpt-4", hosting="openai", status=JobStatus.COMPLETED
    )
    old_completed_job.created_at = current_time - 30 * 3600  # 30 hours old
    old_completed_job.completed_at = current_time - 25 * 3600  # completed 25 hours ago

    recent_completed_job = Job(
        prompt="Recent completed", model="gpt-4", hosting="openai", status=JobStatus.COMPLETED
    )
    recent_completed_job.created_at = current_time - 30 * 3600  # 30 hours old
    recent_completed_job.completed_at = current_time - 10 * 3600  # completed 10 hours ago

    old_pending_job = Job(
        prompt="Old pending", model="gpt-4", hosting="openai", status=JobStatus.PENDING
    )
    old_pending_job.created_at = current_time - 30 * 3600  # 30 hours old

    old_processing_job = Job(
        prompt="Old processing", model="gpt-4", hosting="openai", status=JobStatus.PROCESSING
    )
    old_processing_job.created_at = current_time - 30 * 3600  # 30 hours old
    mock_task = AsyncMock(spec=asyncio.Task)
    mock_task.done.return_value = False
    old_processing_job.task = mock_task

    recent_job = Job(prompt="Recent", model="gpt-4", hosting="openai")
    recent_job.created_at = current_time - 10 * 3600  # 10 hours old

    job_manager.jobs = {
        old_completed_job.id: old_completed_job,
        recent_completed_job.id: recent_completed_job,
        old_pending_job.id: old_pending_job,
        old_processing_job.id: old_processing_job,
        recent_job.id: recent_job,
    }

    # Test cleanup with default 24 hours max age
    removed_count = await job_manager.cleanup_old_jobs()
    assert (
        removed_count == 4
    )  # should remove old_completed_job, recent_completed_job, old_pending_job, old_processing_job
    assert len(job_manager.jobs) == 1
    assert old_completed_job.id not in job_manager.jobs
    assert recent_completed_job.id not in job_manager.jobs
    assert old_pending_job.id not in job_manager.jobs
    assert old_processing_job.id not in job_manager.jobs
    assert recent_job.id in job_manager.jobs

    # Verify task was cancelled
    mock_task.cancel.assert_called_once()


def test_get_job_summary(job_manager, sample_job, sample_job_result):
    # Setup job with all fields populated
    sample_job.started_at = time.time() - 3600  # 1 hour ago
    sample_job.completed_at = time.time() - 1800  # 30 minutes ago
    sample_job.result = sample_job_result

    summary = job_manager.get_job_summary(sample_job)

    assert summary["id"] == sample_job.id
    assert summary["agent_id"] == sample_job.agent_id
    assert summary["status"] == sample_job.status.value
    assert summary["prompt"] == sample_job.prompt
    assert summary["model"] == sample_job.model
    assert summary["hosting"] == sample_job.hosting

    # Check datetime formatting
    assert isinstance(summary["created_at"], str)
    assert isinstance(summary["started_at"], str)
    assert isinstance(summary["completed_at"], str)

    # Check result serialization
    assert summary["result"] == sample_job_result.model_dump()

    # Test with missing optional fields
    minimal_job = Job(prompt="Test", model="gpt-4", hosting="openai")
    minimal_summary = job_manager.get_job_summary(minimal_job)
    assert minimal_summary["started_at"] is None
    assert minimal_summary["completed_at"] is None
    assert minimal_summary["result"] is None


@pytest.mark.asyncio
async def test_job_validator():
    # Test valid task
    mock_task = AsyncMock(spec=asyncio.Task)
    job = Job(prompt="Test", model="gpt-4", hosting="openai", task=mock_task)
    assert job.task == mock_task

    # Test invalid task
    with pytest.raises(ValueError):
        Job(prompt="Test", model="gpt-4", hosting="openai", task="not a task")  # type: ignore
