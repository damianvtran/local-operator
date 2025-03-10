"""
Tests for the job endpoints of the FastAPI server.

This module contains tests for job-related functionality, including
retrieving, listing, cancelling, and cleaning up jobs.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from local_operator.jobs import Job, JobContextRecord, JobResult, JobStatus
from local_operator.types import ConversationRole


@pytest.fixture
def sample_job():
    """Create a sample job for testing."""
    now = datetime.now(timezone.utc)
    return Job(
        id="test-job-123",
        agent_id="test-agent-456",
        prompt="Test prompt",
        model="gpt-4",
        hosting="openai",
        status=JobStatus.COMPLETED,
        created_at=(now - timedelta(minutes=15)).timestamp(),
        started_at=(now - timedelta(minutes=14)).timestamp(),
        completed_at=(now - timedelta(minutes=13)).timestamp(),
        result=JobResult(
            response="Test response",
            context=[JobContextRecord(role=ConversationRole.USER, content="Test prompt", files=[])],
            stats={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        ),
    )


@pytest.mark.asyncio
async def test_get_job_status_success(test_app_client, mock_job_manager, sample_job):
    """Test retrieving a job's status successfully."""
    job_id = sample_job.id
    mock_job_manager.get_job.return_value = sample_job
    mock_job_manager.get_job_summary.return_value = {
        "id": job_id,
        "agent_id": sample_job.agent_id,
        "status": sample_job.status.value,
        "prompt": sample_job.prompt,
        "model": sample_job.model,
        "hosting": sample_job.hosting,
        "created_at": datetime.fromtimestamp(sample_job.created_at, tz=timezone.utc).isoformat(),
        "started_at": datetime.fromtimestamp(sample_job.started_at, tz=timezone.utc).isoformat(),
        "completed_at": datetime.fromtimestamp(
            sample_job.completed_at, tz=timezone.utc
        ).isoformat(),
        "result": sample_job.result.model_dump(),
    }

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.get(f"/v1/jobs/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Job status retrieved"
    result = data["result"]
    assert result["id"] == job_id
    assert result["agent_id"] == sample_job.agent_id
    assert result["status"] == sample_job.status.value
    assert result["prompt"] == sample_job.prompt
    assert result["model"] == sample_job.model
    assert result["hosting"] == sample_job.hosting
    assert "created_at" in result
    assert "started_at" in result
    assert "completed_at" in result
    assert result["result"]["response"] == "Test response"


@pytest.mark.asyncio
async def test_get_job_status_not_found(test_app_client, mock_job_manager):
    """Test retrieving a non-existent job."""
    job_id = "nonexistent-job"
    mock_job_manager.get_job.side_effect = KeyError(f'Job with ID "{job_id}" not found')

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.get(f"/v1/jobs/{job_id}")

    assert response.status_code == 404
    data = response.json()
    assert f'Job with ID "{job_id}" not found' in data.get("detail", "")


@pytest.mark.asyncio
async def test_get_job_status_error(test_app_client, mock_job_manager):
    """Test error handling when retrieving a job."""
    job_id = "error-job"
    mock_job_manager.get_job.side_effect = Exception("Test error")

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.get(f"/v1/jobs/{job_id}")

    assert response.status_code == 500
    data = response.json()
    assert "Internal Server Error" in data.get("detail", "")


@pytest.mark.asyncio
async def test_list_jobs_success(test_app_client, mock_job_manager, sample_job):
    """Test listing jobs successfully."""
    mock_job_manager.list_jobs.return_value = [sample_job]
    mock_job_manager.get_job_summary.return_value = {
        "id": sample_job.id,
        "agent_id": sample_job.agent_id,
        "status": sample_job.status.value,
        "prompt": sample_job.prompt,
        "model": sample_job.model,
        "hosting": sample_job.hosting,
        "created_at": datetime.fromtimestamp(sample_job.created_at, tz=timezone.utc).isoformat(),
        "started_at": datetime.fromtimestamp(sample_job.started_at, tz=timezone.utc).isoformat(),
        "completed_at": datetime.fromtimestamp(
            sample_job.completed_at, tz=timezone.utc
        ).isoformat(),
        "result": sample_job.result.model_dump() if sample_job.result else None,
    }

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.get("/v1/jobs")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Jobs retrieved successfully"
    result = data["result"]
    assert result["count"] == 1
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["id"] == sample_job.id


@pytest.mark.asyncio
async def test_list_jobs_with_filters(test_app_client, mock_job_manager, sample_job):
    """Test listing jobs with agent_id and status filters."""
    mock_job_manager.list_jobs.return_value = [sample_job]
    mock_job_manager.get_job_summary.return_value = {
        "id": sample_job.id,
        "agent_id": sample_job.agent_id,
        "status": sample_job.status.value,
        "prompt": sample_job.prompt,
        "model": sample_job.model,
        "hosting": sample_job.hosting,
        "created_at": datetime.fromtimestamp(sample_job.created_at, tz=timezone.utc).isoformat(),
        "started_at": datetime.fromtimestamp(sample_job.started_at, tz=timezone.utc).isoformat(),
        "completed_at": datetime.fromtimestamp(
            sample_job.completed_at, tz=timezone.utc
        ).isoformat(),
        "result": sample_job.result.model_dump() if sample_job.result else None,
    }

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.get(
            f"/v1/jobs?agent_id={sample_job.agent_id}&status={sample_job.status.value}"
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    mock_job_manager.list_jobs.assert_called_once_with(
        agent_id=sample_job.agent_id, status=sample_job.status
    )


@pytest.mark.asyncio
async def test_list_jobs_error(test_app_client, mock_job_manager):
    """Test error handling when listing jobs."""
    mock_job_manager.list_jobs.side_effect = Exception("Test error")

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.get("/v1/jobs")

    assert response.status_code == 500
    data = response.json()
    assert "Internal Server Error" in data.get("detail", "")


@pytest.mark.asyncio
async def test_cancel_job_success(test_app_client, mock_job_manager, sample_job):
    """Test cancelling a job successfully."""
    job_id = sample_job.id
    mock_job_manager.cancel_job.return_value = True

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.delete(f"/v1/jobs/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == f"Job {job_id} cancelled successfully"
    mock_job_manager.cancel_job.assert_called_once_with(job_id)


@pytest.mark.asyncio
async def test_cancel_job_not_found(test_app_client, mock_job_manager):
    """Test cancelling a non-existent job."""
    job_id = "nonexistent-job"
    mock_job_manager.cancel_job.side_effect = KeyError(f'Job with ID "{job_id}" not found')

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.delete(f"/v1/jobs/{job_id}")

    assert response.status_code == 404
    data = response.json()
    assert f'Job with ID "{job_id}" not found' in data.get("detail", "")


@pytest.mark.asyncio
async def test_cancel_job_cannot_cancel(test_app_client, mock_job_manager, sample_job):
    """Test cancelling a job that cannot be cancelled."""
    job_id = sample_job.id
    mock_job_manager.cancel_job.return_value = False

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.delete(f"/v1/jobs/{job_id}")

    assert response.status_code == 400
    data = response.json()
    assert f"Job {job_id} cannot be cancelled" in data.get("detail", "")


@pytest.mark.asyncio
async def test_cancel_job_error(test_app_client, mock_job_manager, sample_job):
    """Test error handling when cancelling a job."""
    job_id = sample_job.id
    mock_job_manager.cancel_job.side_effect = Exception("Test error")

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.delete(f"/v1/jobs/{job_id}")

    assert response.status_code == 500
    data = response.json()
    assert "Internal Server Error" in data.get("detail", "")


@pytest.mark.asyncio
async def test_cleanup_jobs_success(test_app_client, mock_job_manager):
    """Test cleaning up old jobs successfully."""
    mock_job_manager.cleanup_old_jobs.return_value = 5

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.post("/v1/jobs/cleanup?max_age_hours=48")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Cleanup completed successfully"
    assert data["result"]["removed_count"] == 5
    mock_job_manager.cleanup_old_jobs.assert_called_once_with(max_age_hours=48)


@pytest.mark.asyncio
async def test_cleanup_jobs_default_age(test_app_client, mock_job_manager):
    """Test cleaning up old jobs with default age parameter."""
    mock_job_manager.cleanup_old_jobs.return_value = 3

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.post("/v1/jobs/cleanup")

    assert response.status_code == 200
    data = response.json()
    assert data["result"]["removed_count"] == 3
    mock_job_manager.cleanup_old_jobs.assert_called_once_with(max_age_hours=24)


@pytest.mark.asyncio
async def test_cleanup_jobs_error(test_app_client, mock_job_manager):
    """Test error handling when cleaning up jobs."""
    mock_job_manager.cleanup_old_jobs.side_effect = Exception("Test error")

    with patch("local_operator.server.routes.jobs.get_job_manager", return_value=mock_job_manager):
        response = await test_app_client.post("/v1/jobs/cleanup")

    assert response.status_code == 500
    data = response.json()
    assert "Internal Server Error" in data.get("detail", "")
