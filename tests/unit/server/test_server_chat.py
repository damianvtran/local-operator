"""
Tests for the chat endpoints of the FastAPI server.

This module contains tests for the chat functionality, including
regular chat and agent-specific chat endpoints.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from local_operator.agents import AgentEditFields
from local_operator.jobs import JobStatus
from local_operator.server.models.schemas import AgentChatRequest, ChatRequest
from local_operator.types import ConversationRecord, ConversationRole


@pytest.mark.asyncio
async def test_chat_success(
    test_app_client,
    dummy_executor,
    mock_create_operator,
):
    """Test the chat endpoint using the test_app_client fixture."""
    test_prompt = "Hello, how are you?"

    # Use an empty context to trigger insertion of the system prompt by the server.
    payload = {
        "hosting": "openai",
        "model": "gpt-4o",
        "prompt": test_prompt,
        "context": [],
    }

    response = await test_app_client.post("/v1/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    # Verify that the response contains the dummy operator response.
    assert result.get("response") == "dummy operator response"
    conversation = result.get("context")
    assert isinstance(conversation, list)

    # Count occurrences of the test prompt in the conversation
    prompt_count = sum(1 for msg in conversation if msg.get("content") == test_prompt)
    assert prompt_count == 1, "Test prompt should appear exactly once in conversation"

    # Verify expected roles are present
    roles = [msg.get("role") for msg in conversation]
    assert ConversationRole.SYSTEM.value in roles
    assert ConversationRole.USER.value in roles
    assert ConversationRole.ASSISTANT.value in roles

    # Verify token stats are present.
    stats = result.get("stats")
    assert stats is not None
    assert stats.get("total_tokens") > 0
    assert stats.get("prompt_tokens") > 0
    assert stats.get("completion_tokens") > 0


@pytest.mark.asyncio
async def test_chat_sync_with_agent_success(
    test_app_client,
    dummy_executor,
    dummy_registry,
    mock_create_operator,
):
    """Test chat with a specific agent."""

    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Test Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
        )
    )
    agent_id = agent.id

    test_prompt = "Hello agent, how are you?"
    payload = ChatRequest(
        hosting="openai",
        model="gpt-4",
        prompt=test_prompt,
        context=[],
    )

    response = await test_app_client.post(f"/v1/chat/agents/{agent_id}", json=payload.model_dump())

    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result.get("response") == "dummy operator response"
    conversation = result.get("context")
    assert isinstance(conversation, list)

    # Verify token stats
    stats = result.get("stats")
    assert stats is not None
    assert stats.get("total_tokens") > 0
    assert stats.get("prompt_tokens") > 0
    assert stats.get("completion_tokens") > 0


@pytest.mark.asyncio
async def test_chat_with_agent_persist_conversation(
    test_app_client,
    dummy_executor,
    dummy_registry,
    mock_create_operator,
):
    """Test chat with a specific agent with conversation persistence across multiple requests."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Test Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
        )
    )
    agent_id = agent.id

    # First request
    first_prompt = "Hello agent, how are you?"
    first_payload = AgentChatRequest(
        hosting="openai",
        model="gpt-4",
        prompt=first_prompt,
        persist_conversation=True,
    )

    first_response = await test_app_client.post(
        f"/v1/chat/agents/{agent_id}", json=first_payload.model_dump()
    )

    assert first_response.status_code == 200
    first_data = first_response.json()
    first_result = first_data.get("result")
    assert first_result.get("response") == "dummy operator response"
    first_conversation = first_result.get("context")
    assert isinstance(first_conversation, list)

    # Second request - should include history from first request
    second_prompt = "Tell me more about yourself"
    second_payload = AgentChatRequest(
        hosting="openai",
        model="gpt-4",
        prompt=second_prompt,
        persist_conversation=True,
    )

    second_response = await test_app_client.post(
        f"/v1/chat/agents/{agent_id}", json=second_payload.model_dump()
    )

    assert second_response.status_code == 200
    second_data = second_response.json()
    second_result = second_data.get("result")
    assert second_result.get("response") == "dummy operator response"
    second_conversation = second_result.get("context")
    assert isinstance(second_conversation, list)

    # Verify the second conversation contains both prompts
    first_prompt_in_history = any(
        msg.get("content") == first_prompt and msg.get("role") == ConversationRole.USER.value
        for msg in second_conversation
    )
    second_prompt_in_history = any(
        msg.get("content") == second_prompt and msg.get("role") == ConversationRole.USER.value
        for msg in second_conversation
    )

    assert first_prompt_in_history, "First prompt should be in the conversation history"
    assert second_prompt_in_history, "Second prompt should be in the conversation history"

    # The second conversation should be longer than the first
    assert len(second_conversation) > len(
        first_conversation
    ), "Second conversation should contain more messages than the first"


@pytest.mark.asyncio
async def test_chat_sync_with_nonexistent_agent(
    test_app_client,
    dummy_executor,
    dummy_registry,
    mock_create_operator,
):
    """Test chat with a non-existent agent."""
    payload = ChatRequest(
        hosting="openai",
        model="gpt-4",
        prompt="Hello?",
        context=[],
    )

    response = await test_app_client.post("/v1/chat/agents/nonexistent", json=payload.model_dump())

    assert response.status_code == 404
    data = response.json()
    assert "Agent not found" in data.get("detail", "")


# Test when the operator's chat method raises an exception (simulating an error during
# model invocation).
class FailingOperator:
    """Mock operator that simulates a failure in chat."""

    def __init__(self):
        self.executor = None

    async def chat(self):
        raise Exception("Simulated failure in chat")


@pytest.mark.asyncio
async def test_chat_model_failure(test_app_client):
    """Test handling of model failure during chat."""
    with patch("local_operator.server.routes.chat.create_operator", return_value=FailingOperator()):
        payload = ChatRequest(
            hosting="openai",
            model="gpt-4o",
            prompt="This should cause an error",
            context=[
                ConversationRecord(role=ConversationRole.USER, content="This should cause an error")
            ],
        )

        response = await test_app_client.post("/v1/chat", json=payload.model_dump())

        assert response.status_code == 500
        data = response.json()
        # The error detail should indicate an internal server error.
        assert "Internal Server Error" in data.get("detail", "")


@pytest.mark.asyncio
async def test_chat_async_endpoint_success(
    test_app_client,
    mock_create_operator,
    mock_job_manager,
):
    """Test successful asynchronous chat request."""
    # Setup mock job
    mock_job = MagicMock()
    mock_job.id = "test-job-id"
    mock_job.status = JobStatus.PENDING
    mock_job.created_at = datetime.now(timezone.utc).timestamp()
    mock_job.started_at = None
    mock_job.completed_at = None
    mock_job_manager.create_job.return_value = mock_job

    payload = ChatRequest(
        hosting="openai",
        model="gpt-4o",
        prompt="Process this asynchronously",
        context=[],
    )

    response = await test_app_client.post("/v1/chat/async", json=payload.model_dump())

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == 202
    assert data["message"] == "Chat request accepted"
    assert data["result"]["id"] == "test-job-id"
    assert data["result"]["status"] == "pending"
    assert data["result"]["prompt"] == "Process this asynchronously"
    assert data["result"]["model"] == "gpt-4o"
    assert data["result"]["hosting"] == "openai"
    assert mock_job_manager.create_job.called
    assert mock_job_manager.register_task.called


@pytest.mark.asyncio
async def test_chat_async_endpoint_failure(test_app_client):
    """Test handling of failure during async chat job setup."""
    with patch(
        "local_operator.server.routes.chat.create_operator",
        side_effect=Exception("Failed to create operator"),
    ):
        payload = ChatRequest(
            hosting="openai",
            model="gpt-4o",
            prompt="This should fail during setup",
            context=[],
        )

        response = await test_app_client.post("/v1/chat/async", json=payload.model_dump())

        assert response.status_code == 500
        data = response.json()
        assert "Internal Server Error" in data.get("detail", "")


@pytest.mark.asyncio
async def test_chat_async_job_processing(test_app_client, mock_create_operator, mock_job_manager):
    """Test the background processing of an async chat job."""
    # Setup mock job
    mock_job = MagicMock()
    mock_job.id = "test-job-id"
    mock_job.status = JobStatus.PENDING
    mock_job.created_at = datetime.now(timezone.utc).timestamp()
    mock_job.started_at = None
    mock_job.completed_at = None
    mock_job_manager.create_job.return_value = mock_job

    # Store the task for testing
    captured_task = None

    def register_task_side_effect(job_id, task):
        nonlocal captured_task
        captured_task = task
        return None

    mock_job_manager.register_task.side_effect = register_task_side_effect

    payload = ChatRequest(
        hosting="openai",
        model="gpt-4o",
        prompt="Process this asynchronously",
        context=[],
    )

    response = await test_app_client.post("/v1/chat/async", json=payload.model_dump())

    assert response.status_code == 202

    # Verify the task was created and registered
    assert captured_task is not None

    # Run the background task directly
    await captured_task  # type: ignore

    # Verify job status was updated correctly
    assert mock_job_manager.update_job_status.call_count >= 2
    # First call should update to PROCESSING
    assert mock_job_manager.update_job_status.call_args_list[0][0][1] == JobStatus.PROCESSING
    # Last call should update to COMPLETED
    assert mock_job_manager.update_job_status.call_args_list[-1][0][1] == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_chat_with_agent_async_success(
    test_app_client,
    dummy_executor,
    dummy_registry,
    mock_create_operator,
    mock_job_manager,
):
    """Test the async chat with agent endpoint."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Test Agent",
            description="Test agent for async chat",
            security_prompt="Test security prompt",
            hosting="openai",
            model="gpt-4o",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
        )
    )
    agent_id = agent.id
    # Setup mock job
    mock_job = MagicMock()
    mock_job.id = "test-job-id"
    mock_job.status = JobStatus.PENDING
    mock_job.created_at = datetime.now(timezone.utc).timestamp()
    mock_job.started_at = None
    mock_job.completed_at = None
    mock_job_manager.create_job.return_value = mock_job

    # Store the task for testing
    captured_task = None

    def register_task_side_effect(job_id, task):
        nonlocal captured_task
        captured_task = task
        return None

    mock_job_manager.register_task.side_effect = register_task_side_effect

    payload = ChatRequest(
        hosting="openai",
        model="gpt-4o",
        prompt="Process this with an agent asynchronously",
        context=[],
    )

    response = await test_app_client.post(
        f"/v1/chat/agents/{agent_id}/async", json=payload.model_dump()
    )

    assert response.status_code == 202
    data = response.json()
    result = data.get("result")
    assert result.get("id") == "test-job-id"
    assert result.get("agent_id") == agent_id
    assert result.get("status") == JobStatus.PENDING.value

    # Verify the task was created and registered
    assert captured_task is not None

    # Run the background task directly
    await captured_task  # type: ignore

    # Verify job status was updated correctly
    assert mock_job_manager.update_job_status.call_count >= 2
    # First call should update to PROCESSING
    assert mock_job_manager.update_job_status.call_args_list[0][0][1] == JobStatus.PROCESSING
    # Last call should update to COMPLETED
    assert mock_job_manager.update_job_status.call_args_list[-1][0][1] == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_chat_with_agent_async_persist_conversation(
    test_app_client,
    dummy_executor,
    dummy_registry,
    mock_create_operator,
    mock_job_manager,
):
    """Test async chat with agent with conversation persistence across multiple requests."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Test Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description="",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
        )
    )
    agent_id = agent.id

    # Setup for first job
    first_job = MagicMock()
    first_job.id = "first-job-id"
    first_job.status = JobStatus.PENDING
    first_job.created_at = datetime.now(timezone.utc).timestamp()
    first_job.started_at = None
    first_job.completed_at = None
    mock_job_manager.create_job.return_value = first_job

    # Store the first task for testing
    first_task = None

    def register_first_task(job_id, task):
        nonlocal first_task
        first_task = task
        return None

    mock_job_manager.register_task.side_effect = register_first_task

    # First request
    first_prompt = "Hello agent, how are you?"
    first_payload = AgentChatRequest(
        hosting="openai",
        model="gpt-4",
        prompt=first_prompt,
        persist_conversation=True,
    )

    first_response = await test_app_client.post(
        f"/v1/chat/agents/{agent_id}/async", json=first_payload.model_dump()
    )

    assert first_response.status_code == 202
    first_data = first_response.json()
    first_result = first_data.get("result")
    assert first_result.get("id") == "first-job-id"
    assert first_result.get("agent_id") == agent_id
    assert first_result.get("status") == JobStatus.PENDING.value

    # Verify the first task was created and registered
    assert first_task is not None

    # Run the first background task directly
    await first_task  # type: ignore

    # Verify first job status was updated correctly
    assert mock_job_manager.update_job_status.call_count >= 2
    assert mock_job_manager.update_job_status.call_args_list[0][0][1] == JobStatus.PROCESSING
    assert mock_job_manager.update_job_status.call_args_list[-1][0][1] == JobStatus.COMPLETED

    # Setup for second job
    second_job = MagicMock()
    second_job.id = "second-job-id"
    second_job.status = JobStatus.PENDING
    second_job.created_at = datetime.now(timezone.utc).timestamp()
    second_job.started_at = None
    second_job.completed_at = None
    mock_job_manager.create_job.return_value = second_job

    # Store the second task for testing
    second_task = None

    def register_second_task(job_id, task):
        nonlocal second_task
        second_task = task
        return None

    mock_job_manager.register_task.side_effect = register_second_task

    # Second request - should include history from first request
    second_prompt = "Tell me more about yourself"
    second_payload = AgentChatRequest(
        hosting="openai",
        model="gpt-4",
        prompt=second_prompt,
        persist_conversation=True,
    )

    second_response = await test_app_client.post(
        f"/v1/chat/agents/{agent_id}/async", json=second_payload.model_dump()
    )

    assert second_response.status_code == 202
    second_data = second_response.json()
    second_result = second_data.get("result")
    assert second_result.get("id") == "second-job-id"
    assert second_result.get("agent_id") == agent_id
    assert second_result.get("status") == JobStatus.PENDING.value

    # Verify the second task was created and registered
    assert second_task is not None

    # Run the second background task directly
    await second_task  # type: ignore

    # Verify second job status was updated correctly
    assert mock_job_manager.update_job_status.call_count >= 4

    # Extract the conversation history from the operator after the second job
    # This requires accessing the conversation history that was passed to update_job_status
    # Get the last call to update_job_status with COMPLETED status
    completed_calls = [
        call
        for call in mock_job_manager.update_job_status.call_args_list
        if call[0][1] == JobStatus.COMPLETED
    ]
    assert len(completed_calls) >= 2

    # The result passed to the second COMPLETED update should contain our conversation
    second_job_result = completed_calls[-1][0][2]
    assert second_job_result is not None

    # Extract conversation from the result - JobResult is a Pydantic model, not a dict
    conversation = second_job_result.context
    assert isinstance(conversation, list)

    # Verify the conversation contains both prompts
    first_prompt_in_history = any(
        msg.get("content") == first_prompt and msg.get("role") == ConversationRole.USER.value
        for msg in conversation
    )
    second_prompt_in_history = any(
        msg.get("content") == second_prompt and msg.get("role") == ConversationRole.USER.value
        for msg in conversation
    )

    assert first_prompt_in_history, "First prompt should be in the conversation history"
    assert second_prompt_in_history, "Second prompt should be in the conversation history"


@pytest.mark.asyncio
async def test_chat_with_agent_async_agent_not_found(test_app_client, mock_job_manager):
    """Test the async chat with agent endpoint when agent is not found."""
    agent_id = "non-existent-agent"

    payload = ChatRequest(
        hosting="openai",
        model="gpt-4o",
        prompt="This should fail because the agent doesn't exist",
        context=[],
    )

    response = await test_app_client.post(
        f"/v1/chat/agents/{agent_id}/async", json=payload.model_dump()
    )

    assert response.status_code == 404
    data = response.json()
    assert f"Agent with id {agent_id} not found" in data.get("detail", "")


@pytest.mark.asyncio
async def test_chat_with_agent_async_with_context(
    test_app_client,
    dummy_executor,
    dummy_registry,
    mock_create_operator,
    mock_job_manager,
):
    """Test the async chat with agent endpoint with custom context."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Test Agent with Context",
            description="Test agent for async chat with context",
            security_prompt="Test security prompt",
            hosting="openai",
            model="gpt-4o",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
        )
    )
    agent_id = agent.id

    # Setup mock job
    mock_job = MagicMock()
    mock_job.id = "test-job-context"
    mock_job.status = JobStatus.PENDING
    mock_job.created_at = datetime.now(timezone.utc).timestamp()
    mock_job.started_at = None
    mock_job.completed_at = None
    mock_job_manager.create_job.return_value = mock_job

    # Store the task for testing
    captured_task = None

    def register_task_side_effect(job_id, task):
        nonlocal captured_task
        captured_task = task
        return None

    mock_job_manager.register_task.side_effect = register_task_side_effect

    # Create a custom context
    custom_context = [
        ConversationRecord(role=ConversationRole.SYSTEM, content="Custom system prompt"),
        ConversationRecord(role=ConversationRole.USER, content="Previous user message"),
        ConversationRecord(role=ConversationRole.ASSISTANT, content="Previous assistant response"),
    ]

    payload = ChatRequest(
        hosting="openai",
        model="gpt-4o",
        prompt="Process this with custom context",
        context=custom_context,
    )

    response = await test_app_client.post(
        f"/v1/chat/agents/{agent_id}/async", json=payload.model_dump()
    )

    assert response.status_code == 202
    data = response.json()
    result = data.get("result")
    assert result.get("agent_id") == agent_id

    # Run the background task directly
    await captured_task  # type: ignore

    # Verify job was completed
    assert mock_job_manager.update_job_status.call_args_list[-1][0][1] == JobStatus.COMPLETED
