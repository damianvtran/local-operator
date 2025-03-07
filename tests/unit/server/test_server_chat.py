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
            current_working_directory=None,
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
            current_working_directory=None,
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
async def test_chat_async_process_execution(
    test_app_client,
    mock_create_operator,
    mock_job_manager,
):
    """Test that async chat jobs are executed in separate processes."""
    # Setup mock job
    mock_job = MagicMock()
    mock_job.id = "test-process-job-id"
    mock_job.status = JobStatus.PENDING
    mock_job.created_at = datetime.now(timezone.utc).timestamp()
    mock_job.started_at = None
    mock_job.completed_at = None
    mock_job_manager.create_job.return_value = mock_job

    # Mock the job processor functions
    mock_process = MagicMock()
    mock_create_and_start_job_process_with_queue = MagicMock(
        return_value=(mock_process, MagicMock())
    )

    # Mock the job processor functions to avoid pickling issues in tests
    with patch(
        "local_operator.server.routes.chat.create_and_start_job_process_with_queue",
        mock_create_and_start_job_process_with_queue,
    ):
        with patch("local_operator.server.routes.chat.run_job_in_process_with_queue"):
            payload = ChatRequest(
                hosting="openai",
                model="gpt-4o",
                prompt="Process this in a separate process",
                context=[],
            )

            response = await test_app_client.post("/v1/chat/async", json=payload.model_dump())

            assert response.status_code == 202

            # Verify that create_and_start_job_process_with_queue was called
            assert mock_create_and_start_job_process_with_queue.called

            # Verify the job was created
            mock_job_manager.create_job.assert_called_once()


@pytest.mark.asyncio
async def test_chat_async_endpoint_success(
    test_app_client,
    mock_create_operator,
    mock_job_manager,
):
    """Test successful asynchronous chat request."""
    # Mock the job processor functions
    mock_process = MagicMock()
    mock_monitor_task = MagicMock()
    mock_create_and_start_job_process_with_queue = MagicMock(
        return_value=(mock_process, mock_monitor_task)
    )

    # Setup mock job
    mock_job = MagicMock()
    mock_job.id = "test-job-id"
    mock_job.status = JobStatus.PENDING
    mock_job.created_at = datetime.now(timezone.utc).timestamp()
    mock_job.started_at = None
    mock_job.completed_at = None
    mock_job_manager.create_job.return_value = mock_job

    # Mock the job processor functions to avoid pickling issues in tests
    with patch(
        "local_operator.server.routes.chat.create_and_start_job_process_with_queue",
        mock_create_and_start_job_process_with_queue,
    ):
        with patch("local_operator.server.routes.chat.run_job_in_process_with_queue"):
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

            # Verify the job was created
            mock_job_manager.create_job.assert_called_once()

            # In the actual implementation, register_task is called inside
            # create_and_start_job_process_with_queue
            # So we don't need to verify it was called directly
            mock_create_and_start_job_process_with_queue.assert_called_once()


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

    # Create a mock process that will be returned by create_and_start_job_process_with_queue
    mock_process = MagicMock()
    mock_monitor_task = MagicMock()

    # This is the key change - we need to simulate what create_and_start_job_process_with_queue does
    # In the real implementation, it starts the process and registers it
    def mock_create_and_start_side_effect(*args, **kwargs):
        # Start the process
        mock_process.start()
        # Register the process with the job manager
        job_id = args[0] if args else kwargs.get("job_id")
        mock_job_manager.register_process(job_id, mock_process)
        # Return the process and monitor task
        return mock_process, mock_monitor_task

    mock_create_and_start = MagicMock(side_effect=mock_create_and_start_side_effect)

    with patch(
        "local_operator.server.routes.chat.create_and_start_job_process_with_queue",
        mock_create_and_start,
    ):
        with patch("local_operator.server.routes.chat.run_job_in_process_with_queue"):
            payload = ChatRequest(
                hosting="openai",
                model="gpt-4o",
                prompt="Process this asynchronously",
                context=[],
            )

            response = await test_app_client.post("/v1/chat/async", json=payload.model_dump())

            assert response.status_code == 202

            # Verify create_and_start_job_process_with_queue was called
            mock_create_and_start.assert_called_once()

            # Verify the process was started
            mock_process.start.assert_called_once()

            # Verify the process was registered with the job manager
            mock_job_manager.register_process.assert_called_once_with(mock_job.id, mock_process)


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
            current_working_directory=None,
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

    # Mock the job processor functions
    mock_process = MagicMock()
    mock_create_and_start_job_process_with_queue = MagicMock(
        return_value=(mock_process, MagicMock())
    )

    # Mock the job processor functions to avoid pickling issues in tests
    with patch(
        "local_operator.server.routes.chat.create_and_start_job_process_with_queue",
        mock_create_and_start_job_process_with_queue,
    ):
        with patch("local_operator.server.routes.chat.run_agent_job_in_process_with_queue"):
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

            # Verify that create_and_start_job_process_with_queue was called
            assert mock_create_and_start_job_process_with_queue.called


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
            current_working_directory=None,
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

    # Mock the job processor functions
    mock_process = MagicMock()
    mock_create_and_start_job_process_with_queue = MagicMock(
        return_value=(mock_process, MagicMock())
    )

    # Mock the job processor functions to avoid pickling issues in tests
    with patch(
        "local_operator.server.routes.chat.create_and_start_job_process_with_queue",
        mock_create_and_start_job_process_with_queue,
    ):
        with patch("local_operator.server.routes.chat.run_agent_job_in_process_with_queue"):
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

            # Verify that create_and_start_job_process_with_queue was called
            assert mock_create_and_start_job_process_with_queue.called

            # Setup for second job
            second_job = MagicMock()
            second_job.id = "second-job-id"
            second_job.status = JobStatus.PENDING
            second_job.created_at = datetime.now(timezone.utc).timestamp()
            second_job.started_at = None
            second_job.completed_at = None
            mock_job_manager.create_job.return_value = second_job

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

            # Verify that create_and_start_job_process_with_queue was called again
            assert mock_create_and_start_job_process_with_queue.call_count == 2


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
            current_working_directory=None,
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

    # Mock the job processor functions
    mock_process = MagicMock()
    mock_create_and_start_job_process_with_queue = MagicMock(
        return_value=(mock_process, MagicMock())
    )

    # Create a custom context
    custom_context = [
        ConversationRecord(role=ConversationRole.SYSTEM, content="Custom system prompt"),
        ConversationRecord(role=ConversationRole.USER, content="Previous user message"),
        ConversationRecord(role=ConversationRole.ASSISTANT, content="Previous assistant response"),
    ]

    # Mock the job processor functions to avoid pickling issues in tests
    with patch(
        "local_operator.server.routes.chat.create_and_start_job_process_with_queue",
        mock_create_and_start_job_process_with_queue,
    ):
        with patch("local_operator.server.routes.chat.run_agent_job_in_process_with_queue"):
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

            # Verify that create_and_start_job_process_with_queue was called
            assert mock_create_and_start_job_process_with_queue.called
