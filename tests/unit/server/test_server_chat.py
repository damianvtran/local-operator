"""
Tests for the chat endpoints of the FastAPI server.

This module contains tests for the chat functionality, including
regular chat and agent-specific chat endpoints.
"""

from unittest.mock import patch

import pytest

from local_operator.server.models.schemas import ChatRequest
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
    from local_operator.agents import AgentEditFields

    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Test Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description="",
            last_message="",
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

    response = await test_app_client.post(
        f"/v1/chat/agents/{agent_id}/sync", json=payload.model_dump()
    )

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

    response = await test_app_client.post(
        "/v1/chat/agents/nonexistent/sync", json=payload.model_dump()
    )

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
