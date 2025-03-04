"""
Tests for the agent endpoints of the FastAPI server.

This module contains tests for agent-related functionality, including
creating, updating, deleting, and listing agents.
"""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.server.app import app
from local_operator.server.models.schemas import AgentCreate, AgentUpdate
from local_operator.types import ConversationRecord, ConversationRole


@pytest.mark.asyncio
async def test_update_agent_success(test_app_client, dummy_registry: AgentRegistry):
    """Test updating an agent using the test_app_client fixture."""
    # Create a dummy agent
    agent_registry = app.state.agent_registry
    agent = agent_registry.create_agent(
        AgentEditFields(
            name="Original Name",
            security_prompt="Original Security",
            hosting="openai",
            model="gpt-4",
        )
    )
    agent_id = agent.id

    update_payload = AgentUpdate(
        name="Updated Name",
        security_prompt="Updated Security",
        hosting="anthropic",
        model="claude-2",
    )

    response = await test_app_client.patch(
        f"/v1/agents/{agent_id}", json=update_payload.model_dump()
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent updated successfully"
    result = data.get("result")
    assert result["id"] == agent_id
    assert result["name"] == "Updated Name"
    assert result["security_prompt"] == "Updated Security"
    assert result["hosting"] == "anthropic"
    assert result["model"] == "claude-2"


@pytest.mark.asyncio
async def test_update_agent_single_field(test_app_client, dummy_registry: AgentRegistry):
    """Test updating only a single field of an agent."""
    # Create a dummy agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Original Name",
            security_prompt="Original Security",
            hosting="openai",
            model="gpt-4",
        )
    )
    agent_id = agent.id

    update_payload = AgentEditFields(
        name="Updated Name Only", security_prompt=None, hosting=None, model=None
    )

    response = await test_app_client.patch(
        f"/v1/agents/{agent_id}", json=update_payload.model_dump()
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent updated successfully"
    result = data.get("result")
    assert result["id"] == agent_id
    assert result["name"] == "Updated Name Only"
    assert result["security_prompt"] == "Original Security"
    assert result["hosting"] == "openai"
    assert result["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_update_agent_not_found(test_app_client, dummy_registry: AgentRegistry):
    """Test updating a non-existent agent."""
    update_payload = AgentEditFields(
        name="Non-existent Update", security_prompt=None, hosting=None, model=None
    )
    non_existent_agent_id = "nonexistent"

    response = await test_app_client.patch(
        f"/v1/agents/{non_existent_agent_id}", json=update_payload.model_dump()
    )

    assert response.status_code == 404
    data = response.json()
    assert "Agent not found" in data.get("detail", "")


@pytest.mark.asyncio
async def test_delete_agent_success(test_app_client, dummy_registry: AgentRegistry):
    """Test deleting an agent."""
    # Create a dummy agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Agent to Delete",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
        )
    )
    agent_id = agent.id

    response = await test_app_client.delete(f"/v1/agents/{agent_id}")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent deleted successfully"
    assert data.get("result") == {}

    # Verify agent was actually deleted
    with pytest.raises(KeyError):
        dummy_registry.get_agent(agent_id)


@pytest.mark.asyncio
async def test_delete_agent_not_found(test_app_client, dummy_registry: AgentRegistry):
    """Test deleting a non-existent agent."""
    non_existent_agent_id = "nonexistent"
    response = await test_app_client.delete(f"/v1/agents/{non_existent_agent_id}")

    assert response.status_code == 404
    data = response.json()
    assert "Agent not found" in data.get("detail", "")


@pytest.mark.asyncio
async def test_create_agent_success(dummy_registry: AgentRegistry):
    """Test creating a new agent."""
    create_payload = AgentCreate(
        name="New Test Agent", security_prompt="Test Security", hosting="openai", model="gpt-4"
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/v1/agents", json=create_payload.model_dump())

    assert response.status_code == 201
    data = response.json()
    assert data.get("status") == 201
    assert data.get("message") == "Agent created successfully"
    result = data.get("result")
    assert result["name"] == create_payload.name
    assert result["security_prompt"] == create_payload.security_prompt
    assert result["hosting"] == create_payload.hosting
    assert result["model"] == create_payload.model
    assert "id" in result


@pytest.mark.asyncio
async def test_create_agent_invalid_data(dummy_registry: AgentRegistry):
    """Test creating an agent with invalid data."""
    invalid_payload = AgentCreate(
        name="", security_prompt="", hosting="", model=""
    )  # Invalid empty name

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/v1/agents", json=invalid_payload.model_dump())

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_list_agents_pagination(test_app_client, dummy_registry: AgentRegistry):
    """Test listing agents with pagination."""
    # Create multiple test agents
    for i in range(15):
        dummy_registry.create_agent(
            AgentEditFields(
                name=f"Agent {i}", security_prompt=f"Security {i}", hosting="openai", model="gpt-4"
            )
        )

    # Test first page
    response = await test_app_client.get("/v1/agents?page=1&per_page=10")
    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["total"] == 15
    assert result["page"] == 1
    assert result["per_page"] == 10
    assert len(result["agents"]["agents"]) == 10

    # Test second page
    response = await test_app_client.get("/v1/agents?page=2&per_page=10")
    data = response.json()
    result = data.get("result")
    assert len(result["agents"]["agents"]) == 5


@pytest.mark.asyncio
async def test_get_agent_success(test_app_client, dummy_registry: AgentRegistry):
    """Test getting a specific agent."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Test Agent", security_prompt="Test Security", hosting="openai", model="gpt-4"
        )
    )
    agent_id = agent.id

    response = await test_app_client.get(f"/v1/agents/{agent_id}")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["id"] == agent_id
    assert result["name"] == "Test Agent"
    assert result["security_prompt"] == "Test Security"
    assert result["hosting"] == "openai"
    assert result["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_get_agent_not_found(test_app_client, dummy_registry: AgentRegistry):
    """Test getting a non-existent agent."""
    non_existent_id = "nonexistent"

    response = await test_app_client.get(f"/v1/agents/{non_existent_id}")

    assert response.status_code == 404
    data = response.json()
    assert "Agent not found" in data.get("detail", "")


@pytest.mark.asyncio
async def test_get_agent_conversation(test_app_client, dummy_registry: AgentRegistry):
    """Test retrieving conversation history for a specific agent."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Conversation Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
        )
    )
    agent_id = agent.id

    # Get conversation (should be empty initially)
    response = await test_app_client.get(f"/v1/agents/{agent_id}/conversation")

    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == agent_id
    assert "first_message_datetime" in data
    assert "last_message_datetime" in data
    assert "messages" in data
    assert len(data["messages"]) == 0

    mock_conversation = [
        ConversationRecord(
            role=ConversationRole.SYSTEM,
            content="You are a helpful assistant",
            should_summarize=False,
            timestamp=datetime.now(timezone.utc),
        ),
        ConversationRecord(
            role=ConversationRole.USER,
            content="Hello, can you help me?",
            should_summarize=True,
            timestamp=datetime.now(timezone.utc),
        ),
        ConversationRecord(
            role=ConversationRole.ASSISTANT,
            content="Yes, I'd be happy to help. What do you need?",
            should_summarize=True,
            timestamp=datetime.now(timezone.utc),
        ),
    ]

    mock_execution_history = []

    # Save the mock conversation
    dummy_registry.save_agent_conversation(agent_id, mock_conversation, mock_execution_history)

    # Get conversation again (should now have the mock data)
    response = await test_app_client.get(f"/v1/agents/{agent_id}/conversation")

    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == agent_id
    assert len(data["messages"]) == 3
    assert data["messages"][0]["role"] == "system"
    assert data["messages"][1]["role"] == "user"
    assert data["messages"][2]["role"] == "assistant"
    assert "You are a helpful assistant" in data["messages"][0]["content"]
    assert "Hello, can you help me?" in data["messages"][1]["content"]

    # Test with non-existent agent
    non_existent_id = "nonexistent"
    response = await test_app_client.get(f"/v1/agents/{non_existent_id}/conversation")

    assert response.status_code == 404
    data = response.json()
    assert f"Agent with ID {non_existent_id} not found" in data.get("detail", "")
