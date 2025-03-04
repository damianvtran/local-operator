"""
Tests for the agent endpoints of the FastAPI server.

This module contains tests for agent-related functionality, including
creating, updating, deleting, and listing agents.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.server.app import app
from local_operator.server.models.schemas import AgentCreate, AgentUpdate


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
async def test_update_agent_registry_not_initialized(test_app_client):
    """Test updating an agent when the registry is not initialized."""
    # Safely get the original agent_registry without risking a KeyError.
    original_registry = getattr(app.state, "agent_registry", None)
    app.state.agent_registry = None
    update_payload = AgentEditFields(name="New Name", security_prompt="", hosting="", model="")
    agent_id = "agent1"
    response = await test_app_client.patch(
        f"/v1/agents/{agent_id}", json=update_payload.model_dump()
    )
    # Restore the original state of agent_registry.
    if original_registry is not None:
        app.state.agent_registry = original_registry
    else:
        # Remove agent_registry from the underlying state dict if it exists.
        if "agent_registry" in app.state._state:
            del app.state._state["agent_registry"]
    assert response.status_code == 500
    data = response.json()
    assert "Agent registry not initialized" in data.get("detail", "")


@pytest.mark.asyncio
async def test_delete_agent_registry_not_initialized():
    """Test deleting an agent when the registry is not initialized."""
    original_registry = getattr(app.state, "agent_registry", None)
    app.state.agent_registry = None
    agent_id = "agent1"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.delete(f"/v1/agents/{agent_id}")
    if original_registry is not None:
        app.state.agent_registry = original_registry
    else:
        if "agent_registry" in app.state._state:
            del app.state._state["agent_registry"]
    assert response.status_code == 500
    data = response.json()
    assert "Agent registry not initialized" in data.get("detail", "")


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
