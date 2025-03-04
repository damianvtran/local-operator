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
from local_operator.types import (
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
    ProcessResponseStatus,
)


@pytest.mark.asyncio
async def test_update_agent_success(test_app_client, dummy_registry: AgentRegistry):
    """Test updating an agent using the test_app_client fixture."""
    # Create a dummy agent
    agent_registry = app.state.agent_registry
    original_agent = agent_registry.create_agent(
        AgentEditFields(
            name="Original Name",
            security_prompt="Original Security",
            hosting="openai",
            model="gpt-4",
            description="Original description",
            last_message="Original last message",
        )
    )
    agent_id = original_agent.id

    update_payload = AgentUpdate(
        name="Updated Name",
        security_prompt="Updated Security",
        hosting="anthropic",
        model="claude-2",
        description="New description",
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
    assert result["description"] == "New description"
    assert result["last_message"] == "Original last message"
    # Pydantic serializes datetime to ISO 8601 format with 'T' separator and 'Z' for UTC
    assert result[
        "last_message_datetime"
    ] == original_agent.last_message_datetime.isoformat().replace("+00:00", "Z")


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
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    update_payload = AgentEditFields(
        name="Updated Name Only",
        security_prompt=None,
        hosting=None,
        model=None,
        description=None,
        last_message=None,
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
        name="Non-existent Update",
        security_prompt=None,
        hosting=None,
        model=None,
        description=None,
        last_message=None,
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
            description=None,
            last_message=None,
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
        name="New Test Agent",
        security_prompt="Test Security",
        hosting="openai",
        model="gpt-4",
        description="",
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
        name="",
        security_prompt="",
        hosting="",
        model="",
        description="",
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
                name=f"Agent {i}",
                security_prompt=f"Security {i}",
                hosting="openai",
                model="gpt-4",
                description=None,
                last_message=None,
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
    assert len(result["agents"]) == 10

    # Test second page
    response = await test_app_client.get("/v1/agents?page=2&per_page=10")
    data = response.json()
    result = data.get("result")
    assert len(result["agents"]) == 5


@pytest.mark.asyncio
async def test_get_agent_success(test_app_client, dummy_registry: AgentRegistry):
    """Test getting a specific agent."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Test Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
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
async def test_get_agent_conversation_empty(test_app_client, dummy_registry: AgentRegistry):
    """Test retrieving empty conversation history for an agent."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Conversation Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    # Get conversation (should be empty initially)
    response = await test_app_client.get(f"/v1/agents/{agent_id}/conversation")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["agent_id"] == agent_id
    assert "first_message_datetime" in result
    assert "last_message_datetime" in result
    assert "messages" in result
    assert len(result["messages"]) == 0
    assert result["page"] == 1
    assert result["per_page"] == 10
    assert result["total"] == 0
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_get_agent_conversation_pagination_default(
    test_app_client, dummy_registry: AgentRegistry
):
    """Test default pagination for agent conversation history."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Conversation Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    # Create 15 mock conversation records for pagination testing
    mock_conversation = []
    for i in range(15):
        mock_conversation.append(
            ConversationRecord(
                role=ConversationRole.USER if i % 2 == 0 else ConversationRole.ASSISTANT,
                content=f"Message {i+1}",
                should_summarize=True,
                timestamp=datetime.now(timezone.utc),
            )
        )

    # Save the mock conversation
    dummy_registry.save_agent_conversation(agent_id, mock_conversation, [])

    # Test default pagination (page 1, per_page 10)
    response = await test_app_client.get(f"/v1/agents/{agent_id}/conversation")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["agent_id"] == agent_id
    assert len(result["messages"]) == 10
    assert result["page"] == 1
    assert result["per_page"] == 10
    assert result["total"] == 15
    assert result["count"] == 10
    assert result["messages"][0]["content"] == "Message 6"
    assert result["messages"][9]["content"] == "Message 15"


@pytest.mark.asyncio
async def test_get_agent_conversation_pagination_second_page(
    test_app_client, dummy_registry: AgentRegistry
):
    """Test second page pagination for agent conversation history."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Conversation Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    # Create 15 mock conversation records
    mock_conversation = []
    for i in range(15):
        mock_conversation.append(
            ConversationRecord(
                role=ConversationRole.USER if i % 2 == 0 else ConversationRole.ASSISTANT,
                content=f"Message {i+1}",
                should_summarize=True,
                timestamp=datetime.now(timezone.utc),
            )
        )

    # Save the mock conversation
    dummy_registry.save_agent_conversation(agent_id, mock_conversation, [])

    # Test second page
    response = await test_app_client.get(f"/v1/agents/{agent_id}/conversation?page=2&per_page=10")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["agent_id"] == agent_id
    assert len(result["messages"]) == 5
    assert result["page"] == 2
    assert result["per_page"] == 10
    assert result["total"] == 15
    assert result["count"] == 5
    assert result["messages"][0]["content"] == "Message 1"
    assert result["messages"][4]["content"] == "Message 5"


@pytest.mark.asyncio
async def test_get_agent_conversation_custom_per_page(
    test_app_client, dummy_registry: AgentRegistry
):
    """Test custom per_page parameter for agent conversation history."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Conversation Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    # Create 15 mock conversation records
    mock_conversation = []
    for i in range(15):
        mock_conversation.append(
            ConversationRecord(
                role=ConversationRole.USER if i % 2 == 0 else ConversationRole.ASSISTANT,
                content=f"Message {i+1}",
                should_summarize=True,
                timestamp=datetime.now(timezone.utc),
            )
        )

    # Save the mock conversation
    dummy_registry.save_agent_conversation(agent_id, mock_conversation, [])

    # Test custom per_page
    response = await test_app_client.get(f"/v1/agents/{agent_id}/conversation?per_page=5")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["agent_id"] == agent_id
    assert len(result["messages"]) == 5
    assert result["page"] == 1
    assert result["per_page"] == 5
    assert result["count"] == 5
    assert result["messages"][0]["content"] == "Message 11"
    assert result["messages"][4]["content"] == "Message 15"


@pytest.mark.asyncio
async def test_get_agent_conversation_page_out_of_bounds(
    test_app_client, dummy_registry: AgentRegistry
):
    """Test out of bounds page parameter for agent conversation history."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Conversation Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    # Create 15 mock conversation records
    mock_conversation = []
    for i in range(15):
        mock_conversation.append(
            ConversationRecord(
                role=ConversationRole.USER if i % 2 == 0 else ConversationRole.ASSISTANT,
                content=f"Message {i+1}",
                should_summarize=True,
                timestamp=datetime.now(timezone.utc),
            )
        )

    # Save the mock conversation
    dummy_registry.save_agent_conversation(agent_id, mock_conversation, [])

    # Test page out of bounds
    response = await test_app_client.get(f"/v1/agents/{agent_id}/conversation?page=4&per_page=5")

    assert response.status_code == 400
    data = response.json()
    assert "Page 4 is out of bounds" in data.get("detail", "")


@pytest.mark.asyncio
async def test_get_agent_conversation_not_found(test_app_client, dummy_registry: AgentRegistry):
    """Test retrieving conversation for a non-existent agent."""
    non_existent_id = "nonexistent"
    response = await test_app_client.get(f"/v1/agents/{non_existent_id}/conversation")

    assert response.status_code == 404
    data = response.json()
    assert f"Agent with ID {non_existent_id} not found" in data.get("detail", "")


@pytest.mark.asyncio
async def test_get_agent_execution_history(test_app_client, dummy_registry: AgentRegistry):
    """Test retrieving execution history for an agent."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Execution History Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    # Create 5 mock execution records
    mock_executions = []
    for i in range(5):
        mock_executions.append(
            CodeExecutionResult(
                code=f"print('Execution {i+1}')",
                stdout=f"Execution {i+1}",
                stderr="",
                logging="",
                message="",
                formatted_print="",
                role=ConversationRole.SYSTEM,
                status=ProcessResponseStatus.SUCCESS,
                timestamp=datetime.now(timezone.utc),
            )
        )

    # Save the mock executions
    dummy_registry.save_agent_conversation(agent_id, [], mock_executions)

    # Test retrieving execution history
    response = await test_app_client.get(f"/v1/agents/{agent_id}/history")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result", {})

    assert result.get("agent_id") == agent_id
    assert result.get("total") == 5
    assert result.get("page") == 1
    assert result.get("per_page") == 10
    assert result.get("count") == 5
    assert len(result.get("history", [])) == 5


@pytest.mark.asyncio
async def test_get_agent_execution_history_pagination(
    test_app_client, dummy_registry: AgentRegistry
):
    """Test pagination for agent execution history."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Execution History Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    # Create 15 mock execution records
    mock_executions = []
    for i in range(15):
        mock_executions.append(
            CodeExecutionResult(
                code=f"print('Execution {i+1}')",
                stdout=f"Execution {i+1}",
                stderr="",
                logging="",
                message="",
                formatted_print="",
                role=ConversationRole.SYSTEM,
                status=ProcessResponseStatus.SUCCESS,
                timestamp=datetime.now(timezone.utc),
            )
        )

    # Save the mock executions
    dummy_registry.save_agent_conversation(agent_id, [], mock_executions)  # type: ignore

    # Test pagination - page 1 with 5 per page
    response = await test_app_client.get(f"/v1/agents/{agent_id}/history?page=1&per_page=5")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result", {})

    assert result.get("agent_id") == agent_id
    assert result.get("total") == 15
    assert result.get("page") == 1
    assert result.get("per_page") == 5
    assert result.get("count") == 5
    assert len(result.get("history", [])) == 5

    # Test pagination - page 3 with 5 per page
    response = await test_app_client.get(f"/v1/agents/{agent_id}/history?page=3&per_page=5")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result", {})

    assert result.get("page") == 3
    assert result.get("count") == 5
    assert len(result.get("history", [])) == 5


@pytest.mark.asyncio
async def test_get_agent_execution_history_page_out_of_bounds(
    test_app_client, dummy_registry: AgentRegistry
):
    """Test out of bounds page parameter for agent execution history."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Execution History Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    # Create 15 mock execution records
    mock_executions = []
    for i in range(15):
        mock_executions.append(
            CodeExecutionResult(
                code=f"print('Execution {i+1}')",
                stdout=f"Execution {i+1}",
                stderr="",
                logging="",
                message="",
                formatted_print="",
                role=ConversationRole.SYSTEM,
                status=ProcessResponseStatus.SUCCESS,
                timestamp=datetime.now(timezone.utc),
            )
        )

    # Save the mock executions
    dummy_registry.save_agent_conversation(agent_id, [], mock_executions)  # type: ignore

    # Test page out of bounds
    response = await test_app_client.get(f"/v1/agents/{agent_id}/history?page=4&per_page=5")

    assert response.status_code == 400
    data = response.json()
    assert "Page 4 is out of bounds" in data.get("detail", "")


@pytest.mark.asyncio
async def test_get_agent_execution_history_not_found(
    test_app_client, dummy_registry: AgentRegistry
):
    """Test retrieving execution history for a non-existent agent."""
    non_existent_id = "nonexistent"
    response = await test_app_client.get(f"/v1/agents/{non_existent_id}/history")

    assert response.status_code == 404
    data = response.json()
    assert f"Agent with ID {non_existent_id} not found" in data.get("detail", "")


@pytest.mark.asyncio
async def test_get_agent_execution_history_empty(test_app_client, dummy_registry: AgentRegistry):
    """Test retrieving execution history for an agent with no executions."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Empty Execution History Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
        )
    )
    agent_id = agent.id

    # Test retrieving empty execution history
    response = await test_app_client.get(f"/v1/agents/{agent_id}/history")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result", {})

    assert result.get("agent_id") == agent_id
    assert result.get("total") == 0
    assert result.get("count") == 0
    assert len(result.get("history", [])) == 0
