"""
Tests for the agent endpoints of the FastAPI server.

This module contains tests for agent-related functionality, including
creating, updating, deleting, and listing agents.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.server.app import app
from local_operator.server.models.schemas import AgentCreate, AgentUpdate
from local_operator.types import (
    AgentState,
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
            temperature=0.2,
            top_p=0.5,
            top_k=10,
            max_tokens=100,
            stop=["\n"],
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=1234567890,
            current_working_directory=".",
        )
    )
    agent_id = original_agent.id

    update_payload = AgentUpdate(
        name="Updated Name",
        security_prompt="Updated Security",
        hosting="anthropic",
        model="claude-2",
        description="New description",
        temperature=0.3,
        top_p=0.6,
        top_k=15,
        max_tokens=150,
        stop=["\n"],
        frequency_penalty=0.1,
        presence_penalty=0.1,
        seed=1234567890,
        current_working_directory="/tmp/path",
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
    assert result["temperature"] == 0.3
    assert result["top_p"] == 0.6
    assert result["top_k"] == 15
    assert result["max_tokens"] == 150
    assert result["stop"] == ["\n"]
    assert result["frequency_penalty"] == 0.1
    assert result["presence_penalty"] == 0.1
    assert result["seed"] == 1234567890
    assert result["current_working_directory"] == "/tmp/path"
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
            temperature=0.7,
            top_p=1.0,
            top_k=None,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
            current_working_directory=".",
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
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
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
    assert result["temperature"] == 0.7
    assert result["top_p"] == 1.0
    assert result["max_tokens"] == 2048
    assert result["frequency_penalty"] == 0.0
    assert result["presence_penalty"] == 0.0
    assert result["current_working_directory"] == "."


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
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
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
    assert result["temperature"] == create_payload.temperature
    assert result["top_p"] == create_payload.top_p
    assert result["max_tokens"] == create_payload.max_tokens
    assert result["frequency_penalty"] == create_payload.frequency_penalty
    assert result["presence_penalty"] == create_payload.presence_penalty
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
        temperature=0.7,
        top_p=1.0,
        top_k=None,
        max_tokens=2048,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=None,
        current_working_directory=None,
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
async def test_list_agents_name_filter(test_app_client, dummy_registry: AgentRegistry):
    """Test filtering agents by name."""
    # Create test agents with different naming patterns
    test_agents = [
        "SearchAgent",
        "Research Assistant",
        "Code Helper",
        "Search Engine",
        "Assistant Bot",
    ]

    for name in test_agents:
        dummy_registry.create_agent(
            AgentEditFields(
                name=name,
                security_prompt="Test Security",
                hosting="openai",
                model="gpt-4",
                description=None,
                last_message=None,
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

    # Test filtering by "Search" - should return 2 agents
    response = await test_app_client.get("/v1/agents?name=search")
    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["total"] == 3
    assert len(result["agents"]) == 3
    agent_names = [agent["name"] for agent in result["agents"]]
    assert "SearchAgent" in agent_names
    assert "Search Engine" in agent_names

    # Test filtering by "Assistant" - should return 2 agents
    response = await test_app_client.get("/v1/agents?name=assistant")
    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["total"] == 2
    assert len(result["agents"]) == 2
    agent_names = [agent["name"] for agent in result["agents"]]
    assert "Research Assistant" in agent_names
    assert "Assistant Bot" in agent_names

    # Test filtering by a non-existent name - should return 0 agents
    response = await test_app_client.get("/v1/agents?name=NonExistent")
    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["total"] == 0
    assert len(result["agents"]) == 0


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

    response = await test_app_client.get(f"/v1/agents/{agent_id}")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["id"] == agent_id
    assert result["name"] == "Test Agent"
    assert result["security_prompt"] == "Test Security"
    assert result["hosting"] == "openai"
    assert result["model"] == "gpt-4"
    assert result["temperature"] == 0.7
    assert result["top_p"] == 1.0
    assert result["max_tokens"] == 2048
    assert result["frequency_penalty"] == 0.0
    assert result["presence_penalty"] == 0.0


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
    dummy_registry.save_agent_state(
        agent_id=agent_id,
        agent_state=AgentState(
            version="",
            conversation=mock_conversation,
            execution_history=[],
            learnings=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
        ),
    )

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
    dummy_registry.save_agent_state(
        agent_id=agent_id,
        agent_state=AgentState(
            version="",
            conversation=mock_conversation,
            execution_history=[],
            learnings=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
        ),
    )

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
    dummy_registry.save_agent_state(
        agent_id=agent_id,
        agent_state=AgentState(
            version="",
            conversation=mock_conversation,
            execution_history=[],
            learnings=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
        ),
    )

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
    dummy_registry.save_agent_state(
        agent_id=agent_id,
        agent_state=AgentState(
            version="",
            conversation=mock_conversation,
            execution_history=[],
            learnings=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
        ),
    )
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
                files=[],
            )
        )

    # Save the mock executions
    dummy_registry.save_agent_state(
        agent_id=agent_id,
        agent_state=AgentState(
            version="",
            conversation=[],
            execution_history=mock_executions,
            learnings=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
        ),
    )

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
                files=[],
            )
        )

    # Save the mock executions
    dummy_registry.save_agent_state(
        agent_id=agent_id,
        agent_state=AgentState(
            version="",
            conversation=[],
            execution_history=mock_executions,
            learnings=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
        ),
    )

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
                files=[],
            )
        )

    # Save the mock executions
    dummy_registry.save_agent_state(
        agent_id=agent_id,
        agent_state=AgentState(
            version="",
            conversation=[],
            execution_history=mock_executions,
            learnings=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
        ),
    )

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

    # Test retrieving empty execution history
    response = await test_app_client.get(f"/v1/agents/{agent_id}/history")

    assert response.status_code == 200
    data = response.json()
    result = data.get("result", {})

    assert result.get("agent_id") == agent_id
    assert result.get("total") == 0
    assert result.get("count") == 0
    assert len(result.get("history", [])) == 0


@pytest.mark.asyncio
async def test_clear_agent_conversation(test_app_client, dummy_registry: AgentRegistry):
    """Test clearing an agent's conversation history."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Conversation Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
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

    # Create mock conversation records
    mock_conversation = []
    for i in range(5):
        mock_conversation.append(
            ConversationRecord(
                role=ConversationRole.USER if i % 2 == 0 else ConversationRole.ASSISTANT,
                content=f"Message {i+1}",
                should_summarize=True,
                timestamp=datetime.now(timezone.utc),
            )
        )

    # Save the mock conversation
    dummy_registry.save_agent_state(
        agent_id=agent_id,
        agent_state=AgentState(
            version="",
            conversation=mock_conversation,
            execution_history=[],
            learnings=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
        ),
    )

    # Verify conversation exists
    response = await test_app_client.get(f"/v1/agents/{agent_id}/conversation")
    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["total"] == 5

    # Clear the conversation
    response = await test_app_client.delete(f"/v1/agents/{agent_id}/conversation")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent conversation cleared successfully"
    assert data.get("result") == {}

    # Verify conversation is cleared
    response = await test_app_client.get(f"/v1/agents/{agent_id}/conversation")
    assert response.status_code == 200
    data = response.json()
    result = data.get("result")
    assert result["total"] == 0
    assert len(result["messages"]) == 0


@pytest.mark.asyncio
async def test_clear_agent_conversation_not_found(test_app_client, dummy_registry: AgentRegistry):
    """Test clearing conversation for a non-existent agent."""
    non_existent_id = "nonexistent"
    response = await test_app_client.delete(f"/v1/agents/{non_existent_id}/conversation")

    assert response.status_code == 404
    data = response.json()
    assert f"Agent with ID {non_existent_id} not found" in data.get("detail", "")


@pytest.mark.asyncio
async def test_get_agent_system_prompt(test_app_client, dummy_registry: AgentRegistry):
    """Test retrieving an agent's system prompt."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="System Prompt Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
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

    # Write a system prompt for the agent
    test_system_prompt = "This is a test system prompt for the agent."
    dummy_registry.set_agent_system_prompt(agent_id, test_system_prompt)

    # Get the system prompt
    response = await test_app_client.get(f"/v1/agents/{agent_id}/system-prompt")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent system prompt retrieved successfully"
    assert data.get("result") == {"system_prompt": test_system_prompt}


@pytest.mark.asyncio
async def test_get_agent_system_prompt_not_found(test_app_client, dummy_registry: AgentRegistry):
    """Test retrieving system prompt for a non-existent agent."""
    non_existent_id = "nonexistent"
    response = await test_app_client.get(f"/v1/agents/{non_existent_id}/system-prompt")

    assert response.status_code == 404
    data = response.json()
    assert f"Agent with ID {non_existent_id} not found" in data.get("detail", "")


@pytest.mark.asyncio
async def test_update_agent_system_prompt(test_app_client, dummy_registry: AgentRegistry):
    """Test updating an agent's system prompt."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Update System Prompt Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
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

    # Initial system prompt
    initial_prompt = "Initial system prompt"
    dummy_registry.set_agent_system_prompt(agent_id, initial_prompt)

    # Update the system prompt
    new_prompt = "This is the updated system prompt."
    update_payload = {"system_prompt": new_prompt}
    response = await test_app_client.put(
        f"/v1/agents/{agent_id}/system-prompt", json=update_payload
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent system prompt updated successfully"
    assert data.get("result") == {}

    # Verify the system prompt was updated
    response = await test_app_client.get(f"/v1/agents/{agent_id}/system-prompt")
    assert response.status_code == 200
    data = response.json()
    assert data.get("result") == {"system_prompt": new_prompt}


@pytest.mark.asyncio
async def test_update_agent_system_prompt_not_found(test_app_client, dummy_registry: AgentRegistry):
    """Test updating system prompt for a non-existent agent."""
    non_existent_id = "nonexistent"
    update_payload = {"system_prompt": "New system prompt"}
    response = await test_app_client.put(
        f"/v1/agents/{non_existent_id}/system-prompt", json=update_payload
    )

    assert response.status_code == 404
    data = response.json()
    assert f"Agent with ID {non_existent_id} not found" in data.get("detail", "")


@pytest.mark.asyncio
async def test_update_agent_system_prompt_validation_error(
    test_app_client, dummy_registry: AgentRegistry
):
    """Test updating system prompt with invalid payload."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Validation Error Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description=None,
            last_message=None,
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

    # Invalid payload (missing system_prompt field)
    invalid_payload = {"wrong_field": "This won't work"}
    response = await test_app_client.put(
        f"/v1/agents/{agent_id}/system-prompt", json=invalid_payload
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_upload_agent_to_radient_success(test_app_client, dummy_registry: AgentRegistry):
    """
    Test successful upload of an agent to the Radient marketplace.

    Args:
        test_app_client: The test HTTP client.
        dummy_registry: The agent registry fixture.

    Raises:
        AssertionError: If the response does not indicate success.
    """
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Radient Upload Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description="Test agent for Radient upload",
            last_message=None,
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

    # Patch credential and config managers, and upload method
    with (
        patch("local_operator.server.routes.agents.CredentialManager") as mock_cred_mgr,
        patch("local_operator.server.routes.agents.ConfigManager") as mock_cfg_mgr,
        patch("local_operator.server.routes.agents.RadientClient") as mock_radient_client,
        patch.object(dummy_registry, "export_agent", return_value=(MagicMock(), None)),
        patch.object(
            dummy_registry, "upload_agent_to_radient", return_value={"agent_id": agent_id}
        ),
    ):

        mock_cred_mgr.return_value.get_credential.return_value = "dummy-api-key"
        mock_cfg_mgr.return_value.get_config_value.return_value = "https://api.radienthq.com"
        mock_radient_client.return_value = MagicMock()

        response = await test_app_client.post(f"/v1/agents/{agent_id}/upload")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent uploaded to Radient successfully"
    assert data.get("result", {}).get("agent_id") == agent_id


@pytest.mark.asyncio
async def test_upload_agent_to_radient_missing_api_key(
    test_app_client, dummy_registry: AgentRegistry
):
    """
    Test upload to Radient fails with missing RADIENT_API_KEY.

    Args:
        test_app_client: The test HTTP client.
        dummy_registry: The agent registry fixture.

    Raises:
        AssertionError: If the response does not indicate unauthorized.
    """
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Radient Upload Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description="Test agent for Radient upload",
            last_message=None,
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

    with (
        patch("local_operator.server.routes.agents.CredentialManager") as mock_cred_mgr,
        patch("local_operator.server.routes.agents.ConfigManager") as mock_cfg_mgr,
    ):
        mock_cred_mgr.return_value.get_credential.return_value = None
        mock_cfg_mgr.return_value.get_config_value.return_value = "https://api.radienthq.com"

        response = await test_app_client.post(f"/v1/agents/{agent_id}/upload")

    assert response.status_code == 401 or response.status_code == 400
    data = response.json()
    # Accept either 401 or 400 for missing API key, depending on implementation
    assert (
        "RADIENT_API_KEY is required" in data.get("detail", "")
        or "Missing RADIENT_API_KEY" in data.get("detail", "")
        or "Missing required credential" in data.get("detail", "")
    )


@pytest.mark.asyncio
async def test_upload_agent_to_radient_agent_not_found(
    test_app_client, dummy_registry: AgentRegistry
):
    """
    Test upload to Radient fails if agent does not exist.

    Args:
        test_app_client: The test HTTP client.
        dummy_registry: The agent registry fixture.

    Raises:
        AssertionError: If the response does not indicate not found.
    """
    non_existent_id = "nonexistent"
    with (
        patch("local_operator.server.routes.agents.CredentialManager") as mock_cred_mgr,
        patch("local_operator.server.routes.agents.ConfigManager") as mock_cfg_mgr,
    ):
        mock_cred_mgr.return_value.get_credential.return_value = "dummy-api-key"
        mock_cfg_mgr.return_value.get_config_value.return_value = "https://api.radienthq.com"

        response = await test_app_client.post(f"/v1/agents/{non_existent_id}/upload")

    assert response.status_code == 404
    data = response.json()
    assert (
        f"Agent with ID {non_existent_id} not found" in data.get("detail", "")
        or "not found" in data.get("detail", "").lower()
    )


@pytest.mark.asyncio
async def test_upload_agent_to_radient_error(test_app_client, dummy_registry: AgentRegistry):
    """
    Test upload to Radient fails with an internal error.

    Args:
        test_app_client: The test HTTP client.
        dummy_registry: The agent registry fixture.

    Raises:
        AssertionError: If the response does not indicate a bad request.
    """
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Radient Upload Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
            description="Test agent for Radient upload",
            last_message=None,
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

    with (
        patch("local_operator.server.routes.agents.CredentialManager") as mock_cred_mgr,
        patch("local_operator.server.routes.agents.ConfigManager") as mock_cfg_mgr,
        patch.object(dummy_registry, "export_agent", return_value=(MagicMock(), None)),
        patch.object(
            dummy_registry, "upload_agent_to_radient", side_effect=Exception("Upload failed")
        ),
    ):
        mock_cred_mgr.return_value.get_credential.return_value = "dummy-api-key"
        mock_cfg_mgr.return_value.get_config_value.return_value = "https://api.radienthq.com"

        response = await test_app_client.post(f"/v1/agents/{agent_id}/upload")

    assert response.status_code == 400 or response.status_code == 500
    data = response.json()
    assert "Error uploading agent to Radient" in data.get(
        "detail", ""
    ) or "Upload failed" in data.get("detail", "")


@pytest.mark.asyncio
async def test_download_agent_from_radient_success(test_app_client, dummy_registry: AgentRegistry):
    """
    Test successful download of an agent from the Radient marketplace.

    Args:
        test_app_client: The test HTTP client.
        dummy_registry: The agent registry fixture.

    Raises:
        AssertionError: If the response does not indicate success.
    """
    agent_id = "radient-agent-123"
    mock_agent = MagicMock()
    mock_agent.model_dump.return_value = {
        "id": "imported-agent-123",
        "name": "Imported Agent",
        "created_date": "2024-01-01T00:00:00",
        "version": "0.2.16",
        "security_prompt": "Example security prompt",
        "hosting": "openrouter",
        "model": "openai/gpt-4o-mini",
        "description": "An imported agent",
        "last_message": "",
        "last_message_datetime": "2024-01-01T00:00:00",
    }

    with (
        patch("local_operator.server.routes.agents.ConfigManager") as mock_cfg_mgr,
        patch("local_operator.server.routes.agents.RadientClient") as mock_radient_client,
        patch.object(dummy_registry, "download_agent_from_radient", return_value=mock_agent),
    ):
        mock_cfg_mgr.return_value.get_config_value.return_value = "https://api.radienthq.com"
        mock_radient_client.return_value = MagicMock()

        response = await test_app_client.get(f"/v1/agents/{agent_id}/download")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent downloaded from Radient successfully"
    result = data.get("result")
    assert result["id"] == "imported-agent-123"
    assert result["name"] == "Imported Agent"


@pytest.mark.asyncio
async def test_download_agent_from_radient_error(test_app_client, dummy_registry: AgentRegistry):
    """
    Test download from Radient fails with an internal error.

    Args:
        test_app_client: The test HTTP client.
        dummy_registry: The agent registry fixture.

    Raises:
        AssertionError: If the response does not indicate a bad request.
    """
    agent_id = "radient-agent-123"
    with (
        patch("local_operator.server.routes.agents.ConfigManager") as mock_cfg_mgr,
        patch("local_operator.server.routes.agents.RadientClient") as mock_radient_client,
        patch.object(
            dummy_registry, "download_agent_from_radient", side_effect=Exception("Download failed")
        ),
    ):
        mock_cfg_mgr.return_value.get_config_value.return_value = "https://api.radienthq.com"
        mock_radient_client.return_value = MagicMock()

        response = await test_app_client.get(f"/v1/agents/{agent_id}/download")

    assert response.status_code == 400 or response.status_code == 500
    data = response.json()
    assert "Error downloading agent from Radient" in data.get(
        "detail", ""
    ) or "Download failed" in data.get("detail", "")
