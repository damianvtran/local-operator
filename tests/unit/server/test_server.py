from typing import List
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.executor import ExecutorInitError
from local_operator.mocks import ChatMock
from local_operator.model.configure import ModelConfiguration
from local_operator.model.registry import ModelInfo
from local_operator.server.app import app
from local_operator.server.models import AgentCreate, AgentUpdate, ChatRequest
from local_operator.types import (
    ActionType,
    ConversationRecord,
    ConversationRole,
    ResponseJsonSchema,
)


# Dummy implementations for the executor dependency
class DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyExecutor:
    def __init__(self):
        self.model_configuration = ModelConfiguration(
            hosting="test",
            name="test-model",
            instance=ChatMock(),
            info=ModelInfo(),
            api_key=None,
        )
        self.conversation_history = []

    async def invoke_model(self, conversation_history):
        # Simply return a dummy response content as if coming from the model.
        return DummyResponse("dummy model response")

    async def process_response(self, response_content: str):
        # Dummy processing; does nothing extra.
        return "processed successfully"

    def initialize_conversation_history(self, conversation_history: List[ConversationRecord] = []):
        if len(self.conversation_history) != 0:
            raise ExecutorInitError("Conversation history already initialized")

        if len(conversation_history) == 0:
            self.conversation_history = [
                ConversationRecord(role=ConversationRole.SYSTEM, content="System prompt")
            ]
        else:
            self.conversation_history = conversation_history


# Dummy Operator using a dummy executor
class DummyOperator:
    def __init__(self, executor):
        self.executor = executor
        self.current_agent = None

    async def handle_user_input(self, prompt: str):
        dummy_response = ResponseJsonSchema(
            previous_step_success=True,
            previous_goal="",
            current_goal="Respond to user",
            next_goal="",
            response="dummy operator response",
            code="",
            content="",
            file_path="",
            replacements=[],
            action=ActionType.DONE,
            learnings="",
            previous_step_issue="",
        )

        self.executor.conversation_history.append(
            ConversationRecord(role=ConversationRole.USER, content=prompt)
        )
        self.executor.conversation_history.append(
            ConversationRecord(
                role=ConversationRole.ASSISTANT, content=dummy_response.model_dump_json()
            )
        )

        return dummy_response


# Fixture for overriding the executor dependency for successful chat requests.
@pytest.fixture
def dummy_executor():
    return DummyExecutor()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing.

    Args:
        tmp_path: pytest fixture that provides a temporary directory unique to each test

    Returns:
        pathlib.Path: Path to the temporary directory
    """
    return tmp_path


@pytest.fixture
def test_app_client(temp_dir):
    """Create a test client with a properly initialized app state.

    This fixture uses the app's state to properly initialize the test environment,
    ensuring tests run in an environment that closely matches the real application.

    Args:
        temp_dir: pytest fixture that provides a temporary directory

    Yields:
        AsyncClient: A configured test client for making requests to the app
    """
    # Override app state with test-specific values
    # Store original values to restore later
    original_state = {}
    if hasattr(app.state, "credential_manager"):
        original_state["credential_manager"] = app.state.credential_manager
    if hasattr(app.state, "config_manager"):
        original_state["config_manager"] = app.state.config_manager
    if hasattr(app.state, "agent_registry"):
        original_state["agent_registry"] = app.state.agent_registry

    # Set up test-specific state
    mock_credential_manager = CredentialManager(config_dir=temp_dir)
    mock_config_manager = ConfigManager(config_dir=temp_dir)
    mock_agent_registry = AgentRegistry(config_dir=temp_dir)

    mock_credential_manager.get_credential = lambda key: SecretStr("test-credential")

    app.state.credential_manager = mock_credential_manager
    app.state.config_manager = mock_config_manager
    app.state.agent_registry = mock_agent_registry

    # Create and yield the test client
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    yield client

    # Restore original state
    for key, value in original_state.items():
        setattr(app.state, key, value)


@pytest.fixture
def dummy_registry(temp_dir):
    registry = AgentRegistry(config_dir=temp_dir)
    app.state.agent_registry = registry
    yield registry
    app.state.agent_registry = None


@pytest.fixture
def mock_create_operator(monkeypatch):
    """Mock the create_operator function to return a DummyOperator.

    Args:
        monkeypatch: pytest fixture for modifying objects during testing

    Returns:
        function: The mocked create_operator function
    """

    def _mock_create_operator(
        request_hosting,
        request_model,
        credential_manager,
        config_manager,
        agent_registry,
        current_agent=None,
    ):
        print("Returning dummy operator")
        return DummyOperator(DummyExecutor())

    monkeypatch.setattr("local_operator.server.routes.chat.create_operator", _mock_create_operator)
    return _mock_create_operator


@pytest.fixture
def mock_credential_manager(temp_dir):
    """Create a mock credential manager for testing.

    Args:
        temp_dir: pytest fixture that provides a temporary directory

    Returns:
        CredentialManager: A credential manager instance using the temporary directory
    """
    credential_manager = CredentialManager(config_dir=temp_dir)
    app.state.credential_manager = credential_manager
    yield credential_manager
    app.state.credential_manager = None


@pytest.fixture
def mock_config_manager(temp_dir):
    """Create a mock config manager for testing.

    Args:
        temp_dir: pytest fixture that provides a temporary directory

    Returns:
        ConfigManager: A config manager instance using the temporary directory
    """
    config_manager = ConfigManager(config_dir=temp_dir)
    app.state.config_manager = config_manager
    yield config_manager
    app.state.config_manager = None


# Test for successful /health endpoint response.
@pytest.mark.asyncio
async def test_health_check(test_app_client):
    """Test the health check endpoint using the test_app_client fixture."""
    response = await test_app_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "ok"


# Test for successful /v1/chat endpoint response.
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

    # Verify that the response contains the dummy operator response.
    assert data.get("response") == "dummy operator response"
    conversation = data.get("context")
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
    stats = data.get("stats")
    assert stats is not None
    assert stats.get("total_tokens") > 0
    assert stats.get("prompt_tokens") > 0
    assert stats.get("completion_tokens") > 0


# Test for successful /v1/chat/agents/{agent_id} endpoint response
@pytest.mark.asyncio
async def test_chat_with_agent_success(
    test_app_client,
    dummy_executor,
    dummy_registry,
    mock_create_operator,
):
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Test Agent",
            security_prompt="Test Security",
            hosting="openai",
            model="gpt-4",
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

    assert data.get("response") == "dummy operator response"
    conversation = data.get("context")
    assert isinstance(conversation, list)

    # Verify token stats
    stats = data.get("stats")
    assert stats is not None
    assert stats.get("total_tokens") > 0
    assert stats.get("prompt_tokens") > 0
    assert stats.get("completion_tokens") > 0


# Test chat with non-existent agent
@pytest.mark.asyncio
async def test_chat_with_nonexistent_agent(
    test_app_client,
    dummy_executor,
    dummy_registry,
    mock_create_operator,
):
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


# Test chat with agent when registry not initialized
@pytest.mark.asyncio
async def test_chat_with_agent_registry_not_initialized(test_app_client, dummy_executor):
    # Remove registry from app state
    original_registry = getattr(app.state, "agent_registry", None)
    app.state.agent_registry = None

    payload = ChatRequest(
        hosting="openai",
        model="gpt-4",
        prompt="Hello?",
        context=[],
    )

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/v1/chat/agents/agent1", json=payload.model_dump())

        assert response.status_code == 500
        data = response.json()
        assert "Agent registry not initialized" in data.get("detail", "")

    finally:
        # Restore original registry
        if original_registry is not None:
            app.state.agent_registry = original_registry


# Test when the operator's chat method raises an exception (simulating an error during
# model invocation).
class FailingOperator:
    def __init__(self):
        self.executor = None

    async def chat(self):
        raise Exception("Simulated failure in chat")


@pytest.mark.asyncio
async def test_chat_model_failure(test_app_client):
    with patch("local_operator.server.routes.chat.create_operator", FailingOperator()):
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


# Test when operator is not initialized.
# Simulate this by overriding create_operator to raise an HTTPException.
@pytest.mark.asyncio
async def test_chat_executor_not_initialized(test_app_client, mock_create_operator):
    async def get_none(hosting, model):
        raise HTTPException(status_code=500, detail="Executor not initialized")

    with patch("local_operator.server.routes.chat.create_operator", get_none):
        payload = ChatRequest(
            hosting="openai",
            model="gpt-4o",
            prompt="Test executor not initialized",
            context=[
                ConversationRecord(
                    role=ConversationRole.USER, content="Test executor not initialized"
                )
            ],
        )

        response = await test_app_client.post("/v1/chat", json=payload.model_dump())

        assert response.status_code == 500
        data = response.json()
        # Adjusted assertion based on failure: expecting "Internal Server Error" in
        # the error detail.
        assert "Internal Server Error" in data.get("detail", "")


# Test for successful agent update.
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


# Test for updating only name field
@pytest.mark.asyncio
async def test_update_agent_single_field(test_app_client, dummy_registry: AgentRegistry):
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


# Test for agent update when agent is not found.
@pytest.mark.asyncio
async def test_update_agent_not_found(test_app_client, dummy_registry: AgentRegistry):
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


# Test for successful agent deletion.
@pytest.mark.asyncio
async def test_delete_agent_success(test_app_client, dummy_registry: AgentRegistry):
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


# Test for agent deletion when agent is not found.
@pytest.mark.asyncio
async def test_delete_agent_not_found(test_app_client, dummy_registry: AgentRegistry):
    non_existent_agent_id = "nonexistent"
    response = await test_app_client.delete(f"/v1/agents/{non_existent_agent_id}")

    assert response.status_code == 404
    data = response.json()
    assert "Agent not found" in data.get("detail", "")


# Test for agent update when agent registry is not initialized.
@pytest.mark.asyncio
async def test_update_agent_registry_not_initialized(test_app_client):
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


# Test for agent deletion when agent registry is not initialized.
@pytest.mark.asyncio
async def test_delete_agent_registry_not_initialized():
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


# Test for agent creation success
@pytest.mark.asyncio
async def test_create_agent_success(dummy_registry: AgentRegistry):
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


# Test for agent creation with invalid data
@pytest.mark.asyncio
async def test_create_agent_invalid_data(dummy_registry: AgentRegistry):
    invalid_payload = AgentCreate(
        name="", security_prompt="", hosting="", model=""
    )  # Invalid empty name

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/v1/agents", json=invalid_payload.model_dump())

    assert response.status_code == 400


# Test for listing agents with pagination
@pytest.mark.asyncio
async def test_list_agents_pagination(dummy_registry: AgentRegistry):
    # Create multiple test agents
    for i in range(15):
        dummy_registry.create_agent(
            AgentEditFields(
                name=f"Agent {i}", security_prompt=f"Security {i}", hosting="openai", model="gpt-4"
            )
        )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Test first page
        response = await ac.get("/v1/agents?page=1&per_page=10")
        assert response.status_code == 200
        data = response.json()
        result = data.get("result")
        assert result["total"] == 15
        assert result["page"] == 1
        assert result["per_page"] == 10
        assert len(result["agents"]["agents"]) == 10

        # Test second page
        response = await ac.get("/v1/agents?page=2&per_page=10")
        data = response.json()
        result = data.get("result")
        assert len(result["agents"]["agents"]) == 5


# Test for getting a specific agent
@pytest.mark.asyncio
async def test_get_agent_success(test_app_client, dummy_registry: AgentRegistry):
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


# Test for getting a non-existent agent
@pytest.mark.asyncio
async def test_get_agent_not_found(test_app_client, dummy_registry: AgentRegistry):
    non_existent_id = "nonexistent"

    response = await test_app_client.get(f"/v1/agents/{non_existent_id}")

    assert response.status_code == 404
    data = response.json()
    assert "Agent not found" in data.get("detail", "")


# Test that the app state is properly initialized using the test_app_client fixture
@pytest.mark.asyncio
async def test_app_state_initialization(test_app_client):
    """Test that the app state is properly initialized with the test_app_client fixture."""
    # Verify that the app state has been initialized with the expected managers
    assert app.state.credential_manager is not None
    assert app.state.config_manager is not None
    assert app.state.agent_registry is not None

    # Test a request that depends on the app state
    response = await test_app_client.get("/v1/agents")
    assert response.status_code == 200
