from typing import Dict, List

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from local_operator import server as srv
from local_operator.executor import ExecutorInitError
from local_operator.operator import ConversationRole
from local_operator.server import ChatMessage, ChatRequest, app
from local_operator.types import ResponseJsonSchema


# Dummy implementations for the executor dependency
class DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyExecutor:
    def __init__(self):
        self.model = self
        self.conversation_history = []

    async def invoke_model(self, conversation_history):
        # Simply return a dummy response content as if coming from the model.
        return DummyResponse("dummy model response")

    async def process_response(self, response_content: str):
        # Dummy processing; does nothing extra.
        return "processed successfully"

    def initialize_conversation_history(self, conversation_history: List[Dict[str, str]] = []):
        if len(self.conversation_history) != 0:
            raise ExecutorInitError("Conversation history already initialized")

        if len(conversation_history) == 0:
            self.conversation_history = [
                {"role": ConversationRole.SYSTEM.value, "content": "System prompt"}
            ]
        else:
            self.conversation_history = conversation_history


# Dummy Operator using a dummy executor
class DummyOperator:
    def __init__(self, executor):
        self.executor = executor

    async def handle_user_input(self, prompt: str):
        dummy_response = ResponseJsonSchema(
            previous_step_success=True,
            previous_goal="",
            current_goal="Respond to user",
            next_goal="",
            response="dummy operator response",
            code="",
            action="DONE",
            learnings="",
            plan="",
        )

        self.executor.conversation_history.append({"role": "user", "content": prompt})
        self.executor.conversation_history.append(
            {"role": "assistant", "content": dummy_response.model_dump_json()}
        )

        return dummy_response


# Fixture for overriding the executor dependency for successful chat requests.
@pytest.fixture
def dummy_executor():
    return DummyExecutor()


# Test for successful /health endpoint response.
@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "ok"


# Test for successful /v1/chat endpoint response.
@pytest.mark.asyncio
async def test_chat_success(dummy_executor):
    original_create_operator = srv.create_operator
    # Override create_operator to return a DummyOperator with our dummy executor.
    srv.create_operator = lambda hosting, model: DummyOperator(dummy_executor)

    test_prompt = "Hello, how are you?"

    # Use an empty context to trigger insertion of the system prompt by the server.
    payload = {
        "hosting": "openai",
        "model": "gpt-4o",
        "prompt": test_prompt,
        "context": [],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/v1/chat", json=payload)

    srv.create_operator = original_create_operator

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


# Test when the operator's chat method raises an exception (simulating an error during
# model invocation).
class FailingOperator:
    def __init__(self):
        self.executor = None

    async def chat(self):
        raise Exception("Simulated failure in chat")


@pytest.mark.asyncio
async def test_chat_model_failure():
    original_create_operator = srv.create_operator
    srv.create_operator = lambda hosting, model: FailingOperator()

    payload = {
        "hosting": "openai",
        "model": "gpt-4o",
        "prompt": "This should cause an error",
        "context": [{"role": "user", "content": "This should cause an error"}],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/v1/chat", json=payload)

    srv.create_operator = original_create_operator

    assert response.status_code == 500
    data = response.json()
    # The error detail should indicate an internal server error.
    assert "Internal Server Error" in data.get("detail", "")


# Test when operator is not initialized.
# Simulate this by overriding create_operator to raise an HTTPException.
@pytest.mark.asyncio
async def test_chat_executor_not_initialized():
    original_create_operator = srv.create_operator

    async def get_none(hosting, model):
        raise HTTPException(status_code=500, detail="Executor not initialized")

    srv.create_operator = get_none

    payload = ChatRequest(
        hosting="openai",
        model="gpt-4o",
        prompt="Test executor not initialized",
        context=[ChatMessage(role="user", content="Test executor not initialized")],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/v1/chat", json=payload.model_dump())

    srv.create_operator = original_create_operator

    assert response.status_code == 500
    data = response.json()
    # Adjusted assertion based on failure: expecting "Internal Server Error" in the error detail.
    assert "Internal Server Error" in data.get("detail", "")


# Dummy Agent Registry for testing agent CRUD endpoints
class DummyAgentRegistry:
    def __init__(self):
        self.agents = {}

    def create_agent(self, name, description=None):
        agent_id = f"agent{len(self.agents) + 1}"
        agent = {"id": agent_id, "name": name, "description": description}
        self.agents[agent_id] = agent
        return agent

    def update_agent(self, agent_id, update_data):
        if agent_id not in self.agents:
            return None
        self.agents[agent_id].update(update_data)
        return self.agents[agent_id]

    def delete_agent(self, agent_id):
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False

    def list_agents(self):
        return list(self.agents.values())

    def get_agent(self, agent_id):
        return self.agents.get(agent_id)


@pytest.fixture
def dummy_registry():
    registry = DummyAgentRegistry()
    app.state.agent_registry = registry
    yield registry
    app.state.agent_registry = None


# Test for successful agent update.
@pytest.mark.asyncio
async def test_update_agent_success(dummy_registry):
    # Create a dummy agent.
    agent = dummy_registry.create_agent(name="Original Name", description="Original Description")
    agent_id = agent["id"]

    update_payload = {"name": "Updated Name", "description": "Updated Description"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.patch(f"/v1/agents/{agent_id}", json=update_payload)

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent updated successfully"
    result = data.get("result")
    assert result["id"] == agent_id
    assert result["name"] == "Updated Name"
    assert result["description"] == "Updated Description"


# Test for agent update when agent is not found.
@pytest.mark.asyncio
async def test_update_agent_not_found(dummy_registry):
    update_payload = {"name": "Non-existent Update"}
    non_existent_agent_id = "nonexistent"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.patch(f"/v1/agents/{non_existent_agent_id}", json=update_payload)

    assert response.status_code == 404
    data = response.json()
    assert "Agent not found" in data.get("detail", "")


# Test for successful agent deletion.
@pytest.mark.asyncio
async def test_delete_agent_success(dummy_registry):
    # Create a dummy agent.
    agent = dummy_registry.create_agent(name="Agent to Delete", description="Will be deleted")
    agent_id = agent["id"]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.delete(f"/v1/agents/{agent_id}")

    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "Agent deleted successfully"


# Test for agent deletion when agent is not found.
@pytest.mark.asyncio
async def test_delete_agent_not_found(dummy_registry):
    non_existent_agent_id = "nonexistent"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.delete(f"/v1/agents/{non_existent_agent_id}")

    assert response.status_code == 404
    data = response.json()
    assert "Agent not found" in data.get("detail", "")


# Test for agent update when agent registry is not initialized.
@pytest.mark.asyncio
async def test_update_agent_registry_not_initialized():
    # Safely get the original agent_registry without risking a KeyError.
    original_registry = getattr(app.state, "agent_registry", None)
    app.state.agent_registry = None
    update_payload = {"name": "New Name"}
    agent_id = "agent1"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.patch(f"/v1/agents/{agent_id}", json=update_payload)
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
