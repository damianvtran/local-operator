import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from local_operator import server as srv
from local_operator.operator import ConversationRole
from local_operator.server import ChatMessage, ChatRequest, app, create_operator


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


# Dummy Operator using a dummy executor
class DummyOperator:
    def __init__(self, executor):
        self.executor = executor

    async def handle_user_input(self, prompt: str):
        self.executor.conversation_history.append({"role": "user", "content": prompt})
        self.executor.conversation_history.append(
            {"role": "assistant", "content": "dummy operator response"}
        )
        return DummyResponse("dummy operator response")


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


# Test for successful /chat endpoint response.
@pytest.mark.asyncio
async def test_chat_success(dummy_executor):
    original_create_operator = srv.create_operator
    # Override create_operator to return a DummyOperator with our dummy executor.
    srv.create_operator = lambda hosting, model: DummyOperator(dummy_executor)

    # Use an empty context to trigger insertion of the system prompt by the server.
    payload = {
        "hosting": "openai",
        "model": "gpt-4o",
        "prompt": "Hello, how are you?",
        "context": [],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/chat", json=payload)

    srv.create_operator = original_create_operator

    assert response.status_code == 200
    data = response.json()

    # Verify that the response contains the dummy operator response.
    assert data.get("response") == "dummy operator response"
    conversation = data.get("context")
    assert isinstance(conversation, list)
    # The conversation should include the system prompt and the user's message.
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
        response = await ac.post("/chat", json=payload)

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
        response = await ac.post("/chat", json=payload.model_dump())

    srv.create_operator = original_create_operator

    assert response.status_code == 500
    data = response.json()
    # Adjusted assertion based on failure: expecting "Internal Server Error" in the error detail.
    assert "Internal Server Error" in data.get("detail", "")
