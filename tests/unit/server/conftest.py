"""
Fixtures for server tests.

This module provides pytest fixtures for testing the FastAPI server components,
including mock clients, executors, and dependencies needed for API testing.
"""

import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.executor import ExecutorInitError
from local_operator.jobs import JobManager
from local_operator.mocks import ChatMock
from local_operator.model.configure import ModelConfiguration
from local_operator.model.registry import ModelInfo
from local_operator.server.app import app
from local_operator.types import (
    ActionType,
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
    ProcessResponseStatus,
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
        self.code_history = []

    async def invoke_model(self, conversation_history):
        # Simply return a dummy response content as if coming from the model.
        return DummyResponse("dummy model response")

    async def process_response(self, response_content: str):
        # Dummy processing; does nothing extra.
        return "processed successfully"

    def initialize_conversation_history(
        self, new_conversation_history: List[ConversationRecord] = [], overwrite: bool = False
    ):
        if overwrite:
            self.conversation_history = []

        if len(self.conversation_history) != 0:
            raise ExecutorInitError("Conversation history already initialized")

        history = [
            ConversationRecord(
                role=ConversationRole.SYSTEM, content="System prompt", is_system_prompt=True
            )
        ]

        if len(new_conversation_history) == 0:
            self.conversation_history = history
        else:
            filtered_history = [
                record for record in new_conversation_history if not record.is_system_prompt
            ]
            self.conversation_history = history + filtered_history

    def add_to_code_history(self, code_execution_result: CodeExecutionResult, response):
        self.code_history.append(code_execution_result)


# Dummy Operator using a dummy executor
class DummyOperator:
    def __init__(self, executor):
        self.executor = executor
        self.current_agent = None

    async def handle_user_input(self, prompt: str, user_message_id: str | None = None):
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
        self.executor.add_to_code_history(
            CodeExecutionResult(
                id=user_message_id if user_message_id else str(uuid.uuid4()),
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message=prompt,
                role=ConversationRole.USER,
                status=ProcessResponseStatus.SUCCESS,
            ),
            None,
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
    # Use a shorter refresh interval for tests to ensure changes are quickly reflected
    mock_agent_registry = AgentRegistry(config_dir=temp_dir, refresh_interval=1.0)
    mock_job_manager = JobManager()

    mock_credential_manager.get_credential = lambda key: SecretStr("test-credential")

    app.state.credential_manager = mock_credential_manager
    app.state.config_manager = mock_config_manager
    app.state.agent_registry = mock_agent_registry
    app.state.job_manager = mock_job_manager

    # Create and yield the test client
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    yield client

    # Restore original state
    for key, value in original_state.items():
        setattr(app.state, key, value)


@pytest.fixture
def dummy_registry(temp_dir):
    # Use a shorter refresh interval for tests to ensure changes are quickly reflected
    registry = AgentRegistry(config_dir=temp_dir, refresh_interval=1.0)
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
    dummy_operator = DummyOperator(DummyExecutor())
    with patch("local_operator.server.routes.chat.create_operator", return_value=dummy_operator):
        yield dummy_operator


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


@pytest.fixture
def mock_job_manager():
    """Create a mock job manager for testing."""
    manager = MagicMock(spec=JobManager)
    manager.get_job = AsyncMock()
    manager.list_jobs = AsyncMock()
    manager.cancel_job = AsyncMock()
    manager.cleanup_old_jobs = AsyncMock()
    manager.get_job_summary = MagicMock()
    manager.create_job = AsyncMock()
    manager.register_task = AsyncMock()
    manager.update_job_status = AsyncMock()

    app.state.job_manager = manager
    yield manager
    app.state.job_manager = None
