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

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.env import EnvConfig
from local_operator.jobs import JobManager
from local_operator.model.configure import ModelConfiguration
from local_operator.model.registry import ModelInfo
from local_operator.scheduler_service import SchedulerService
from local_operator.server.app import app
from local_operator.server.utils.event_broker import EventBroker
from local_operator.server.utils.operator import ExecutorInitError
from local_operator.types import (
    ActionType,
    AgentState,
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
    OperatorType,
    ProcessResponseStatus,
    ResponseJsonSchema,
)


# Dummy implementations of the session-facade surface the chat routes use.
# They stand in for ``server.utils.operator.ServerOperator`` /
# ``ServerExecutor`` so route tests exercise the envelope, not the engine.
class DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyExecutor:
    def __init__(self):
        # ``instance`` is None in the rewritten engine — streaming goes
        # through the provider wire clients, not a chat-model object.
        self.model_configuration = ModelConfiguration(
            hosting="test",
            name="test-model",
            info=ModelInfo(
                id="test-model",
                name="test-model",
                description="Mock model",
                recommended=True,
            ),
            api_key=None,
        )
        self.agent_state = AgentState(
            version="",
            conversation=[],
            execution_history=[],
            learnings=[],
            current_plan=None,
            instruction_details=None,
            agent_system_prompt=None,
        )

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
            self.agent_state.conversation = []

        if len(self.agent_state.conversation) != 0:
            raise ExecutorInitError("Conversation history already initialized")

        history = [
            ConversationRecord(
                role=ConversationRole.SYSTEM, content="System prompt", is_system_prompt=True
            )
        ]

        if len(new_conversation_history) == 0:
            self.agent_state.conversation = history
        else:
            filtered_history = [
                record for record in new_conversation_history if not record.is_system_prompt
            ]
            self.agent_state.conversation = history + filtered_history

    def add_to_code_history(self, code_execution_result: CodeExecutionResult, response):
        self.agent_state.execution_history.append(code_execution_result)


# Dummy Operator using a dummy executor
class DummyOperator:
    def __init__(self, executor):
        self.executor = executor
        self.current_agent = None

    async def handle_user_input(
        self, prompt: str, user_message_id: str | None = None, attachments: List[str] = []
    ):

        dummy_response = ResponseJsonSchema(
            response="dummy operator response",
            code="",
            content="",
            file_path="",
            mentioned_files=[],
            replacements=[],
            action=ActionType.DONE,
            learnings="",
        )

        self.executor.agent_state.conversation.append(
            ConversationRecord(role=ConversationRole.USER, content=prompt, files=attachments)
        )
        self.executor.agent_state.conversation.append(
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
                files=attachments,
                role=ConversationRole.USER,
                status=ProcessResponseStatus.SUCCESS,
            ),
            None,
        )

        return dummy_response, "dummy operator response"


@pytest.fixture(autouse=True)
def _no_live_api_key_check(monkeypatch):
    """Keep ``PUT /v1/auth/providers/{id}/key`` off the network in unit tests.

    The route asks the provider whether a key is valid before storing it
    (``providers.key_check``). Server tests save fabricated keys to exercise the
    STORE and the defaults policy, and a live check would turn every one of them
    into a real request that the provider rightly answers 401 (-> 422 here), and
    into a test that depends on the network. The verdict is stubbed to "could not
    check", which is the path that stores the key; the check's own behaviour is
    pinned in ``tests/unit/providers/test_key_check.py`` against a mock transport,
    and the route's reaction to each verdict in ``test_desktop_controls.py``.
    """
    from local_operator.providers import key_check

    async def _unchecked(_provider_id, _key, **_kwargs):
        return key_check.KeyCheck(None, None)

    monkeypatch.setattr(key_check, "check_api_key", _unchecked)


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
    original_desktop_auth = getattr(app.state, "desktop_auth", None)
    # These clients bypass lifespan. A shared provider store must follow this
    # fixture's config root rather than survive into the next test's managers.
    app.state.desktop_auth = None
    if hasattr(app.state, "config_manager"):
        original_state["config_manager"] = app.state.config_manager
    if hasattr(app.state, "agent_registry"):
        original_state["agent_registry"] = app.state.agent_registry
    if hasattr(app.state, "job_manager"):
        original_state["job_manager"] = app.state.job_manager
    if hasattr(app.state, "event_broker"):
        original_state["event_broker"] = app.state.event_broker
    if hasattr(app.state, "env_config"):
        original_state["env_config"] = app.state.env_config

    # Set up test-specific state
    mock_config_manager = ConfigManager(config_dir=temp_dir)
    # Use a shorter refresh interval for tests to ensure changes are quickly reflected
    mock_agent_registry = AgentRegistry(config_dir=temp_dir, refresh_interval=1.0)
    mock_job_manager = JobManager()
    # The SSE fan-out is app state the async chat routes depend on, so a client
    # built without lifespan must supply it. It is the ONLY streaming fan-out:
    # the websocket manager this fixture used to build went with the removal of
    # the /v1/ws transport.
    mock_event_broker = EventBroker()
    mock_env_config = EnvConfig(
        radient_api_base_url="https://api.radienthq.com/v1",
    )
    mock_scheduler_service = SchedulerService(
        agent_registry=mock_agent_registry,
        job_manager=mock_job_manager,
        config_manager=mock_config_manager,
        env_config=mock_env_config,
        operator_type=OperatorType.SERVER,
        verbosity_level=VerbosityLevel.QUIET,
    )
    _ = mock_scheduler_service.start()

    # PR2a: the plaintext/legacy getter is gone, so the fixture arms a
    # provider-class STORE ROW for RADIENT_API_KEY — the tier the Radient
    # resolver actually reads now. Written through the production writer so
    # the row name and namespace cannot drift from the reader.
    from local_operator.providers.registry import store_provider_key
    from local_operator.secrets.access import open_store

    open_store(temp_dir, create=True)
    store_provider_key("RADIENT_API_KEY", "test-credential", base=temp_dir)

    app.state.config_manager = mock_config_manager
    app.state.agent_registry = mock_agent_registry
    app.state.job_manager = mock_job_manager
    app.state.event_broker = mock_event_broker
    app.state.env_config = mock_env_config
    app.state.scheduler_service = mock_scheduler_service

    # Create and yield the test client
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    yield client

    # Restore original state
    created_auth = getattr(app.state, "desktop_auth", None)
    if created_auth is not None:
        created_auth.store.close()
    app.state.desktop_auth = original_desktop_auth
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
def mock_credential_manager(temp_dir):
    """The config manager the store-first readers resolve their root through.

    The fixture name is kept because the server tests depend on it by name; what
    it yields is the ``ConfigManager`` whose ``config_dir`` the readers take,
    now that PR2b deleted the ``CredentialManager`` this used to hand back (and
    the ``app.state.credential_manager`` slot it used to fill).

    ``temp_dir`` and NOT the HOME-derived ``paths.config_dir()``: the tests arm
    their rows through this manager (``_store_key``) and the app under test is
    given ``ConfigManager(config_dir=temp_dir)``, so the same root has to be on
    both sides or a row is written under one root and looked up under another.
    A HOME-derived root is the worse failure of the two — an unisolated run then
    writes into the operator's real store.

    IT ALSO FILLS ``app.state.config_manager``, and that is not decoration
    (review round 2, M2). On the base the model route read
    ``app.state.credential_manager`` — the slot this fixture used to fill — and
    after PR2b it takes its root through ``Depends(get_config_manager)``, which
    reads ``app.state.config_manager``. Not filling it left the slot to whatever
    an EARLIER test in the same worker had seeded, so
    ``test_server_models.py`` passed only in a shared run and failed 9/11 when
    run alone (``KeyError: 'config_manager'``). The previous value is restored
    rather than cleared to ``None``, so a test that composes both this and
    ``mock_config_manager`` does not tear down the other's slot.
    """
    config_manager = ConfigManager(config_dir=temp_dir)
    previous = getattr(app.state, "config_manager", None)
    app.state.config_manager = config_manager
    try:
        yield config_manager
    finally:
        app.state.config_manager = previous


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
