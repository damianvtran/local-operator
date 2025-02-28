from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_operator.executor import LocalCodeExecutor
from local_operator.model.configure import configure_model
from local_operator.operator import Operator, OperatorType
from local_operator.tools import ToolRegistry
from local_operator.types import ActionType, ResponseJsonSchema


@pytest.fixture
def mock_model_config():
    model_configuration = MagicMock()
    model_configuration.instance = AsyncMock()
    model_configuration.instance.ainvoke = AsyncMock()
    return model_configuration


@pytest.fixture
def executor(mock_model_config):
    executor = LocalCodeExecutor(model_configuration=mock_model_config)
    executor.conversation_history = []
    executor.tool_registry = ToolRegistry()
    return executor


@pytest.fixture
def cli_operator(mock_model_config, executor):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    config_manager = MagicMock()
    config_manager.get_config_value = MagicMock(return_value="test_value")

    agent_registry = MagicMock()
    agent_registry.list_agents = MagicMock(return_value=[])

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_configuration=mock_model_config,
        config_manager=config_manager,
        type=OperatorType.CLI,
        agent_registry=agent_registry,
        current_agent=None,
        training_mode=False,
    )

    operator._get_input_with_history = MagicMock(return_value="noop")

    return operator


def test_cli_operator_init(mock_model_config, executor):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    config_manager = MagicMock()
    config_manager.get_config_value = MagicMock(return_value="test_value")

    agent_registry = MagicMock()
    agent_registry.list_agents = MagicMock(return_value=[])

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_configuration=mock_model_config,
        config_manager=config_manager,
        type=OperatorType.CLI,
        agent_registry=agent_registry,
        current_agent=None,
        training_mode=False,
    )

    assert operator.model_configuration == mock_model_config
    assert operator.credential_manager == credential_manager
    assert operator.executor is not None


@pytest.mark.asyncio
async def test_cli_operator_chat(cli_operator, mock_model_config):
    mock_response = ResponseJsonSchema(
        previous_step_success=True,
        previous_goal="",
        current_goal="",
        next_goal="",
        response="I'm done",
        code="",
        content="",
        file_path="",
        replacements=[],
        action=ActionType.DONE,
        learnings="",
        previous_step_issue="",
    )
    mock_model_config.instance.ainvoke.return_value.content = mock_response.model_dump_json()
    cli_operator._agent_should_exit = MagicMock(return_value=True)

    with patch("builtins.input", return_value="exit"):
        await cli_operator.chat()

        assert (
            cli_operator.executor.conversation_history[-1].content
            == mock_response.model_dump_json()
        )


def test_agent_is_done(cli_operator):
    test_cases = [
        {"name": "None response", "response": None, "expected": False},
        {
            "name": "DONE action",
            "response": ResponseJsonSchema(
                previous_step_success=True,
                previous_goal="",
                current_goal="",
                next_goal="",
                response="",
                code="",
                content="",
                file_path="",
                replacements=[],
                action=ActionType.DONE,
                learnings="",
                previous_step_issue="",
            ),
            "expected": True,
        },
        {
            "name": "Other action",
            "response": ResponseJsonSchema(
                previous_step_success=True,
                previous_goal="",
                current_goal="",
                next_goal="",
                response="",
                code="",
                content="",
                file_path="",
                replacements=[],
                action=ActionType.CODE,
                learnings="",
                previous_step_issue="",
            ),
            "expected": False,
        },
    ]

    for test_case in test_cases:
        cli_operator._agent_should_exit = MagicMock(return_value=False)
        assert (
            cli_operator._agent_is_done(test_case["response"]) == test_case["expected"]
        ), f"Failed test case: {test_case['name']}"

        # Test with agent_should_exit returning True
        if test_case["response"] is not None:
            cli_operator._agent_should_exit = MagicMock(return_value=True)
            assert (
                cli_operator._agent_is_done(test_case["response"]) is True
            ), f"Failed test case: {test_case['name']} with agent_should_exit=True"


def test_agent_requires_user_input(cli_operator):
    test_cases = [
        {"name": "None response", "response": None, "expected": False},
        {
            "name": "ASK action",
            "response": ResponseJsonSchema(
                previous_step_success=True,
                previous_goal="",
                current_goal="",
                next_goal="",
                response="",
                code="",
                content="",
                file_path="",
                replacements=[],
                action=ActionType.ASK,
                learnings="",
                previous_step_issue="",
            ),
            "expected": True,
        },
        {
            "name": "Other action",
            "response": ResponseJsonSchema(
                previous_step_success=True,
                previous_goal="",
                current_goal="",
                next_goal="",
                response="",
                code="",
                content="",
                file_path="",
                replacements=[],
                action=ActionType.DONE,
                learnings="",
                previous_step_issue="",
            ),
            "expected": False,
        },
    ]

    for test_case in test_cases:
        assert (
            cli_operator._agent_requires_user_input(test_case["response"]) == test_case["expected"]
        ), f"Failed test case: {test_case['name']}"


def test_agent_should_exit(cli_operator):
    test_cases = [
        {"name": "None response", "response": None, "expected": False},
        {
            "name": "BYE action",
            "response": ResponseJsonSchema(
                previous_step_success=True,
                previous_goal="",
                current_goal="",
                next_goal="",
                response="",
                code="",
                content="",
                file_path="",
                replacements=[],
                action=ActionType.BYE,
                learnings="",
                previous_step_issue="",
            ),
            "expected": True,
        },
        {
            "name": "Other action",
            "response": ResponseJsonSchema(
                previous_step_success=True,
                previous_goal="",
                current_goal="",
                next_goal="",
                response="",
                code="",
                content="",
                file_path="",
                replacements=[],
                action=ActionType.CODE,
                learnings="",
                previous_step_issue="",
            ),
            "expected": False,
        },
    ]

    for test_case in test_cases:
        assert (
            cli_operator._agent_should_exit(test_case["response"]) == test_case["expected"]
        ), f"Failed test case: {test_case['name']}"


@pytest.mark.asyncio
async def test_operator_print_hello_world(cli_operator):
    """Test that operator correctly handles 'print hello world' command and output
    using ChatMock."""
    # Configure mock model
    mock_model_config = configure_model("test", "", MagicMock())

    mock_executor = LocalCodeExecutor(mock_model_config)
    mock_executor.tool_registry = ToolRegistry()
    cli_operator.executor = mock_executor
    cli_operator.model_configuration = mock_model_config

    # Execute command and get response
    await cli_operator.handle_user_input("print hello world")

    # Verify conversation history was updated
    assert len(cli_operator.executor.conversation_history) > 0
    last_message = cli_operator.executor.conversation_history[-1]
    last_message_content = ResponseJsonSchema.model_validate_json(last_message.content)

    assert last_message_content is not None
    assert last_message_content.previous_step_success is True
    assert last_message_content.previous_goal == "Print Hello World"
    assert last_message_content.current_goal == "Complete task"
    assert last_message_content.response == "I have printed 'Hello World' to the console."
    assert last_message_content.code == ""
    assert last_message_content.action == ActionType.DONE
