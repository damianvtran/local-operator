from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_operator.model import configure_model
from local_operator.operator import LocalCodeExecutor, Operator, OperatorType
from local_operator.types import ResponseJsonSchema


@pytest.fixture
def mock_model():
    model = AsyncMock()
    model.ainvoke = AsyncMock()
    return model


@pytest.fixture
def executor(mock_model):
    return LocalCodeExecutor(mock_model)


@pytest.fixture
def cli_operator(mock_model, executor):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    config_manager = MagicMock()
    config_manager.get_config_value = MagicMock(return_value="test_value")

    agent_registry = MagicMock()
    agent_registry.list_agents = MagicMock(return_value=[])

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_instance=mock_model,
        config_manager=config_manager,
        type=OperatorType.CLI,
        agent_registry=agent_registry,
        current_agent=None,
        training_mode=False,
    )

    operator._get_input_with_history = MagicMock(return_value="noop")

    return operator


def test_cli_operator_init(mock_model, executor):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    config_manager = MagicMock()
    config_manager.get_config_value = MagicMock(return_value="test_value")

    agent_registry = MagicMock()
    agent_registry.list_agents = MagicMock(return_value=[])

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_instance=mock_model,
        config_manager=config_manager,
        type=OperatorType.CLI,
        agent_registry=agent_registry,
        current_agent=None,
        training_mode=False,
    )

    assert operator.model == mock_model
    assert operator.credential_manager == credential_manager
    assert operator.executor is not None


@pytest.mark.asyncio
async def test_cli_operator_chat(cli_operator, mock_model):
    mock_response = ResponseJsonSchema(
        previous_step_success=True,
        previous_goal="",
        current_goal="",
        next_goal="",
        response="I'm done",
        code="",
        action="DONE",
        learnings="",
        plan="",
    )
    mock_model.ainvoke.return_value.content = mock_response.model_dump_json()
    cli_operator._agent_should_exit = MagicMock(return_value=True)

    with patch("builtins.input", return_value="exit"):
        await cli_operator.chat()

        assert (
            cli_operator.executor.conversation_history[-1]["content"]
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
                action="DONE",
                learnings="",
                plan="",
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
                action="CONTINUE",
                learnings="",
                plan="",
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
                action="ASK",
                learnings="",
                plan="",
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
                action="CONTINUE",
                learnings="",
                plan="",
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
                action="BYE",
                learnings="",
                plan="",
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
                action="CONTINUE",
                learnings="",
                plan="",
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
    mock_model, _ = configure_model("test", "", None)

    mock_executor = LocalCodeExecutor(mock_model)
    cli_operator.executor = mock_executor
    cli_operator.model = mock_model

    # Execute command and get response
    await cli_operator.handle_user_input("print hello world")

    # Verify conversation history was updated
    assert len(cli_operator.executor.conversation_history) > 0
    last_message = cli_operator.executor.conversation_history[-1]
    last_message_content = ResponseJsonSchema.model_validate_json(last_message["content"])

    assert last_message_content is not None
    assert last_message_content.previous_step_success is True
    assert last_message_content.previous_goal == "Print Hello World"
    assert last_message_content.current_goal == "Complete task"
    assert last_message_content.response == "I have printed 'Hello World' to the console."
    assert last_message_content.code == ""
    assert last_message_content.action == "DONE"
