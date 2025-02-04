from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_operator.operator import (
    LocalCodeExecutor,
    Operator,
    OperatorType,
    get_tools_str,
)


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

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_instance=mock_model,
        config_manager=config_manager,
        type=OperatorType.CLI,
    )

    operator._get_input_with_history = MagicMock(return_value="noop")

    return operator


def test_cli_operator_init(mock_model, executor):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    config_manager = MagicMock()
    config_manager.get_config_value = MagicMock(return_value="test_value")

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_instance=mock_model,
        config_manager=config_manager,
        type=OperatorType.CLI,
    )

    assert operator.model == mock_model
    assert operator.credential_manager == credential_manager
    assert operator.executor is not None


@pytest.mark.asyncio
async def test_cli_operator_chat(cli_operator, mock_model):
    mock_model.ainvoke.return_value.content = "[DONE]"
    cli_operator._agent_should_exit = MagicMock(return_value=True)

    with patch("builtins.input", return_value="exit"):
        await cli_operator.chat()

        assert cli_operator.executor.conversation_history[-1]["content"] == "[DONE]"


def test_agent_is_done(cli_operator):
    test_cases = [
        {"name": "DONE keyword", "content": "Some output\n[DONE]", "expected": True},
        {"name": "ASK keyword", "content": "Some output\n[ASK]", "expected": False},
        {"name": "No special keyword", "content": "Some output\nregular text", "expected": False},
    ]

    for test_case in test_cases:
        mock_response = MagicMock()
        mock_response.content = test_case["content"]
        assert (
            cli_operator._agent_is_done(mock_response) == test_case["expected"]
        ), f"Failed test case: {test_case['name']}"


def test_agent_requires_user_input(cli_operator):
    test_cases = [
        {"name": "ASK keyword", "content": "Some output\n[ASK]", "expected": True},
        {"name": "DONE keyword", "content": "Some output\n[DONE]", "expected": False},
        {"name": "No special keyword", "content": "Some output\nregular text", "expected": False},
    ]

    for test_case in test_cases:
        mock_response = MagicMock()
        mock_response.content = test_case["content"]
        assert (
            cli_operator._agent_requires_user_input(mock_response) == test_case["expected"]
        ), f"Failed test case: {test_case['name']}"


def test_agent_should_exit(cli_operator):
    mock_response = MagicMock()
    mock_response.content = "Some output\n[BYE]"
    assert cli_operator._agent_should_exit(mock_response) is True


def test_get_tools_str():
    test_cases = [
        {"name": "No module provided", "module": None, "expected": ""},
        {
            "name": "Empty module (0 functions)",
            "module": MagicMock(__dir__=MagicMock(return_value=[])),
            "expected": "",
        },
        {
            "name": "One function module",
            "module": MagicMock(),
            "expected": "- test_func(param1: str, param2: int) -> bool: Test function description",
        },
        {
            "name": "Two function module",
            "module": MagicMock(),
            "expected": (
                "- test_func(param1: str, param2: int) -> bool: Test function description\n"
                "- other_func(name: str) -> str: Another test function"
            ),
        },
        {
            "name": "Async function module",
            "module": MagicMock(),
            "expected": "- async async_func(url: str) -> str: Async test function",
        },
    ]

    # Set up test functions
    def test_func(param1: str, param2: int) -> bool:
        """Test function description"""
        return True

    test_func.__name__ = "test_func"
    test_func.__doc__ = "Test function description"

    def other_func(name: str) -> str:
        """Another test function"""
        return name

    other_func.__name__ = "other_func"
    other_func.__doc__ = "Another test function"

    async def async_func(url: str) -> str:
        """Async test function"""
        return url

    async_func.__name__ = "async_func"
    async_func.__doc__ = "Async test function"

    # Configure the one function module
    test_cases[2]["module"].test_func = test_func
    test_cases[2]["module"].__dir__ = MagicMock(return_value=["test_func"])

    # Configure the two function module
    test_cases[3]["module"].test_func = test_func
    test_cases[3]["module"].other_func = other_func
    test_cases[3]["module"].__dir__ = MagicMock(
        return_value=["test_func", "other_func", "_private"]
    )

    # Configure the async function module
    test_cases[4]["module"].async_func = async_func
    test_cases[4]["module"].__dir__ = MagicMock(return_value=["async_func"])

    # Run test cases
    for case in test_cases:
        result = get_tools_str(case["module"])
        result_lines = sorted(result.split("\n")) if result else []
        expected_lines = sorted(case["expected"].split("\n")) if case["expected"] else []
        assert (
            result_lines == expected_lines
        ), f"Failed test case: {case['name']}\nExpected: {case['expected']}\nGot: {result}"
