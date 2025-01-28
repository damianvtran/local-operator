import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_operator.agent import CliOperator, LocalCodeExecutor


@pytest.fixture
def mock_model():
    model = AsyncMock()
    model.ainvoke = AsyncMock()
    return model


@pytest.fixture
def executor(mock_model):
    return LocalCodeExecutor(mock_model)


@pytest.fixture
def cli_operator(mock_model):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    operator = CliOperator(
        credential_manager=credential_manager,
        model_instance=mock_model,
    )

    operator._get_input_with_history = MagicMock(return_value="noop")

    return operator


def test_extract_code_blocks(executor):
    test_text = """
    Some text
    ```python
    print('hello')
    ```
    More text
    ```python
    x = 1 + 1
    ```
    """
    result = executor.extract_code_blocks(test_text)
    assert len(result) == 2
    assert "print('hello')" in result[0]
    assert "x = 1 + 1" in result[1]


@pytest.mark.asyncio
async def test_check_code_safety_safe(executor, mock_model):
    mock_model.ainvoke.return_value.content = "no"
    code = "print('hello')"
    result = await executor.check_code_safety(code)
    assert result is False
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_check_code_safety_unsafe(executor, mock_model):
    mock_model.ainvoke.return_value.content = "yes"
    code = "import os; os.remove('important_file.txt')"
    result = await executor.check_code_safety(code)
    assert result is True
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_execute_code_success(executor, mock_model):
    mock_model.ainvoke.return_value.content = "no"  # Safety check passes
    code = "print('hello')"

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.execute_code(code)
        assert "✓ Code Execution Successful" in result
        assert "hello" in result


@pytest.mark.asyncio
async def test_execute_code_no_output(executor, mock_model):
    mock_model.ainvoke.return_value.content = "no"  # Safety check passes
    code = "x = 1 + 1"  # Code that produces no output

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.execute_code(code)
        assert "✓ Code Execution Successful" in result
        assert "[No output]" in result


@pytest.mark.asyncio
async def test_execute_code_timeout(executor, mock_model):
    mock_model.ainvoke.return_value.content = "no"
    code = "import time; time.sleep(5)"

    result = await executor.execute_code(code, max_retries=0, timeout=0.1)

    assert "✗ Code Execution Failed" in result
    assert "timed out after 0.1 seconds" in result
    assert len(executor.conversation_history) > 0
    assert "timed out" in executor.conversation_history[-1]["content"]


@pytest.mark.asyncio
async def test_process_response(executor, mock_model):
    response = """
    Here's some code:
    ```python
    print('hello world')
    ```
    """
    mock_model.ainvoke.return_value.content = "no"

    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        await executor.process_response(response)
        output = mock_stdout.getvalue()
        assert "Executing Code Blocks" in output
        assert "hello world" in output


def test_cli_operator_init(mock_model):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    operator = CliOperator(
        credential_manager=credential_manager,
        model_instance=mock_model,
    )

    assert operator.model == mock_model
    assert operator.credential_manager == credential_manager
    assert operator.executor is not None


@pytest.mark.asyncio
async def test_cli_operator_chat(cli_operator, mock_model):
    mock_model.ainvoke.return_value.content = "DONE"
    cli_operator._agent_should_exit = MagicMock(return_value=True)

    with patch("builtins.input", return_value="exit"):
        await cli_operator.chat()

        assert cli_operator.executor.conversation_history[-1]["content"] == "DONE"


def test_agent_is_done(cli_operator):
    mock_response = MagicMock()
    mock_response.content = "Some output\nDONE"
    assert cli_operator._agent_is_done(mock_response) is True


def test_agent_should_exit(cli_operator):
    mock_response = MagicMock()
    mock_response.content = "Some output\nBye!"
    assert cli_operator._agent_should_exit(mock_response) is True
