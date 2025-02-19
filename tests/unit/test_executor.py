import io
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIError

from local_operator.executor import (
    ConfirmSafetyResult,
    LocalCodeExecutor,
    ProcessResponseStatus,
    get_confirm_safety_result,
    get_context_vars_str,
    process_json_response,
)
from local_operator.operator import Operator, OperatorType
from local_operator.types import (
    ActionType,
    ConversationRecord,
    ConversationRole,
    ResponseJsonSchema,
)


# Helper function to normalize code blocks.
def normalize_code_block(code: str) -> str:
    return textwrap.dedent(code).strip()


@pytest.fixture
def tmp_path() -> Generator[Path, None, None]:
    """
    Fixture to provide a temporary directory path.

    Returns:
        Path: The path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model_config():
    model_configuration = MagicMock()
    model_configuration.instance = AsyncMock()
    model_configuration.instance.ainvoke = AsyncMock()
    return model_configuration


@pytest.fixture
def executor(mock_model_config):
    agent = MagicMock()
    agent.id = "test_agent"
    agent.name = "Test Agent"
    agent.version = "1.0.0"
    agent.security_prompt = ""

    return LocalCodeExecutor(mock_model_config, agent=agent)


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


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "Multiple code blocks",
            "input": """
            Some text
            ```python
            print('hello')
            ```
            More text
            ```python
            x = 1 + 1
            ```
        """,
            "expected_blocks": ["print('hello')", "x = 1 + 1"],
            "expected_count": 2,
        },
        {
            "name": "Multi-line code block",
            "input": """
            Here's a multi-line code block:
            ```python
            def calculate_sum(a, b):
                result = a + b
                return result

            total = calculate_sum(5, 3)
            print(f"The sum is: {total}")
            ```
        """,
            "expected_blocks": [
                """def calculate_sum(a, b):
                result = a + b
                return result

            total = calculate_sum(5, 3)
            print(f"The sum is: {total}")"""
            ],
            "expected_count": 1,
        },
        {
            "name": "No code blocks",
            "input": """
            No code blocks here
            Just plain text
        """,
            "expected_blocks": [],
            "expected_count": 0,
        },
        {
            "name": "Starts with code block",
            "input": """```python
            import os
            os.chdir(os.path.expanduser("~/local-operator-site"))
            print(f"Current working directory is now: {os.getcwd()}")
            ```
            [DONE]""",
            "expected_blocks": [
                """import os
            os.chdir(os.path.expanduser("~/local-operator-site"))
            print(f"Current working directory is now: {os.getcwd()}")"""
            ],
            "expected_count": 1,
        },
        {
            "name": "Starts with code block, space at terminator",
            "input": """```python
            import os
            os.chdir(os.path.expanduser("~/local-operator-site"))
            print(f"Current working directory is now: {os.getcwd()}")
            ```\t
            [DONE]""",
            "expected_blocks": [
                """import os
            os.chdir(os.path.expanduser("~/local-operator-site"))
            print(f"Current working directory is now: {os.getcwd()}")"""
            ],
            "expected_count": 1,
        },
        {
            "name": "Only commented block",
            "input": """
            # ```python
            # This is a comment
            # Another comment
            # Yet another comment
            # ```
            """,
            "expected_blocks": [],
            "expected_count": 0,
        },
        {
            "name": "Code block with commented block and valid code",
            "input": """
            # ```python
            # Block in a comment
            # ```
            ```python
            def real_code():
                pass
            ```
            """,
            "expected_blocks": ["def real_code():\n                pass"],
            "expected_count": 1,
        },
        {
            "name": "Code block with single comment",
            "input": """
            ```python
            # This function calculates the square
            def square(x):
                return x * x
            ```
            """,
            "expected_blocks": [
                "# This function calculates the square\n"
                "            def square(x):\n"
                "                return x * x"
            ],
            "expected_count": 1,
        },
        {
            "name": "Code block with git diff markers",
            "input": """
            - ```python
            - def old_function():
            -     return "old"
            + def new_function():
            +     return "new"
            + ```
            ```python
            def actual_code():
                return True
            ```
            """,
            "expected_blocks": ["def actual_code():\n                return True"],
            "expected_count": 1,
        },
        {
            "name": "Code block with git diff and conflict markers",
            "input": """
            + ```python
            <<<<<<< HEAD
            - def main():
            -     return "local"
            =======
            + def main():
            +     return "remote"
            >>>>>>> feature-branch
            + ```
            + ```python
            + def valid_code():
            +     return "this is valid"
            + ```
            """,
            "expected_blocks": [],
            "expected_count": 0,
        },
        {
            "name": "Code block with nested code enclosure",
            "input": '''
            Here's a code example:
            ```python
            def outer_function():
                print("""
                ```python
                def inner_code():
                    return None
                ```
                """)
                return True
            ```
            ''',
            "expected_blocks": [
                "def outer_function():\n"
                '                print("""\n'
                "                ```python\n"
                "                def inner_code():\n"
                "                    return None\n"
                "                ```\n"
                '                """)\n'
                "                return True",
            ],
            "expected_count": 1,
        },
        {
            "name": "Code block with triple nested code enclosure",
            "input": '''
            Here's a code example:
            ```python
            def outer_function():
                print("""
                ```markdown
                # Title
                ## Subtitle
                ```python
                def inner_code():
                    return None
                ```

                End of markdown
                ```
                """)
                return True
            ```
            ''',
            "expected_blocks": [
                "def outer_function():\n"
                '                print("""\n'
                "                ```markdown\n"
                "                # Title\n"
                "                ## Subtitle\n"
                "                ```python\n"
                "                def inner_code():\n"
                "                    return None\n"
                "                ```\n"
                "\n"
                "                End of markdown\n"
                "                ```\n"
                '                """)\n'
                "                return True",
            ],
            "expected_count": 1,
        },
        {
            "name": "Code block with nested code enclosure and subsequent code block",
            "input": '''
            Here's a code example:
            ```python
            def outer_function():
                print("""
                ```python
                def inner_code():
                    return None
                ```
                """)
                return True
            ```

            Here's another example:
            ```python
            x = 1 + 1
            ```
            ''',
            "expected_blocks": [
                "def outer_function():\n"
                '                print("""\n'
                "                ```python\n"
                "                def inner_code():\n"
                "                    return None\n"
                "                ```\n"
                '                """)\n'
                "                return True",
                "x = 1 + 1",
            ],
            "expected_count": 2,
        },
        {
            "name": "Text ending with code block",
            "input": """
            Here's some text that ends with a code block:
            ```python
            def final_function():
                return "last block"
            ```""",
            "expected_blocks": ['def final_function():\n                return "last block"'],
            "expected_count": 1,
        },
    ],
)
def test_extract_code_blocks(executor, case):
    result = executor.extract_code_blocks(case["input"])
    error_msg = (
        f"Test case '{case['name']}' failed: "
        f"expected {case['expected_count']} code blocks but got {len(result)}"
    )
    assert len(result) == case["expected_count"], error_msg

    for expected, actual in zip(case["expected_blocks"], result):
        norm_expected = normalize_code_block(expected)
        norm_actual = normalize_code_block(actual)
        error_msg = (
            f"Test case '{case['name']}' failed:\n"
            f"Expected:\n{norm_expected}\n"
            f"Actual:\n{norm_actual}"
        )
        assert norm_expected == norm_actual, error_msg


@pytest.mark.asyncio
async def test_check_code_safety_safe(executor, mock_model_config):
    mock_model_config.instance.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"
    code = "print('hello')"
    result = await executor.check_code_safety(code)
    assert result == ConfirmSafetyResult.SAFE
    mock_model_config.instance.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_check_code_safety_unsafe(executor, mock_model_config):
    # Test the default path when can_prompt_user is True
    mock_model_config.instance.ainvoke.return_value.content = (
        "The code is unsafe because it deletes important files\n\n[UNSAFE]"
    )
    code = "x = 1 + 1"
    result = await executor.check_code_safety(code)
    assert result == ConfirmSafetyResult.UNSAFE
    mock_model_config.instance.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_check_code_safety_override(executor, mock_model_config):
    mock_model_config.instance.ainvoke.return_value.content = (
        "The code is safe with security override\n\n[OVERRIDE]"
    )
    code = "x = 1 + 1"
    result = await executor.check_code_safety(code)
    assert result == ConfirmSafetyResult.OVERRIDE
    mock_model_config.instance.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_check_code_safety_unsafe_without_prompt(executor, mock_model_config):
    # Test the branch when can_prompt_user is False
    executor.can_prompt_user = False
    mock_model_config.instance.ainvoke.return_value.content = (
        "The code is unsafe because it deletes important files\n\n[UNSAFE]"
    )
    code = "x = 1 + 1"
    result = await executor.check_code_safety(code)
    assert result == ConfirmSafetyResult.UNSAFE
    mock_model_config.instance.ainvoke.assert_called_once()

    # Conversation history is converted to a list of Dict
    assert executor.conversation_history[-1].role == ConversationRole.ASSISTANT
    assert (
        "The code is unsafe because it deletes important files"
        in executor.conversation_history[-1].content
    )


@pytest.mark.asyncio
async def test_execute_code_success(executor, mock_model_config):
    mock_model_config.instance.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"
    code = "print('hello')"

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.execute_code(code)
        assert "✓ Code Execution Complete" in result
        assert "hello" in result


@pytest.mark.asyncio
async def test_execute_code_no_output(executor, mock_model_config):
    mock_model_config.instance.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"
    code = "x = 1 + 1"  # Code that produces no output

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.execute_code(code)
        assert "✓ Code Execution Complete" in result
        assert "[No output]" in result


@pytest.mark.asyncio
async def test_execute_code_safety_no_prompt(executor, mock_model_config):
    executor.can_prompt_user = False
    mock_model_config.instance.ainvoke.return_value.content = (
        "The code is unsafe because it deletes important files\n\n[UNSAFE]"
    )
    code = "import os; os.remove('file.txt')"  # Potentially dangerous code

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.execute_code(code)

        # Should not cancel execution but add warning to conversation history
        assert "requires further confirmation" in result
        assert len(executor.conversation_history) > 0
        last_message = executor.conversation_history[-1]
        assert last_message.role == ConversationRole.ASSISTANT
        assert "potentially dangerous operation" in last_message.content


@pytest.mark.asyncio
async def test_execute_code_safety_with_prompt(executor, mock_model_config):
    # Default can_prompt_user is True
    mock_model_config.instance.ainvoke.return_value.content = (
        "The code is unsafe because it deletes important files\n\n[UNSAFE]"
    )
    code = "import os; os.remove('file.txt')"  # Potentially dangerous code

    with (
        patch("sys.stdout", new_callable=io.StringIO),
        patch("builtins.input", return_value="n"),
    ):  # User responds "n" to safety prompt
        result = await executor.execute_code(code)

        # Should cancel execution when user declines
        assert "Code execution canceled by user" in result
        assert len(executor.conversation_history) > 0
        last_message = executor.conversation_history[-1]
        assert last_message.role == ConversationRole.USER
        assert "dangerous operation" in last_message.content


@pytest.mark.asyncio
async def test_execute_code_safety_with_prompt_approved(executor, mock_model_config):
    # Default can_prompt_user is True
    mock_model_config.instance.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"
    code = "x = 1 + 1"

    with (
        patch("sys.stdout", new_callable=io.StringIO),
        patch("builtins.input", return_value="y"),  # User responds "y" to safety prompt
    ):
        result = await executor.execute_code(code)

        # Should proceed with execution when user approves
        assert "Code Execution Complete" in result


@pytest.mark.asyncio
async def test_execute_code_safety_with_override(executor, mock_model_config):
    # Default can_prompt_user is True
    mock_model_config.instance.ainvoke.return_value.content = (
        "The code is unsafe but has security override\n\n[OVERRIDE]"
    )
    code = "x = 1 + 1"

    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        result = await executor.execute_code(code)

        # Should proceed with execution and log override
        assert "Code Execution Complete" in result
        output = mock_stdout.getvalue()
        assert "Code safety override applied" in output


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "Safe response",
            "input": "The code looks safe to execute\n[SAFE]",
            "expected": ConfirmSafetyResult.SAFE,
        },
        {
            "name": "Unsafe response",
            "input": "This code contains dangerous operations\n[UNSAFE]",
            "expected": ConfirmSafetyResult.UNSAFE,
        },
        {
            "name": "Override response",
            "input": "Code is normally unsafe but allowed by security settings\n[OVERRIDE]",
            "expected": ConfirmSafetyResult.OVERRIDE,
        },
        {
            "name": "Default to safe",
            "input": "Some response without any safety markers",
            "expected": ConfirmSafetyResult.SAFE,
        },
        {
            "name": "Empty string",
            "input": "",
            "expected": ConfirmSafetyResult.SAFE,
        },
        {
            "name": "Just whitespace",
            "input": "   \n  ",
            "expected": ConfirmSafetyResult.SAFE,
        },
        {
            "name": "None input",
            "input": None,
            "expected": ConfirmSafetyResult.SAFE,
        },
        {
            "name": "Case insensitive SAFE",
            "input": "[safe]",
            "expected": ConfirmSafetyResult.SAFE,
        },
        {
            "name": "Case insensitive UNSAFE",
            "input": "[unsafe]",
            "expected": ConfirmSafetyResult.UNSAFE,
        },
        {
            "name": "Case insensitive OVERRIDE",
            "input": "[override]",
            "expected": ConfirmSafetyResult.OVERRIDE,
        },
    ],
)
def test_get_confirm_safety_result(case):
    result = get_confirm_safety_result(case["input"])
    assert result == case["expected"], f"Failed {case['name']}"


@pytest.mark.asyncio
async def test_process_response(executor, mock_model_config):
    response = ResponseJsonSchema(
        previous_step_success=True,
        previous_goal="",
        current_goal="Print hello world",
        next_goal="",
        response="Here's some code:",
        code="print('hello world')",
        action=ActionType.CODE,
        learnings="",
        plan="",
        content="",
        file_path="",
        replacements=[],
    )
    mock_model_config.instance.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"

    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        await executor.process_response(response)
        output = mock_stdout.getvalue()
        assert "Executing Code" in output
        assert "hello world" in output


def test_limit_conversation_history(executor):
    test_cases = [
        {"name": "Empty history", "initial": [], "expected": []},
        {
            "name": "Only system prompt",
            "initial": [ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt")],
            "expected": [ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt")],
        },
        {
            "name": "History within limit",
            "initial": [
                ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt"),
                ConversationRecord(role=ConversationRole.USER, content="msg1"),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="msg2"),
            ],
            "expected": [
                ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt"),
                ConversationRecord(role=ConversationRole.USER, content="msg1"),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="msg2"),
            ],
        },
        {
            "name": "History exceeding limit",
            "initial": [
                ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt"),
                ConversationRecord(role=ConversationRole.USER, content="msg1"),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="msg2"),
                ConversationRecord(role=ConversationRole.USER, content="msg3"),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="msg4"),
            ],
            "expected": [
                ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt"),
                ConversationRecord(role=ConversationRole.USER, content="msg3"),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="msg4"),
            ],
        },
    ]
    for test_case in test_cases:
        executor.max_conversation_history = 3
        executor.conversation_history = test_case["initial"]
        executor._limit_conversation_history()

        expected_len = len(test_case["expected"])
        actual_len = len(executor.conversation_history)
        assert (
            expected_len == actual_len
        ), f"{test_case['name']}: Expected length {expected_len} but got {actual_len}"

        for i, msg in enumerate(test_case["expected"]):
            expected_role = msg.role
            actual_role = executor.conversation_history[i].role
            assert expected_role == actual_role, (
                f"{test_case['name']}: Expected role {expected_role} but got {actual_role} "
                f"at position {i}"
            )

            expected_content = msg.content
            actual_content = executor.conversation_history[i].content
            assert expected_content == actual_content, (
                f"{test_case['name']}: Expected content {expected_content} "
                f"but got {actual_content} at position {i}"
            )


@pytest.mark.asyncio
async def test_summarize_old_steps(mock_model_config):
    executor = LocalCodeExecutor(
        model_configuration=mock_model_config, detail_conversation_length=2
    )

    # Mock the summarization response
    mock_model_config.instance.ainvoke.return_value.content = "[SUMMARY] This is a summary"

    test_cases = [
        {
            "name": "Empty history",
            "initial": [],
            "expected": [],
        },
        {
            "name": "Only system prompt",
            "initial": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="system prompt",
                    should_summarize=True,
                    summarized=False,
                )
            ],
            "expected": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="system prompt",
                    should_summarize=True,
                    summarized=False,
                )
            ],
        },
        {
            "name": "Within detail length",
            "initial": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="system prompt",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="msg1",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="msg2",
                    should_summarize=True,
                    summarized=False,
                ),
            ],
            "expected": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="system prompt",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="msg1",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="msg2",
                    should_summarize=True,
                    summarized=False,
                ),
            ],
        },
        {
            "name": "Beyond detail length with skip conditions",
            "initial": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="system prompt",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.USER,
                    content="user msg",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="skip me",
                    should_summarize=False,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="already summarized",
                    should_summarize=True,
                    summarized=True,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="summarize me",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="recent1",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="recent2",
                    should_summarize=True,
                    summarized=False,
                ),
            ],
            "expected": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="system prompt",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.USER,
                    content="[SUMMARY] This is a summary",
                    should_summarize=True,
                    summarized=True,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="skip me",
                    should_summarize=False,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="already summarized",
                    should_summarize=True,
                    summarized=True,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="[SUMMARY] This is a summary",
                    should_summarize=True,
                    summarized=True,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="recent1",
                    should_summarize=True,
                    summarized=False,
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="recent2",
                    should_summarize=True,
                    summarized=False,
                ),
            ],
        },
    ]

    for test_case in test_cases:
        executor.conversation_history = test_case["initial"]
        executor.detail_conversation_length = 2
        await executor._summarize_old_steps()

        assert executor.conversation_history == test_case["expected"], (
            f"{test_case['name']}: Expected conversation history to match "
            f"but got {executor.conversation_history}"
        )


@pytest.mark.asyncio
async def test_summarize_old_steps_all_detail(executor):
    executor.detail_conversation_length = -1
    executor.conversation_history = [
        ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt"),
        ConversationRecord(role=ConversationRole.USER, content="user msg"),
    ]

    await executor._summarize_old_steps()

    assert executor.conversation_history == [
        ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt"),
        ConversationRecord(role=ConversationRole.USER, content="user msg"),
    ]


@pytest.mark.asyncio
async def test_invoke_model_api_error(executor):
    mock_request = MagicMock()
    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        executor.model_configuration.instance.ainvoke.side_effect = APIError(
            message="API Error",
            request=mock_request,
            body={"code": "error_code", "type": "error_type"},
        )

        with pytest.raises(APIError) as exc_info:
            await executor.invoke_model(
                [ConversationRecord(role=ConversationRole.USER, content="test")]
            )

        assert str(exc_info.value) == "API Error"
        assert exc_info.value.code == "error_code"
        assert exc_info.value.type == "error_type"


@pytest.mark.asyncio
async def test_invoke_model_rate_limit(executor):
    mock_request = MagicMock()
    executor.model_configuration.instance.ainvoke.side_effect = APIError(
        message="Rate limit exceeded",
        request=mock_request,
        body={"code": "rate_limit_exceeded", "type": "rate_limit_error"},
    )

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(APIError) as exc_info:
            await executor.invoke_model(
                [ConversationRecord(role=ConversationRole.USER, content="test")]
            )

    assert str(exc_info.value) == "Rate limit exceeded"
    assert exc_info.value.code == "rate_limit_exceeded"
    assert exc_info.value.type == "rate_limit_error"


@pytest.mark.asyncio
async def test_invoke_model_context_length(executor):
    mock_request = MagicMock()
    executor.model_configuration.instance.ainvoke.side_effect = APIError(
        message="Maximum context length exceeded",
        request=mock_request,
        body={"code": "context_length_exceeded", "type": "invalid_request_error"},
    )

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(APIError) as exc_info:
            await executor.invoke_model(
                [ConversationRecord(role=ConversationRole.USER, content="test")]
            )

    assert str(exc_info.value) == "Maximum context length exceeded"
    assert exc_info.value.code == "context_length_exceeded"
    assert exc_info.value.type == "invalid_request_error"


@pytest.mark.asyncio
async def test_invoke_model_general_exception(executor):
    executor.model_configuration.instance.ainvoke.side_effect = Exception("Unexpected error")

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(Exception) as exc_info:
            await executor.invoke_model(
                [ConversationRecord(role=ConversationRole.USER, content="test")]
            )

    assert str(exc_info.value) == "Unexpected error"


@pytest.mark.asyncio
async def test_invoke_model_timeout(executor):
    executor.model_configuration.instance.ainvoke.side_effect = TimeoutError("Request timed out")

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(TimeoutError) as exc_info:
            await executor.invoke_model(
                [ConversationRecord(role=ConversationRole.USER, content="test")]
            )

    assert str(exc_info.value) == "Request timed out"


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "No working directory",
            "conversation": [
                ConversationRecord(role=ConversationRole.SYSTEM, content="Some system message"),
                ConversationRecord(role=ConversationRole.USER, content="Test message"),
            ],
            "expected": None,
        },
        {
            "name": "Single working directory",
            "conversation": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM, content="Current working directory: /test/path"
                ),
                ConversationRecord(role=ConversationRole.USER, content="Test message"),
            ],
            "expected": Path("/test/path"),
        },
        {
            "name": "Multiple working directories - returns most recent",
            "conversation": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM, content="Current working directory: /old/path"
                ),
                ConversationRecord(role=ConversationRole.USER, content="Test message"),
                ConversationRecord(
                    role=ConversationRole.SYSTEM, content="Current working directory: /new/path"
                ),
            ],
            "expected": Path("/new/path"),
        },
        {
            "name": "Case insensitive working directory",
            "conversation": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM, content="CURRENT WORKING DIRECTORY: /test/path"
                ),
                ConversationRecord(role=ConversationRole.USER, content="Test message"),
            ],
            "expected": Path("/test/path"),
        },
        {
            "name": "Message with working directory but not at start",
            "conversation": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="Some text before Current working directory: /test/path",
                ),
                ConversationRecord(role=ConversationRole.USER, content="Test message"),
            ],
            "expected": None,
        },
        {
            "name": "Multiple entries - returns last in list",
            "conversation": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="Current working directory: /path/one",
                ),
                ConversationRecord(role=ConversationRole.USER, content="Test message"),
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="Current working directory: /path/two",
                ),
                ConversationRecord(role=ConversationRole.USER, content="Another message"),
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="Current working directory: /path/three",
                ),
            ],
            "expected": Path("/path/three"),
        },
        {
            "name": "First entry with working directory",
            "conversation": [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="Current working directory: /first/path",
                ),
                ConversationRecord(role=ConversationRole.USER, content="Test message"),
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="Some other system message",
                ),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="Response"),
            ],
            "expected": Path("/first/path"),
        },
        {
            "name": "Empty conversation",
            "conversation": [],
            "expected": None,
        },
    ],
)
def test_get_conversation_working_directory(executor, test_case):
    executor.conversation_history = test_case["conversation"]
    result = executor.get_conversation_working_directory()
    assert result == test_case["expected"]


@pytest.mark.parametrize(
    "context_vars, function_name, expected_output",
    [
        (
            {"a": 1, "b": "hello", "c": [1, 2, 3]},
            "",
            "a: 1\nb: hello\nc: [1, 2, 3]\n",
        ),
        (
            {"__builtins__": None, "a": 1},
            "",
            "a: 1\n",
        ),
        (
            {},
            "",
            "",
        ),
        (
            {},
            "function_1",
            "function_1: function_1(x: _empty) -> _empty: No description available\n",
        ),
        (
            {},
            "function_2",
            "function_2: function_2(x: _empty) -> _empty: This is a docstring\n",
        ),
        (
            {},
            "function_3",
            "function_3: async function_3(x: _empty) -> _empty: No description available\n",
        ),
        (
            {"a": 1, "b": "hello", "c": [1, 2, 3]},
            "function_1",
            "a: 1\nb: hello\nc: [1, 2, 3]\nfunction_1: function_1(x: _empty) -> _empty: "
            "No description available\n",
        ),
    ],
)
def test_get_context_vars_str(
    context_vars: Dict[str, Any], function_name: str, expected_output: str
) -> None:
    """
    Test the get_context_vars_str function with various inputs.

    Args:
        context_vars: A dictionary representing the context variables.
        expected_output: The expected string representation of the context variables.
    """

    def function_1(x):
        return x * 2

    def function_2(x):
        """This is a docstring"""
        return x * 2

    async def function_3(x):
        return x * 2

    if function_name == "function_1":
        context_vars["function_1"] = function_1
    elif function_name == "function_2":
        context_vars["function_2"] = function_2
    elif function_name == "function_3":
        context_vars["function_3"] = function_3

    result = get_context_vars_str(context_vars)
    assert result == expected_output


@pytest.mark.parametrize(
    "response_str, expected_code, expected_action",
    [
        (
            '{"previous_step_success": true, "previous_goal": "", '
            '"current_goal": "Print hello world", "next_goal": "", '
            '"response": "Here\'s some code:", "code": "print(\'hello world\')", '
            '"action": "CODE", "learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": []}',
            "print('hello world')",
            ActionType.CODE,
        ),
        (
            '```json\n{"previous_step_success": true, "previous_goal": "", '
            '"current_goal": "Print hello world", "next_goal": "", '
            '"response": "Here\'s some code:", "code": "print(\'hello world\')", '
            '"action": "CODE", "learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": []}\n```',
            "print('hello world')",
            ActionType.CODE,
        ),
        (
            '```json\n{"previous_step_success": true, "previous_goal": "", '
            '"current_goal": "Write a file", "next_goal": "", '
            '"response": "I will write a file", "code": "", "action": "WRITE", '
            '"learnings": "", "plan": "", "content": "file content", '
            '"file_path": "output.txt", "replacements": []}\n```',
            "",
            ActionType.WRITE,
        ),
        (
            'Some text before ```json\n{"previous_step_success": true, "previous_goal": "", '
            '"current_goal": "Print hello world", "next_goal": "", '
            '"response": "Here\'s some code:", "code": "print(\'hello world\')", '
            '"action": "CODE", "learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": []}\n```',
            "print('hello world')",
            ActionType.CODE,
        ),
    ],
)
def test_process_json_response(
    response_str: str, expected_code: str, expected_action: ActionType
) -> None:
    """
    Test the process_json_response function with various inputs.

    Args:
        response_str: A JSON string representing the response from the language model.
        expected_code: The expected code extracted from the JSON response.
        expected_action: The expected action type extracted from the JSON response.
    """
    response = process_json_response(response_str)
    assert response.code == expected_code
    assert response.action == expected_action


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_type, code, file_path, content, replacements, expected_output",
    [
        (ActionType.CODE, "print('hello')", None, None, None, "Executing Code"),
        (ActionType.CHECK, "x = 1", None, None, None, "Executing Check"),
        (ActionType.WRITE, None, "test.txt", "test content", None, "Executing Write"),
        (
            ActionType.EDIT,
            None,
            "test.txt",
            "new content",
            [{"old": "old", "new": "new"}],
            "Executing Edit",
        ),
        (ActionType.READ, None, "test.txt", None, None, "Executing Read"),
    ],
)
async def test_perform_action(
    executor: LocalCodeExecutor,
    action_type: ActionType,
    code: str | None,
    file_path: str | None,
    content: str | None,
    replacements: list[dict[str, str]] | None,
    expected_output: str,
) -> None:
    response = ResponseJsonSchema(
        previous_step_success=True,
        previous_goal="",
        current_goal="Test goal",
        next_goal="",
        response="Test response",
        code=code or "",
        action=action_type,
        learnings="",
        plan="",
        content=content or "",
        file_path=file_path or "",
        replacements=replacements or [],
    )

    original_read_file = executor.read_file
    original_write_file = executor.write_file
    original_edit_file = executor.edit_file
    original_execute_code = executor.execute_code

    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        if action_type == ActionType.READ:
            executor.read_file = AsyncMock(return_value="File content")
        elif action_type == ActionType.WRITE:
            executor.write_file = AsyncMock(return_value="File written")
        elif action_type == ActionType.EDIT:
            executor.edit_file = AsyncMock(return_value="File edited")
        else:
            executor.execute_code = AsyncMock(return_value="Code executed")

        result = await executor.perform_action(response)
        assert result is not None
        assert result.status == ProcessResponseStatus.SUCCESS
        assert expected_output in mock_stdout.getvalue()

    executor.read_file = original_read_file
    executor.write_file = original_write_file
    executor.edit_file = original_edit_file
    executor.execute_code = original_execute_code


@pytest.mark.asyncio
async def test_perform_action_handles_exception(executor: LocalCodeExecutor):
    response = ResponseJsonSchema(
        previous_step_success=True,
        previous_goal="",
        current_goal="Test goal",
        next_goal="",
        response="Test response",
        code="invalid code",
        action=ActionType.CODE,
        learnings="",
        plan="",
        content="",
        file_path="",
        replacements=[],
    )

    executor.execute_code = AsyncMock(side_effect=Exception("Execution failed"))

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.perform_action(response)
        assert "Error: Execution failed" in executor.conversation_history[-1].content
        assert result is not None
        assert result.status == ProcessResponseStatus.SUCCESS


@pytest.mark.asyncio
async def test_read_file_action(executor: LocalCodeExecutor, tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Test file content")

    result = await executor.read_file(str(file_path))

    assert "Successfully read file" in result
    assert str(file_path) in executor.conversation_history[-1].content
    assert "Test file content" in executor.conversation_history[-1].content


@pytest.mark.asyncio
async def test_write_file_action(executor: LocalCodeExecutor, tmp_path: Path):
    file_path = tmp_path / "test_file.txt"

    result = await executor.write_file(str(file_path), "New file content")

    with open(file_path, "r") as f:
        file_content = f.read()
    assert file_content == "New file content"

    assert "Successfully wrote to file" in result
    assert str(file_path) in executor.conversation_history[-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_content, replacements, expected_content, should_raise",
    [
        (
            "Original content",
            [{"find": "Original content", "replace": "Replacement content"}],
            "Replacement content",
            False,
        ),
        (
            "Line 1\nLine 2\nLine 3",
            [{"find": "Line 2", "replace": "Replaced Line 2"}],
            "Line 1\nReplaced Line 2\nLine 3",
            False,
        ),
        (
            "Multiple copies of the same word word word",
            [{"find": "word", "replace": "replaced"}, {"find": "word", "replace": "replaced"}],
            "Multiple copies of the same replaced replaced word",
            False,
        ),
        (
            "First line\nSecond line",
            [
                {"find": "First line", "replace": "New first line"},
                {"find": "Second line", "replace": "New second line"},
            ],
            "New first line\nNew second line",
            False,
        ),
        (
            "No match",
            [{"find": "Nonexistent", "replace": "Replacement"}],
            "No match",
            True,
        ),
        (
            "",
            [{"find": "", "replace": "Replacement"}],
            "Replacement",
            False,
        ),
        (
            "Initial content",
            [{"find": "Initial content", "replace": ""}],
            "",
            False,
        ),
    ],
)
async def test_edit_file_action(
    executor: LocalCodeExecutor,
    tmp_path: Path,
    initial_content: str,
    replacements: list[dict[str, str]],
    expected_content: str,
    should_raise: bool,
) -> None:
    """
    Test the edit_file action with various scenarios.

    Args:
        executor: The LocalCodeExecutor fixture.
        tmp_path: The temporary directory path fixture.
        initial_content: The initial content of the file.
        replacements: A list of dictionaries, where each dictionary
            contains a "find" key and a "replace" key.
        expected_content: The expected content of the file after the replacements.
        should_raise: A boolean indicating whether the test should raise an exception.
    """
    file_path = tmp_path / "test_file.txt"
    file_path.write_text(initial_content)

    if should_raise:
        with pytest.raises(ValueError):
            await executor.edit_file(str(file_path), replacements)
    else:
        result = await executor.edit_file(str(file_path), replacements)
        with open(file_path, "r") as f:
            file_content = f.read()
        assert file_content == expected_content
        assert result is not None
        assert "Successfully edited file" in result
        assert str(file_path) in executor.conversation_history[-1].content
