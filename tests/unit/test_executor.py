import io
import subprocess
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import BaseMessage
from openai import APIError

from local_operator.executor import (
    CodeExecutionError,
    CodeExecutionResult,
    ConfirmSafetyResult,
    LocalCodeExecutor,
    get_confirm_safety_result,
    get_context_vars_str,
    process_json_response,
)
from local_operator.operator import Operator, OperatorType
from local_operator.tools import ToolRegistry
from local_operator.types import (
    ActionType,
    AgentState,
    ConversationRecord,
    ConversationRole,
    ExecutionType,
    ProcessResponseStatus,
    RequestClassification,
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
    model_configuration.instance.astream = AsyncMock()
    return model_configuration


@pytest.fixture
def test_tool_registry():
    """Fixture for an empty tool registry."""
    return ToolRegistry()


@pytest.fixture
def env_config():
    return {
        "RADIENT_API_KEY": "test_key",
    }


@pytest.fixture
def executor(mock_model_config, test_tool_registry):
    agent = MagicMock()
    agent.id = "test_agent"
    agent.name = "Test Agent"
    agent.version = "1.0.0"
    agent.security_prompt = ""

    # Create a fresh AgentState for each test
    fresh_agent_state = AgentState(
        version="",
        conversation=[],
        execution_history=[],
        learnings=[],
        current_plan=None,
        instruction_details=None,
        agent_system_prompt=None,
    )

    mock_executor = LocalCodeExecutor(mock_model_config, agent=agent)
    mock_executor.tool_registry = test_tool_registry
    mock_executor.agent_state = fresh_agent_state

    yield mock_executor


@pytest.fixture
def cli_operator(mock_model_config, executor, env_config):
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
        env_config=env_config,
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

    async def mock_astream(*args, **kwargs):
        message = BaseMessage(content="The code is safe\n\n[SAFE]", type="assistant")
        yield message

    mock_model_config.instance.astream = mock_astream

    response = ResponseJsonSchema(
        code="print('hello')",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )
    result = await executor.check_response_safety(response)
    assert result == ConfirmSafetyResult.SAFE


@pytest.mark.asyncio
async def test_check_code_safety_unsafe(executor, mock_model_config):
    # Test the default path when can_prompt_user is True

    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is unsafe because it deletes important files\n\n[UNSAFE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream

    response = ResponseJsonSchema(
        code="x = 1 + 1",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )
    result = await executor.check_response_safety(response)
    assert result == ConfirmSafetyResult.UNSAFE


@pytest.mark.asyncio
async def test_check_code_safety_override(executor, mock_model_config):
    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is safe with security override\n\n[OVERRIDE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream

    response = ResponseJsonSchema(
        code="x = 1 + 1",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )
    result = await executor.check_response_safety(response)
    assert result == ConfirmSafetyResult.OVERRIDE


@pytest.mark.asyncio
async def test_check_code_safety_unsafe_without_prompt(executor, mock_model_config):
    # Test the branch when can_prompt_user is False
    executor.can_prompt_user = False

    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is unsafe because it deletes important files\n\n[UNSAFE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream

    response = ResponseJsonSchema(
        code="x = 1 + 1",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )
    result = await executor.check_response_safety(response)
    assert result == ConfirmSafetyResult.UNSAFE


@pytest.mark.asyncio
async def test_execute_code_success(executor, mock_model_config):
    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is safe\n\n[SAFE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream

    response = ResponseJsonSchema(
        code="print('hello')",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )

    with patch("sys.stdout", new_callable=io.StringIO):
        execution_result = await executor.execute_code(response)
        assert "✓ Code Execution Complete" in execution_result.formatted_print
        assert "hello" in execution_result.formatted_print


@pytest.mark.asyncio
async def test_execute_code_no_output(executor, mock_model_config):
    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is safe\n\n[SAFE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream
    response = ResponseJsonSchema(
        code="x = 1 + 1",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )

    with patch("sys.stdout", new_callable=io.StringIO):
        execution_result = await executor.execute_code(response)
        assert "✓ Code Execution Complete" in execution_result.formatted_print
        assert "[No output]" in execution_result.formatted_print


@pytest.mark.asyncio
async def test_execute_code_safety_no_prompt(executor, mock_model_config):
    executor.can_prompt_user = False

    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is unsafe because it deletes important files\n\n[UNSAFE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream

    response = ResponseJsonSchema(
        code="import os; os.remove('file.txt')",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )

    with patch("sys.stdout", new_callable=io.StringIO):
        safety_result = await executor.check_and_confirm_safety(response)
        execution_result = await executor.handle_safety_result(safety_result, response)
        assert safety_result == ConfirmSafetyResult.UNSAFE

        # Should not cancel execution but add warning to conversation history
        assert "The code is unsafe because it deletes important files" in execution_result.message
        assert len(executor.agent_state.conversation) > 0
        last_message = executor.agent_state.conversation[-1]
        assert last_message.role == ConversationRole.ASSISTANT
        assert "The code is unsafe because it deletes important files" in last_message.content


@pytest.mark.asyncio
async def test_execute_code_safety_with_prompt(executor, mock_model_config):
    # Default can_prompt_user is True
    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is unsafe because it deletes important files\n\n[UNSAFE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream
    response = ResponseJsonSchema(
        code="import os; os.remove('file.txt')",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )

    with (
        patch("sys.stdout", new_callable=io.StringIO),
        patch("builtins.input", return_value="n"),
    ):  # User responds "n" to safety prompt
        safety_result = await executor.check_and_confirm_safety(response)
        execution_result = await executor.handle_safety_result(safety_result, response)
        assert safety_result == ConfirmSafetyResult.UNSAFE

        # Should cancel execution when user declines
        assert "Code execution canceled by user" in execution_result.message
        assert len(executor.agent_state.conversation) > 0
        last_message = executor.agent_state.conversation[-1]
        assert last_message.role == ConversationRole.USER
        assert "this is a dangerous operation" in last_message.content


@pytest.mark.asyncio
async def test_execute_code_safety_with_prompt_approved(executor, mock_model_config):
    # Default can_prompt_user is True
    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is safe\n\n[SAFE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream
    response = ResponseJsonSchema(
        code="x = 1 + 1",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )

    with (
        patch("sys.stdout", new_callable=io.StringIO),
        patch("builtins.input", return_value="y"),  # User responds "y" to safety prompt
    ):
        execution_result = await executor.execute_code(response)

        # Should proceed with execution when user approves
        assert "Code Execution Complete" in execution_result.formatted_print


@pytest.mark.asyncio
async def test_execute_code_safety_with_override(executor, mock_model_config):
    # Default can_prompt_user is True
    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is unsafe but has security override\n\n[OVERRIDE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream

    response = ResponseJsonSchema(
        code="x = 1 + 1",
        action=ActionType.CODE,
        content="",
        file_path="",
        learnings="",
        mentioned_files=[],
        replacements=[],
        response="",
    )

    with patch("sys.stdout", new_callable=io.StringIO):
        safety_result = await executor.check_and_confirm_safety(response)
        execution_result = await executor.handle_safety_result(safety_result, response)

        assert safety_result == ConfirmSafetyResult.OVERRIDE
        assert execution_result is None


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
        response="Here's some code:",
        code="print('hello world')",
        action=ActionType.CODE,
        learnings="",
        content="",
        mentioned_files=[],
        file_path="",
        replacements=[],
    )

    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="The code is safe\n\n[SAFE]",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream
    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        await executor.process_response(response, RequestClassification(type="data_science"))
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
                ConversationRecord(role=ConversationRole.USER, content="msg3"),
            ],
            "expected": [
                ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt"),
                ConversationRecord(role=ConversationRole.USER, content="msg1"),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="msg2"),
                ConversationRecord(role=ConversationRole.USER, content="msg3"),
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
                ConversationRecord(
                    role=ConversationRole.USER,
                    content="[Some conversation history has been truncated for brevity]",
                    should_summarize=False,
                ),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="msg4"),
            ],
        },
    ]
    for test_case in test_cases:
        executor.max_conversation_history = 3
        executor.agent_state.conversation = test_case["initial"]
        executor._limit_conversation_history()

        expected_len = len(test_case["expected"])
        actual_len = len(executor.agent_state.conversation)
        assert (
            expected_len == actual_len
        ), f"{test_case['name']}: Expected length {expected_len} but got {actual_len}"

        for i, msg in enumerate(test_case["expected"]):
            expected_role = msg.role
            actual_role = executor.agent_state.conversation[i].role
            assert expected_role == actual_role, (
                f"{test_case['name']}: Expected role {expected_role} but got {actual_role} "
                f"at position {i}"
            )

            expected_content = msg.content
            actual_content = executor.agent_state.conversation[i].content
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
    async def mock_astream(*args, **kwargs):
        message = BaseMessage(
            content="[SUMMARY] This is a summary",
            type="assistant",
        )
        yield message

    mock_model_config.instance.astream = mock_astream

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
        executor.agent_state.conversation = test_case["initial"]
        executor.detail_conversation_length = 2
        await executor._summarize_old_steps()

        assert executor.agent_state.conversation == test_case["expected"], (
            f"{test_case['name']}: Expected conversation history to match "
            f"but got {executor.agent_state.conversation}"
        )


@pytest.mark.asyncio
async def test_summarize_old_steps_all_detail(executor):
    executor.detail_conversation_length = -1
    executor.agent_state.conversation = [
        ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt"),
        ConversationRecord(role=ConversationRole.USER, content="user msg"),
    ]

    await executor._summarize_old_steps()

    assert executor.agent_state.conversation == [
        ConversationRecord(role=ConversationRole.SYSTEM, content="system prompt"),
        ConversationRecord(role=ConversationRole.USER, content="user msg"),
    ]


@pytest.mark.asyncio
async def test_invoke_model_api_error(executor):
    mock_request = MagicMock()
    with patch("asyncio.sleep", AsyncMock(return_value=None)):

        async def mock_error_astream(*args, **kwargs):
            message = BaseMessage(
                content="API Error",
                type="assistant",
            )
            raise APIError(
                message="API Error",
                request=mock_request,
                body={"code": "error_code", "type": "error_type"},
            )
            yield message

        executor.model_configuration.instance.astream = mock_error_astream

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

    async def mock_error_astream(*args, **kwargs):
        message = BaseMessage(
            content="Rate limit exceeded",
            type="assistant",
        )
        raise APIError(
            message="Rate limit exceeded",
            request=mock_request,
            body={"code": "rate_limit_exceeded", "type": "rate_limit_error"},
        )
        yield message

    executor.model_configuration.instance.astream = mock_error_astream

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

    async def mock_error_astream(*args, **kwargs):
        message = BaseMessage(
            content="Maximum context length exceeded",
            type="assistant",
        )
        raise APIError(
            message="Maximum context length exceeded",
            request=mock_request,
            body={"code": "context_length_exceeded", "type": "invalid_request_error"},
        )
        yield message

    executor.model_configuration.instance.astream = mock_error_astream

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
    async def mock_error_astream(*args, **kwargs):
        message = BaseMessage(
            content="Unexpected error",
            type="assistant",
        )
        raise Exception("Unexpected error")
        yield message

    executor.model_configuration.instance.astream = mock_error_astream

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(Exception) as exc_info:
            await executor.invoke_model(
                [ConversationRecord(role=ConversationRole.USER, content="test")]
            )

    assert str(exc_info.value) == "Unexpected error"


@pytest.mark.asyncio
async def test_invoke_model_timeout(executor):
    async def mock_error_astream(*args, **kwargs):
        message = BaseMessage(
            content="Request timed out",
            type="assistant",
        )
        raise TimeoutError("Request timed out")
        yield message

    executor.model_configuration.instance.astream = mock_error_astream

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(TimeoutError) as exc_info:
            await executor.invoke_model(
                [ConversationRecord(role=ConversationRole.USER, content="test")]
            )

    assert str(exc_info.value) == "Request timed out"


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
        (
            {"time": datetime(2024, 1, 1, 12, 0, 0)},
            "",
            "time: 2024-01-01 12:00:00\n",
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
        function_name: The name of the function to add to the context variables.
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
            '{"previous_step_success": true, "previous_step_issue": "", "previous_goal": "", '
            '"current_goal": "Print hello world", "next_goal": "", '
            '"response": "Here\'s some code:", "code": "print(\'hello world\')", '
            '"action": "CODE", "learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": [], "mentioned_files": []}',
            "print('hello world')",
            ActionType.CODE,
        ),
        (
            '```json\n{"previous_step_success": true, "previous_step_issue": "", '
            '"previous_goal": "", "current_goal": "Print hello world", '
            '"next_goal": "", "response": "Here\'s some code:", '
            '"code": "print(\'hello world\')", "action": "CODE", '
            '"learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": [], "mentioned_files": []}\n```',
            "print('hello world')",
            ActionType.CODE,
        ),
        (
            '```json\n{"previous_step_success": true, "previous_step_issue": "", '
            '"previous_goal": "", "current_goal": "Write a file", '
            '"next_goal": "", "response": "I will write a file", '
            '"code": "", "action": "WRITE", "learnings": "", '
            '"plan": "", "content": "file content", '
            '"file_path": "output.txt", "replacements": [], "mentioned_files": []}\n```',
            "",
            ActionType.WRITE,
        ),
        (
            'Some text before ```json\n{"previous_step_success": true, '
            '"previous_step_issue": "", "previous_goal": "", '
            '"current_goal": "Print hello world", "next_goal": "", '
            '"response": "Here\'s some code:", "code": "print(\'hello world\')", '
            '"action": "CODE", "learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": [], "mentioned_files": []}\n```',
            "print('hello world')",
            ActionType.CODE,
        ),
        (
            'Text before {"previous_step_success": true, "previous_step_issue": "", '
            '"previous_goal": "", "current_goal": "Print hello world", '
            '"next_goal": "", "response": "Here\'s some code:", '
            '"code": "print(\'hello world\')", "action": "CODE", '
            '"learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": [], "mentioned_files": []}',
            "print('hello world')",
            ActionType.CODE,
        ),
        (
            '{"previous_step_success": true, "previous_step_issue": "", '
            '"previous_goal": "", "current_goal": "Print hello world", '
            '"next_goal": "", "response": "Here\'s some code:", '
            '"code": "print(\'hello world\')", "action": "CODE", '
            '"learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": [], "mentioned_files": []} Text after',
            "print('hello world')",
            ActionType.CODE,
        ),
        (
            'Text before {"previous_step_success": true, "previous_step_issue": "", '
            '"previous_goal": "", "current_goal": "Print hello world", '
            '"next_goal": "", "response": "Here\'s some code:", '
            '"code": "print(\'hello world\')", "action": "CODE", '
            '"learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": [], "mentioned_files": []} Text after',
            "print('hello world')",
            ActionType.CODE,
        ),
        (
            '<think>Some thinking process here</think> {"previous_step_success": true, '
            '"previous_step_issue": "", "previous_goal": "", '
            '"current_goal": "Process with thinking", "next_goal": "", '
            '"response": "Here\'s the result after thinking", '
            '"code": "print(\'thought result\')", "action": "CODE", '
            '"learnings": "", "plan": "", "content": "", '
            '"file_path": "", "replacements": [], "mentioned_files": []}',
            "print('thought result')",
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
        (ActionType.DELEGATE, None, None, None, None, "Executing Delegate"),
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

    file_path = file_path or ""

    response = ResponseJsonSchema(
        response="Test response",
        code=code or "",
        action=action_type,
        learnings="",
        content=content or "",
        file_path=file_path,
        mentioned_files=[],
        replacements=replacements or [],
    )

    original_read_file = executor.read_file
    original_write_file = executor.write_file
    original_edit_file = executor.edit_file
    original_execute_code = executor.execute_code
    original_delegate_to_agent = executor.delegate_to_agent

    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        if action_type == ActionType.READ:
            executor.read_file = AsyncMock(
                return_value=CodeExecutionResult(
                    stdout="File content",
                    stderr="",
                    logging="",
                    formatted_print="File content",
                    code="",
                    message="",
                    role=ConversationRole.SYSTEM,
                    status=ProcessResponseStatus.SUCCESS,
                    files=[file_path],
                    execution_type=ExecutionType.ACTION,
                    action=action_type,
                )
            )
        elif action_type == ActionType.WRITE:
            executor.write_file = AsyncMock(
                return_value=CodeExecutionResult(
                    stdout="File written",
                    stderr="",
                    logging="",
                    formatted_print="File written",
                    code="",
                    message="",
                    role=ConversationRole.SYSTEM,
                    status=ProcessResponseStatus.SUCCESS,
                    files=[file_path],
                    execution_type=ExecutionType.ACTION,
                    action=action_type,
                )
            )
        elif action_type == ActionType.EDIT:
            executor.edit_file = AsyncMock(
                return_value=CodeExecutionResult(
                    stdout="File edited",
                    stderr="",
                    logging="",
                    formatted_print="File edited",
                    code="",
                    message="",
                    role=ConversationRole.SYSTEM,
                    status=ProcessResponseStatus.SUCCESS,
                    files=[file_path],
                    execution_type=ExecutionType.ACTION,
                    action=action_type,
                )
            )
        elif action_type == ActionType.DELEGATE:
            executor.delegate_to_agent = AsyncMock(
                return_value=CodeExecutionResult(
                    stdout="Delegated to agent",
                    stderr="",
                    logging="",
                    formatted_print="Delegated to agent",
                    code="",
                    message="Delegated to agent",
                    role=ConversationRole.SYSTEM,
                    status=ProcessResponseStatus.SUCCESS,
                    files=[],
                    execution_type=ExecutionType.ACTION,
                    action=action_type,
                )
            )
        else:
            executor.execute_code = AsyncMock(
                return_value=CodeExecutionResult(
                    stdout="Code executed",
                    stderr="",
                    logging="",
                    formatted_print="Code executed",
                    code="",
                    message="",
                    role=ConversationRole.SYSTEM,
                    status=ProcessResponseStatus.SUCCESS,
                    files=[],
                    execution_type=ExecutionType.ACTION,
                    action=action_type,
                )
            )

        result = await executor.perform_action(response, RequestClassification(type="data_science"))
        assert result is not None
        assert result.status == ProcessResponseStatus.SUCCESS
        assert expected_output in mock_stdout.getvalue()

    executor.read_file = original_read_file
    executor.write_file = original_write_file
    executor.edit_file = original_edit_file
    executor.execute_code = original_execute_code
    executor.delegate_to_agent = original_delegate_to_agent


@pytest.mark.asyncio
async def test_perform_action_handles_exception(executor: LocalCodeExecutor):
    response = ResponseJsonSchema(
        response="Test response",
        code="invalid code",
        action=ActionType.CODE,
        learnings="",
        content="",
        file_path="",
        mentioned_files=[],
        replacements=[],
    )

    executor.execute_code = AsyncMock(side_effect=Exception("Execution failed"))

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.perform_action(response, RequestClassification(type="data_science"))
        assert "error encountered" in executor.agent_state.conversation[-1].content
        assert result is not None
        assert result.status == ProcessResponseStatus.SUCCESS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_content",
    [
        "",
        "Single line content",
        "First line\nSecond line\nThird line",
        "\n\n\n",
        "   ",
    ],
    ids=[
        "empty",
        "single_line",
        "multi_line",
        "new_lines_only",
        "spaces_only",
    ],
)
async def test_read_file_action(executor: LocalCodeExecutor, tmp_path: Path, file_content: str):
    """
    Test the read_file action with different file contents.

    Args:
        executor (LocalCodeExecutor): The executor instance.
        tmp_path (Path): pytest fixture for a temporary directory.
        file_content (str): The content to write to the test file.
    """
    file_path = tmp_path / "test_file.txt"
    file_path.write_text(file_content)

    result = await executor.read_file(str(file_path))

    assert "Successfully read file" in result.formatted_print
    assert str(file_path) in executor.agent_state.conversation[-1].content

    with open(file_path, "r") as f:
        lines = f.readlines()

    expected_content = ""
    for i, line in enumerate(lines):
        line_number = i + 1
        line_length = len(line) - (
            len(line.rstrip("\r\n")) - len(line)
        )  # Handle different line endings
        expected_content += f"{line_number:>4} | {line_length:>4} | {line}"

    assert expected_content in executor.agent_state.conversation[-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_content, expected_content",
    [
        ("New file content", "New file content"),
        ("```\nMarkdown content\n```", "Markdown content"),
        ("```python\nCode content\n```", "Code content"),
        ("```\nLine 1\nLine 2\n```", "Line 1\nLine 2"),
        ("No markdown here", "No markdown here"),
        ("```", ""),
        ("", ""),
    ],
    ids=[
        "no_markdown",
        "basic_markdown",
        "python_markdown",
        "multiline_markdown",
        "no_markdown_present",
        "just_markdown_delimiters",
        "empty_string",
    ],
)
async def test_write_file_action(
    executor: LocalCodeExecutor, tmp_path: Path, file_content: str, expected_content: str
) -> None:
    """
    Test the write_file action with different file contents, including markdown.

    Args:
        executor (LocalCodeExecutor): The executor instance.
        tmp_path (Path): pytest fixture for a temporary directory.
        file_content (str): The content to write to the test file, potentially with markdown.
        expected_content (str): The expected content of the file after writing.
    """
    file_path = tmp_path / "test_file.txt"

    result = await executor.write_file(str(file_path), file_content)

    with open(file_path, "r") as f:
        actual_content = f.read()

    assert actual_content == expected_content
    assert "Successfully wrote to file" in result.formatted_print
    assert str(file_path) in executor.agent_state.conversation[-1].content


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
        assert "Successfully edited file" in result.formatted_print
        assert str(file_path) in executor.agent_state.conversation[-1].content


@pytest.mark.parametrize(
    "initial_history, expected_history",
    [
        (
            [],
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="SYSTEM_PROMPT",
                    is_system_prompt=True,
                )
            ],
        ),
        (
            [
                ConversationRecord(role=ConversationRole.USER, content="User message 1"),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="Assistant message 1"),
            ],
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="SYSTEM_PROMPT",
                    is_system_prompt=True,
                ),
                ConversationRecord(role=ConversationRole.USER, content="User message 1"),
                ConversationRecord(role=ConversationRole.ASSISTANT, content="Assistant message 1"),
            ],
        ),
        (
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="Old system prompt",
                    is_system_prompt=True,
                ),
                ConversationRecord(role=ConversationRole.USER, content="User message 1"),
            ],
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="SYSTEM_PROMPT",
                    is_system_prompt=True,
                ),
                ConversationRecord(role=ConversationRole.USER, content="User message 1"),
            ],
        ),
        (
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="SYSTEM_PROMPT",
                    is_system_prompt=True,
                ),
                ConversationRecord(role=ConversationRole.USER, content="User message 1"),
            ],
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="SYSTEM_PROMPT",
                    is_system_prompt=True,
                ),
                ConversationRecord(role=ConversationRole.USER, content="User message 1"),
            ],
        ),
    ],
    ids=[
        "empty_history",
        "user_and_assistant_messages",
        "old_system_prompt",
        "existing_system_prompt",
    ],
)
def test_initialize_conversation_history(executor, initial_history, expected_history):
    """
    Test the initialize_conversation_history method.

    Args:
        executor: The LocalCodeExecutor fixture.
        initial_history: The initial conversation history.
        expected_history: The expected conversation history after initialization.
    """
    with patch("local_operator.executor.create_system_prompt", return_value="SYSTEM_PROMPT"):
        executor.initialize_conversation_history(initial_history)
        assert executor.agent_state.conversation == expected_history


@pytest.mark.parametrize(
    "initial_history, new_history, expected_history",
    [
        (
            [],
            [],
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="SYSTEM_PROMPT",
                    is_system_prompt=True,
                    should_cache=True,
                ),
            ],
        ),
        (
            [],
            [
                ConversationRecord(
                    role=ConversationRole.USER, content="User message 1", should_cache=False
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="Assistant message 1",
                    should_cache=False,
                ),
            ],
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="SYSTEM_PROMPT",
                    is_system_prompt=True,
                    should_cache=True,
                ),
                ConversationRecord(
                    role=ConversationRole.USER, content="User message 1", should_cache=False
                ),
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content="Assistant message 1",
                    should_cache=False,
                ),
            ],
        ),
        (
            [],
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="Old system prompt",
                    is_system_prompt=True,
                    should_cache=True,
                ),
                ConversationRecord(
                    role=ConversationRole.USER, content="User message 1", should_cache=False
                ),
            ],
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="SYSTEM_PROMPT",
                    is_system_prompt=True,
                    should_cache=True,
                ),
                ConversationRecord(
                    role=ConversationRole.USER, content="User message 1", should_cache=False
                ),
            ],
        ),
        (
            [
                ConversationRecord(
                    role=ConversationRole.USER, content="Existing user message", should_cache=False
                )
            ],
            [],
            [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content="SYSTEM_PROMPT",
                    is_system_prompt=True,
                    should_cache=True,
                ),
            ],
        ),
    ],
    ids=[
        "empty_histories",
        "new_history_with_user_and_assistant",
        "new_history_with_old_system_prompt",
        "initial_history_with_existing_user_message",
    ],
)
def test_load_conversation_history(executor, initial_history, new_history, expected_history):
    """
    Test the load_conversation_history method.

    Args:
        executor: The LocalCodeExecutor fixture.
        initial_history: The initial conversation history.
        new_history: The new conversation history to load.
        expected_history: The expected conversation history after loading.
    """
    with patch("local_operator.executor.create_system_prompt", return_value="SYSTEM_PROMPT"):
        executor.agent_state.conversation = initial_history

        new_agent_state = AgentState(
            version="",
            conversation=new_history,
            execution_history=[],
            learnings=[],
            current_plan=None,
            instruction_details=None,
            agent_system_prompt=None,
        )

        executor.load_agent_state(new_agent_state)
        assert executor.agent_state.conversation == expected_history


def test_get_environment_details(executor, monkeypatch, tmp_path):
    """Test that get_environment_details returns correct environment information."""
    # Mock directory indexing
    mock_index = {
        ".": [
            ("test.py", "code", 1024),
            ("doc.txt", "doc", 500),
            ("image.png", "image", 2048576),
            ("other.bin", "other", 750),
        ]
    }
    monkeypatch.setattr("local_operator.executor.list_working_directory", lambda: mock_index)

    # Mock git status
    def mock_check_output(*args, **kwargs):
        if args[0][0] == "which" or args[0][0] == "where":
            return b"/usr/bin/git"
        if args[0][0] == "git" and args[0][1] == "status":
            return b"On branch main\nnothing to commit, working tree clean\n"
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    # Mock current working directory and datetime
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))
    fixed_datetime = datetime(2024, 1, 1, 12, 0, 0)
    mock_datetime = MagicMock()
    mock_datetime.now.return_value = fixed_datetime
    monkeypatch.setattr("local_operator.executor.datetime", mock_datetime)

    # Mock context variables
    executor.context = {"test_var": "test_value"}

    # Get environment details
    env_details = executor.get_environment_details()

    # Verify output contains expected information
    assert str(tmp_path) in env_details
    assert "2024-01-01 12:00:00" in env_details
    assert "On branch main" in env_details
    assert "nothing to commit, working tree clean" in env_details
    assert "test_var: test_value" in env_details

    # Verify directory tree formatting
    assert "📁 ./" in env_details
    assert "📄 test.py (code, 1.0KB)" in env_details
    assert "📝 doc.txt (doc, 500B)" in env_details
    assert "🖼️ image.png (image, 2.0MB)" in env_details
    assert "📎 other.bin (other, 750B)" in env_details


def test_get_environment_details_not_git_repo(executor, monkeypatch, tmp_path):
    """Test get_environment_details when not in a git repository."""
    # Mock directory indexing with empty directory
    monkeypatch.setattr("local_operator.executor.list_working_directory", lambda: {})

    # Mock git branch check to fail
    def mock_check_output(*args, **kwargs):
        if args[0][0] == "which" or args[0][0] == "where":
            return b"/usr/bin/git"
        if args[0][0] == "git" and args[0][1] == "status":
            raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    # Mock current working directory
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))

    env_details = executor.get_environment_details()

    assert "Not a git repository" in env_details


def test_get_environment_details_no_git(executor, monkeypatch, tmp_path):
    """Test get_environment_details when git is not installed."""
    # Mock directory indexing with empty directory
    monkeypatch.setattr("local_operator.executor.list_working_directory", lambda: {})

    # Mock git branch check to fail
    def mock_check_output(*args, **kwargs):
        if args[0][0] == "which" or args[0][0] == "where":
            return b""
        if args[0][0] == "git" and args[0][1] == "status":
            raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    # Mock current working directory
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))

    env_details = executor.get_environment_details()

    assert "Git is not available on this system" in env_details


def test_get_environment_details_large_directory(executor, monkeypatch):
    """Test get_environment_details handles large directories correctly."""
    # Create mock directory with >300 files
    mock_files = [("file{}.txt".format(i), "doc", 100) for i in range(1000)]
    mock_index = {f"dir{i}": mock_files[i * 100 : (i + 1) * 100] for i in range(10)}
    monkeypatch.setattr("local_operator.executor.list_working_directory", lambda: mock_index)

    # Mock git branch
    def mock_check_output(*args, **kwargs):
        return b"main\n"

    monkeypatch.setattr("subprocess.check_output", mock_check_output)

    env_details = executor.get_environment_details()

    # Verify file count limits are enforced
    assert "... and more files" in env_details
    assert "... and 70 more files" in env_details


@pytest.mark.parametrize(
    "code, error_type, error_msg, expected_content",
    [
        (
            "print(undefined_var)",
            NameError,
            "name 'undefined_var' is not defined",
            ["<error_message>", "name 'undefined_var' is not defined", "err >>", "1 |"],
        ),
        (
            "x = 1/0",
            ZeroDivisionError,
            "division by zero",
            ["<error_message>", "division by zero", "err >>", "1 |"],
        ),
        (
            "def func():\n    x = 1\n    return x + y",
            NameError,
            "name 'y' is not defined",
            ["<error_message>", "name 'y' is not defined"],
        ),
        (
            "import nonexistent_module",
            ModuleNotFoundError,
            "No module named 'nonexistent_module'",
            ["<error_message>", "No module named 'nonexistent_module'", "err >>", "1 |"],
        ),
        (
            "x = [1, 2]\nx[5]",
            IndexError,
            "list index out of range",
            ["<error_message>", "list index out of range", "err >>", "2 |"],
        ),
    ],
    ids=[
        "NameError",
        "ZeroDivisionError",
        "NameErrorMultiline",
        "ModuleNotFoundError",
        "IndexError",
    ],
)
def test_code_execution_error_agent_info_str(
    code, error_type, error_msg, expected_content, monkeypatch
):
    """Test that CodeExecutionError.agent_info_str properly formats error tracebacks."""

    # Create a CodeExecutionError by actually executing the code
    tb = None
    code_execution_error = None
    try:
        # Compile and execute the code to get a real traceback
        compiled_code = compile(code, "<agent_generated_code>", "exec")
        exec(compiled_code)
    except Exception as e:
        if isinstance(e, error_type):
            tb = e.__traceback__
            code_execution_error = CodeExecutionError(str(e), code)
            code_execution_error.__traceback__ = tb
        else:
            # Create an error of the expected type with the real traceback
            temp_error = error_type(error_msg)
            code_execution_error = CodeExecutionError(str(temp_error), code)
            code_execution_error.__traceback__ = temp_error.__traceback__

    # If we didn't get an error (unlikely), create one manually
    if tb is None and code_execution_error is None:
        temp_error = error_type(error_msg)
        code_execution_error = CodeExecutionError(str(temp_error), code)
        # We'll rely on the function to handle missing traceback

    assert code_execution_error is not None

    # Get the annotated error string
    error_str = code_execution_error.agent_info_str()

    # Check that all expected content is in the error message
    for content in expected_content:
        assert content in error_str

    # Verify the structure of the error message
    assert "<error_message>" in error_str
    assert "</error_message>" in error_str
    assert "<agent_generated_code>" in error_str
    assert "<legend>" in error_str
    assert "Error Indicator |Line | Length | Content" in error_str
    assert "</legend>" in error_str
    assert "<code_block>" in error_str
    assert "</code_block>" in error_str
    assert "</agent_generated_code>" in error_str


@pytest.fixture
def executor_with_learnings(executor):
    executor.agent_state.learnings = ["learning1", "learning2"]
    return executor


def test_add_to_learnings_new_learning(executor_with_learnings):
    executor_with_learnings.add_to_learnings("learning3")
    assert "learning3" in executor_with_learnings.agent_state.learnings
    assert len(executor_with_learnings.agent_state.learnings) == 3


def test_add_to_learnings_duplicate_learning(executor_with_learnings):
    executor_with_learnings.add_to_learnings("learning1")
    assert executor_with_learnings.agent_state.learnings.count("learning1") == 1
    assert len(executor_with_learnings.agent_state.learnings) == 2


def test_add_to_learnings_empty_learning(executor_with_learnings):
    executor_with_learnings.add_to_learnings("")
    assert len(executor_with_learnings.agent_state.learnings) == 2


def test_add_to_learnings_none_learning(executor_with_learnings):
    executor_with_learnings.add_to_learnings(None)  # type: ignore
    assert len(executor_with_learnings.agent_state.learnings) == 2


def test_add_to_learnings_exceed_max_learnings(executor):
    executor.max_learnings_history = 2
    executor.agent_state.learnings = ["learning1", "learning2"]
    executor.add_to_learnings("learning3")
    assert "learning1" not in executor.agent_state.learnings
    assert "learning2" in executor.agent_state.learnings
    assert "learning3" in executor.agent_state.learnings
    assert len(executor.agent_state.learnings) == 2


@pytest.mark.parametrize(
    "record_data, expected_length",
    [
        ({"role": ConversationRole.USER, "content": "Test message"}, 1),
        ({"role": ConversationRole.ASSISTANT, "content": "Response message"}, 1),
        ({"role": ConversationRole.SYSTEM, "content": "System message"}, 1),
    ],
)
def test_append_to_history_adds_record(executor, record_data, expected_length):
    # Clear existing history
    executor.agent_state.conversation = []

    # Create record and append
    record = ConversationRecord(**record_data)
    executor.append_to_history(record)

    # Verify record was added
    assert len(executor.agent_state.conversation) == expected_length
    assert executor.agent_state.conversation[0].role == record.role
    assert executor.agent_state.conversation[0].content == record.content
    assert executor.agent_state.conversation[0].timestamp is not None


def test_append_to_history_sets_timestamp_if_none(executor):
    executor.agent_state.conversation = []
    record = ConversationRecord(role=ConversationRole.USER, content="Test message")
    assert record.timestamp is None

    executor.append_to_history(record)

    assert executor.agent_state.conversation[0].timestamp is not None
    assert isinstance(executor.agent_state.conversation[0].timestamp, datetime)


def test_append_to_history_preserves_timestamp(executor):
    executor.agent_state.conversation = []
    timestamp = datetime(2023, 1, 1, 12, 0, 0)
    record = ConversationRecord(
        role=ConversationRole.USER, content="Test message", timestamp=timestamp
    )

    executor.append_to_history(record)

    assert executor.agent_state.conversation[0].timestamp == timestamp


def test_append_to_history_limits_history_length(executor):
    # Set small max history
    executor.max_conversation_history = 3
    executor.agent_state.conversation = [
        ConversationRecord(role=ConversationRole.SYSTEM, content="System prompt")
    ]

    # Add more records than the limit
    for i in range(5):
        record = ConversationRecord(role=ConversationRole.USER, content=f"Message {i+1}")
        executor.append_to_history(record)

    # Verify only the most recent messages are kept
    assert len(executor.agent_state.conversation) == 4
    assert executor.agent_state.conversation[0].content == "System prompt"
    assert (
        executor.agent_state.conversation[1].content
        == "[Some conversation history has been truncated for brevity]"
    )
    assert executor.agent_state.conversation[2].content == "Message 4"
    assert executor.agent_state.conversation[3].content == "Message 5"
