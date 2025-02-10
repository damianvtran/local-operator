import io
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIError

from local_operator.executor import ConfirmSafetyResult
from local_operator.operator import LocalCodeExecutor, Operator, OperatorType
from local_operator.types import ConversationRole, ResponseJsonSchema


# Helper function to normalize code blocks.
def normalize_code_block(code: str) -> str:
    return textwrap.dedent(code).strip()


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
async def test_check_code_safety_safe(executor, mock_model):
    mock_model.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"
    code = "print('hello')"
    result = await executor.check_code_safety(code)
    assert result == ConfirmSafetyResult.SAFE
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_check_code_safety_unsafe(executor, mock_model):
    # Test the default path when can_prompt_user is True
    mock_model.ainvoke.return_value.content = (
        "The code is unsafe because it deletes important files\n\n[UNSAFE]"
    )
    code = "import os; os.remove('important_file.txt')"
    result = await executor.check_code_safety(code)
    assert result == ConfirmSafetyResult.UNSAFE
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_check_code_safety_override(executor, mock_model):
    mock_model.ainvoke.return_value.content = (
        "The code is safe with security override\n\n[OVERRIDE]"
    )
    code = "import os; os.system('some_command')"
    result = await executor.check_code_safety(code)
    assert result == ConfirmSafetyResult.OVERRIDE
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_check_code_safety_unsafe_without_prompt(executor, mock_model):
    # Test the branch when can_prompt_user is False
    executor.can_prompt_user = False
    mock_model.ainvoke.return_value.content = (
        "The code is unsafe because it deletes important files\n\n[UNSAFE]"
    )
    code = "import os; os.remove('important_file.txt')"
    result = await executor.check_code_safety(code)
    assert result == ConfirmSafetyResult.UNSAFE
    mock_model.ainvoke.assert_called_once()
    mock_model.ainvoke.assert_called_with(executor.conversation_history)


@pytest.mark.asyncio
async def test_execute_code_success(executor, mock_model):
    mock_model.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"
    code = "print('hello')"

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.execute_code(code)
        assert "✓ Code Execution Complete" in result
        assert "hello" in result


@pytest.mark.asyncio
async def test_execute_code_no_output(executor, mock_model):
    mock_model.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"
    code = "x = 1 + 1"  # Code that produces no output

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.execute_code(code)
        assert "✓ Code Execution Complete" in result
        assert "[No output]" in result


@pytest.mark.asyncio
async def test_execute_code_safety_no_prompt(executor, mock_model):
    executor.can_prompt_user = False
    mock_model.ainvoke.return_value.content = (
        "The code is unsafe because it deletes important files\n\n[UNSAFE]"
    )
    code = "import os; os.remove('file.txt')"  # Potentially dangerous code

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.execute_code(code)

        # Should not cancel execution but add warning to conversation history
        assert "requires further confirmation" in result
        assert len(executor.conversation_history) > 0
        last_message = executor.conversation_history[-1]
        assert last_message["role"] == "assistant"
        assert "potentially dangerous operation" in last_message["content"]


@pytest.mark.asyncio
async def test_execute_code_safety_with_prompt(executor, mock_model):
    # Default can_prompt_user is True
    mock_model.ainvoke.return_value.content = (
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
        assert last_message["role"] == "user"
        assert "dangerous operation" in last_message["content"]


@pytest.mark.asyncio
async def test_execute_code_safety_with_prompt_approved(executor, mock_model):
    # Default can_prompt_user is True
    mock_model.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"
    code = "x = 1 + 1"

    with (
        patch("sys.stdout", new_callable=io.StringIO),
        patch("builtins.input", return_value="y"),  # User responds "y" to safety prompt
    ):
        result = await executor.execute_code(code)

        # Should proceed with execution when user approves
        assert "Code Execution Complete" in result


@pytest.mark.asyncio
async def test_execute_code_safety_with_override(executor, mock_model):
    # Default can_prompt_user is True
    mock_model.ainvoke.return_value.content = (
        "The code is unsafe but has security override\n\n[OVERRIDE]"
    )
    code = "x = 1 + 1"

    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        result = await executor.execute_code(code)

        # Should proceed with execution and log override
        assert "Code Execution Complete" in result
        output = mock_stdout.getvalue()
        assert "Code safety override applied" in output


@pytest.mark.asyncio
async def test_process_response(executor, mock_model):
    response = ResponseJsonSchema(
        previous_step_success=True,
        previous_goal="",
        current_goal="Print hello world",
        next_goal="",
        response="Here's some code:",
        code="print('hello world')",
        action="CONTINUE",
        learnings="",
        plan="",
    )
    mock_model.ainvoke.return_value.content = "The code is safe\n\n[SAFE]"

    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        await executor.process_response(response)
        output = mock_stdout.getvalue()
        assert "Executing Code Blocks" in output
        assert "hello world" in output


def test_limit_conversation_history(executor):
    test_cases = [
        {"name": "Empty history", "initial": [], "expected": []},
        {
            "name": "Only system prompt",
            "initial": [{"role": "system", "content": "system prompt"}],
            "expected": [{"role": "system", "content": "system prompt"}],
        },
        {
            "name": "History within limit",
            "initial": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "msg2"},
            ],
            "expected": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "msg2"},
            ],
        },
        {
            "name": "History exceeding limit",
            "initial": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "msg2"},
                {"role": "user", "content": "msg3"},
                {"role": "assistant", "content": "msg4"},
            ],
            "expected": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "msg3"},
                {"role": "assistant", "content": "msg4"},
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
            expected_role = msg["role"]
            actual_role = executor.conversation_history[i]["role"]
            assert expected_role == actual_role, (
                f"{test_case['name']}: Expected role {expected_role} but got {actual_role} "
                f"at position {i}"
            )

            expected_content = msg["content"]
            actual_content = executor.conversation_history[i]["content"]
            assert expected_content == actual_content, (
                f"{test_case['name']}: Expected content {expected_content} "
                f"but got {actual_content} at position {i}"
            )


@pytest.mark.asyncio
async def test_summarize_old_steps(mock_model):
    executor = LocalCodeExecutor(model=mock_model, detail_conversation_length=2)

    # Mock the summarization response
    mock_model.ainvoke.return_value.content = "[SUMMARY] This is a summary"

    test_cases = [
        {
            "name": "Empty history",
            "initial": [],
            "expected": [],
        },
        {
            "name": "Only system prompt",
            "initial": [
                {
                    "role": "system",
                    "content": "system prompt",
                    "summarized": "False",
                    "should_summarize": "True",
                }
            ],
            "expected": [
                {
                    "role": "system",
                    "content": "system prompt",
                    "summarized": "False",
                    "should_summarize": "True",
                }
            ],
        },
        {
            "name": "Within detail length",
            "initial": [
                {
                    "role": "system",
                    "content": "system prompt",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "msg1",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "msg2",
                    "summarized": "False",
                    "should_summarize": "True",
                },
            ],
            "expected": [
                {
                    "role": "system",
                    "content": "system prompt",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "msg1",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "msg2",
                    "summarized": "False",
                    "should_summarize": "True",
                },
            ],
        },
        {
            "name": "Beyond detail length with skip conditions",
            "initial": [
                {
                    "role": "system",
                    "content": "system prompt",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "user",
                    "content": "user msg",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "skip me",
                    "summarized": "False",
                    "should_summarize": "False",
                },
                {
                    "role": "assistant",
                    "content": "already summarized",
                    "summarized": "True",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "summarize me",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "recent1",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "recent2",
                    "summarized": "False",
                    "should_summarize": "True",
                },
            ],
            "expected": [
                {
                    "role": "system",
                    "content": "system prompt",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "user",
                    "content": "user msg",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "skip me",
                    "summarized": "False",
                    "should_summarize": "False",
                },
                {
                    "role": "assistant",
                    "content": "already summarized",
                    "summarized": "True",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "[SUMMARY] This is a summary",
                    "summarized": "True",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "recent1",
                    "summarized": "False",
                    "should_summarize": "True",
                },
                {
                    "role": "assistant",
                    "content": "recent2",
                    "summarized": "False",
                    "should_summarize": "True",
                },
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
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user msg"},
    ]

    await executor._summarize_old_steps()

    assert executor.conversation_history == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user msg"},
    ]


@pytest.mark.asyncio
async def test_invoke_model_api_error(executor):
    mock_request = MagicMock()
    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        executor.model.ainvoke.side_effect = APIError(
            message="API Error",
            request=mock_request,
            body={"code": "error_code", "type": "error_type"},
        )

        with pytest.raises(APIError) as exc_info:
            await executor.invoke_model([{"role": "user", "content": "test"}])

        assert str(exc_info.value) == "API Error"
        assert exc_info.value.code == "error_code"
        assert exc_info.value.type == "error_type"


@pytest.mark.asyncio
async def test_invoke_model_rate_limit(executor):
    mock_request = MagicMock()
    executor.model.ainvoke.side_effect = APIError(
        message="Rate limit exceeded",
        request=mock_request,
        body={"code": "rate_limit_exceeded", "type": "rate_limit_error"},
    )

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(APIError) as exc_info:
            await executor.invoke_model([{"role": "user", "content": "test"}])

    assert str(exc_info.value) == "Rate limit exceeded"
    assert exc_info.value.code == "rate_limit_exceeded"
    assert exc_info.value.type == "rate_limit_error"


@pytest.mark.asyncio
async def test_invoke_model_context_length(executor):
    mock_request = MagicMock()
    executor.model.ainvoke.side_effect = APIError(
        message="Maximum context length exceeded",
        request=mock_request,
        body={"code": "context_length_exceeded", "type": "invalid_request_error"},
    )

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(APIError) as exc_info:
            await executor.invoke_model([{"role": "user", "content": "test"}])

    assert str(exc_info.value) == "Maximum context length exceeded"
    assert exc_info.value.code == "context_length_exceeded"
    assert exc_info.value.type == "invalid_request_error"


@pytest.mark.asyncio
async def test_invoke_model_general_exception(executor):
    executor.model.ainvoke.side_effect = Exception("Unexpected error")

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(Exception) as exc_info:
            await executor.invoke_model([{"role": "user", "content": "test"}])

    assert str(exc_info.value) == "Unexpected error"


@pytest.mark.asyncio
async def test_invoke_model_timeout(executor):
    executor.model.ainvoke.side_effect = TimeoutError("Request timed out")

    with patch("asyncio.sleep", AsyncMock(return_value=None)):
        with pytest.raises(TimeoutError) as exc_info:
            await executor.invoke_model([{"role": "user", "content": "test"}])

    assert str(exc_info.value) == "Request timed out"


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "No working directory",
            "conversation": [
                {"role": ConversationRole.SYSTEM.value, "content": "Some system message"},
                {"role": ConversationRole.USER.value, "content": "Test message"},
            ],
            "expected": None,
        },
        {
            "name": "Single working directory",
            "conversation": [
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "Current working directory: /test/path",
                },
                {"role": ConversationRole.USER.value, "content": "Test message"},
            ],
            "expected": Path("/test/path"),
        },
        {
            "name": "Multiple working directories - returns most recent",
            "conversation": [
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "Current working directory: /old/path",
                },
                {"role": ConversationRole.USER.value, "content": "Test message"},
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "Current working directory: /new/path",
                },
            ],
            "expected": Path("/new/path"),
        },
        {
            "name": "Case insensitive working directory",
            "conversation": [
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "CURRENT WORKING DIRECTORY: /test/path",
                },
                {"role": ConversationRole.USER.value, "content": "Test message"},
            ],
            "expected": Path("/test/path"),
        },
        {
            "name": "Message with working directory but not at start",
            "conversation": [
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "Some text before Current working directory: /test/path",
                },
                {"role": ConversationRole.USER.value, "content": "Test message"},
            ],
            "expected": None,
        },
        {
            "name": "Multiple entries - returns last in list",
            "conversation": [
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "Current working directory: /path/one",
                },
                {"role": ConversationRole.USER.value, "content": "Test message"},
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "Current working directory: /path/two",
                },
                {"role": ConversationRole.USER.value, "content": "Another message"},
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "Current working directory: /path/three",
                },
            ],
            "expected": Path("/path/three"),
        },
        {
            "name": "First entry with working directory",
            "conversation": [
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "Current working directory: /first/path",
                },
                {"role": ConversationRole.USER.value, "content": "Test message"},
                {
                    "role": ConversationRole.SYSTEM.value,
                    "content": "Some other system message",
                },
                {"role": ConversationRole.ASSISTANT.value, "content": "Response"},
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
