import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_operator.agent import CliOperator, ConversationRole, LocalCodeExecutor


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

    config_manager = MagicMock()
    config_manager.get_config_value = MagicMock(return_value="test_value")

    operator = CliOperator(
        credential_manager=credential_manager,
        model_instance=mock_model,
        config_manager=config_manager,
    )

    operator._get_input_with_history = MagicMock(return_value="noop")

    return operator


def test_extract_code_blocks(executor):
    test_cases = [
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
                "def calculate_sum(a, b):\n                result = a + b\n"
                "                return result\n\n            total = calculate_sum(5, 3)\n"
                '            print(f"The sum is: {total}")'
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
    ]

    for case in test_cases:
        result = executor.extract_code_blocks(case["input"])
        assert len(result) == case["expected_count"], (
            f"Test case '{case['name']}' failed: "
            f"expected {case['expected_count']} code blocks but got {len(result)}"
        )
        for expected, actual in zip(case["expected_blocks"], result):
            assert expected in actual.strip(), (
                f"Test case '{case['name']}' failed: "
                f"expected code block '{expected}' not found in actual result '{actual.strip()}'"
            )


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
        assert "✓ Code Execution Complete" in result
        assert "hello" in result


@pytest.mark.asyncio
async def test_execute_code_no_output(executor, mock_model):
    mock_model.ainvoke.return_value.content = "no"  # Safety check passes
    code = "x = 1 + 1"  # Code that produces no output

    with patch("sys.stdout", new_callable=io.StringIO):
        result = await executor.execute_code(code)
        assert "✓ Code Execution Complete" in result
        assert "[No output]" in result


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


def test_cli_operator_init(mock_model):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    config_manager = MagicMock()
    config_manager.get_config_value = MagicMock(return_value="test_value")

    operator = CliOperator(
        credential_manager=credential_manager,
        model_instance=mock_model,
        config_manager=config_manager,
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
