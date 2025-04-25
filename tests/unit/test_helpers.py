import os
import subprocess  # Added import
from unittest.mock import MagicMock, patch

import pytest

from local_operator.helpers import (
    clean_json_response,
    clean_plain_text_response,
    get_user_shell_path,
    remove_think_tags,
    setup_subprocess_environment,
)


@pytest.mark.parametrize(
    "response_content, expected_output",
    [
        ("This is a test <think>with think tags</think>.", "This is a test ."),
        ("No think tags here.", "No think tags here."),
        ("<think>Only think tags</think>", ""),
        ("<think>Think tags leading content</think> content", "content"),
        ("", ""),
    ],
)
def test_remove_think_tags(response_content, expected_output):
    """
    Test the remove_think_tags function with various inputs.

    Args:
        response_content (str): The input string containing <think> tags.
        expected_output (str): The expected output string after removing <think> tags.
    """
    assert remove_think_tags(response_content) == expected_output


@pytest.mark.parametrize(
    "response_content, expected_output",
    [
        ("", ""),
        ("Just plain text", "Just plain text"),
        ("Line 1\nLine 2", "Line 1\nLine 2"),
        ("Text with  multiple   spaces", "Text with multiple spaces"),
        ("Text with trailing spaces  \n  Line 2  ", "Text with trailing spaces\nLine 2"),
        (
            "Here's a reflection:\n```python\nprint('hello world')\n```\nThis is important.",
            "Here's a reflection:\n\nThis is important.",
        ),
        (
            "Here's a reflection:\n```python\nprint('hello world')\n",
            "Here's a reflection:",
        ),
        ("Normal text without code blocks.", "Normal text without code blocks."),
        ('```json\n{"key": "value"}\n```\nAfter the code block.', "After the code block."),
        ("Before\n```\nUnspecified code block\n```\nAfter", "Before\n\nAfter"),
        ("```python\nMultiple\nLines\nOf\nCode\n```", ""),
        (
            '{\n"action": "EXECUTE_CODE",\n"code": "print(\'test\')"\n}',
            "",
        ),
        ('main content {"json": "value" }', "main content"),
        ('main content {"json": "value" } following text', "main content following text"),
        ('{"simple_key": "value"}', ""),
        (
            'Multiline\n\nInput\n\nI will respond with a message explaining my action. {"response":"Example response","code":"","content":"","file_path":"","mentioned_files":[],"replacements":[],"action":"DONE","learnings":"Example learnings"}',  # noqa: E501
            "Multiline\n\nInput\n\nI will respond with a message explaining my action.",
        ),
    ],
)
def test_clean_plain_text_response(response_content, expected_output):
    """
    Test the clean_plain_text_response function with various inputs.

    Args:
        response_content (str): The input string containing code blocks or JSON.
        expected_output (str): The expected output string after cleaning.
    """
    assert clean_plain_text_response(response_content) == expected_output


@pytest.mark.parametrize(
    "response_content, expected_output",
    [
        pytest.param(
            '```json\n{"action": "EXECUTE_CODE", "code": "print(\'test\')"}\n```',
            '{"action": "EXECUTE_CODE", "code": "print(\'test\')"}',
            id="json_in_code_block",
        ),
        pytest.param(
            '{"action": "EXECUTE_CODE", "code": "print(\'test\')"}',
            '{"action": "EXECUTE_CODE", "code": "print(\'test\')"}',
            id="plain_json",
        ),
        pytest.param(
            'Some text before ```json\n{"action": "EXECUTE_CODE"}\n``` and after',
            '{"action": "EXECUTE_CODE"}',
            id="json_with_surrounding_text",
        ),
        pytest.param(
            'Text {"action": "EXECUTE_CODE"} more text',
            '{"action": "EXECUTE_CODE"}',
            id="json_embedded_in_text",
        ),
        pytest.param(
            '<think>Thinking...</think>{"action": "EXECUTE_CODE"}',
            '{"action": "EXECUTE_CODE"}',
            id="json_with_think_tags",
        ),
        pytest.param(
            '```json\n{"nested": {"key": "value"}}\n```',
            '{"nested": {"key": "value"}}',
            id="nested_json_in_code_block",
        ),
        pytest.param('{"incomplete": "json"', '{"incomplete": "json"', id="incomplete_json"),
        pytest.param("No JSON content", "No JSON content", id="no_json_content"),
        pytest.param(
            "Braces {in text} but not JSON",
            "Braces {in text} but not JSON",
            id="braces_in_text_not_json",
        ),
        pytest.param(
            "```python\nprint('hello world')\n```",
            "```python\nprint('hello world')\n```",
            id="python_code_block",
        ),
        pytest.param(
            'JSON response content: ```json\n{"action": "WRITE", "learnings": "I learned...", "response": "Writing the first 10 Fibonacci numbers to a markdown file in a readable format.", "code": "", "content": "# First 10 Fibonacci Numbers\\n\\nThe following are the first 10 Fibonacci numbers, starting with F(0) = 0 and F(1) = 1. Each subsequent number is the sum of the two preceding numbers.\\n\\n```\\n0, 1, 1, 2, 3, 5, 8, 13, 21, 34\\n```", "file_path": "fibonacci_numbers.md", "mentioned_files": [], "replacements": []}\n```',  # noqa: E501
            '{"action": "WRITE", "learnings": "I learned...", "response": "Writing the first 10 Fibonacci numbers to a markdown file in a readable format.", "code": "", "content": "# First 10 Fibonacci Numbers\\n\\nThe following are the first 10 Fibonacci numbers, starting with F(0) = 0 and F(1) = 1. Each subsequent number is the sum of the two preceding numbers.\\n\\n```\\n0, 1, 1, 2, 3, 5, 8, 13, 21, 34\\n```", "file_path": "fibonacci_numbers.md", "mentioned_files": [], "replacements": []}',  # noqa: E501
            id="json_response_content_marker",
        ),
        pytest.param(
            '{"find": "old_content", "replace": "```json{"new_content": "value"}```"}',
            '{"find": "old_content", "replace": "```json{"new_content": "value"}```"}',
            id="json_in_replace_block_with_newlines",
        ),
        pytest.param(
            '```json\n{"find": "old_content", "replace": "```json{"new_content": "value"}```"}\n```',  # noqa: E501
            '{"find": "old_content", "replace": "```json{"new_content": "value"}```"}',
            id="json_in_replace_block_with_newlines_and_outer_marker",
        ),
        pytest.param(
            'This is some text before\n\n```json\n{"find": "old_content", "replace": "```json{"new_content": "value"}```"}\n```',  # noqa: E501
            '{"find": "old_content", "replace": "```json{"new_content": "value"}```"}',
            id="json_in_replace_block_with_newlines_and_outer_marker_with_leading_text",
        ),
    ],
)
def test_clean_json_response(response_content: str, expected_output: str) -> None:
    """Test the clean_json_response function with various inputs.

    Args:
        response_content (str): Input string containing JSON with potential code blocks
        or think tags
        expected_output (str): Expected cleaned JSON output
    """
    assert clean_json_response(response_content) == expected_output


# --- Tests for Environment Setup ---


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run")
@patch("local_operator.helpers.os.path.exists")
@patch("local_operator.helpers.os.getlogin")
def test_get_user_shell_path_windows(
    mock_getlogin, mock_exists, mock_run, mock_environ, mock_system
):
    """Test get_user_shell_path on Windows."""
    mock_system.return_value = "Windows"
    assert get_user_shell_path() is None
    mock_run.assert_not_called()


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run")
def test_get_user_shell_path_unsupported_os(mock_run, mock_environ, mock_system):
    """Test get_user_shell_path on an unsupported OS."""
    mock_system.return_value = "Solaris"
    assert get_user_shell_path() is None
    mock_run.assert_not_called()


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run")
@patch("local_operator.helpers.os.path.exists")
@patch("local_operator.helpers.os.getlogin")
def test_get_user_shell_path_macos_success(
    mock_getlogin, mock_exists, mock_run, mock_environ, mock_system
):
    """Test get_user_shell_path on macOS successfully retrieves PATH."""
    mock_system.return_value = "Darwin"
    mock_environ["SHELL"] = "/bin/zsh"
    mock_exists.return_value = True
    mock_run.return_value = MagicMock(stdout="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin\n")
    expected_path = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
    assert get_user_shell_path() == expected_path
    mock_run.assert_called_once_with(
        '"/bin/zsh" -ilc "echo $PATH"',
        capture_output=True,
        text=True,
        check=True,
        shell=True,
        env=mock_environ,
    )


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run")
@patch("local_operator.helpers.os.path.exists")
@patch("local_operator.helpers.os.getlogin")
def test_get_user_shell_path_linux_no_shell_env_var(
    mock_getlogin, mock_exists, mock_run, mock_environ, mock_system
):
    """Test get_user_shell_path on Linux when SHELL env var is not set."""
    mock_system.return_value = "Linux"
    # SHELL is not in mock_environ
    mock_exists.return_value = True
    mock_run.return_value = MagicMock(stdout="/usr/local/bin:/usr/bin:/bin\n")
    expected_path = "/usr/local/bin:/usr/bin:/bin"
    assert get_user_shell_path() == expected_path
    # Should default to /bin/bash on Linux
    mock_run.assert_called_once_with(
        '"/bin/bash" -ilc "echo $PATH"',
        capture_output=True,
        text=True,
        check=True,
        shell=True,
        env=mock_environ,
    )


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run")
@patch("local_operator.helpers.os.path.exists")
@patch("local_operator.helpers.os.getlogin")
def test_get_user_shell_path_command_fails(
    mock_getlogin, mock_exists, mock_run, mock_environ, mock_system
):
    """Test get_user_shell_path when the shell command fails."""
    mock_system.return_value = "Darwin"
    mock_environ["SHELL"] = "/bin/zsh"
    mock_exists.return_value = True
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="Error")
    assert get_user_shell_path() is None


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run")
@patch("local_operator.helpers.os.path.exists")
@patch("local_operator.helpers.os.getlogin")
def test_get_user_shell_path_empty_output(
    mock_getlogin, mock_exists, mock_run, mock_environ, mock_system
):
    """Test get_user_shell_path when the shell command returns empty output."""
    mock_system.return_value = "Linux"
    mock_environ["SHELL"] = "/bin/bash"
    mock_exists.return_value = True
    mock_run.return_value = MagicMock(stdout="\n")  # Empty or just newline
    assert get_user_shell_path() is None


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.get_user_shell_path")
@patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True)
def test_setup_subprocess_environment_windows(mock_get_path, mock_system):
    """Test setup_subprocess_environment on Windows."""
    mock_system.return_value = "Windows"
    initial_path = os.environ["PATH"]
    setup_subprocess_environment()
    assert os.environ["PATH"] == initial_path  # PATH should not change
    mock_get_path.assert_not_called()


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.get_user_shell_path")
@patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True)
def test_setup_subprocess_environment_updates_path(mock_get_path, mock_system):
    """Test setup_subprocess_environment updates PATH when different."""
    mock_system.return_value = "Darwin"
    new_path = "/opt/homebrew/bin:/usr/bin:/bin"
    mock_get_path.return_value = new_path
    setup_subprocess_environment()
    assert os.environ["PATH"] == new_path


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.get_user_shell_path")
@patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True)
def test_setup_subprocess_environment_same_path(mock_get_path, mock_system):
    """Test setup_subprocess_environment does nothing if PATH is the same."""
    mock_system.return_value = "Linux"
    same_path = "/usr/bin:/bin"
    mock_get_path.return_value = same_path
    initial_path = os.environ["PATH"]
    setup_subprocess_environment()
    assert os.environ["PATH"] == initial_path  # PATH should not change


@patch("local_operator.helpers.platform.system")
@patch("local_operator.helpers.get_user_shell_path")
@patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True)
def test_setup_subprocess_environment_get_path_fails(mock_get_path, mock_system):
    """Test setup_subprocess_environment does nothing if get_user_shell_path returns None."""
    mock_system.return_value = "Darwin"
    mock_get_path.return_value = None
    initial_path = os.environ["PATH"]
    setup_subprocess_environment()
    assert os.environ["PATH"] == initial_path  # PATH should not change
