import os
import platform
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from local_operator.helpers import (
    is_marker_inside_json,  # Keep existing test helper if needed
)
from local_operator.helpers import parse_agent_action_xml  # Added import
from local_operator.helpers import (
    clean_json_response,
    clean_plain_text_response,
    get_posix_shell_path,
    get_windows_registry_path,
    remove_think_tags,
    setup_cross_platform_environment,
)

# Pre-define potential winreg attributes for type checkers
HKEY_CURRENT_USER = None
HKEY_LOCAL_MACHINE = None
KEY_READ = None
OpenKey = None
QueryValueEx = None
winreg = None

# Mock winreg for non-Windows platforms
if platform.system() != "Windows":
    # Create a mock object for the winreg module
    winreg_mock = MagicMock()
    sys.modules["winreg"] = winreg_mock

    # Assign constants and functions to the mock object
    winreg_mock.HKEY_CURRENT_USER = 1
    winreg_mock.HKEY_LOCAL_MACHINE = 2
    winreg_mock.KEY_READ = 3
    winreg_mock.OpenKey = MagicMock()
    winreg_mock.QueryValueEx = MagicMock()

    # Also assign to top-level names for patching convenience if needed elsewhere
    # (though patching sys.modules['winreg'] is usually sufficient)
    HKEY_CURRENT_USER = winreg_mock.HKEY_CURRENT_USER
    HKEY_LOCAL_MACHINE = winreg_mock.HKEY_LOCAL_MACHINE
    KEY_READ = winreg_mock.KEY_READ
    OpenKey = winreg_mock.OpenKey
    QueryValueEx = winreg_mock.QueryValueEx
    winreg = winreg_mock  # Assign the mock to the winreg name
else:
    import winreg  # Import real winreg on Windows

    # Assign real values to top-level names
    HKEY_CURRENT_USER = winreg.HKEY_CURRENT_USER  # type: ignore
    HKEY_LOCAL_MACHINE = winreg.HKEY_LOCAL_MACHINE  # type: ignore
    KEY_READ = winreg.KEY_READ  # type: ignore
    OpenKey = winreg.OpenKey  # type: ignore

# --- Tests for Response Cleaning (Keep Existing) ---


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
    """Test the remove_think_tags function."""
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
    """Test the clean_plain_text_response function."""
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
    """Test the clean_json_response function."""
    assert clean_json_response(response_content) == expected_output


@pytest.mark.parametrize(
    "text, marker, expected",
    [
        ('{"key": "value `marker`"}', "`marker`", True),
        ('```json\n{\n  "key": "value"\n}\n```', "```", False),
        ("Some text ``` marker ``` more text", "```", False),
        ('{"outer": {"inner": "```"}}', "```", True),
    ],
)
def test_is_marker_inside_json(text, marker, expected):
    """Test the is_marker_inside_json function."""
    assert is_marker_inside_json(text, marker) == expected


# --- Tests for parse_agent_action_xml ---
@pytest.mark.parametrize(
    "xml_string, expected_dict",
    [
        pytest.param(
            "<action>CODE</action><learnings>Learned something</learnings><response>Running code</response><code>print('Hello')</code>",  # noqa: E501
            {
                "action": "CODE",
                "learnings": "Learned something",
                "response": "Running code",
                "code": "print('Hello')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="basic_code_action",
        ),
        pytest.param(
            (
                "Some text before <action_response><action>WRITE</action>"
                "<content>File content</content><file_path>test.txt</file_path>"
                "</action_response> and text after."
            ),
            {
                "action": "WRITE",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "File content",
                "file_path": "test.txt",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="write_action_with_surrounding_text_and_action_response_tag",
        ),
        pytest.param(
            "```xml\n<action>EDIT</action><file_path>doc.md</file_path><replacements>\n- old line\n- another old line\n+ new line 1\n+ new line 2\n</replacements>\n```",  # noqa: E501
            {
                "action": "EDIT",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "doc.md",
                "replacements": [
                    {"find": "old line\nanother old line", "replace": "new line 1\nnew line 2"}
                ],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="edit_action_with_markdown_wrapper_and_replacements",
        ),
        pytest.param(
            "```\n<action>DONE</action><response>Task complete</response>\n```",
            {
                "action": "DONE",
                "learnings": "",
                "response": "Task complete",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="done_action_with_generic_markdown_wrapper",
        ),
        pytest.param(
            "<action>ASK</action><response>Need more info</response>",
            {
                "action": "ASK",
                "learnings": "",
                "response": "Need more info",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="ask_action",
        ),
        pytest.param(
            "<action>BYE</action>",
            {
                "action": "BYE",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="bye_action_empty_tags",
        ),
        pytest.param(
            "<action>DELEGATE</action><agent>OtherAgent</agent><message>Please help</message>",
            {
                "action": "DELEGATE",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "OtherAgent",
                "message": "Please help",
                "mentioned_files": [],
            },
            id="delegate_action",
        ),
        pytest.param(
            "<action>CODE</action><learnings></learnings><response></response><code></code>",
            {
                "action": "CODE",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="empty_tags_for_code",
        ),
        pytest.param(
            (
                "<action>EDIT</action><file_path>file.txt</file_path><replacements>\n"
                "- find1\n+ replace1\n- find2\n+ replace2\n</replacements>"
            ),
            {
                "action": "EDIT",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "file.txt",
                "replacements": [
                    {"find": "find1", "replace": "replace1"},
                    {"find": "find2", "replace": "replace2"},
                ],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="multiple_replacements",
        ),
        pytest.param(
            "<action>READ</action><file_path>data.csv</file_path><response>Reading file</response>",
            {
                "action": "READ",
                "learnings": "",
                "response": "Reading file",
                "code": "",
                "content": "",
                "file_path": "data.csv",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="read_action",
        ),
        pytest.param(
            "<outer><action>CODE</action><code>print(1)</code></outer>",  # Simple nesting
            {
                "action": "CODE",  # Expects to find the first valid action tag
                "learnings": "",
                "response": "",
                "code": "print(1)",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="simple_nested_xml",
        ),
        pytest.param(
            "<action>EDIT</action><file_path>test.txt</file_path><replacements></replacements>",
            {
                "action": "EDIT",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "test.txt",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="edit_action_empty_replacements_tag",
        ),
        pytest.param(
            "<action>CODE</action><code>print('<tag>value</tag>')</code>",
            {
                "action": "CODE",
                "learnings": "",
                "response": "",
                "code": "print('<tag>value</tag>')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="code_with_xml_entities",
        ),
        pytest.param(
            "<action>WRITE</action><content><doc><title>My Doc</title></doc></content><file_path>my.xml</file_path>",  # noqa: E501
            {
                "action": "WRITE",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "<doc><title>My Doc</title></doc>",
                "file_path": "my.xml",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="write_xml_content_as_entities",
        ),
        pytest.param(
            "<action>WRITE</action><content><![CDATA[<doc><title>My Doc</title></doc>]]></content><file_path>my.xml</file_path>",  # noqa: E501
            {
                "action": "WRITE",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "<![CDATA[<doc><title>My Doc</title></doc>]]>",
                "file_path": "my.xml",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="write_xml_content_with_cdata",
        ),
        pytest.param(
            "<action>EDIT</action><file_path>config.xml</file_path><replacements>\n- <old_setting>true</old_setting>\n+ <new_setting>false</new_setting>\n</replacements>",  # noqa: E501
            {
                "action": "EDIT",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "config.xml",
                "replacements": [
                    {
                        "find": "<old_setting>true</old_setting>",
                        "replace": "<new_setting>false</new_setting>",
                    }
                ],
                "agent": "",
                "message": "",
                "mentioned_files": [],
            },
            id="edit_replacing_xml_entities",
        ),
    ],
)
def test_parse_agent_action_xml(xml_string, expected_dict):
    """Test the parse_agent_action_xml function."""
    assert parse_agent_action_xml(xml_string) == expected_dict


# --- Environment Setup Tests (Keep Existing) ---
@patch("local_operator.helpers.platform.system", return_value="Windows")
@patch("winreg.OpenKey")  # type: ignore
@patch("winreg.QueryValueEx")  # type: ignore
def test_get_windows_registry_path_success(mock_query, mock_open_key, mock_system):
    """Test successful PATH retrieval from Windows registry."""
    import os
    from unittest.mock import patch

    # Patch os.pathsep to ':' for the test context
    with patch.object(os, "pathsep", ":", create=True):
        # Mock context manager returned by OpenKey
        mock_key_user = MagicMock()
        mock_key_system = MagicMock()
        mock_open_key.side_effect = [mock_key_user, mock_key_system]

        # Mock QueryValueEx return values - Use os.pathsep dynamically
        user_path_mock = f"C:\\UserPath1{os.pathsep}C:\\UserPath2"
        system_path_mock = f"C:\\SystemPath1{os.pathsep}C:\\Windows"
        mock_query.side_effect = [
            (user_path_mock, 1),  # User Path
            (system_path_mock, 1),  # System Path
        ]

        # Construct expected path directly using os.pathsep, matching function output
        expected_path = "C:\\UserPath1:\\UserPath2:\\SystemPath1:\\Windows"

        assert get_windows_registry_path() == expected_path
        assert mock_open_key.call_count == 2
        assert mock_query.call_count == 2


@patch("local_operator.helpers.platform.system", return_value="Windows")
@patch("winreg.OpenKey")  # type: ignore
@patch("winreg.QueryValueEx")  # type: ignore
def test_get_windows_registry_path_no_user_path(mock_query, mock_open_key, mock_system):
    """Test registry retrieval when User PATH doesn't exist."""
    mock_key_system = MagicMock()
    # Simulate FileNotFoundError for user key, then success for system key
    mock_open_key.side_effect = [FileNotFoundError, mock_key_system]
    mock_query.return_value = ("C:\\SystemPath1", 1)  # Only system path queried

    expected_path = "C:\\SystemPath1"
    assert get_windows_registry_path() == expected_path
    assert mock_open_key.call_count == 2
    mock_query.assert_called_once()  # Only called for system path


@patch("local_operator.helpers.platform.system", return_value="Windows")
@patch("winreg.OpenKey", side_effect=Exception("Registry Error"))  # type: ignore
def test_get_windows_registry_path_registry_error(mock_open_key, mock_system):
    """Test registry retrieval failure."""
    assert get_windows_registry_path() is None


@patch("local_operator.helpers.platform.system", return_value="Linux")
def test_get_windows_registry_path_not_windows(mock_system):
    """Test function returns None on non-Windows OS."""
    assert get_windows_registry_path() is None


# --- get_posix_shell_path Tests ---


@patch("local_operator.helpers.platform.system", return_value="Darwin")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run")
def test_get_posix_shell_path_macos_success(mock_run, mock_environ, mock_system):
    """Test successful PATH retrieval on macOS."""
    mock_environ["SHELL"] = "/bin/zsh"
    mock_run.return_value = MagicMock(stdout="/usr/bin:/bin:/opt/homebrew/bin\n")
    expected_path = "/usr/bin:/bin:/opt/homebrew/bin"
    assert get_posix_shell_path() == expected_path
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args == "'/bin/zsh' -l -c 'echo \"$PATH\"'"


@patch("local_operator.helpers.platform.system", return_value="Linux")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run")
@patch("local_operator.helpers.os.path.expanduser", return_value="/home/user/.local/bin")
@patch("local_operator.helpers.os.path.isdir", return_value=True)
def test_get_posix_shell_path_linux_adds_local_bin(
    mock_isdir, mock_expanduser, mock_run, mock_environ, mock_system
):
    """Test that ~/.local/bin is added on Linux if missing."""
    mock_environ["SHELL"] = "/bin/bash"
    mock_run.return_value = MagicMock(stdout="/usr/bin:/bin\n")
    expected_path = f"/home/user/.local/bin{os.pathsep}/usr/bin:/bin"
    assert get_posix_shell_path() == expected_path
    mock_run.assert_called_once()
    mock_expanduser.assert_called_once_with("~/.local/bin")
    mock_isdir.assert_called_once_with("/home/user/.local/bin")


@patch("local_operator.helpers.platform.system", return_value="Linux")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run")
@patch("local_operator.helpers.os.path.expanduser", return_value="/home/user/.local/bin")
@patch("local_operator.helpers.os.path.isdir", return_value=True)
def test_get_posix_shell_path_linux_local_bin_exists(
    mock_isdir, mock_expanduser, mock_run, mock_environ, mock_system
):
    """Test that ~/.local/bin is not added if already present."""
    mock_environ["SHELL"] = "/bin/bash"
    existing_path = "/usr/bin:/bin:/home/user/.local/bin"  # No f needed here
    # Simulate stdout without trailing newline as .strip() is used in the function
    mock_run.return_value = MagicMock(stdout=existing_path)
    assert get_posix_shell_path() == existing_path  # Path should be unchanged
    mock_run.assert_called_once()
    mock_expanduser.assert_called_once_with("~/.local/bin")
    mock_isdir.assert_called_once_with("/home/user/.local/bin")


@patch("local_operator.helpers.platform.system", return_value="Darwin")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch("local_operator.helpers.subprocess.run", side_effect=FileNotFoundError)
def test_get_posix_shell_path_shell_not_found(mock_run, mock_environ, mock_system):
    """Test failure when the shell executable is not found."""
    mock_environ["SHELL"] = "/non/existent/shell"
    assert get_posix_shell_path() is None


@patch("local_operator.helpers.platform.system", return_value="Linux")
@patch("local_operator.helpers.os.environ", new_callable=dict)
@patch(
    "local_operator.helpers.subprocess.run",
    side_effect=subprocess.CalledProcessError(1, "cmd", stderr="Error"),
)
def test_get_posix_shell_path_command_fails(mock_run, mock_environ, mock_system):
    """Test failure when the shell command returns an error."""
    mock_environ["SHELL"] = "/bin/bash"
    assert get_posix_shell_path() is None


@patch("local_operator.helpers.platform.system", return_value="Windows")
def test_get_posix_shell_path_not_posix(mock_system):
    """Test function returns None on non-POSIX OS."""
    assert get_posix_shell_path() is None


# --- setup_cross_platform_environment Tests ---


@patch("local_operator.helpers.platform.system", return_value="Windows")
@patch("local_operator.helpers.get_windows_registry_path")
@patch("local_operator.helpers.get_posix_shell_path")
@patch.dict(os.environ, {"PATH": "C:\\Initial"}, clear=True)
def test_setup_env_windows_updates(mock_get_posix, mock_get_win, mock_system):
    """Test setup updates PATH on Windows when different."""
    new_path = "C:\\User;C:\\System"
    mock_get_win.return_value = new_path
    setup_cross_platform_environment()
    assert os.environ["PATH"] == new_path
    mock_get_posix.assert_not_called()
    mock_get_win.assert_called_once()


@patch("local_operator.helpers.platform.system", return_value="Darwin")
@patch("local_operator.helpers.get_windows_registry_path")
@patch("local_operator.helpers.get_posix_shell_path")
@patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True)
def test_setup_env_macos_updates(mock_get_posix, mock_get_win, mock_system):
    """Test setup updates PATH on macOS when different."""
    new_path = "/opt/homebrew/bin:/usr/bin"
    mock_get_posix.return_value = new_path
    setup_cross_platform_environment()
    assert os.environ["PATH"] == new_path
    mock_get_win.assert_not_called()
    mock_get_posix.assert_called_once()


@patch("local_operator.helpers.platform.system", return_value="Linux")
@patch("local_operator.helpers.get_windows_registry_path")
@patch("local_operator.helpers.get_posix_shell_path")
@patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True)
def test_setup_env_linux_same_path(mock_get_posix, mock_get_win, mock_system):
    """Test setup does not update PATH on Linux when it's the same."""
    same_path = "/usr/bin"
    mock_get_posix.return_value = same_path
    initial_path = os.environ["PATH"]
    setup_cross_platform_environment()
    assert os.environ["PATH"] == initial_path
    mock_get_win.assert_not_called()
    mock_get_posix.assert_called_once()


@patch("local_operator.helpers.platform.system", return_value="Windows")
@patch("local_operator.helpers.get_windows_registry_path", return_value=None)
@patch("local_operator.helpers.get_posix_shell_path")
@patch.dict(os.environ, {"PATH": "C:\\Initial"}, clear=True)
def test_setup_env_windows_retrieval_fails(mock_get_posix, mock_get_win, mock_system):
    """Test setup does not update PATH on Windows if retrieval fails."""
    initial_path = os.environ["PATH"]
    setup_cross_platform_environment()
    assert os.environ["PATH"] == initial_path
    mock_get_posix.assert_not_called()
    mock_get_win.assert_called_once()


@patch("local_operator.helpers.platform.system", return_value="FreeBSD")  # Unsupported
@patch("local_operator.helpers.get_windows_registry_path")
@patch("local_operator.helpers.get_posix_shell_path")
@patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True)
def test_setup_env_unsupported_os(mock_get_posix, mock_get_win, mock_system):
    """Test setup does nothing on unsupported OS."""
    initial_path = os.environ["PATH"]
    setup_cross_platform_environment()
    assert os.environ["PATH"] == initial_path
    mock_get_win.assert_not_called()
    mock_get_posix.assert_not_called()
