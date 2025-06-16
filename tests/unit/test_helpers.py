import os
import platform
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from local_operator.helpers import (  # Added import
    _extract_initial_think_tags,
    clean_json_response,
    clean_plain_text_response,
    get_posix_shell_path,
    get_windows_registry_path,
    is_marker_inside_json,
    parse_agent_action_xml,
    parse_replacements,
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
        (
            "This is a test <think>with think tags</think>.",
            "This is a test .",
        ),  # Keep trailing period
        ("No think tags here.", "No think tags here."),
        ("<think>Only think tags</think>", ""),
        (
            "<think>Think tags leading content</think> content",
            "content",
        ),  # Single space after removal
        (
            "Text with <thinking>some thoughts</thinking> and more text.",
            "Text with and more text.",
        ),  # Single space
        ("<thinking>Just thoughts</thinking>", ""),
        (
            "Leading <think>thought 1</think> middle <thinking>thought 2</thinking> trailing",
            "Leading middle trailing",  # Single space between words
        ),
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
            '<thinking>Thinking about JSON...</thinking>{"action": "CODE"}',
            '{"action": "CODE"}',
            id="json_with_thinking_tags_variant",
        ),
        pytest.param(
            '  <think>Leading space then think</think> {"key": "value"}',
            '{"key": "value"}',
            id="json_with_leading_space_and_think_tag",
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
            "Plain text response",
            {
                "action": "",
                "learnings": "",
                "response": "Plain text response",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="plain_text_response",
        ),
        pytest.param(
            "Plain text response with a url and backticks: `https://www.google.com`",
            {
                "action": "",
                "learnings": "",
                "response": "Plain text response with a url and backticks: `https://www.google.com`",  # noqa: E501
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="plain_text_response_with_url_and_backticks",
        ),
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
                "thinking": "",
            },
            id="basic_code_action",
        ),
        pytest.param(
            "<think>This is a thought.</think><action>CODE</action><code>print('Hello')</code>",
            {
                "action": "CODE",
                "learnings": "",
                "response": "",
                "code": "print('Hello')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "This is a thought.",
            },
            id="code_action_with_think_tag",
        ),
        pytest.param(
            "  <thinking>Another thought here.</thinking>  <action>WRITE</action><content>File content</content>",  # noqa E501
            {
                "action": "WRITE",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "File content",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "Another thought here.",
            },
            id="write_action_with_thinking_tag_and_spaces",
        ),
        pytest.param(
            "I am writing a think tag example <action>WRITE</action><content>This is the file content <think>This is a thought</think></content>",  # noqa E501
            {
                "action": "WRITE",
                "learnings": "",
                "response": "I am writing a think tag example",
                "code": "",
                "content": "This is the file content <think>This is a thought</think>",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="writing_action_writing_think_tag",
        ),
        pytest.param(
            "<think>I am thinking about writing thinking</think> I am writing a think tag example <action>WRITE</action><content>This is the file content <think>This is a thought</think></content>",  # noqa E501
            {
                "action": "WRITE",
                "learnings": "",
                "response": "I am writing a think tag example",
                "code": "",
                "content": "This is the file content <think>This is a thought</think>",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "I am thinking about writing thinking",
            },
            id="writing_action_writing_think_tag_with_thinking",
        ),
        pytest.param(
            "<think>Thought 1</think>Message part 1 <action>CODE</action> Message part 2",
            {
                "action": "CODE",
                "learnings": "",
                "response": "Message part 1 Message part 2",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "Thought 1",
            },
            id="think_tag_with_message_around_action",
        ),
        pytest.param(
            "This is a message before ```xml\n<action>CODE</action><learnings>Learned something</learnings><code>print('Hello')</code>\n```",  # noqa: E501
            {
                "action": "CODE",
                "learnings": "Learned something",
                "response": "This is a message before",
                "code": "print('Hello')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="basic_code_action_with_code_block_marker",
        ),
        pytest.param(
            "<think>A plan.</think>This is a message before ```xml\n<action>CODE</action><code>print('H')</code>\n```",  # noqa: E501
            {
                "action": "CODE",
                "learnings": "",
                "response": "This is a message before",
                "code": "print('H')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "A plan.",
            },
            id="think_tag_before_fenced_action",
        ),
        pytest.param(
            "This is a message before <action_response><action>WRITE</action><file_path>test.txt</file_path><content>```mermaid\ngraph TD\nA --> B\n```</content></action_response>",  # noqa: E501
            {
                "action": "WRITE",
                "learnings": "",
                "response": "This is a message before",
                "code": "",
                "content": "```mermaid\ngraph TD\nA --> B\n```",
                "file_path": "test.txt",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="basic_code_action_with_nested_code_block_marker",
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
                "response": "Some text before and text after.",
                "code": "",
                "content": "File content",
                "file_path": "test.txt",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="write_action_with_surrounding_text_and_action_response_tag",
        ),
        pytest.param(
            "```xml\n<action>EDIT</action><file_path>doc.md</file_path><replacements>\n<<<<<<< SEARCH\nold line\nanother old line\n=======\nnew line 1\nnew line 2\n>>>>>>> REPLACE\n</replacements>\n```",  # noqa: E501
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
                "thinking": "",
            },
            id="edit_action_with_markdown_wrapper_and_replacements",
        ),
        pytest.param(
            "```xml\n<action>EDIT</action><file_path>doc.md</file_path><replacements>\n<<<<<<< SEARCH\nold line\n\nanother old line\n=======\nnew line 1\n\n\nnew line 2\n>>>>>>> REPLACE\n</replacements>\n```",  # noqa: E501
            {
                "action": "EDIT",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "doc.md",
                "replacements": [
                    {
                        "find": "old line\n\nanother old line",
                        "replace": "new line 1\n\n\nnew line 2",
                    }
                ],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="edit_action_with_newlines_in_replacements",
        ),
        pytest.param(
            "```xml\n<action>EDIT</action><file_path>doc.md</file_path><replacements>\n<<<<<<< SEARCH\nold line\n\n<<<<<<< SEARCH\nrepl======\n>>>>>> REPLACE\nanother old line\n=======\nnew line 1\n\n\nnew line 2\n>>>>>>> REPLACE\n</replacements>\n```",  # noqa: E501
            {
                "action": "EDIT",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "doc.md",
                "replacements": [
                    {
                        "find": "old line\n\n<<<<<<< SEARCH\nrepl======\n>>>>>> REPLACE\nanother old line",  # noqa: E501
                        "replace": "new line 1\n\n\nnew line 2",
                    }
                ],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="edit_action_with_recursive_replacements",
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
                "thinking": "",
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
                "thinking": "",
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
                "thinking": "",
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
                "thinking": "",
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
                "thinking": "",
            },
            id="empty_tags_for_code",
        ),
        pytest.param(
            (
                "<action>EDIT</action><file_path>file.txt</file_path><replacements>\n"
                "<<<<<<< SEARCH\nfind1\n=======\nreplace1\n>>>>>>> REPLACE\n<<<<<<< SEARCH\nfind2\n=======\nreplace2\n>>>>>>> REPLACE\n</replacements>"  # noqa: E501
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
                "thinking": "",
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
                "thinking": "",
            },
            id="read_action",
        ),
        pytest.param(
            "<outer><action>CODE</action><code>print(1)</code></outer>",  # Simple nesting
            {
                "action": "CODE",
                "learnings": "",
                "response": "",
                "code": "print(1)",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
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
                "thinking": "",
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
                "thinking": "",
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
                "thinking": "",
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
                "thinking": "",
            },
            id="write_xml_content_with_cdata",
        ),
        pytest.param(
            "<action>EDIT</action><file_path>config.xml</file_path><replacements>\n<<<<<<< SEARCH\n<old_setting>true</old_setting>\n=======\n<new_setting>false</new_setting>\n>>>>>>> REPLACE\n</replacements>",  # noqa: E501
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
                "thinking": "",
            },
            id="edit_replacing_xml_entities",
        ),
        pytest.param(
            "Just some text",
            {
                "action": "",
                "learnings": "",
                "response": "Just some text",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="only_text_no_xml",
        ),
        pytest.param(
            "Prefix <action>tool</action>",
            {
                "action": "tool",
                "learnings": "",
                "response": "Prefix",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="text_before_xml",
        ),
        pytest.param(
            "<action>tool</action> Suffix",
            {
                "action": "tool",
                "learnings": "",
                "response": "Suffix",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="text_after_xml",
        ),
        pytest.param(
            "Prefix <action>tool</action> Suffix",
            {
                "action": "tool",
                "learnings": "",
                "response": "Prefix Suffix",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="text_before_and_after_xml",
        ),
        pytest.param(
            "Outer <response>Inner</response>",
            {
                "action": "",
                "learnings": "",
                "response": "Inner",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="outer_text_with_explicit_response_tag",
        ),
        pytest.param(
            "Outer <message>InnerMsg</message>",
            {
                "action": "",
                "learnings": "",
                "response": "Outer",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "InnerMsg",
                "mentioned_files": [],
                "thinking": "",
            },
            id="outer_text_with_explicit_message_tag",
        ),
        pytest.param(
            "Outer <response>InnerResp</response> <message>InnerMsg</message>",  # noqa: E501
            {
                "action": "",
                "learnings": "",
                "response": "InnerResp",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "InnerMsg",
                "mentioned_files": [],
                "thinking": "",
            },
            id="outer_text_with_explicit_response_and_message_tags",
        ),
        pytest.param(
            "<message>Msg only</message>",
            {
                "action": "",
                "learnings": "",
                "response": "",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "Msg only",
                "mentioned_files": [],
                "thinking": "",
            },
            id="only_message_tag_no_outer_text",
        ),
        pytest.param(
            "<response>Resp only</response>",
            {
                "action": "",
                "learnings": "",
                "response": "Resp only",
                "code": "",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="only_response_tag_no_outer_text",
        ),
        pytest.param(
            "<action>CODE</action><code>print('hi')</code><mentioned_files></mentioned_files>",  # noqa: E501
            {
                "action": "CODE",
                "learnings": "",
                "response": "",
                "code": "print('hi')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="mentioned_files_empty_tag",
        ),
        pytest.param(
            "<action>CODE</action><code>print('hi')</code><mentioned_files>file1.txt</mentioned_files>",  # noqa: E501
            {
                "action": "CODE",
                "learnings": "",
                "response": "",
                "code": "print('hi')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": ["file1.txt"],
                "thinking": "",
            },
            id="mentioned_files_single_file",
        ),
        pytest.param(
            "<action>CODE</action><code>print('hi')</code><mentioned_files>\nfile1.txt\nfile2.py\n</mentioned_files>",  # noqa: E501
            {
                "action": "CODE",
                "learnings": "",
                "response": "",
                "code": "print('hi')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": ["file1.txt", "file2.py"],
                "thinking": "",
            },
            id="mentioned_files_multiple_files_newline_separated",
        ),
        pytest.param(
            "<action>CODE</action><code>print('hi')</code><mentioned_files>\n  file1.txt  \n\n file2.py \n</mentioned_files>",  # noqa: E501
            {
                "action": "CODE",
                "learnings": "",
                "response": "",
                "code": "print('hi')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": ["file1.txt", "file2.py"],
                "thinking": "",
            },
            id="mentioned_files_with_whitespace_and_empty_lines",
        ),
        pytest.param(
            "<action>CODE</action><code>print('hi')</code>",
            {
                "action": "CODE",
                "learnings": "",
                "response": "",
                "code": "print('hi')",
                "content": "",
                "file_path": "",
                "replacements": [],
                "agent": "",
                "message": "",
                "mentioned_files": [],
                "thinking": "",
            },
            id="no_mentioned_files_tag",
        ),
    ],
)
def test_parse_agent_action_xml(xml_string, expected_dict):
    """Test the parse_agent_action_xml function."""
    # Ensure the 'thinking' key exists in expected_dict for all tests after this change
    if "thinking" not in expected_dict:
        expected_dict["thinking"] = ""
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


@patch("local_operator.helpers.platform.system", return_value="Windows")
@patch("local_operator.helpers.get_windows_registry_path")
@patch("local_operator.helpers.get_posix_shell_path")
@patch.dict(os.environ, {"PATH": "C:\\Initial"}, clear=True)
def test_setup_env_windows_updates(mock_get_posix, mock_get_win, mock_system):
    """Test setup updates PATH on Windows when different."""
    new_path = "C:\\User;C:\\System"
    mock_get_win.return_value = new_path

    with patch("os.path.isdir", return_value=True):
        setup_cross_platform_environment()
        assert new_path in os.environ["PATH"]
        assert "Local Operator" in os.environ["PATH"] and "bin" in os.environ["PATH"]
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

    with patch("os.path.isdir", return_value=True):
        setup_cross_platform_environment()
        assert new_path in os.environ["PATH"]
        assert "Local Operator" in os.environ["PATH"] and "bin" in os.environ["PATH"]
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

    with patch("os.path.isdir", return_value=True):
        setup_cross_platform_environment()
        assert initial_path in os.environ["PATH"]
        assert "Local Operator" in os.environ["PATH"] and "bin" in os.environ["PATH"]
        mock_get_win.assert_not_called()
        mock_get_posix.assert_called_once()


@patch("local_operator.helpers.platform.system", return_value="Windows")
@patch("local_operator.helpers.get_windows_registry_path", return_value=None)
@patch("local_operator.helpers.get_posix_shell_path")
@patch.dict(os.environ, {"PATH": "C:\\Initial"}, clear=True)
def test_setup_env_windows_retrieval_fails(mock_get_posix, mock_get_win, mock_system):
    """Test setup does not update PATH on Windows if retrieval fails."""
    initial_path = os.environ["PATH"]

    with patch("os.path.isdir", return_value=True):
        setup_cross_platform_environment()
        assert initial_path in os.environ["PATH"]
        assert "Local Operator" in os.environ["PATH"] and "bin" in os.environ["PATH"]
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


# --- Tests for _extract_initial_think_tags ---
@pytest.mark.parametrize(
    "text, expected_thinking, expected_remaining",
    [
        ("<think>My thoughts</think>Rest of message", "My thoughts", "Rest of message"),
        (
            "  <think>  Thoughts with spaces  </think>  Message  ",
            "Thoughts with spaces",
            "  Message  ",
        ),
        (
            "<thinking>My thinking process</thinking>Action here",
            "My thinking process",
            "Action here",
        ),
        (
            "  <thinking>  Thinking with spaces  </thinking>  Action  ",
            "Thinking with spaces",
            "  Action  ",
        ),
        ("No think tag here", "", "No think tag here"),
        ("<think>Unclosed thought", "", "<think>Unclosed thought"),  # Malformed
        (
            "Text before <think>thought</think>",
            "",
            "Text before <think>thought</think>",
        ),  # Not initial
        (
            "<think>Thought 1</think><think>Thought 2</think>",
            "Thought 1",
            "<think>Thought 2</think>",
        ),  # Only first
        ("<think></think>Empty", "", "Empty"),
        ("<thinking></thinking>Empty", "", "Empty"),
        ("<think>  </think>Spaces inside", "", "Spaces inside"),  # Content is just spaces
        ("  <think>Content</think>", "Content", ""),  # Think tag at end after spaces
        ("<think>Content</think>", "Content", ""),  # Think tag at end
    ],
)
def test_extract_initial_think_tags(text, expected_thinking, expected_remaining):
    thinking, remaining = _extract_initial_think_tags(text)
    assert thinking == expected_thinking
    assert remaining == expected_remaining


# --- Tests for parse_replacements ---
@pytest.mark.parametrize(
    "replacements_str, expected_output",
    [
        pytest.param(
            "",
            [],
            id="empty_string",
        ),
        pytest.param(
            "   \n  \n   ",
            [],
            id="whitespace_only",
        ),
        pytest.param(
            "<<<<<<< SEARCH\nold_content\n=======\nnew_content\n>>>>>>> REPLACE",
            [{"find": "old_content", "replace": "new_content"}],
            id="basic_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\nold_content\nwith multiple lines\n=======\nnew_content\nwith multiple lines\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "old_content\nwith multiple lines",
                    "replace": "new_content\nwith multiple lines",
                }
            ],
            id="multiline_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\nfirst_old\n=======\nfirst_new\n>>>>>>> REPLACE\n\n<<<<<<< SEARCH\nsecond_old\n=======\nsecond_new\n>>>>>>> REPLACE",  # noqa: E501
            [
                {"find": "first_old", "replace": "first_new"},
                {"find": "second_old", "replace": "second_new"},
            ],
            id="multiple_replacements",
        ),
        pytest.param(
            "<<<<<<< SEARCH\nouter_old\n<<<<<<< SEARCH\ninner_old\n=======\ninner_new\n>>>>>>> REPLACE\n=======\nouter_new\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "outer_old\n<<<<<<< SEARCH\ninner_old\n=======\ninner_new\n>>>>>>> REPLACE",  # noqa: E501
                    "replace": "outer_new",
                }
            ],
            id="nested_search_blocks",
        ),
        pytest.param(
            "<<<<<<< SEARCH\ncontent\n=======\n\n>>>>>>> REPLACE",
            [{"find": "content", "replace": ""}],
            id="empty_replace_content",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n\n=======\nnew_content\n>>>>>>> REPLACE",
            [{"find": "", "replace": "new_content"}],
            id="empty_find_content",
        ),
        pytest.param(
            "<<<<<<< SEARCH\nold_content\n>>>>>>> REPLACE",
            [],
            id="missing_separator",
        ),
        pytest.param(
            "<<<<<<< SEARCH\nold_content\n=======\nnew_content",
            [],
            id="missing_end_marker",
        ),
        pytest.param(
            "old_content\n=======\nnew_content\n>>>>>>> REPLACE",
            [],
            id="missing_start_marker",
        ),
        pytest.param(
            "Some random text\n<<<<<<< SEARCH\nold_content\n=======\nnew_content\n>>>>>>> REPLACE\nMore text",  # noqa: E501
            [{"find": "old_content", "replace": "new_content"}],
            id="replacement_with_surrounding_text",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n  old_content  \n=======\n  new_content  \n>>>>>>> REPLACE",
            [{"find": "old_content", "replace": "new_content"}],
            id="content_with_whitespace_stripped",
        ),
        pytest.param(
            "<<<<<<< SEARCH\ncode = 'old'\nprint(code)\n=======\ncode = 'new'\nprint(code)\n>>>>>>> REPLACE",  # noqa: E501
            [{"find": "code = 'old'\nprint(code)", "replace": "code = 'new'\nprint(code)"}],
            id="code_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n# Old comment\ndef old_function():\n    pass\n=======\n# New comment\ndef new_function():\n    return True\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "# Old comment\ndef old_function():\n    pass",
                    "replace": "# New comment\ndef new_function():\n    return True",
                }
            ],
            id="function_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n- Old bullet point\n- Another old point\n- Third old point\n=======\n- New bullet point\n- Another new point\n- Third new point\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "- Old bullet point\n- Another old point\n- Third old point",  # noqa: E501
                    "replace": "- New bullet point\n- Another new point\n- Third new point",  # noqa: E501
                }
            ],
            id="markdown_bullet_list_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n* Item 1\n  * Nested item 1.1\n  * Nested item 1.2\n* Item 2\n=======\n* Updated Item 1\n  * Updated nested item 1.1\n  * Updated nested item 1.2\n* Updated Item 2\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "* Item 1\n  * Nested item 1.1\n  * Nested item 1.2\n* Item 2",  # noqa: E501
                    "replace": "* Updated Item 1\n  * Updated nested item 1.1\n  * Updated nested item 1.2\n* Updated Item 2",  # noqa: E501
                }
            ],
            id="markdown_nested_bullet_list_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n| Column 1 | Column 2 | Column 3 |\n|----------|----------|----------|\n| Old Val1 | Old Val2 | Old Val3 |\n| Old Val4 | Old Val5 | Old Val6 |\n=======\n| Column 1 | Column 2 | Column 3 |\n|----------|----------|----------|\n| New Val1 | New Val2 | New Val3 |\n| New Val4 | New Val5 | New Val6 |\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "| Column 1 | Column 2 | Column 3 |\n|----------|----------|----------|\n| Old Val1 | Old Val2 | Old Val3 |\n| Old Val4 | Old Val5 | Old Val6 |",  # noqa: E501
                    "replace": "| Column 1 | Column 2 | Column 3 |\n|----------|----------|----------|\n| New Val1 | New Val2 | New Val3 |\n| New Val4 | New Val5 | New Val6 |",  # noqa: E501
                }
            ],
            id="markdown_table_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n1. First numbered item\n2. Second numbered item\n   - Sub bullet under second\n   - Another sub bullet\n3. Third numbered item\n=======\n1. Updated first numbered item\n2. Updated second numbered item\n   - Updated sub bullet under second\n   - Updated another sub bullet\n3. Updated third numbered item\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "1. First numbered item\n2. Second numbered item\n   - Sub bullet under second\n   - Another sub bullet\n3. Third numbered item",  # noqa: E501
                    "replace": "1. Updated first numbered item\n2. Updated second numbered item\n   - Updated sub bullet under second\n   - Updated another sub bullet\n3. Updated third numbered item",  # noqa: E501
                }
            ],
            id="markdown_numbered_list_with_sub_bullets_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n> This is a blockquote\n> with multiple lines\n> and some **bold** text\n>\n> And a new paragraph in the quote\n=======\n> This is an updated blockquote\n> with multiple lines\n> and some **updated bold** text\n>\n> And an updated new paragraph in the quote\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "> This is a blockquote\n> with multiple lines\n> and some **bold** text\n>\n> And a new paragraph in the quote",  # noqa: E501
                    "replace": "> This is an updated blockquote\n> with multiple lines\n> and some **updated bold** text\n>\n> And an updated new paragraph in the quote",  # noqa: E501
                }
            ],
            id="markdown_multiline_blockquote_replacement",
        ),
        pytest.param(
            '<<<<<<< SEARCH\n```python\ndef old_function(x, y):\n    """Old docstring"""\n    result = x + y\n    return result\n\nprint(old_function(1, 2))\n```\n=======\n```python\ndef new_function(x, y, z=0):\n    """New docstring with additional parameter"""\n    result = x + y + z\n    return result\n\nprint(new_function(1, 2, 3))\n```\n>>>>>>> REPLACE',  # noqa: E501
            [
                {
                    "find": '```python\ndef old_function(x, y):\n    """Old docstring"""\n    result = x + y\n    return result\n\nprint(old_function(1, 2))\n```',  # noqa: E501
                    "replace": '```python\ndef new_function(x, y, z=0):\n    """New docstring with additional parameter"""\n    result = x + y + z\n    return result\n\nprint(new_function(1, 2, 3))\n```',  # noqa: E501
                }
            ],
            id="markdown_code_block_with_multiline_function_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n## Old Section Header\n\nSome content under the old header.\n\n### Old Subsection\n\n- Old bullet\n- Another old bullet\n\n=======\n## New Section Header\n\nSome updated content under the new header.\n\n### New Subsection\n\n- New bullet\n- Another new bullet\n\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "## Old Section Header\n\nSome content under the old header.\n\n### Old Subsection\n\n- Old bullet\n- Another old bullet",  # noqa: E501
                    "replace": "## New Section Header\n\nSome updated content under the new header.\n\n### New Subsection\n\n- New bullet\n- Another new bullet",  # noqa: E501
                }
            ],
            id="markdown_section_with_headers_and_content_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n- [ ] Unchecked task item\n- [x] Checked task item\n- [ ] Another unchecked item\n  - [ ] Nested unchecked subtask\n  - [x] Nested checked subtask\n=======\n- [x] Updated checked task item\n- [x] Updated checked task item\n- [ ] Updated unchecked item\n  - [x] Updated nested checked subtask\n  - [x] Updated nested checked subtask\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "- [ ] Unchecked task item\n- [x] Checked task item\n- [ ] Another unchecked item\n  - [ ] Nested unchecked subtask\n  - [x] Nested checked subtask",  # noqa: E501
                    "replace": "- [x] Updated checked task item\n- [x] Updated checked task item\n- [ ] Updated unchecked item\n  - [x] Updated nested checked subtask\n  - [x] Updated nested checked subtask",  # noqa: E501
                }
            ],
            id="markdown_checkbox_list_with_nested_items_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n    - [ ] Deeply indented unchecked item\n    - [x] Deeply indented checked item\n        - [ ] Even deeper unchecked\n        - [x] Even deeper checked\n            - [ ] Maximum depth unchecked\n=======\n    - [x] Updated deeply indented checked item\n    - [x] Updated deeply indented checked item\n        - [x] Updated even deeper checked\n        - [x] Updated even deeper checked\n            - [x] Updated maximum depth checked\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "    - [ ] Deeply indented unchecked item\n    - [x] Deeply indented checked item\n        - [ ] Even deeper unchecked\n        - [x] Even deeper checked\n            - [ ] Maximum depth unchecked",  # noqa: E501
                    "replace": "    - [x] Updated deeply indented checked item\n    - [x] Updated deeply indented checked item\n        - [x] Updated even deeper checked\n        - [x] Updated even deeper checked\n            - [x] Updated maximum depth checked",  # noqa: E501
                }
            ],
            id="markdown_deeply_indented_checkbox_list_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n1. [ ] Numbered checkbox item\n2. [x] Numbered checked item\n   - [ ] Sub checkbox under numbered\n   - [x] Sub checked under numbered\n3. [ ] Third numbered checkbox\n=======\n1. [x] Updated numbered checked item\n2. [x] Updated numbered checked item\n   - [x] Updated sub checked under numbered\n   - [x] Updated sub checked under numbered\n3. [x] Updated third numbered checked\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "1. [ ] Numbered checkbox item\n2. [x] Numbered checked item\n   - [ ] Sub checkbox under numbered\n   - [x] Sub checked under numbered\n3. [ ] Third numbered checkbox",  # noqa: E501
                    "replace": "1. [x] Updated numbered checked item\n2. [x] Updated numbered checked item\n   - [x] Updated sub checked under numbered\n   - [x] Updated sub checked under numbered\n3. [x] Updated third numbered checked",  # noqa: E501
                }
            ],
            id="markdown_numbered_checkbox_list_with_sub_items_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n        * [ ] Tab-indented checkbox\n        * [x] Tab-indented checked\n            * [ ] Double tab checkbox\n                * [x] Triple tab checked\n=======\n        * [x] Updated tab-indented checked\n        * [x] Updated tab-indented checked\n            * [x] Updated double tab checked\n                * [x] Updated triple tab checked\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "        * [ ] Tab-indented checkbox\n        * [x] Tab-indented checked\n            * [ ] Double tab checkbox\n                * [x] Triple tab checked",  # noqa: E501
                    "replace": "        * [x] Updated tab-indented checked\n        * [x] Updated tab-indented checked\n            * [x] Updated double tab checked\n                * [x] Updated triple tab checked",  # noqa: E501
                }
            ],
            id="markdown_tab_indented_checkbox_list_replacement",
        ),
        pytest.param(
            "<<<<<<< SEARCH\n  - [ ] Mixed indentation checkbox\n    - [x] Different indent level\n      - [ ] Yet another indent\n        - [x] Final indent level\n=======\n  - [x] Updated mixed indentation checked\n    - [x] Updated different indent level\n      - [x] Updated yet another indent\n        - [x] Updated final indent level\n>>>>>>> REPLACE",  # noqa: E501
            [
                {
                    "find": "  - [ ] Mixed indentation checkbox\n    - [x] Different indent level\n      - [ ] Yet another indent\n        - [x] Final indent level",  # noqa: E501
                    "replace": "  - [x] Updated mixed indentation checked\n    - [x] Updated different indent level\n      - [x] Updated yet another indent\n        - [x] Updated final indent level",  # noqa: E501
                }
            ],
            id="markdown_mixed_indentation_checkbox_list_replacement",
        ),
    ],
)
def test_parse_replacements(replacements_str, expected_output):
    """Test the parse_replacements function."""
    assert parse_replacements(replacements_str) == expected_output
