import pytest

from local_operator.stream import stream_action_buffer
from local_operator.types import ActionType, CodeExecutionResult


def run_stream(input_text):
    """
    Helper to stream input_text char-by-char through stream_action_buffer.
    Returns the final CodeExecutionResult and other state.
    """
    buffer = []
    result = CodeExecutionResult()
    in_action_response = False
    current_tag = None
    tag_buffer = []
    finished = False
    finished_once = False

    for idx, token in enumerate(input_text):
        if finished_once:
            continue
        finished, result, in_action_response, current_tag, tag_buffer = stream_action_buffer(
            token, buffer, result, in_action_response, current_tag, tag_buffer
        )
        if finished:
            finished_once = True
    return result, finished_once, in_action_response, current_tag, tag_buffer


@pytest.mark.parametrize(
    "input_text,expected_message,expected_action,expected_content,expected_code,expected_replacements,expected_files,expect_error,expected_finish",  # noqa: 501
    [
        pytest.param(
            "Hello, this is a plain text message.",
            "Hello, this is a plain text message.",
            None,
            "",
            "",
            "",
            [],
            False,
            False,
            id="plain_text",
        ),
        pytest.param(
            "<action_response><action>CODE</action></action_response>",
            "",
            ActionType.CODE,
            "",
            "",
            "",
            [],
            False,
            True,
            id="xml_only_action",
        ),
        pytest.param(
            "Hello <action_response><action>WRITE</action></action_response> world",
            "Hello ",
            ActionType.WRITE,
            "",
            "",
            "",
            [],
            False,
            True,
            id="mixed_text_xml",
        ),
        pytest.param(
            "Writing hello world<action_response><action>WRITE</action><content>Writing hello world</content></action_response>",  # noqa: E501
            "Writing hello world",
            ActionType.WRITE,
            "Writing hello world",
            "",
            "",
            [],
            False,
            True,
            id="mixed_text_xml_write_duplicate",
        ),
        pytest.param(
            "Hello Hello <action_response><action>WRITE</action></action_response> world",
            "Hello Hello ",
            ActionType.WRITE,
            "",
            "",
            "",
            [],
            False,
            True,
            id="mixed_text_xml_pre_duplicate",
        ),
        pytest.param(
            "<action_response>"
            "<action>EDIT</action>"
            "<content>Some content</content>"
            "<code>print(1)</code>"
            "<replacements>foo</replacements>"
            "<mentioned_files>file1.py\nfile2.py\n</mentioned_files>"
            "</action_response>",
            "",
            ActionType.EDIT,
            "Some content",
            "print(1)",
            "foo",
            ["file1.py", "file2.py"],
            False,
            True,
            id="all_xml_fields",
        ),
        pytest.param(
            "<action_response>" "<action>EDIT</action>" "<mentioned_files>file1.py\nfil",
            "",
            ActionType.EDIT,
            "",
            "",
            "",
            ["file1.py"],
            False,
            False,
            id="partial_files",
        ),
        pytest.param(
            "<action_response>" "<action>ED",
            "",
            None,
            "",
            "",
            "",
            [],
            False,
            False,
            id="partial_action",
        ),
        pytest.param(
            "Start <action_response><action>READ</action><content>abc</content><mentioned_files>fileA.txt\n</mentioned_files></action_response> End.",  # noqa: E501
            "Start ",
            ActionType.READ,
            "abc",
            "",
            "",
            ["fileA.txt"],
            False,
            True,
            id="mixed_text_before_after_xml",
        ),
        pytest.param(
            "<action_response><mentioned_files>foo.py\nbar.py\nbaz.py\n</mentioned_files></action_response>",  # noqa: E501
            "",
            None,
            "",
            "",
            "",
            ["foo.py", "bar.py", "baz.py"],
            False,
            True,
            id="multiple_mentioned_files",
        ),
        pytest.param(
            "<action_response><content>hello</content><code>print('hi')</code><content> world</content></action_response>",  # noqa: E501
            "",
            None,
            "hello world",
            "print('hi')",
            "",
            [],
            False,
            True,
            id="interleaved_content_and_code",
        ),
        pytest.param(
            "<action_response><action>INVALID</action></action_response>",
            None,
            None,
            None,
            None,
            None,
            None,
            True,
            True,
            id="invalid_action_type",
        ),
    ],
)
def test_stream_action_buffer_cases(
    input_text,
    expected_message,
    expected_action,
    expected_content,
    expected_code,
    expected_replacements,
    expected_files,
    expect_error,
    expected_finish,
):
    if expect_error:
        with pytest.raises(ValueError):
            run_stream(input_text)
    else:
        result, finished, *_ = run_stream(input_text)
        assert finished is expected_finish
        if expected_message is not None:
            assert result.message == expected_message
        if expected_action is not None:
            assert result.action == expected_action
        else:
            assert result.action is None
        if expected_content is not None:
            assert result.content == expected_content
        if expected_code is not None:
            assert result.code == expected_code
        if expected_replacements is not None:
            assert result.replacements == expected_replacements
        if expected_files is not None:
            assert result.files == expected_files
