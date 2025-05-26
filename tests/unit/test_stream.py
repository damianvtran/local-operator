import pytest

from local_operator.stream import stream_action_buffer
from local_operator.types import ActionType


def run_stream(input_text):
    """
    Helper to simulate streaming input_text through stream_action_buffer.
    Returns the final CodeExecutionResult and other state.
    """
    # Simulate streaming by calling stream_action_buffer with the full text
    finished, result = stream_action_buffer(input_text)
    return result, finished, False, None, []


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
            "Hello, this is a plain text message. <action_re",
            "Hello, this is ",
            None,
            "",
            "",
            "",
            [],
            False,
            False,
            id="plain_text_with_partial_action_tag",
        ),
        pytest.param(
            "Hello, this is a plain text message. <action_response>",
            "Hello, this is a plain text message. ",
            None,
            "",
            "",
            "",
            [],
            False,
            False,
            id="plain_text_with_full_action_tag",
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
            "<action_response><action>CODE</action><code>print(1)",
            "",
            ActionType.CODE,
            "",
            "print(1)",
            "",
            [],
            False,
            False,
            id="partial_code",
        ),
        pytest.param(
            "Generating code ```xml\n<action_response><action>CODE",
            "Generating code ",
            ActionType.CODE,
            "",
            "",
            "",
            [],
            False,
            True,
            id="xml_only_action_with_partial_xml_fence",
        ),
        pytest.param(
            "Generating code ```xml\n<action_response><action>CODE</action></action_response>",
            "Generating code ",
            ActionType.CODE,
            "",
            "",
            "",
            [],
            False,
            True,
            id="xml_only_action_with_xml_fence",
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
        pytest.param(
            """
To gather the latest news on Donald Trump, I will perform a web search using multiple queries to get a broad range of information. I'll start with a general search and then follow up with more specific queries if needed.

Let's begin with the initial search.

<action_response>
<action>CODE</action>

<learnings>
</learnings>

<code>
search_queries = [
    "Donald Trump latest news",
    "Donald Trump recent updates",
    "Donald Trump current events"
]

search_results = []
for query in search_queries:
    result = tools.search_web(query, max_results=10)
    search_results.append(result)

print(search_results)
</code>

<mentioned_files>
</mentioned_files>
</action_response>""",  # noqa: E501
            """
To gather the latest news on Donald Trump, I will perform a web search using multiple queries to get a broad range of information. I'll start with a general search and then follow up with more specific queries if needed.

Let's begin with the initial search.

""",  # noqa: E501
            ActionType.CODE,
            "",
            """
search_queries = [
    "Donald Trump latest news",
    "Donald Trump recent updates",
    "Donald Trump current events"
]

search_results = []
for query in search_queries:
    result = tools.search_web(query, max_results=10)
    search_results.append(result)

print(search_results)
""",
            "",
            [],
            False,
            True,
            id="code_with_learnings",
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
