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
    "input_text,expected_thinking,expected_message,expected_action,expected_content,expected_code,expected_replacements,expected_files,expect_error,expected_finish",  # noqa: E501
    [
        pytest.param(
            "Hello, this is a plain text message.",
            "",  # expected_thinking
            "Hello, this is a plain text message.",  # expected_message
            None,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            False,  # expected_finish
            id="plain_text",
        ),
        pytest.param(
            "<think>My thoughts here.</think>Just a message.",
            "My thoughts here.",  # expected_thinking
            "Just a message.",  # expected_message
            None,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            False,  # expected_finish
            id="think_tag_only_with_message",
        ),
        pytest.param(
            "<thinking>Deep thoughts.</thinking>Message <action_response><action>CODE</action><code>1+1</code></action_response>",  # noqa: E501
            "Deep thoughts.",  # expected_thinking
            "Message ",  # expected_message
            ActionType.CODE,  # expected_action
            "",  # expected_content
            "1+1",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="thinking_tag_with_message_and_action",
        ),
        pytest.param(
            "  <think> Spaced thoughts. </think>  Spaced message.  ",
            "Spaced thoughts.",  # expected_thinking
            "Spaced message.  ",
            None,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            False,  # expected_finish
            id="think_tag_with_leading_trailing_spaces",
        ),
        pytest.param(
            "<think>Thought before action.</think><action_response><action>WRITE</action><content>abc</content></action_response>",  # noqa: E501
            "Thought before action.",  # expected_thinking
            "",  # expected_message
            ActionType.WRITE,  # expected_action
            "abc",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="think_tag_before_action_no_message",
        ),
        pytest.param(
            "Writing thinking to a file <action_response><action>WRITE</action><content>Here is the file <think>This is a thought</think></content></action_response>",  # noqa: E501
            "",  # expected_thinking
            "Writing thinking to a file ",  # expected_message
            ActionType.WRITE,  # expected_action
            "Here is the file <think>This is a thought</think>",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="writing_action_writing_think_tag",
        ),
        pytest.param(
            "Writing thinking to a file <action_response><action>WRITE</action><content>Here is the file <think>This is",  # noqa: E501
            "",  # expected_thinking
            "Writing thinking to a file ",  # expected_message
            ActionType.WRITE,  # expected_action
            "Here is the file <think>This is",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            False,  # expected_finish
            id="writing_action_writing_think_tag_partial_content",
        ),
        pytest.param(
            "Hello, this is a plain text message. <action_re",
            "",  # expected_thinking
            "Hello, this is ",  # expected_message
            None,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            False,  # expected_finish
            id="plain_text_with_partial_action_tag",
        ),
        pytest.param(
            "Hello, this is a plain text message. <action_response>",
            "",  # expected_thinking
            "Hello, this is a plain text message. ",  # expected_message
            None,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            False,  # expected_finish
            id="plain_text_with_full_action_tag",
        ),
        pytest.param(
            "<action_response><action>CODE</action></action_response>",
            "",  # expected_thinking
            "",  # expected_message
            ActionType.CODE,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="xml_only_action",
        ),
        pytest.param(
            "<action_response><action>CODE</action><code>print(1)",
            "",  # expected_thinking
            "",  # expected_message
            ActionType.CODE,  # expected_action
            "",  # expected_content
            "print(1)",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            False,  # expected_finish
            id="partial_code",
        ),
        pytest.param(
            "Generating code ```xml\n<action_response><action>CODE",
            "",  # expected_thinking
            "Generating code ",  # expected_message
            ActionType.CODE,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="xml_only_action_with_partial_xml_fence",
        ),
        pytest.param(
            "Generating code ```xml\n<action_response><action>CODE</action></action_response>",
            "",  # expected_thinking
            "Generating code ",  # expected_message
            ActionType.CODE,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="xml_only_action_with_xml_fence",
        ),
        pytest.param(
            "Generating code <action_response><action>WRITE</action><content>```mermaid\ngraph TD\nA --> B\n```</content></action_response>",  # noqa: E501
            "",  # expected_thinking
            "Generating code ",  # expected_message
            ActionType.WRITE,  # expected_action
            "```mermaid\ngraph TD\nA --> B\n```",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="write_action_with_mermaid_fence",
        ),
        pytest.param(
            "Hello <action_response><action>WRITE</action></action_response> world",
            "",  # expected_thinking
            "Hello ",  # expected_message
            ActionType.WRITE,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="mixed_text_xml",
        ),
        pytest.param(
            "Writing hello world<action_response><action>WRITE</action><content>Writing hello world</content></action_response>",  # noqa: E501
            "",  # expected_thinking
            "Writing hello world",  # expected_message
            ActionType.WRITE,  # expected_action
            "Writing hello world",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="mixed_text_xml_write_duplicate",
        ),
        pytest.param(
            "Hello Hello <action_response><action>WRITE</action></action_response> world",
            "",  # expected_thinking
            "Hello Hello ",  # expected_message
            ActionType.WRITE,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
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
            "",  # expected_thinking
            "",  # expected_message
            ActionType.EDIT,  # expected_action
            "Some content",  # expected_content
            "print(1)",  # expected_code
            "foo",  # expected_replacements
            ["file1.py", "file2.py"],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="all_xml_fields",
        ),
        pytest.param(
            "<action_response>" "<action>EDIT</action>" "<mentioned_files>file1.py\nfil",
            "",  # expected_thinking
            "",  # expected_message
            ActionType.EDIT,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            ["file1.py"],  # expected_files
            False,  # expect_error
            False,  # expected_finish
            id="partial_files",
        ),
        pytest.param(
            "<action_response>" "<action>ED",
            "",  # expected_thinking
            "",  # expected_message
            None,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            False,  # expected_finish
            id="partial_action",
        ),
        pytest.param(
            "Start <action_response><action>READ</action><content>abc</content><mentioned_files>fileA.txt\n</mentioned_files></action_response> End.",  # noqa: E501
            "",  # expected_thinking
            "Start ",  # expected_message
            ActionType.READ,  # expected_action
            "abc",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            ["fileA.txt"],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="mixed_text_before_after_xml",
        ),
        pytest.param(
            "<action_response><mentioned_files>foo.py\nbar.py\nbaz.py\n</mentioned_files></action_response>",  # noqa: E501
            "",  # expected_thinking
            "",  # expected_message
            None,  # expected_action
            "",  # expected_content
            "",  # expected_code
            "",  # expected_replacements
            ["foo.py", "bar.py", "baz.py"],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="multiple_mentioned_files",
        ),
        pytest.param(
            "<action_response><content>hello</content><code>print('hi')</code><content> world</content></action_response>",  # noqa: E501
            "",  # expected_thinking
            "",  # expected_message
            None,  # expected_action
            "hello world",  # expected_content
            "print('hi')",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="interleaved_content_and_code",
        ),
        pytest.param(
            "<action_response><action>INVALID</action></action_response>",
            "",  # expected_thinking
            None,  # expected_message - will be empty string
            None,  # expected_action
            None,  # expected_content
            None,  # expected_code
            None,  # expected_replacements
            None,  # expected_files
            True,  # expect_error
            True,  # expected_finish
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
            "",  # expected_thinking
            """To gather the latest news on Donald Trump, I will perform a web search using multiple queries to get a broad range of information. I'll start with a general search and then follow up with more specific queries if needed.

Let's begin with the initial search.

""",  # noqa: E501
            ActionType.CODE,  # expected_action
            "",  # expected_content
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
""",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="code_with_learnings",
        ),
        pytest.param(
            "<think>My thinking process for the search.</think>"
            """
To gather the latest news on Donald Trump, I will perform a web search using multiple queries to get a broad range of information. I'll start with a general search and then follow up with more specific queries if needed.

Let's begin with the initial search.

<action_response>
<action>CODE</action>
<code>
print("Searching...")
</code>
</action_response>""",  # noqa: E501
            "My thinking process for the search.",  # expected_thinking
            """To gather the latest news on Donald Trump, I will perform a web search using multiple queries to get a broad range of information. I'll start with a general search and then follow up with more specific queries if needed.

Let's begin with the initial search.

""",  # noqa: E501
            ActionType.CODE,  # expected_action
            "",  # expected_content
            """
print("Searching...")
""",  # expected_code
            "",  # expected_replacements
            [],  # expected_files
            False,  # expect_error
            True,  # expected_finish
            id="think_tag_with_multiline_message_and_action",
        ),
    ],
)
def test_stream_action_buffer_cases(
    input_text,
    expected_thinking,
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
        assert result.thinking == expected_thinking
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
