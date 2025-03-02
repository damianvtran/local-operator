import pytest

from local_operator.helpers import remove_think_tags


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
