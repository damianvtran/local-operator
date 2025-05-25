from typing import Tuple

from local_operator.types import ActionType, CodeExecutionResult

DEFAULT_LOOKAHEAD_LENGTH = 32


def stream_action_buffer(
    accumulated_text: str,
    lookahead_length: int = DEFAULT_LOOKAHEAD_LENGTH,
) -> Tuple[bool, CodeExecutionResult]:
    """
    Processes accumulated text and formats it into an action response based on lookahead.

    Args:
        accumulated_text (str): The complete accumulated text up to this point.
        lookahead_length (int): The number of characters to reserve as lookahead buffer.

    Returns:
        Tuple[bool, CodeExecutionResult]:
            (finished, result) where finished is True if </action_response> tag has been
            fully processed, otherwise False.

    Raises:
        ValueError: If an invalid ActionType is encountered in <action> tag.
    """
    result = CodeExecutionResult()

    # Check if we have action_response tags
    action_start_idx = accumulated_text.find("<action_response>")
    action_end_idx = accumulated_text.find("</action_response>")

    if action_start_idx == -1:
        # No action_response found
        if len(accumulated_text) <= lookahead_length:
            # Check if the lookahead contains potential action tag beginnings
            lookahead_text = accumulated_text
            if "<" not in lookahead_text:
                # No potential tags, return all text
                result.message = accumulated_text
                return False, result
            else:
                # Potential tag beginning found, don't return anything yet
                return False, result

        # Process text minus lookahead, but check if lookahead contains potential tags
        processing_boundary = len(accumulated_text) - lookahead_length
        lookahead_text = accumulated_text[processing_boundary:]

        # If lookahead doesn't contain potential action tag beginnings, include more text
        if "<" not in lookahead_text:
            result.message = accumulated_text
        else:
            result.message = accumulated_text[:processing_boundary]
        return False, result

    if action_end_idx == -1:
        # action_response started but not finished
        # Set message to everything before action_response
        result.message = accumulated_text[:action_start_idx]

        # Parse partial action content if we have enough
        action_content = accumulated_text[action_start_idx + len("<action_response>") :]
        if action_content:
            _parse_action_content(action_content, result, partial=True)

        return False, result

    # Complete action_response found
    pre_action = accumulated_text[:action_start_idx]
    action_content = accumulated_text[action_start_idx + len("<action_response>") : action_end_idx]

    # Set the message to everything before action_response
    result.message = pre_action

    # Parse the action_response content
    _parse_action_content(action_content, result, partial=False)

    return True, result


def _parse_action_content(content: str, result: CodeExecutionResult, partial: bool = False) -> None:
    """
    Parses the content within action_response tags and populates the result object.

    Args:
        content (str): The content between <action_response> and </action_response> tags.
        result (CodeExecutionResult): The result object to populate.
        partial (bool): Whether this is a partial parse (incomplete action_response).

    Raises:
        ValueError: If an invalid ActionType is encountered.
    """
    tags = ["action", "content", "code", "replacements", "mentioned_files", "learnings"]

    for tag in tags:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"

        start_idx = 0
        while True:
            open_idx = content.find(open_tag, start_idx)
            if open_idx == -1:
                break

            close_idx = content.find(close_tag, open_idx)
            if close_idx == -1:
                # If we're doing partial parsing and there's no closing tag,
                # extract the content after the opening tag
                if partial:
                    partial_content = content[open_idx + len(open_tag) :]
                    if tag == "mentioned_files":
                        _handle_partial_mentioned_files(partial_content, result)
                    else:
                        # For other tags, assign the partial content directly
                        _assign_tag_content(tag, partial_content, result, partial=True)
                break

            tag_content = content[open_idx + len(open_tag) : close_idx]
            _assign_tag_content(tag, tag_content, result, partial=False)

            start_idx = close_idx + len(close_tag)


def _handle_partial_mentioned_files(content: str, result: CodeExecutionResult) -> None:
    """
    Handle partial mentioned_files content that may not have a closing tag yet.

    Args:
        content (str): The partial content after <mentioned_files>
        result (CodeExecutionResult): The result object to update.
    """
    # Split by newlines and process complete lines
    lines = content.split("\n")

    # Process all complete lines (all but the last one if it doesn't end with newline)
    complete_lines = lines[:-1] if not content.endswith("\n") else lines

    for line in complete_lines:
        file_candidate = line.strip()
        if file_candidate:
            if not hasattr(result, "files") or result.files is None:
                result.files = []
            result.files.append(file_candidate)


def _assign_tag_content(
    tag: str, content: str, result: CodeExecutionResult, partial: bool = False
) -> None:
    """
    Assigns the content to the appropriate field in the result object based on tag.

    Args:
        tag (str): The tag name.
        content (str): The content to assign.
        result (CodeExecutionResult): The result object to update.
        partial (bool): Whether this is partial content (for graceful error handling).

    Raises:
        ValueError: If an invalid ActionType is encountered.
    """
    try:
        if tag == "action":
            action_value = content.strip()
            try:
                result.action = ActionType(action_value)
            except ValueError:
                # For partial content, if action is invalid, don't set it
                if not partial:
                    raise
        elif tag == "content":
            result.content += content
        elif tag == "code":
            result.code += content
        elif tag == "replacements":
            result.replacements += content
        elif tag == "mentioned_files":
            # Handle mentioned_files line by line
            lines = content.strip().split("\n")
            for line in lines:
                file_candidate = line.strip()
                if file_candidate:
                    if not hasattr(result, "files") or result.files is None:
                        result.files = []
                    result.files.append(file_candidate)
        elif tag == "learnings":
            result.learnings += content
    except Exception as exc:
        raise ValueError(f"Failed to assign tag content for <{tag}>: {exc}") from exc
