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
    XML_FENCE = "```xml\n"

    # Extract thinking content first
    thinking_content, remaining_text_after_thinking = _extract_thinking_content(accumulated_text)
    if thinking_content:
        result.thinking = thinking_content

    # Use remaining_text_after_thinking for further processing
    current_text_to_process = remaining_text_after_thinking.lstrip()

    # Check if we have action_response tags in the remaining text
    action_start_idx = current_text_to_process.find("<action_response>")
    action_end_idx = current_text_to_process.find("</action_response>")
    is_fenced_action = False

    if action_start_idx != -1:
        # Check if the action_response is preceded by ```xml\n
        fence_idx = current_text_to_process.rfind(XML_FENCE, 0, action_start_idx)
        if fence_idx != -1 and fence_idx + len(XML_FENCE) == action_start_idx:
            is_fenced_action = True
            # Message is text before the fence
            result.message = current_text_to_process[:fence_idx]
        else:
            # Message is text before <action_response>
            result.message = current_text_to_process[:action_start_idx]
    else:
        # No action_response found in the remaining text
        # Check for potential action tag beginnings or ```
        potential_tag_chars = ["<", "`"]
        if len(current_text_to_process) <= lookahead_length:
            lookahead_text_segment = current_text_to_process
            if not any(char in lookahead_text_segment for char in potential_tag_chars):
                result.message = current_text_to_process
                return False, result
            else:
                # If there's only thinking content and no message/action, return
                if not result.message and not action_start_idx != -1:
                    return False, result
        else:  # len(current_text_to_process) > lookahead_length
            processing_boundary = len(current_text_to_process) - lookahead_length
            lookahead_text_segment = current_text_to_process[processing_boundary:]

            if not any(char in lookahead_text_segment for char in potential_tag_chars):
                result.message = current_text_to_process
            else:
                result.message = current_text_to_process[:processing_boundary]
        return False, result

    # Determine the content to parse for action
    if action_start_idx != -1:
        if action_end_idx != -1:
            # Complete action_response found
            action_block_content = current_text_to_process[
                action_start_idx + len("<action_response>") : action_end_idx
            ]
            _parse_action_content(action_block_content, result, partial=False)
            return True, result
        else:
            # action_response started but not finished
            action_block_content = current_text_to_process[
                action_start_idx + len("<action_response>") :
            ]
            _parse_action_content(action_block_content, result, partial=True)
            # If it's a fenced action and we have an action type, consider it finished
            if is_fenced_action and result.action is not None:
                return True, result
            return False, result

    # This part should ideally not be reached if action_start_idx was -1 due to earlier return
    return False, result


def _extract_thinking_content(text: str) -> Tuple[str, str]:
    """
    Extracts content from <think> or <thinking> tags at the beginning of the text.

    Args:
        text (str): The input text.

    Returns:
        Tuple[str, str]: (thinking_content, remaining_text)
                         thinking_content is the extracted content from the think tag.
                         remaining_text is the text after the think tag.
                         If no think tag is found at the beginning, thinking_content is ""
                         and remaining_text is the original text.
    """
    think_tags = [
        ("<think>", "</think>"),
        ("<thinking>", "</thinking>"),
    ]

    stripped_text = text.lstrip()  # Handle leading whitespace before think tag

    for open_tag, close_tag in think_tags:
        if stripped_text.startswith(open_tag):
            end_tag_idx = stripped_text.find(close_tag)
            if end_tag_idx != -1:
                # Calculate the actual start of the content inside the tag
                content_start_idx = len(open_tag)
                thinking_content = stripped_text[content_start_idx:end_tag_idx].strip()
                # Calculate the end of the closing tag in the original non-stripped text
                # This requires finding the original start of the open_tag
                original_open_tag_start_idx = text.find(open_tag)
                if original_open_tag_start_idx != -1:
                    # Find the close_tag after the original_open_tag_start_idx
                    original_end_tag_idx = text.find(
                        close_tag, original_open_tag_start_idx + len(open_tag)
                    )
                    if original_end_tag_idx != -1:
                        remaining_text_start_idx = original_end_tag_idx + len(close_tag)
                        return thinking_content, text[remaining_text_start_idx:]
    return "", text  # No think tag found or malformed


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
    tags = [
        "action",
        "content",
        "code",
        "replacements",
        "mentioned_files",
        "learnings",
        "file_path",
        "agent",
    ]

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
        elif tag == "file_path":
            result.file_path += content
        elif tag == "agent":
            result.agent += content
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
