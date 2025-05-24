from typing import List, Optional, Tuple

from local_operator.types import ActionType, CodeExecutionResult


def stream_action_buffer(
    token: str,
    buffer: List[str],
    result: CodeExecutionResult,
    in_action_response: bool,
    current_tag: Optional[str],
    tag_buffer: List[str],
) -> Tuple[bool, CodeExecutionResult, bool, Optional[str], List[str]]:
    """
    Streams tokens into a buffer and updates the CodeExecutionResult object in-place
    according to the XML tags encountered in the stream.

    Args:
        token (str): The new token to process.
        buffer (List[str]): The rolling buffer of tokens (will be updated in-place).
        result (CodeExecutionResult): The result object to update in-place.
        in_action_response (bool): Whether currently inside <action_response>.
        current_tag (Optional[str]): The current tag being processed.
        tag_buffer (List[str]): Buffer for accumulating tag content.

    Returns:
        Tuple[bool, CodeExecutionResult, bool, Optional[str], List[str]]:
            (finished, result, in_action_response, current_tag, tag_buffer)
            where finished is True if the </action_response> tag has been fully processed,
            otherwise False.

    Raises:
        ValueError: If an invalid ActionType is encountered in <action> tag.
    """
    # Update buffer
    buffer.append(token)
    if len(buffer) > 4096:
        buffer.pop(0)

    finished = False

    def buffer_str():
        return "".join(buffer)

    buf = buffer_str()

    if not in_action_response:
        while True:
            idx = buf.find("<action_response>")
            if idx == -1:
                # No <action_response> yet, append token to message (plain text mode)
                partial_tag = "<action_response>"
                should_append = True
                for i in range(1, len(partial_tag)):
                    if buf.endswith(partial_tag[:i]):
                        should_append = False
                        break
                if "<action_response>" not in buf and should_append:
                    result.message += token
                return False, result, in_action_response, current_tag, tag_buffer
            else:
                # Found start of <action_response>
                in_action_response = True
                # Only flush everything up to <action_response> to message (not the tag itself)
                pre = buf[:idx]
                if pre and pre not in result.message:
                    result.message += pre
                # Remove everything up to and including <action_response> from buffer
                del buffer[: idx + len("<action_response>")]
                # After removing the tag, do not append the current token to the message
                buf = buffer_str()
                # If another <action_response> is at the start, keep removing
                if buf.startswith("<action_response>"):
                    continue
                break
        current_tag = None
        tag_buffer = []
        buf = buffer_str()

    # If in action_response, look for tags
    if in_action_response:
        # Check for end of action_response
        while "</action_response>" in buf:
            idx = buf.find("</action_response>")
            before = buf[:idx]
            if current_tag:
                tag_buffer.append(before)
                tag_content = "".join(tag_buffer)
                _assign_tag_content(current_tag, tag_content, result)
                current_tag = None
                tag_buffer = []
            else:
                result.message += before
            finished = True
            del buffer[: idx + len("</action_response>")]
            # Reset state for post-XML streaming
            in_action_response = False
            current_tag = None
            tag_buffer = []
            buffer.clear()
            # Do not process any more tokens after finishing
            buf = buffer_str()
            return True, result, in_action_response, current_tag, tag_buffer

        tags = ["action", "content", "code", "replacements", "mentioned_files"]
        while True:
            if not current_tag:
                found_tag = False
                for tag in tags:
                    open_tag = f"<{tag}>"
                    if open_tag in buf:
                        idx = buf.find(open_tag)
                        if idx > 0:
                            result.message += buf[:idx]
                        # Remove up to and including <tag>
                        del buffer[: idx + len(open_tag)]
                        current_tag = tag
                        tag_buffer = []
                        buf = buffer_str()
                        found_tag = True
                        break
                if not found_tag:
                    break  # No more tags found, exit loop

            if current_tag:
                tag = current_tag
                close_tag = f"</{tag}>"
                if close_tag in buf:
                    idx = buf.find(close_tag)
                    tag_content = buf[:idx]
                    _assign_tag_content(tag, tag_content, result)
                    # Remove up to and including </tag>
                    del buffer[: idx + len(close_tag)]
                    current_tag = None
                    tag_buffer = []
                    buf = buffer_str()
                    continue  # There may be more tags in the buffer, so loop again
                else:
                    # For mentioned_files, handle line-by-line
                    if tag == "mentioned_files":
                        if "\n" in buf:
                            idx = buf.find("\n")
                            file_candidate = buf[:idx].strip()
                            if file_candidate:
                                if not hasattr(result, "files") or result.files is None:
                                    result.files = []
                                result.files.append(file_candidate)
                            del buffer[: idx + 1]
                            buf = buffer_str()
                            tag_buffer = []
                            continue
                        else:
                            tag_buffer.append(token)
                    else:
                        tag_buffer.append(token)
                    break  # Wait for more tokens
            else:
                break  # No current tag, exit loop
    return finished, result, in_action_response, current_tag, tag_buffer


def _assign_tag_content(tag: str, content: str, result: CodeExecutionResult):
    """
    Assigns the content to the appropriate field in the result object based on tag.

    Args:
        tag (str): The tag name.
        content (str): The content to assign.
        result (CodeExecutionResult): The result object to update.

    Raises:
        ValueError: If an invalid ActionType is encountered.
    """
    try:
        if tag == "action":
            action_value = content.strip()
            result.action = ActionType(action_value)
        elif tag == "content":
            result.content += content
        elif tag == "code":
            result.code += content
        elif tag == "replacements":
            result.replacements += content
        elif tag == "mentioned_files":
            file_candidate = content.strip()
            if file_candidate:
                if not hasattr(result, "files") or result.files is None:
                    result.files = []
                result.files.append(file_candidate)
    except Exception as exc:
        raise ValueError(f"Failed to assign tag content for <{tag}>: {exc}") from exc
