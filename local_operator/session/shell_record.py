"""Stable identities for an explicitly user-run shell command's receipt.

The terminal executes; only the owner journals. Retrying an acknowledged receipt
must not create a second synthetic exchange, including after an owner restart.
The tool call ID is minted once at submission and is the idempotency identity.
"""

from local_operator.harness.types import Message, ToolCall, ToolResult


def shell_record_messages(command: str, result: ToolResult) -> list[Message]:
    user = Message.user(f"! {command}")
    assistant = Message.assistant("")
    assistant.tool_calls = [
        ToolCall(id=result.tool_call_id, name="bash", arguments={"command": command})
    ]
    tool = Message.tool_result(result)
    for message, suffix in zip((user, assistant, tool), ("user", "call", "result")):
        message.id = f"shell:{result.tool_call_id}:{suffix}"
    return [user, assistant, tool]
