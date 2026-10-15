"""lop harness events → ACP ``session/update`` payloads, as pure functions.

The table these implement is in ``docs/design/acp-agent.md`` ("Event
mapping"). Everything here is synchronous and side-effect free apart from the
small per-prompt bookkeeping :class:`EventMapper` keeps (which messages have
streamed, which tool calls are still open), so a unit test can drive the whole
table with real harness event objects and no connection.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from acp import helpers
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ContentToolCallContent,
    StopReason,
    ToolCallProgress,
    ToolCallStart,
    ToolKind,
)

from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    Message,
    MessageEndEvent,
    MessageUpdateEvent,
    ReasoningDeltaEvent,
    SubagentEndEvent,
    SubagentStartEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)

#: Upper bound on the tool-result text sent back in a ``tool_call_update``. The
#: model receives the full result regardless; the client only renders it, and
#: a multi-megabyte ``cat`` echoed into a voice client's stream helps nobody.
MAX_RESULT_CHARS = 8_000

SessionUpdate = AgentMessageChunk | AgentThoughtChunk | ToolCallStart | ToolCallProgress

_TOOL_KINDS: dict[str, ToolKind] = {
    "read": "read",
    "grep": "search",
    "glob": "search",
    "edit": "edit",
    "write": "edit",
    "bash": "execute",
    "eval": "execute",
    "web_fetch": "fetch",
}


class TurnError(RuntimeError):
    """The turn ended with an error; the prompt answers with JSON-RPC -32603."""


def tool_kind(tool_name: str) -> ToolKind:
    """The ACP ``ToolKind`` a lop tool renders as; ``other`` when unknown."""
    return _TOOL_KINDS.get(tool_name, "other")


def _text_content(text: str) -> list[ContentToolCallContent]:
    return [helpers.tool_content(helpers.text_block(text))]


def _limited(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return f"{text[:MAX_RESULT_CHARS]}\n… [{len(text) - MAX_RESULT_CHARS} more characters]"


def _blocks_text(content: Sequence[Any]) -> str:
    return "".join(block.text for block in content if isinstance(block, TextContent))


def stop_reason_for(event: AgentEndEvent) -> StopReason:
    """D9: the ``stopReason`` a settled turn reports, or :class:`TurnError`.

    ``refusal`` is read from the loop's own mark on the final assistant
    message (``stop_reason == "refusal"``), which the loop sets together with
    the error text — so a refusal is a stop reason, not an error.
    """
    if event.aborted:
        return "cancelled"
    if event.error:
        last = event.messages[-1] if event.messages else None
        if getattr(last, "stop_reason", None) == "refusal":
            return "refusal"
        raise TurnError(event.error)
    return "end_turn"


class EventMapper:
    """Per-session mapping state: streamed message ids and open tool calls."""

    def __init__(self) -> None:
        self._streamed: set[str] = set()
        #: Started-but-unfinished calls, oldest first: (tool_call_id, tool_name).
        self._open_calls: list[tuple[str, str]] = []
        #: Open calls a permission request has already been matched to.
        self._claimed: set[str] = set()

    def updates_for(self, event: AgentEvent) -> list[SessionUpdate]:
        """The ACP updates one lop event produces (often none)."""
        if isinstance(event, MessageUpdateEvent):
            if not event.delta:
                return []
            self._streamed.add(event.message.id)
            return [helpers.update_agent_message_text(event.delta)]
        if isinstance(event, MessageEndEvent):
            message = event.message
            if not isinstance(message, Message) or message.role != "assistant":
                return []
            if message.id in self._streamed:
                return []
            text = message.text
            return [helpers.update_agent_message_text(text)] if text else []
        if isinstance(event, ReasoningDeltaEvent):
            return [helpers.update_agent_thought_text(event.delta)] if event.delta else []
        if isinstance(event, ToolExecutionStartEvent):
            self._open_calls.append((event.tool_call_id, event.tool_name))
            return [
                helpers.start_tool_call(
                    event.tool_call_id,
                    event.intent or event.tool_name,
                    kind=tool_kind(event.tool_name),
                    status="in_progress",
                    raw_input=event.args or None,
                )
            ]
        if isinstance(event, ToolExecutionUpdateEvent):
            text = _blocks_text(event.partial_result.content)
            if not text:
                return []
            return [helpers.update_tool_call(event.tool_call_id, content=_text_content(text))]
        if isinstance(event, ToolExecutionEndEvent):
            self._close(event.tool_call_id)
            text = event.result.text
            return [
                helpers.update_tool_call(
                    event.tool_call_id,
                    status="failed" if event.is_error else "completed",
                    content=_text_content(_limited(text)) if text else None,
                )
            ]
        if isinstance(event, SubagentStartEvent):
            return [
                helpers.start_tool_call(
                    f"subagent-{event.job_id}", event.label, kind="other", status="in_progress"
                )
            ]
        if isinstance(event, SubagentEndEvent):
            text = event.result_text or event.error_text or ""
            return [
                helpers.update_tool_call(
                    f"subagent-{event.job_id}",
                    status="completed" if event.status == "completed" else "failed",
                    content=_text_content(_limited(text)) if text else None,
                )
            ]
        return []

    def _close(self, tool_call_id: str) -> None:
        self._open_calls = [entry for entry in self._open_calls if entry[0] != tool_call_id]
        self._claimed.discard(tool_call_id)

    def permission_call_id(self, tool_name: str, job_id: str | None) -> tuple[str, bool]:
        """The tool-call id a permission request names, and whether it is real.

        The approval gate is not given the call id (only tool name, summary
        and job id), and it runs after ``tool_execution_start``. So the oldest
        open, not-yet-claimed call with the same tool name is the one being
        asked about. A subagent's call (``job_id`` set) never appears on this
        session's stream, and neither does an unmatched name: both get a
        synthetic ``perm-<hex>`` id.
        """
        if job_id is None:
            for call_id, name in self._open_calls:
                if name == tool_name and call_id not in self._claimed:
                    self._claimed.add(call_id)
                    return call_id, True
        return f"perm-{uuid.uuid4().hex[:12]}", False


def escape_secret_references(value: str) -> str:
    """Make a client-supplied value pass through lop's ``${NAME}`` resolver literally.

    lop resolves ``${NAME}`` in MCP ``headers`` against the operator's secret
    store. A header the CLIENT supplied must never do that — it would let the
    client read the operator's secrets by naming them in a header sent to its
    own URL — so every ``${`` becomes the resolver's documented escape ``$${``.
    """
    return value.replace("${", "$${")


def mcp_server_configs(servers: Sequence[Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """D8 (revised): ``session/new`` ``mcpServers`` → lop server entries.

    Returns ``(configs, ignored)``. HTTP servers become lop ``http`` entries
    (the streamable-HTTP transport) with the client's headers passed through;
    everything else — stdio, SSE, ACP-transport servers — is ignored in the
    POC and returned by name so the caller can log it.
    """
    configs: dict[str, dict[str, Any]] = {}
    ignored: list[str] = []
    for server in servers:
        name = str(getattr(server, "name", "") or "")
        if getattr(server, "type", None) == "http" and name:
            headers: Mapping[str, str] = {
                header.name: escape_secret_references(header.value)
                for header in (getattr(server, "headers", None) or [])
            }
            configs[name] = {"type": "http", "url": server.url, "headers": dict(headers)}
        else:
            ignored.append(f"{name or '?'} ({getattr(server, 'type', None) or 'stdio'})")
    return configs, ignored
