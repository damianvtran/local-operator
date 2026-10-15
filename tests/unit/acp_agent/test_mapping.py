"""The lop event → ACP ``session/update`` table (docs/design/acp-agent.md).

One case per row of the mapping table, built from the REAL harness event
classes so a renamed field fails here rather than in a client's renderer.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("acp")

from local_operator.acp_agent.mapping import (  # noqa: E402
    MAX_RESULT_CHARS,
    EventMapper,
    TurnError,
    stop_reason_for,
    tool_kind,
)
from local_operator.harness.types import (  # noqa: E402
    AgentEndEvent,
    AgentToolUpdate,
    CompactionStartEvent,
    Message,
    MessageEndEvent,
    MessageUpdateEvent,
    NoticeEvent,
    ReasoningDeltaEvent,
    SubagentEndEvent,
    SubagentStartEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResult,
)


def _assistant(text: str = "", message_id: str = "m1") -> Message:
    return Message(role="assistant", content=[TextContent(text=text)], id=message_id)


def _dump(update: object) -> dict[str, Any]:
    return update.model_dump(by_alias=True, exclude_none=True)  # type: ignore[attr-defined]


def test_message_delta_becomes_an_agent_message_chunk() -> None:
    mapper = EventMapper()
    [update] = mapper.updates_for(MessageUpdateEvent(message=_assistant(), delta="Hel"))
    assert _dump(update) == {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": "Hel"},
    }


def test_message_end_sends_full_text_only_when_nothing_streamed() -> None:
    mapper = EventMapper()
    mapper.updates_for(MessageUpdateEvent(message=_assistant(message_id="a"), delta="x"))
    assert mapper.updates_for(MessageEndEvent(message=_assistant("x", message_id="a"))) == []
    [update] = mapper.updates_for(MessageEndEvent(message=_assistant("whole", message_id="b")))
    assert _dump(update)["content"]["text"] == "whole"


def test_message_end_of_a_user_or_empty_message_sends_nothing() -> None:
    mapper = EventMapper()
    user = Message(role="user", content=[TextContent(text="hi")])
    assert mapper.updates_for(MessageEndEvent(message=user)) == []
    assert mapper.updates_for(MessageEndEvent(message=_assistant(""))) == []


def test_reasoning_delta_becomes_a_thought_chunk() -> None:
    [update] = EventMapper().updates_for(ReasoningDeltaEvent(message_id="m", delta="hmm"))
    assert _dump(update) == {
        "sessionUpdate": "agent_thought_chunk",
        "content": {"type": "text", "text": "hmm"},
    }


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("read", "read"),
        ("grep", "search"),
        ("glob", "search"),
        ("edit", "edit"),
        ("write", "edit"),
        ("bash", "execute"),
        ("eval", "execute"),
        ("web_fetch", "fetch"),
        ("todo", "other"),
    ],
)
def test_tool_kind(name: str, kind: str) -> None:
    assert tool_kind(name) == kind


def test_tool_start_becomes_an_in_progress_tool_call() -> None:
    event = ToolExecutionStartEvent(
        tool_call_id="c1", tool_name="bash", args={"command": "ls"}, intent="Listing files"
    )
    [update] = EventMapper().updates_for(event)
    assert _dump(update) == {
        "sessionUpdate": "tool_call",
        "toolCallId": "c1",
        "title": "Listing files",
        "kind": "execute",
        "status": "in_progress",
        "rawInput": {"command": "ls"},
    }


def test_tool_start_without_intent_is_titled_by_the_tool_name() -> None:
    [update] = EventMapper().updates_for(
        ToolExecutionStartEvent(tool_call_id="c", tool_name="read")
    )
    assert _dump(update)["title"] == "read"


def test_tool_update_carries_the_partial_text() -> None:
    event = ToolExecutionUpdateEvent(
        tool_call_id="c1",
        tool_name="bash",
        partial_result=AgentToolUpdate(content=[TextContent(text="line 1")]),
    )
    [update] = EventMapper().updates_for(event)
    assert _dump(update) == {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "c1",
        "content": [{"type": "content", "content": {"type": "text", "text": "line 1"}}],
    }


@pytest.mark.parametrize(("is_error", "status"), [(False, "completed"), (True, "failed")])
def test_tool_end_settles_the_call(is_error: bool, status: str) -> None:
    result = ToolResult(tool_call_id="c1", content=[TextContent(text="out")], is_error=is_error)
    [update] = EventMapper().updates_for(
        ToolExecutionEndEvent(tool_call_id="c1", tool_name="bash", result=result)
    )
    dumped = _dump(update)
    assert dumped["status"] == status
    assert dumped["content"][0]["content"]["text"] == "out"


def test_tool_end_result_text_is_length_limited() -> None:
    result = ToolResult(tool_call_id="c1", content=[TextContent(text="x" * (MAX_RESULT_CHARS * 2))])
    [update] = EventMapper().updates_for(
        ToolExecutionEndEvent(tool_call_id="c1", tool_name="bash", result=result)
    )
    assert len(_dump(update)["content"][0]["content"]["text"]) <= MAX_RESULT_CHARS + 64


def test_subagent_start_and_end_map_to_an_other_tool_call() -> None:
    mapper = EventMapper()
    [start] = mapper.updates_for(SubagentStartEvent(job_id="j1", label="scout"))
    assert _dump(start) == {
        "sessionUpdate": "tool_call",
        "toolCallId": "subagent-j1",
        "title": "scout",
        "kind": "other",
        "status": "in_progress",
    }
    [end] = mapper.updates_for(
        SubagentEndEvent(job_id="j1", label="scout", status="completed", result_text="done")
    )
    dumped = _dump(end)
    assert dumped["toolCallId"] == "subagent-j1"
    assert dumped["status"] == "completed"
    assert dumped["content"][0]["content"]["text"] == "done"


@pytest.mark.parametrize(
    "event",
    [
        NoticeEvent(text="hello", kind="info"),
        CompactionStartEvent(reason="manual"),
        AgentEndEvent(),
    ],
)
def test_events_the_client_does_not_see(event: object) -> None:
    assert EventMapper().updates_for(event) == []  # type: ignore[arg-type]


def test_permission_matches_the_oldest_open_call_of_that_tool() -> None:
    mapper = EventMapper()
    mapper.updates_for(ToolExecutionStartEvent(tool_call_id="r1", tool_name="read"))
    mapper.updates_for(ToolExecutionStartEvent(tool_call_id="b1", tool_name="bash"))
    mapper.updates_for(ToolExecutionStartEvent(tool_call_id="b2", tool_name="bash"))
    assert mapper.permission_call_id("bash", None) == ("b1", True)
    # A second concurrent ask for the same tool claims the NEXT open call.
    assert mapper.permission_call_id("bash", None) == ("b2", True)


def test_permission_without_a_match_gets_a_synthetic_id() -> None:
    mapper = EventMapper()
    mapper.updates_for(ToolExecutionStartEvent(tool_call_id="b1", tool_name="bash"))
    call_id, known = mapper.permission_call_id("bash", "job-7")  # a subagent's call
    assert not known and call_id.startswith("perm-")
    call_id, known = mapper.permission_call_id("write", None)
    assert not known and call_id.startswith("perm-")


def test_a_finished_call_is_no_longer_matched() -> None:
    mapper = EventMapper()
    mapper.updates_for(ToolExecutionStartEvent(tool_call_id="b1", tool_name="bash"))
    mapper.updates_for(
        ToolExecutionEndEvent(
            tool_call_id="b1", tool_name="bash", result=ToolResult(tool_call_id="b1")
        )
    )
    assert mapper.permission_call_id("bash", None)[1] is False


def test_stop_reasons() -> None:
    assert stop_reason_for(AgentEndEvent()) == "end_turn"
    assert stop_reason_for(AgentEndEvent(aborted=True)) == "cancelled"
    refused = Message(role="assistant", content=[], stop_reason="refusal")
    assert stop_reason_for(AgentEndEvent(error="no", messages=[refused])) == "refusal"
    with pytest.raises(TurnError, match="provider down"):
        stop_reason_for(AgentEndEvent(error="provider down"))


def test_http_mcp_servers_become_lop_http_entries_and_others_are_ignored() -> None:
    from acp.schema import (
        EnvVariable,
        HttpHeader,
        HttpMcpServer,
        McpServerStdio,
        SseMcpServer,
    )

    from local_operator.acp_agent.mapping import mcp_server_configs

    configs, ignored = mcp_server_configs(
        [
            HttpMcpServer(
                type="http",
                name="gateway",
                url="http://127.0.0.1:9/mcp",
                headers=[HttpHeader(name="Authorization", value="Bearer abc")],
            ),
            McpServerStdio(
                name="local", command="x", args=[], env=[EnvVariable(name="A", value="b")]
            ),
            SseMcpServer(type="sse", name="old", url="http://127.0.0.1:9/sse", headers=[]),
        ]
    )
    assert configs == {
        "gateway": {
            "type": "http",
            "url": "http://127.0.0.1:9/mcp",
            "headers": {"Authorization": "Bearer abc"},
            "preloadTools": True,
        }
    }
    assert ignored == ["local (stdio)", "old (sse)"]


def test_client_headers_cannot_name_the_operators_secrets() -> None:
    """``${NAME}`` in a header is resolved from the operator's store; a client's
    header must stay literal, so it is escaped to the resolver's ``$${``."""
    from acp.schema import HttpHeader, HttpMcpServer

    from local_operator.acp_agent.mapping import mcp_server_configs
    from local_operator.mcp.config import coerce_server_config
    from local_operator.mcp.secret_refs import resolve_config_secrets

    configs, _ = mcp_server_configs(
        [
            HttpMcpServer(
                type="http",
                name="g",
                url="http://127.0.0.1:9/mcp",
                headers=[HttpHeader(name="X-Leak", value="${OPENAI_API_KEY}")],
            )
        ]
    )
    from local_operator.mcp.config import MCPHttpServerConfig

    cfg = coerce_server_config(configs["g"])
    assert isinstance(cfg, MCPHttpServerConfig)
    resolved = resolve_config_secrets("g", cfg, base=None, register=lambda _v: None)
    assert resolved.headers["X-Leak"] == "${OPENAI_API_KEY}"
