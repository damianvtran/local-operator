"""Replay sanitisation for garbled tool names (wire view only).

The durable transcript keeps the garbled name; this proves the copy sent to
the model is rewritten, the original is untouched, and pairing survives.
"""

from __future__ import annotations

import pytest

from local_operator.harness.tool_names import (
    format_malformed_notice,
    normalise_replay_tool_names,
)
from local_operator.harness.types import ImageContent, Message, TextContent, ToolCall

TOOLS = ["bash", "read", "write"]


def garbled_pair(call_id: str = "c1", name: str = "bashLecture") -> list[Message]:
    call = Message(
        role="assistant",
        content=[TextContent(text="running it")],
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments={"command": "echo hi"},
                raw_arguments='{"command": "echo hi"}',
            )
        ],
        provider_payload={"native_replay": {"items": [{"type": "function_call", "name": name}]}},
    )
    result = Message(
        role="tool",
        content=[TextContent(text=f"Tool not found: {name}")],
        tool_call_id=call_id,
        tool_name=name,
        is_error=True,
        provider_payload={"details": {"x": 1}},
    )
    return [Message.user("go"), call, result]


def test_garbled_call_and_result_are_rewritten():
    history = garbled_pair()
    out = normalise_replay_tool_names(history, TOOLS)
    call = next(m for m in out if m.role == "assistant")
    result = next(m for m in out if m.role == "tool")
    assert call.tool_calls[0].name == "bash"
    assert call.tool_calls[0].arguments == {"command": "echo hi"}
    assert call.tool_calls[0].raw_arguments == '{"command": "echo hi"}'
    assert call.provider_payload is None
    assert result.text == format_malformed_notice("bash")
    assert "Lecture" not in result.text
    assert result.tool_name == "bash"
    assert result.tool_call_id == "c1"
    assert result.is_error is True
    # Derived ids on the copies.
    assert call.id.endswith("~nm")
    assert result.id.endswith("~nm")
    # Originals untouched.
    assert history[1].tool_calls[0].name == "bashLecture"
    assert history[2].text == "Tool not found: bashLecture"
    assert history[1].provider_payload is not None


def test_clean_messages_pass_through_by_identity():
    history = [
        Message.user("go"),
        Message(
            role="assistant",
            content=[TextContent(text="ok")],
            tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "echo hi"})],
        ),
        Message(role="tool", content=[TextContent(text="ok")], tool_call_id="c1", tool_name="bash"),
    ]
    out = normalise_replay_tool_names(history, TOOLS)
    assert len(out) == len(history)
    for copied, original in zip(out, history):
        assert copied is original


def test_idempotent():
    history = garbled_pair()
    once = normalise_replay_tool_names(history, TOOLS)
    twice = normalise_replay_tool_names(once, TOOLS)
    assert [m.model_dump() for m in twice] == [m.model_dump() for m in once]
    # The sanitised copies are themselves clean, so a second pass reuses them.
    for first, second in zip(once, twice):
        assert first is second


def test_one_garbled_call_among_several():
    call = Message(
        role="assistant",
        content=[TextContent(text="two things")],
        tool_calls=[
            ToolCall(id="c1", name="bashLecture", arguments={"command": "a"}),
            ToolCall(id="c2", name="read", arguments={"path": "p"}),
        ],
    )
    results = [
        Message(
            role="tool",
            content=[TextContent(text="Tool not found: bashLecture")],
            tool_call_id="c1",
            tool_name="bashLecture",
            is_error=True,
        ),
        Message(
            role="tool",
            content=[TextContent(text="file body")],
            tool_call_id="c2",
            tool_name="read",
        ),
    ]
    history = [Message.user("go"), call, *results]
    out = normalise_replay_tool_names(history, TOOLS)
    rewritten = next(m for m in out if m.role == "assistant")
    assert [c.name for c in rewritten.tool_calls] == ["bash", "read"]
    # The untouched sibling call object rides along by identity.
    assert rewritten.tool_calls[1] is call.tool_calls[1]
    assert rewritten.tool_calls[0] is not call.tool_calls[0]
    assert rewritten.tool_calls[0].arguments == {"command": "a"}
    err = next(m for m in out if m.role == "tool" and m.tool_call_id == "c1")
    ok = next(m for m in out if m.role == "tool" and m.tool_call_id == "c2")
    assert err.text == format_malformed_notice("bash")
    assert ok.text == "file body"
    assert ok is results[1]


def test_pre_fix_transcript_covered_by_id_and_is_error_not_text():
    # The old durable result text is the junk line; pairing must not depend on it.
    history = garbled_pair()
    history[2] = Message(
        role="tool",
        content=[TextContent(text="Tool not found: bashLectureLecture")],
        tool_call_id="c1",
        tool_name="bashLecture",
        is_error=True,
    )
    out = normalise_replay_tool_names(history, TOOLS)
    result = next(m for m in out if m.role == "tool")
    assert result.text == format_malformed_notice("bash")
    assert "Lecture" not in result.text


def test_non_error_result_with_same_id_is_left_alone():
    call = Message(
        role="assistant",
        content=[TextContent(text="t")],
        tool_calls=[ToolCall(id="c1", name="bashLecture", arguments={})],
    )
    result = Message(
        role="tool",
        content=[TextContent(text="actually succeeded")],
        tool_call_id="c1",
        tool_name="bash",
        is_error=False,
    )
    out = normalise_replay_tool_names([Message.user("go"), call, result], TOOLS)
    untouched = next(m for m in out if m.role == "tool")
    assert untouched.text == "actually succeeded"
    assert untouched is result


def test_image_blocks_ride_along():
    call = Message(
        role="assistant",
        content=[TextContent(text="t")],
        tool_calls=[ToolCall(id="c1", name="bashLectureLecture", arguments={})],
    )
    image = ImageContent(data="aGVsbG8=", mime_type="image/png")
    result = Message(
        role="tool",
        content=[TextContent(text="Tool not found: bashLectureLecture"), image],
        tool_call_id="c1",
        tool_name="bashLectureLecture",
        is_error=True,
    )
    out = normalise_replay_tool_names([Message.user("go"), call, result], TOOLS)
    rewritten = next(m for m in out if m.role == "tool")
    assert isinstance(rewritten.content[0], TextContent)
    assert rewritten.content[0].text == format_malformed_notice("bash")
    assert rewritten.content[1] is image


def test_tool_names_accept_tool_objects_and_bare_strings():
    history = garbled_pair()

    class FakeTool:
        name = "bash"

    out = normalise_replay_tool_names(history, [FakeTool()])
    assert next(m for m in out if m.role == "assistant").tool_calls[0].name == "bash"
    out2 = normalise_replay_tool_names(history, ["bash"])
    assert next(m for m in out2 if m.role == "assistant").tool_calls[0].name == "bash"


def test_empty_result_content_gets_the_notice():
    call = Message(
        role="assistant",
        content=[TextContent(text="t")],
        tool_calls=[ToolCall(id="c1", name="bashLecture", arguments={})],
    )
    result = Message(
        role="tool",
        content=[],
        tool_call_id="c1",
        tool_name="bashLecture",
        is_error=True,
    )
    out = normalise_replay_tool_names([Message.user("go"), call, result], TOOLS)
    rewritten = next(m for m in out if m.role == "tool")
    assert rewritten.text == format_malformed_notice("bash")


@pytest.mark.parametrize("name", ["bashLecture", "bashLectureLecture", "bash-x"])
def test_repeated_garble_rewrites_to_bash(name: str):
    history = garbled_pair(name=name)
    out = normalise_replay_tool_names(history, TOOLS)
    assert next(m for m in out if m.role == "assistant").tool_calls[0].name == "bash"


def test_reused_call_id_later_keeps_its_genuine_error():
    """A call id reused later must not consume the earlier garble's rewrite.

    Pairing is in order: the pending rewrite is consumed by the NEXT result
    with that id, later repetition is a new call and keeps its own result.
    """
    history = [
        Message.user("go"),
        Message(
            role="assistant",
            content=[TextContent(text="a")],
            tool_calls=[ToolCall(id="c1", name="bashLecture", arguments={})],
            provider_payload={"x": 1},
        ),
        Message(
            role="tool",
            content=[TextContent(text="Tool not found: bashLecture")],
            tool_call_id="c1",
            tool_name="bashLecture",
            is_error=True,
        ),
        Message(
            role="assistant",
            content=[TextContent(text="b")],
            tool_calls=[ToolCall(id="c1", name="bash", arguments={"y": 2})],
            provider_payload={"x": 2},
        ),
        Message(
            role="tool",
            content=[TextContent(text="exit status 1: permission denied")],
            tool_call_id="c1",
            tool_name="bash",
            is_error=True,
        ),
    ]
    out = normalise_replay_tool_names(history, ["bash"])
    calls = [m for m in out if m.role == "assistant"]
    results = [m for m in out if m.role == "tool"]
    assert calls[0].tool_calls[0].name == "bash"
    assert calls[0].provider_payload is None
    assert results[0].text == format_malformed_notice("bash")
    # Later, genuine pair untouched (by identity).
    assert calls[1] is history[3]
    assert results[1] is history[4]
    assert results[1].text == "exit status 1: permission denied"


def test_replay_warning_is_deduped_per_process(caplog) -> None:
    """One warning per (real name, remainder length) per process.

    Rebuilding the same history must not emit one line per garbled call per
    request; the comment in the normaliser pins that contract.
    """
    import logging

    from local_operator.harness.tool_names import _LOGGED_NEAR_MISSES

    _LOGGED_NEAR_MISSES.clear()
    history = [Message.user("go")]
    for i in range(3):
        history.append(
            Message(
                role="assistant",
                content=[TextContent(text="a")],
                tool_calls=[ToolCall(id=f"c{i}", name="bashLecture", arguments={})],
            )
        )
    for i in range(3):
        history.append(
            Message(
                role="tool",
                content=[TextContent(text="Tool not found: bashLecture")],
                tool_call_id=f"c{i}",
                tool_name="bashLecture",
                is_error=True,
            )
        )
    with caplog.at_level(logging.WARNING, logger="local_operator.harness.tool_names"):
        for _ in range(3):
            normalise_replay_tool_names(history, TOOLS)
    warnings = [r for r in caplog.records if "replay tool name" in r.message]
    assert len(warnings) == 1
