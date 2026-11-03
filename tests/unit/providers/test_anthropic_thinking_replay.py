"""Anthropic thinking blocks: captured verbatim, replayed only where valid.

Anthropic requires the model's ``thinking``/``redacted_thinking`` blocks to be
passed back "complete and unmodified" on later requests of a tool-use turn, and
a signature replayed to another model or account is an HTTP 400. These tests
drive real SSE fixtures through ``AnthropicClient.stream``, persist the Message
through JSON (the transcript/resume representation), then inspect the next
request body. No network.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCall,
)
from local_operator.providers.clients import AnthropicClient
from local_operator.providers.replay import credential_scope

pytestmark = pytest.mark.asyncio

KEY = "sk-ant-fixture"
SPEC = ModelSpec(provider="anthropic", model_id="claude-sonnet-5-5")
THINKING = {"type": "thinking", "thinking": "The median of 3,5,7 is 5.", "signature": "sig-AB"}
REDACTED = {"type": "redacted_thinking", "data": "opaque-encrypted-blob"}


def _sse(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    return "".join(
        f"event: {name}\ndata: {json.dumps({'type': name, **payload})}\n\n"
        for name, payload in events
    ).encode()


def _thinking_turn(*, tool_json: tuple[str, ...] = ('{"cmd"', ': "ls"}')) -> bytes:
    """thinking (two thinking deltas + signature) -> redacted -> text -> tool_use."""
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"message": {"id": "msg_1", "usage": {"input_tokens": 9}}}),
        (
            "content_block_start",
            {"index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "thinking_delta", "thinking": "The median of "}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "thinking_delta", "thinking": "3,5,7 is 5."}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "signature_delta", "signature": "sig-"}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "signature_delta", "signature": "AB"}},
        ),
        ("content_block_stop", {"index": 0}),
        ("content_block_start", {"index": 1, "content_block": dict(REDACTED)}),
        ("content_block_stop", {"index": 1}),
        ("content_block_start", {"index": 2, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 2, "delta": {"type": "text_delta", "text": "Listing."}}),
        ("content_block_stop", {"index": 2}),
        (
            "content_block_start",
            {"index": 3, "content_block": {"type": "tool_use", "id": "tu_1", "name": "bash"}},
        ),
    ]
    events += [
        (
            "content_block_delta",
            {"index": 3, "delta": {"type": "input_json_delta", "partial_json": p}},
        )
        for p in tool_json
    ]
    events += [
        ("content_block_stop", {"index": 3}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}}),
        ("message_stop", {}),
    ]
    return _sse(events)


def _plain_turn() -> bytes:
    return _sse(
        [
            ("message_start", {"message": {"id": "msg_2", "usage": {"input_tokens": 9}}}),
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
            ("content_block_stop", {"index": 0}),
            (
                "message_delta",
                {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            ),
        ]
    )


async def _stream(body: bytes) -> list[Any]:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AnthropicClient(http_client=http)
        request = ChatRequest(model=SPEC, messages=[Message.user("go")])
        return [event async for event in client.stream(request, KEY)]


def _persisted(events: list[Any]) -> Message:
    """Assemble the assistant row as the loop does, then round-trip it through JSON."""
    text = "".join(e.delta for e in events if isinstance(e, StreamTextDelta))
    calls: dict[int, dict[str, Any]] = {}
    for e in events:
        if isinstance(e, StreamToolCallDelta):
            slot = calls.setdefault(e.index, {"id": "", "name": "", "raw": ""})
            slot["id"] = e.id or slot["id"]
            slot["name"] = e.name or slot["name"]
            slot["raw"] += e.argument_delta
    end = next(e for e in reversed(events) if isinstance(e, StreamEndEvent))
    message = Message.assistant(
        text,
        tool_calls=[
            ToolCall(
                id=c["id"], name=c["name"], arguments=json.loads(c["raw"]), raw_arguments=c["raw"]
            )
            for c in calls.values()
        ],
        stop_reason=end.stop_reason,
        provider_payload=end.provider_payload,
    )
    return Message.model_validate_json(message.model_dump_json())


def _history(assistant: Message, model: ModelSpec = SPEC) -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[
            Message.user("go"),
            assistant,
            Message(
                role="tool",
                tool_call_id="tu_1",
                tool_name="bash",
                content=[TextContent(text="a b c")],
            ),
        ],
    )


def _assistant_content(body: dict[str, Any]) -> list[dict[str, Any]]:
    return next(m for m in body["messages"] if m["role"] == "assistant")["content"]


async def test_stream_captures_thinking_signature_and_redacted_verbatim() -> None:
    events = await _stream(_thinking_turn())
    # Thinking is never surfaced as visible text.
    assert "".join(e.delta for e in events if isinstance(e, StreamTextDelta)) == "Listing."
    end = next(e for e in events if isinstance(e, StreamEndEvent))
    native = end.provider_payload["native_replay"]
    assert native["protocol"] == "anthropic-messages"
    assert native["model"] == "claude-sonnet-5-5"
    assert native["credential_scope"] == credential_scope(KEY)
    # Original order, every field intact; tool input lives on the Message, not here.
    assert native["items"] == [
        THINKING,
        REDACTED,
        {"type": "text", "text": "Listing."},
        {"type": "tool_use", "id": "tu_1", "name": "bash"},
    ]


async def test_replay_puts_thinking_first_unchanged_and_survives_resume() -> None:
    assistant = _persisted(await _stream(_thinking_turn()))
    assert assistant.text == "Listing."  # transcript prose excludes thinking
    request = _history(assistant)
    client = AnthropicClient()
    body = client._build_body(request, scope=credential_scope(KEY))
    content = _assistant_content(body)
    assert content[0] == THINKING
    assert content[1] == REDACTED
    assert content[2] == {"type": "text", "text": "Listing."}
    assert content[3] == {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"cmd": "ls"}}
    # The tool_result still follows as the next user message.
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    # No cache marker may land on a thinking block, and rendering must not
    # mutate the stored payload (it is replayed on every later request).
    assert all("cache_control" not in block for block in content[:2])
    stored = assistant.provider_payload["native_replay"]["items"]
    assert all("cache_control" not in item and "input" not in item for item in stored)
    assert client._build_body(request, scope=credential_scope(KEY))["messages"] == body["messages"]


@pytest.mark.parametrize(
    "change",
    ["model", "provider", "credential", "endpoint", "edited_text"],
)
async def test_thinking_dropped_when_provenance_or_visible_turn_changes(change: str) -> None:
    assistant = _persisted(await _stream(_thinking_turn()))
    model, scope, client = SPEC, credential_scope(KEY), AnthropicClient()
    if change == "model":
        model = SPEC.model_copy(update={"model_id": "claude-opus-5-5"})
    elif change == "provider":
        model = SPEC.model_copy(update={"provider": "anthropic-key"})
    elif change == "credential":
        scope = credential_scope("sk-ant-other-account")
    elif change == "endpoint":
        client = AnthropicClient(base_url="https://relay.example")
    else:
        assistant.content = [TextContent(text="Listing (redacted).")]
    body = client._build_body(_history(assistant, model), scope=scope)
    serialized = json.dumps(body)
    assert "sig-AB" not in serialized and "opaque-encrypted-blob" not in serialized
    # Falls back to today's visible render: text + tool_use only.
    assert [b["type"] for b in _assistant_content(body)] == ["text", "tool_use"]


async def test_legacy_session_without_payload_renders_as_before() -> None:
    legacy = Message.model_validate(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Listing."}],
            "tool_calls": [{"id": "tu_1", "name": "bash", "arguments": {"cmd": "ls"}}],
            "stop_reason": "toolUse",
        }
    )
    body = AnthropicClient()._build_body(_history(legacy), scope=credential_scope(KEY))
    assert _assistant_content(body) == [
        {"type": "text", "text": "Listing."},
        {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"cmd": "ls"}},
    ]


async def test_turn_without_thinking_carries_no_payload() -> None:
    end = next(e for e in await _stream(_plain_turn()) if isinstance(e, StreamEndEvent))
    assert end.provider_payload is None


async def test_unparseable_tool_arguments_store_no_payload() -> None:
    # A truncated call is stored with repaired arguments the fingerprint could
    # never match; storing the payload would be dead weight.
    events = await _stream(_thinking_turn(tool_json=('{"cmd": "l',)))
    end = next(e for e in events if isinstance(e, StreamEndEvent))
    assert end.provider_payload is None


async def test_trailing_thinking_block_is_never_replayed() -> None:
    # Anthropic refuses cache_control on a thinking block, and the breakpoint
    # pass marks a message's last block, so a turn ENDING on thinking (output
    # cut after the thinking block) takes the visible render instead.
    events = await _stream(
        _sse(
            [
                ("message_start", {"message": {"id": "m", "usage": {"input_tokens": 1}}}),
                ("content_block_start", {"index": 0, "content_block": dict(THINKING)}),
                ("content_block_stop", {"index": 0}),
                (
                    "content_block_start",
                    {"index": 1, "content_block": {"type": "text", "text": ""}},
                ),
                ("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": " "}}),
                ("content_block_stop", {"index": 1}),
                (
                    "message_delta",
                    {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                ),
            ]
        )
    )
    end = next(e for e in events if isinstance(e, StreamEndEvent))
    assistant = Message.assistant("ok", provider_payload=end.provider_payload)
    request = ChatRequest(model=SPEC, messages=[Message.user("a"), assistant, Message.user("b")])
    assert "sig-AB" not in json.dumps(
        AnthropicClient()._build_body(request, scope=credential_scope(KEY))
    )
