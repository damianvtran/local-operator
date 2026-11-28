"""Foreign tool IDs cross the Anthropic Messages boundary without changing history."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    TextContent,
    ToolCall,
)
from local_operator.providers.clients import ANTHROPIC_API_URL, AnthropicClient
from local_operator.providers.replay import credential_scope, native_payload

SPEC = ModelSpec(provider="anthropic", model_id="claude-test")
KEY = "fixture"


def _sse_ok() -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text='event: message_stop\ndata: {"type": "message_stop"}\n\n',
    )


async def _send(client: AnthropicClient, request: ChatRequest) -> None:
    events = [event async for event in client.stream(request, KEY)]
    assert isinstance(events[-1], StreamEndEvent)


def _tool_history(*ids: str) -> list[Message]:
    return [
        Message.assistant(
            "", tool_calls=[ToolCall(id=call_id, name="lookup", arguments={}) for call_id in ids]
        ),
        *[
            Message(role="tool", tool_call_id=call_id, content=[TextContent(text=f"fixture {i}")])
            for i, call_id in enumerate(ids)
        ],
    ]


def _tool_ids(body: dict[str, Any]) -> list[list[str]]:
    return [
        ids
        for row in body["messages"]
        if (
            ids := [
                block["id" if block["type"] == "tool_use" else "tool_use_id"]
                for block in row["content"]
                if block["type"] in ("tool_use", "tool_result")
            ]
        )
    ]


@pytest.mark.asyncio
async def test_foreign_parallel_tool_ids_are_accepted_at_messages_boundary() -> None:
    valid = "example_1-Z"
    first_turn = _tool_history("example:1", "example/1", valid)
    first_turn[0].provider_payload = native_payload(
        ModelSpec(provider="openai", model_id="fixture-model"),
        "https://api.openai.com/v1",
        "openai-responses",
        [{"type": "function_call", "call_id": "example:1", "name": "lookup", "arguments": "{}"}],
        "",
        [
            {"id": call.id, "name": call.name, "args": call.arguments}
            for call in first_turn[0].tool_calls
        ],
        credential_scope(KEY),
    )
    assert first_turn[0].provider_payload is not None
    # Messages joins adjacent assistant rows into one batch.
    first_turn.insert(1, Message.assistant("Preparing the result."))
    history = [
        Message.user("Look up three items."),
        *first_turn,
        Message.assistant("Found them."),
        Message.user("Look up another item."),
        *_tool_history("example:1"),
    ]
    history[-1].content = [TextContent(text="second turn result")]
    original = [message.model_dump_json() for message in history]
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/messages"
        body = json.loads(request.content)
        ids = _tool_ids(body)
        for row in ids:
            for call_id in row:
                if re.fullmatch(r"[a-zA-Z0-9_-]+", call_id) is None:
                    return httpx.Response(
                        400,
                        json={
                            "error": {"message": f"invalid tool ID {call_id!r}: ^[a-zA-Z0-9_-]+$"}
                        },
                    )
        uses = [call_id for row in ids[::2] for call_id in row]
        assert ids[::2] == ids[1::2]
        assert len(set(uses)) == len(uses)
        assert ids[0][2] == valid
        results = [
            (b["tool_use_id"], b["content"][0]["text"])
            for row in body["messages"]
            for b in row["content"]
            if b["type"] == "tool_result"
        ]
        expected = ["fixture 0", "fixture 1", "fixture 2", "second turn result"]
        if len(sent) == 2:
            expected.append("fixture 0")
        assert results == list(zip(uses, expected, strict=True))
        sent.append(body)
        return _sse_ok()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AnthropicClient(http_client=http)
        request = ChatRequest(model=SPEC, messages=history)
        await _send(client, request)
        await _send(client, request)
        await _send(
            client,
            request.model_copy(
                update={
                    "messages": [*history, Message.user("Another."), *_tool_history("example:1")]
                }
            ),
        )

    assert sent[0] == sent[1]
    # Breakpoints can move with the tail. The already rendered tool IDs may not.
    assert _tool_ids(sent[0]) == _tool_ids(sent[2])[:4]
    assert [message.model_dump_json() for message in request.messages] == original


@pytest.mark.asyncio
async def test_alias_avoids_later_valid_ids_in_native_and_reconstructed_blocks() -> None:
    raw_id = "foreign:1"
    # Force the candidate and its first fallback to be occupied, one by each
    # replay path. Real digest collisions need not occur to test this invariant.
    occupied = "toolu_lop_" + hashlib.sha256(raw_id.encode()).hexdigest()[:32]
    fallback = occupied + "_2"
    native_calls = [{"id": occupied, "name": "lookup", "args": {}}]
    thinking = {
        "type": "thinking",
        "thinking": "Fixture reasoning.",
        "signature": "fixture-signature",
    }
    native_items = [thinking, {"type": "tool_use", "id": occupied, "name": "lookup"}]
    payload = native_payload(
        SPEC,
        ANTHROPIC_API_URL,
        "anthropic-messages",
        native_items,
        "",
        native_calls,
        credential_scope(KEY),
    )
    assert payload is not None
    native_turn = _tool_history(occupied)
    native_turn[0].provider_payload = payload
    history = [
        Message.user("Look up items."),
        *_tool_history(raw_id),
        *native_turn,
        *_tool_history(fallback),
    ]
    original = [message.model_dump_json() for message in history]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        ids = _tool_ids(body)
        alias = ids[0][0]
        assert alias not in (occupied, fallback, raw_id)
        assert ids == [[alias], [alias], [occupied], [occupied], [fallback], [fallback]]
        assert body["messages"][3]["content"] == [thinking, {**native_items[1], "input": {}}]
        return _sse_ok()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await _send(
            AnthropicClient(http_client=http),
            ChatRequest(model=SPEC, messages=history, cache_markers="none"),
        )
    assert [message.model_dump_json() for message in history] == original


@pytest.mark.asyncio
async def test_missing_empty_and_filtered_ids_are_not_made_into_pairs() -> None:
    history = [
        Message.user("Look up items."),
        _tool_history("one:1")[1],  # Nowhere to pair, despite the later call.
        *_tool_history("one:1"),
        Message.assistant("Done."),
        _tool_history("one:1")[1],  # Cannot pair back across an assistant turn.
        *_tool_history(""),
        Message.assistant("", tool_calls=[ToolCall(id="dropped:1", name="", arguments={})]),
        _tool_history("dropped:1")[1],
        *_tool_history("ambiguous:1", "ambiguous:1"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        ids = _tool_ids(json.loads(request.content))
        alias = ids[1][0]
        assert alias != "one:1"
        assert ids == [
            ["one:1"],
            [alias],
            [alias],
            ["one:1"],
            [""],
            [""],
            ["ambiguous:1", "ambiguous:1"],
            ["ambiguous:1", "ambiguous:1"],
        ]
        return _sse_ok()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await _send(AnthropicClient(http_client=http), ChatRequest(model=SPEC, messages=history))
