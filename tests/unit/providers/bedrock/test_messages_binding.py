"""Offline InvokeModel Messages wire tests, with deliberately inert signatures."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from local_operator.harness.render import _default_convert_to_llm
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.providers.bedrock import client as bedrock
from local_operator.providers.bedrock.messages_binding import FIELD, THINKING
from local_operator.providers.context import (
    ContextBinding,
    ContextTokenTracker,
    measure_request,
)
from local_operator.providers.image_rejection import forget_rejected_images
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import build_read_tool
from local_operator.tools.library_tool import build_library_tool

from . import eventstream as es

MODEL = ModelSpec(provider="bedrock", model_id="ca.anthropic.claude-sonnet-5-5")
KEY = "bedrock://env-bearer?region=ca-central-1"


def _thinking(tag: str) -> list[dict[str, Any]]:
    return [
        {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "ponder"}},
        {"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": tag}},
        {"type": "content_block_stop"},
    ]


def _tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "c-read", "name": "read", "input": {}},
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"tool://library"}'},
        },
        {"type": "content_block_stop"},
    ]


def _response(*, tool: bool = False, tag: str = "inert", complete: bool = True) -> httpx.Response:
    blocks = [_thinking(tag)]
    if tool:
        blocks.extend(
            [
                _tool(),
                [
                    {
                        "type": "content_block_start",
                        "content_block": {"type": "redacted_thinking", "data": "opaque-fixture"},
                    },
                    {"type": "content_block_stop"},
                ],
            ]
        )
    events = es.anthropic_events(
        text="portable output", extra_blocks=blocks, stop_reason="tool_use" if tool else "end_turn"
    )
    return httpx.Response(200, content=es.stream_bytes(events if complete else events[:-1]))


@pytest.fixture
def wire(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "inert-no-send")
    monkeypatch.delenv("LOP_BEDROCK_ENDPOINT_URL", raising=False)
    return []


def _client(wire, answer: Callable[[int], httpx.Response]):
    def handler(request: httpx.Request):
        assert request.url.path.endswith("/invoke-with-response-stream")
        wire.append((str(request.url), request.content, json.loads(request.content)))
        return answer(len(wire))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _send(client, request, key=KEY):
    return [event async for event in client.stream(request, key)][-1]


def _request(messages, **options):
    return ChatRequest(model=MODEL, messages=messages, **options)


def _native(body):
    return [
        block
        for message in body["messages"]
        for block in message["content"]
        if block["type"] in THINKING
    ]


def _portable(body):
    """Ignore actual placement of cache breakpoints, not identically named data."""
    return [
        (
            row["role"],
            [
                {k: v for k, v in block.items() if k != "cache_control"}
                for block in row["content"]
                if block["type"] not in THINKING
            ],
        )
        for row in body["messages"]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "thinking"),
    [
        ("same", 2),
        ("markers", 2),
        ("tools-order", 0),
        ("schema-order", 0),
        ("schema-value", 0),
        ("system", 0),
        ("history", 0),
        ("endpoint", 0),
        ("region", 0),
        ("scope", 0),
        ("model", 0),
        ("legacy", 2),
    ],
)
async def test_prefix_splices_only_native_blocks(wire, monkeypatch, mode, thinking):
    read, library = build_read_tool(), build_library_tool()
    # A cache_control property is real tool data, unlike the protocol marker.
    read.parameters = {
        "type": "object",
        "properties": {"path": {}, "cache_control": {"description": "real"}},
    }
    original = _request([Message.user("original")], tools=[read, library], system_blocks=["stable"])
    async with _client(wire, lambda _: _response(tool=True)) as http:
        client = bedrock.BedrockAnthropicClient(http_client=http)
        produced = await _send(client, original)
        replay = produced.provider_payload["native_replay"]
        assert list(replay[FIELD]) == ["0", "2"]
        if mode == "legacy":
            del replay[FIELD]
        call = ToolCall(id="c-read", name="read", arguments={"path": "tool://library"})
        assistant = Message.assistant(
            "portable output", tool_calls=[call], provider_payload=produced.provider_payload
        )
        result = Message.tool_result(
            ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[TextContent(text="local result")],
            )
        )
        followup = original.model_copy(
            update={"messages": [*original.messages, assistant, result, Message.user("continue")]}
        )
        tracker = ContextTokenTracker()
        binding = ContextBinding(tracker, measure_request(followup))
        followup.context_binding = binding
        await _send(client, followup)
        positive = wire[-1][2]
        assert len(_native(positive)) == 2
        tracker.record(binding.measured, produced.usage.model_copy(update={"context_tokens": 99}))
        options: dict[str, Any] = {}
        key = KEY
        if mode == "markers":
            options["cache_markers"] = "none"
        elif mode == "tools-order":
            options["tools"] = [library, read]
        elif mode in {"schema-order", "schema-value"}:
            properties = read.parameters["properties"]
            changed = (
                dict(reversed(list(properties.items())))
                if mode == "schema-order"
                else {**properties, "cache_control": {"description": "changed"}}
            )
            options["tools"] = [
                read.model_copy(update={"parameters": {**read.parameters, "properties": changed}}),
                library,
            ]
        elif mode == "system":
            options["system_blocks"] = ["changed"]
        elif mode in {"history", "legacy"}:
            options["messages"] = [Message.user("changed"), *followup.messages[1:]]
        elif mode == "endpoint":
            monkeypatch.setenv("LOP_BEDROCK_ENDPOINT_URL", "http://127.0.0.1:9")
        elif mode == "region":
            key = "bedrock://env-bearer?region=us-west-2"
        elif mode == "scope":
            # Same endpoint and model, different existing replay scope (different
            # authorized selector; no network/credential lookup in this unit rig).
            from local_operator.providers.replay import credential_scope

            replay["credential_scope"] = credential_scope("different-inert-selector", None)
        elif mode == "model":
            options["model"] = MODEL.model_copy(
                update={"model_id": "us.anthropic.claude-sonnet-5-5"}
            )
        changed_request = followup.model_copy(update=options)
        changed_binding = ContextBinding(tracker, measure_request(changed_request))
        changed_request.context_binding = changed_binding
        await _send(client, changed_request, key)

    candidate = wire[-1][2]
    assert len(_native(candidate)) == thinking
    assert (changed_binding.measured.native_tokens > 0) == bool(thinking)
    calibrated = tracker.reconcile(changed_binding.measured, 1.0)
    assert calibrated == ((99, 99) if mode in {"markers", "same"} else None)
    if mode in {"markers", "same"}:
        assert _native(candidate) == _native(positive)
        assert (wire[-1][1] == wire[-2][1]) == (mode == "same")
    if mode not in {"region", "scope", "model"}:
        # For prefix-only splices (not the existing provenance fallback), preserve
        # the actual provider's portable block order, not merely flattened text.
        assert _portable(candidate)[1:] == _portable(positive)[1:]
    assert assistant.provider_payload is not None
    assert any(part["type"] == "redacted_thinking" for part in replay["items"])


@pytest.mark.asyncio
async def test_compaction_commit_and_canonical_rehydration(wire, tmp_path):
    transcript = Transcript(tmp_path / "session")
    old, keep = Message.user("earlier evidence"), Message.user("keep this evidence")
    await transcript.append_message(old)
    await transcript.append_message(keep)
    async with _client(wire, lambda _: _response()) as http:
        client = bedrock.BedrockAnthropicClient(http_client=http)
        produced = await _send(client, _request([old, keep]))
        assistant = Message.assistant("portable output", provider_payload=produced.provider_payload)
        await transcript.append_message(assistant)
        await transcript.append_message(Message.user("next"))
        warm = _default_convert_to_llm(Transcript(transcript.directory).build_llm_history())
        await _send(client, _request(warm))
        assert len(_native(wire[-1][2])) == 1
        await transcript.append_compaction("recap of earlier evidence", keep.id, tokens_before=99)
        restored = Transcript(transcript.directory).build_llm_history()
        messages = _default_convert_to_llm(restored)
        assert any(
            row.id == assistant.id and row.provider_payload == assistant.provider_payload
            for row in messages
        )
        await _send(client, _request(messages))
    sent = wire[-1][2]
    assert not _native(sent)
    assert [part["text"] for _, blocks in _portable(sent) for part in blocks][-3:] == [
        "keep this evidence",
        "portable output",
        "next",
    ]
    assert "recap of earlier evidence" in wire[-1][1].decode()


@pytest.mark.asyncio
async def test_library_discovery_publishes_next_turn_and_keeps_tool_result(
    wire, tmp_path, monkeypatch
):
    from local_operator.analytics import recorder

    analytics = recorder.AnalyticsRecorder()
    monkeypatch.setattr(recorder, "_recorder", analytics)
    async with _client(wire, lambda n: _response(tool=n == 1, tag=f"inert-{n}")) as http:
        client = bedrock.BedrockAnthropicClient(http_client=http)

        def stream(request: ChatRequest, signal: AbortSignal | None):
            return client.stream(request, KEY)

        session = Session(
            model=MODEL,
            stream_fn=stream,
            tools=[build_read_tool(), build_library_tool()],
            transcript=Transcript(tmp_path / "session"),
            system_blocks_provider=lambda: ["stable"],
        )
        try:
            session._tool_deferral = True  # Isolated machine config cannot choose the fixture.
            assert "library" in session.deferred_tool_names()
            await session.prompt("Find the Library tool documentation")
            await session.prompt("Continue")
        finally:
            await session.dispose()
            analytics.close()
    first, warm, changed = [entry[2] for entry in wire]
    assert "library" not in [tool["name"] for tool in first["tools"]]
    assert first["tools"] == warm["tools"]
    assert len(_native(warm)) == 2
    assert "library" in [tool["name"] for tool in changed["tools"]]
    assert not _native(changed)
    result = next(
        block
        for role, blocks in _portable(changed)
        if role == "user"
        for block in blocks
        if block["type"] == "tool_result"
    )
    assert result["tool_use_id"] == "c-read"
    assert "NEXT TURN" in repr(result["content"])
    assert not result.get("is_error")


@pytest.mark.asyncio
async def test_no_commitment_for_partial_or_armed_stream(wire):
    async with _client(wire, lambda n: _response(complete=n != 1)) as http:
        client = bedrock.BedrockAnthropicClient(http_client=http)
        incomplete = await _send(client, _request([Message.user("hi")]))
        armed = _request(
            [
                Message.user("test"),
                Message.assistant(tool_calls=[ToolCall(id="a", name="read", arguments={})]),
                Message.tool_result(
                    ToolResult(
                        tool_call_id="a", tool_name="read", content=[TextContent(text="redacted")]
                    )
                ),
            ]
        )
        armed.arm_one_turn_secrets({"a": SecretStr("fixture-raw-one-turn")})
        terminal = await _send(client, armed)
    assert b"fixture-raw-one-turn" in wire[-1][1]
    for event in (incomplete, terminal):
        assert FIELD not in event.provider_payload["native_replay"]


@pytest.mark.asyncio
async def test_later_thinking_minted_after_splice_survives(wire):
    async with _client(wire, lambda n: _response(tag=f"inert-{n}")) as http:
        client = bedrock.BedrockAnthropicClient(http_client=http)
        user = Message.user("first")
        first = await _send(client, _request([user]))
        stale = Message.assistant("portable output", provider_payload=first.provider_payload)
        next_turn = _request([user, stale, Message.user("second")], tools=[build_library_tool()])
        second = await _send(client, next_turn)
        assert not _native(wire[-1][2])
        fresh = Message.assistant("portable output", provider_payload=second.provider_payload)
        await _send(
            client,
            next_turn.model_copy(
                update={"messages": [*next_turn.messages, fresh, Message.user("third")]}
            ),
        )
    assert [block["signature"] for block in _native(wire[-1][2])] == ["inert-2"]


@pytest.mark.asyncio
async def test_later_block_proved_dependent_on_an_omission(wire):
    async with _client(wire, lambda n: _response(tag=f"inert-{n}")) as http:
        client = bedrock.BedrockAnthropicClient(http_client=http)
        user = Message.user("first")
        first = await _send(client, _request([user]))
        stale = Message.assistant("portable output", provider_payload=first.provider_payload)
        legacy = stale.model_copy(deep=True)
        del legacy.provider_payload["native_replay"][FIELD]
        tools = [build_library_tool()]
        # Intentionally let the no-send oracle mint a continuation on an
        # unproved legacy prefix. That establishes the causal dependency.
        next_user = Message.user("second")
        next_turn = _request([user, legacy, next_user], tools=tools)
        second = await _send(client, next_turn)
        assert [block["signature"] for block in _native(wire[-1][2])] == ["inert-1"]
        dependent = Message.assistant("portable output", provider_payload=second.provider_payload)
        third = Message.user("third")
        await _send(client, _request([user, stale, next_user, dependent, third], tools=tools))
        assert not _native(wire[-1][2])
        # If the earlier native block is already absent, the provider explicitly
        # allows leading-thinking removal by itself. No proved splice here.
        visible = stale.model_copy(update={"provider_payload": None})
        await _send(client, _request([user, visible, next_user, dependent, third], tools=tools))
    assert [block["signature"] for block in _native(wire[-1][2])] == ["inert-2"]


@pytest.mark.asyncio
async def test_proof_captures_final_body_before_awaiting_response(wire):
    tool = build_library_tool()
    original = tool.model_copy(deep=True)

    def answer(count):
        if count == 1:
            # A shared schema can be updated while HTTP is opening; no extra
            # reference to the request's mutable schema is valid proof.
            tool.parameters["description"] = "changed after the send"
        return _response()

    request = _request([Message.user("first")], tools=[tool])
    async with _client(wire, answer) as http:
        client = bedrock.BedrockAnthropicClient(http_client=http)
        produced = await _send(client, request)
        new_messages = [
            *request.messages,
            Message.assistant("portable output", provider_payload=produced.provider_payload),
            Message.user("next"),
        ]
        await _send(
            client, request.model_copy(update={"messages": new_messages, "tools": [original]})
        )
        await _send(client, request.model_copy(update={"messages": new_messages}))
    assert wire[0][2]["tools"] == wire[1][2]["tools"] != wire[2][2]["tools"]
    assert len(_native(wire[1][2])) == 1
    assert not _native(wire[2][2])


@pytest.mark.asyncio
async def test_rejected_image_retry_binds_only_the_successful_wire_attempt(wire):
    def answer(count):
        if count == 1:
            return httpx.Response(
                400,
                headers={"x-amzn-ErrorType": "ValidationException"},
                json={"message": "messages.0.content.1.image.source.base64: invalid image"},
            )
        return _response()

    user = Message.user("look")
    user.content.append(ImageContent(data="aW5lcnQtaW1hZ2UtZml4dHVyZQ=="))
    try:
        async with _client(wire, answer) as http:
            client = bedrock.BedrockAnthropicClient(http_client=http)
            produced = await _send(client, _request([user]))
            assert len(wire) == 2
            assert wire[0][2]["messages"][0]["content"][1]["type"] == "image"
            assert wire[1][2]["messages"][0]["content"][1]["type"] == "text"
            assert FIELD in produced.provider_payload["native_replay"]
            assistant = Message.assistant(
                "portable output", provider_payload=produced.provider_payload
            )
            await _send(client, _request([user, assistant, Message.user("continue")]))
        # The same replacement recurs before the replay; it must match the
        # second attempt, not the failed image-bearing original.
        assert _portable(wire[1][2])[0] == _portable(wire[2][2])[0]
        assert len(_native(wire[2][2])) == 1
    finally:
        forget_rejected_images()
