"""Real CRC-framed AWS events and JSON Converse share one finalizer."""

import asyncio
import base64
import json

import httpx
import pytest

from local_operator.harness.types import (
    StreamEndEvent,
    StreamStartEvent,
    StreamTextDelta,
)
from local_operator.providers.bedrock.converse import (
    BedrockConverseClient,
    _Accumulator,
)
from local_operator.providers.failover import ProviderError

from .eventstream import exception, frame
from .test_converse_body import LLAMA, NOVA, TOOL, request

KEY = "bedrock://env-bearer?region=us-west-2"


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "synthetic-bedrock-token")
    monkeypatch.delenv("LOP_BEDROCK_ENDPOINT_URL", raising=False)


def wire(events):
    return b"".join(
        frame({":message-type": "event", ":event-type": name}, json.dumps(data).encode())
        for name, data in events
    )


def events():
    return [
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "hello"}}),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "end_turn"}),
        ("metadata", {"usage": {"inputTokens": 10, "outputTokens": 2, "cacheReadInputTokens": 30}}),
    ]


async def collect(handler, req=None):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = [
            event
            async for event in BedrockConverseClient(http_client=http).stream(req or request(), KEY)
        ]
        assert not http.is_closed  # injected pool ownership stays with caller
        return result


def test_stream_usage_after_stop_no_invented_id():
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, content=wire(events()))

    result = asyncio.run(collect(handler))
    assert isinstance(result[0], StreamStartEvent)
    assert result[0].response_id is None
    assert isinstance(result[1], StreamTextDelta)
    assert result[1].delta == "hello"
    assert isinstance(result[-1], StreamEndEvent)
    assert result[-1].usage is not None
    assert result[-1].usage.input_tokens == 10 and result[-1].usage.context_tokens == 40
    assert isinstance(result[-1], StreamEndEvent)
    assert result[-1].stop_reason == "stop"
    assert seen[0].url.path.endswith("/converse-stream")
    assert seen[0].headers["authorization"].startswith("Bearer ")


@pytest.mark.parametrize(
    "model,tools,operation",
    [
        (LLAMA, [TOOL], "converse"),
        (LLAMA, [], "converse-stream"),
        (NOVA, [TOOL], "converse-stream"),
    ],
)
def test_operation_chosen_before_post(model, tools, operation):
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if operation == "converse":
            return httpx.Response(
                200,
                json={
                    "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 10, "outputTokens": 2},
                },
            )
        return httpx.Response(200, content=wire(events()))

    result = asyncio.run(collect(handler, request(model, tools=tools)))
    assert seen[0].endswith("/" + operation) and len(seen) == 1
    assert isinstance(result[-1], StreamEndEvent)
    assert result[-1].stop_reason == "stop"


def test_narrow_stream_fallback():
    paths = []

    def handler(req):
        paths.append(req.url.path)
        if req.url.path.endswith("converse-stream"):
            return httpx.Response(
                400,
                headers={"x-amzn-ErrorType": "ValidationException"},
                json={"message": "This model does not support tool use in streaming"},
            )
        return httpx.Response(
            200,
            json={
                "output": {"message": {"role": "assistant", "content": [{"text": "fallback"}]}},
                "stopReason": "end_turn",
            },
        )

    result = asyncio.run(collect(handler, request("future.chat", tools=[TOOL])))
    assert isinstance(result[1], StreamTextDelta)
    assert result[1].delta == "fallback" and len(paths) == 2


@pytest.mark.parametrize(
    "status,kind,message",
    [
        (400, "ValidationException", "bad schema"),
        (401, "UnrecognizedClientException", "bad credentials"),
        (403, "AccessDeniedException", "no access"),
        (429, "ThrottlingException", "rate limit"),
    ],
)
def test_no_capability_retry_for_generic_errors(status, kind, message):
    count = []

    def handler(req):
        count.append(req)
        return httpx.Response(status, headers={"x-amzn-ErrorType": kind}, json={"message": message})

    with pytest.raises(ProviderError):
        asyncio.run(collect(handler))
    assert len(count) == 1


@pytest.mark.parametrize(
    "data",
    [
        wire(events()[:3]),
        wire(events()) + b"partial",
        wire(events()[:2]) + exception("modelStreamErrorException", "late failure"),
        wire(events())[:-1],
        b"not-a-valid-frame",
    ],
)
def test_malformed_or_partial_never_success(data):
    with pytest.raises(Exception):
        asyncio.run(collect(lambda _: httpx.Response(200, content=data)))


def test_late_exception_after_terminal_is_failure():
    with pytest.raises(ProviderError):
        asyncio.run(
            collect(
                lambda _: httpx.Response(
                    200,
                    content=wire(events()) + exception("modelStreamErrorException", "after stop"),
                )
            )
        )


class Split(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data
        self.closed = False

    async def __aiter__(self):
        for byte in self.data:
            yield bytes([byte])

    async def aclose(self):
        self.closed = True


def test_arbitrary_byte_splits_and_cleanup():
    stream = Split(wire(events()))
    result = asyncio.run(collect(lambda _: httpx.Response(200, stream=stream)))
    assert isinstance(result[-1], StreamEndEvent)
    assert result[-1].stop_reason == "stop" and stream.closed


def test_tool_json_reasoning_and_redacted_fragments():
    acc = _Accumulator(100)
    output = []
    source = [
        ("messageStart", {"role": "assistant"}),
        (
            "contentBlockStart",
            {"contentBlockIndex": 3, "start": {"toolUse": {"toolUseId": "a", "name": "lookup"}}},
        ),
        ("contentBlockDelta", {"contentBlockIndex": 3, "delta": {"toolUse": {"input": '{"x":'}}}),
        (
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "why"}}},
        ),
        (
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}},
        ),
        (
            "contentBlockDelta",
            {"contentBlockIndex": 1, "delta": {"reasoningContent": {"redactedContent": "YQ=="}}},
        ),
        (
            "contentBlockDelta",
            {"contentBlockIndex": 1, "delta": {"reasoningContent": {"redactedContent": "Yg=="}}},
        ),
        ("contentBlockDelta", {"contentBlockIndex": 3, "delta": {"toolUse": {"input": "1}"}}}),
        *[("contentBlockStop", {"contentBlockIndex": index}) for index in (0, 1, 3)],
        ("messageStop", {"stopReason": "tool_use"}),
    ]
    for name, data in source:
        output.extend(acc.event(name, data))
    end = acc.end(request(), "endpoint", "scope", "proof")
    assert end.provider_payload is not None
    native = end.provider_payload["native_replay"]["items"]
    assert native[0]["reasoningContent"]["reasoningText"] == {"text": "why", "signature": "sig"}
    assert base64.b64decode(native[1]["reasoningContent"]["redactedContent"]) == b"ab"
    assert native[2]["toolUse"]["input"] == {"x": 1}
    assert [event.delta for event in output if event.type == "reasoning_delta"] == ["why"]
    assert not any(getattr(event, "delta", "") == "sig" for event in output)


def test_invalid_generated_args_withhold_entire_native_turn():
    acc = _Accumulator(100)
    for name, data in [
        ("messageStart", {"role": "assistant"}),
        (
            "contentBlockStart",
            {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "a", "name": "lookup"}}},
        ),
        (
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"toolUse": {"input": "not json"}}},
        ),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "tool_use"}),
    ]:
        acc.event(name, data)
    payload = acc.end(request(), "endpoint", "scope", "proof").provider_payload
    assert payload is not None and "native_replay" not in payload


def test_output_envelope_covers_hidden_signature():
    acc = _Accumulator(1)
    acc.event("messageStart", {"role": "assistant"})
    with pytest.raises(ProviderError, match="envelope"):
        acc.event(
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "x" * 129}}},
        )
