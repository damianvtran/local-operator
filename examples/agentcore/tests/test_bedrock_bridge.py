"""The OpenAI <-> Bedrock Converse translation, and the bridge's real HTTP path.

The HTTP tests start the actual ``ThreadingHTTPServer`` with an injected fake
``converse_stream`` so the request parsing, SSE framing and error mapping are
exercised end to end without AWS.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterator

import pytest
from bedrock_bridge import (
    StreamError,
    collect_completion,
    error_status,
    start_bridge,
    stream_chunks,
    to_converse,
)
from botocore.exceptions import ClientError

MODEL = "example.provider.model-v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "add two ints",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
        },
    }
]


def test_system_user_and_inference_settings_translate() -> None:
    kwargs = to_converse(
        {
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "developer", "content": [{"type": "text", "text": "and kind"}]},
                {"role": "user", "content": "hi"},
            ],
            "max_completion_tokens": 64,
            "temperature": 0.2,
            "stop": "END",
        },
        MODEL,
    )
    assert kwargs["modelId"] == MODEL
    assert kwargs["system"] == [{"text": "be brief"}, {"text": "and kind"}]
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert kwargs["inferenceConfig"] == {
        "maxTokens": 64,
        "temperature": 0.2,
        "stopSequences": ["END"],
    }
    assert "toolConfig" not in kwargs


def test_tool_round_trip_history_merges_into_alternating_turns() -> None:
    """Parallel tool results become ONE user turn: Converse requires alternation."""
    kwargs = to_converse(
        {
            "messages": [
                {"role": "user", "content": "add twice"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "add", "arguments": '{"a": 1}'}},
                        {"id": "c2", "function": {"name": "add", "arguments": "not json"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "1"},
                {"role": "tool", "tool_call_id": "c2", "content": ""},
                {"role": "user", "content": "thanks"},
            ],
            "tools": TOOLS,
            "tool_choice": "required",
        },
        MODEL,
    )
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles == ["user", "assistant", "user"]
    assistant = kwargs["messages"][1]["content"]
    # Empty assistant text is dropped (Converse rejects blank text blocks).
    assert [b["toolUse"]["toolUseId"] for b in assistant] == ["c1", "c2"]
    assert assistant[0]["toolUse"]["input"] == {"a": 1}
    assert assistant[1]["toolUse"]["input"] == {"_raw_arguments": "not json"}
    results = kwargs["messages"][2]["content"]
    assert results[0]["toolResult"]["content"] == [{"text": "1"}]
    assert results[1]["toolResult"]["content"] == [{"text": "(no output)"}]
    assert results[2] == {"text": "thanks"}
    spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "add" and spec["inputSchema"]["json"]["type"] == "object"
    assert kwargs["toolConfig"]["toolChoice"] == {"any": {}}


def test_tool_choice_none_sends_no_tool_config_so_no_tool_can_be_called() -> None:
    """R3: Converse has no "none" choice; omitting toolConfig is what forbids calls."""
    kwargs = to_converse(
        {
            "messages": [{"role": "user", "content": "just talk"}],
            "tools": TOOLS,
            "tool_choice": "none",
        },
        MODEL,
    )
    assert "toolConfig" not in kwargs


def test_tool_choice_none_after_a_tool_round_keeps_specs_converse_requires() -> None:
    """With toolUse/toolResult in history Converse rejects a missing toolConfig."""
    kwargs = to_converse(
        {
            "messages": [
                {"role": "user", "content": "add"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "add", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "1"},
            ],
            "tools": TOOLS,
            "tool_choice": "none",
        },
        MODEL,
    )
    assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"] == "add"
    assert "toolChoice" not in kwargs["toolConfig"]


def test_data_uri_images_translate_and_remote_urls_are_not_fetched() -> None:
    kwargs = to_converse(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                    ],
                }
            ]
        },
        MODEL,
    )
    blocks = kwargs["messages"][0]["content"]
    assert blocks[0] == {"text": "what is this"}
    assert blocks[1] == {"image": {"format": "png", "source": {"bytes": b"hi"}}}
    assert len(blocks) == 2


def converse_events_with_tool() -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Adding."}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "t1", "name": "add"}},
            }
        },
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"a"'}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": ": 2}"}}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}},
    ]


def test_stream_chunks_number_tool_calls_from_zero_and_finish_after_usage() -> None:
    chunks = list(stream_chunks(converse_events_with_tool(), MODEL))
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    calls = [
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if c["choices"] and "tool_calls" in c["choices"][0]["delta"]
    ]
    # Converse's contentBlockIndex is 1 (text came first); OpenAI's index is 0.
    assert {call["index"] for call in calls} == {0}
    assert calls[0]["id"] == "t1" and calls[0]["function"]["name"] == "add"
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_collect_completion_folds_a_stream_into_one_message() -> None:
    body = collect_completion(stream_chunks(converse_events_with_tool(), MODEL), MODEL)
    message = body["choices"][0]["message"]
    assert message["content"] == "Adding."
    assert message["tool_calls"] == [
        {"id": "t1", "type": "function", "function": {"name": "add", "arguments": '{"a": 2}'}}
    ]
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["usage"]["total_tokens"] == 15


def test_mid_stream_exception_events_raise() -> None:
    events = [{"throttlingException": {"message": "slow down"}}]
    with pytest.raises(StreamError, match="slow down"):
        list(stream_chunks(events, MODEL))


def client_error(code: str, message: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "ConverseStream")


def test_error_status_maps_access_denied_to_403() -> None:
    assert error_status(client_error("AccessDeniedException", "nope")) == (
        403,
        "AccessDeniedException",
        "nope",
    )
    assert error_status(client_error("SomethingNew", "x"))[0] == 502


# ---------------------------------------------------------------------------
# Real HTTP path with a fake Bedrock
# ---------------------------------------------------------------------------


class FakeBedrock:
    def __init__(self, events: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self.events = events or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"stream": iter(self.events)}


@pytest.fixture
def bridge() -> Iterator[Any]:
    servers: list[Any] = []

    def start(fake: FakeBedrock) -> str:
        server = start_bridge(MODEL, converse_stream=fake)
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/v1"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def post(url: str, body: dict[str, Any]) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_models_endpoint_lists_the_configured_model(bridge: Any) -> None:
    base = bridge(FakeBedrock())
    with urllib.request.urlopen(base + "/models", timeout=10) as response:
        assert json.load(response)["data"][0]["id"] == MODEL


def test_streaming_request_returns_sse_ending_in_done(bridge: Any) -> None:
    fake = FakeBedrock(converse_events_with_tool())
    status, raw = post(
        bridge(fake) + "/chat/completions",
        {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "go"}]},
    )
    assert status == 200
    frames = [line[6:] for line in raw.decode().split("\n") if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    assert json.loads(frames[1])["choices"][0]["delta"] == {"content": "Adding."}
    assert fake.calls[0]["modelId"] == MODEL


def test_access_denied_surfaces_as_openai_error_403(bridge: Any) -> None:
    fake = FakeBedrock(error=client_error("AccessDeniedException", "not authorized"))
    status, raw = post(
        bridge(fake) + "/chat/completions",
        {"model": "other.model", "messages": [{"role": "user", "content": "go"}]},
    )
    assert status == 403
    assert json.loads(raw)["error"] == {
        "message": "not authorized",
        "type": "AccessDeniedException",
        "code": "AccessDeniedException",
    }
    # The requested model is forwarded, so the IAM policy is what decides.
    assert fake.calls[0]["modelId"] == "other.model"


def test_malformed_json_is_a_400_not_a_crash(bridge: Any) -> None:
    base = bridge(FakeBedrock())
    request = urllib.request.Request(base + "/chat/completions", data=b"{not json")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)
    assert caught.value.code == 400


def test_botocore_error_raised_mid_stream_becomes_an_error_frame(bridge: Any) -> None:
    """boto3 raises EventStreamError from the iterator itself; never drop the socket."""

    def failing_stream() -> Iterator[dict[str, Any]]:
        yield {"messageStart": {"role": "assistant"}}
        raise client_error("ThrottlingException", "slow down")

    class Fake(FakeBedrock):
        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {"stream": failing_stream()}

    status, raw = post(
        bridge(Fake()) + "/chat/completions",
        {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "go"}]},
    )
    assert status == 200
    frames = [line[6:] for line in raw.decode().split("\n") if line.startswith("data: ")]
    assert json.loads(frames[-2])["error"]["code"] == "ThrottlingException"
    assert frames[-1] == "[DONE]"
