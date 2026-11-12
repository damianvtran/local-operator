"""A provider refusing ONE image replaces it with a note and retries once.

Driven through the real :class:`OpenAICompatClient` (the DeepSeek route) and
:class:`AnthropicClient` with an ``httpx.MockTransport``, so the body the
provider indexes is the body the client actually built. The refusal wording is
DeepSeek's verbatim from the 2026-11-11 episode that motivated this
(``.messages[139].image[0]: You have uploaded an unsupported image...``); the
index is rewritten to the fixture's own wire position.
"""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from local_operator.harness.types import (
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    TextContent,
    ToolCall,
)
from local_operator.model import configure
from local_operator.providers.clients import AnthropicClient, OpenAICompatClient
from local_operator.providers.failover import ProviderError
from local_operator.providers.image_rejection import (
    REJECTED_IMAGE_PLACEHOLDER,
    forget_rejected_images,
)

DEEPSEEK_REFUSAL = (
    ".messages[{index}].image[0]: You have uploaded an unsupported image. Please make "
    "sure your image is valid and has one of the following formats: webp, png, jpeg, "
    "and gif."
)


def _png(shade: int) -> str:
    buffer = io.BytesIO()
    Image.new("L", (40, 20), shade).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


GOOD = _png(10)
BAD = _png(200)  # a VALID png: the provider's refusal, not local damage, picks it


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    configure.invalidate_model_info_cache()
    forget_rejected_images()
    yield
    forget_rejected_images()
    configure.invalidate_model_info_cache()


def _spec(provider: str = "deepseek", model_id: str = "deepseek-flash") -> ModelSpec:
    return ModelSpec(
        provider=provider,
        model_id=model_id,
        context_window=1_000_000,
        max_output_tokens=8192,
        supports_images=True,
    )


def _history() -> list[Message]:
    """Two tool reads, each returning one image, then a user turn."""
    calls = [
        ToolCall(id="a", name="read", arguments={}),
        ToolCall(id="b", name="read", arguments={}),
    ]
    out = [Message.user("look"), Message.assistant(tool_calls=calls[:1])]
    out.append(
        Message(
            role="tool",
            tool_call_id="a",
            tool_name="read",
            content=[TextContent(text="Image a.png"), ImageContent(data=GOOD)],
        )
    )
    out.append(Message.assistant(tool_calls=calls[1:]))
    out.append(
        Message(
            role="tool",
            tool_call_id="b",
            tool_name="read",
            content=[TextContent(text="Image b.png"), ImageContent(data=BAD)],
        )
    )
    out.append(Message.user("and now?"))
    return out


def _sse_ok() -> bytes:
    chunk = {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def _wire_images(body: dict) -> list[str]:
    found = []
    for message in body["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            found += [
                part["image_url"]["url"].split(",", 1)[1]
                for part in content
                if part.get("type") == "image_url"
            ]
    return found


def _index_of(body: dict, payload: str) -> int:
    for index, message in enumerate(body["messages"]):
        content = message.get("content")
        if isinstance(content, list) and any(
            part.get("type") == "image_url" and part["image_url"]["url"].endswith(payload)
            for part in content
        ):
            return index
    raise AssertionError("payload not on the wire")


class _DeepSeek:
    """Refuses any request carrying ``BAD``, naming it by its wire index."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if BAD in _wire_images(body):
            message = DEEPSEEK_REFUSAL.format(index=_index_of(body, BAD))
            return httpx.Response(400, json={"error": {"message": message}})
        return httpx.Response(200, content=_sse_ok())


async def _run(client, request) -> list:
    return [event async for event in client.stream(request, "fixture")]


@pytest.mark.asyncio
async def test_deepseek_refusal_replaces_the_named_image_and_retries_once():
    server = _DeepSeek()
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        request = ChatRequest(model=_spec(), messages=_history())

        events = await _run(client, request)

    assert any(isinstance(event, StreamEndEvent) for event in events)
    assert len(server.bodies) == 2
    retry = server.bodies[1]
    # Only the refused image went; the other one is still sent.
    assert _wire_images(retry) == [GOOD]
    assert REJECTED_IMAGE_PLACEHOLDER in json.dumps(retry)
    # Everything before the refused image's tool result is byte-identical, so
    # the provider's prompt cache still covers that prefix on the retry. (The
    # tool result itself changes: DeepSeek flattens a tool message's text,
    # and the note is now part of it.)
    cut = _index_of(server.bodies[0], BAD) - 1
    assert retry["messages"][:cut] == server.bodies[0]["messages"][:cut]
    assert REJECTED_IMAGE_PLACEHOLDER in retry["messages"][cut]["content"]


@pytest.mark.asyncio
async def test_the_replacement_is_stable_for_later_requests():
    server = _DeepSeek()
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        await _run(client, ChatRequest(model=_spec(), messages=_history()))
        later = _history() + [Message.assistant("ok"), Message.user("next")]

        await _run(client, ChatRequest(model=_spec(), messages=later))

    # The later request goes out ONCE, already carrying the note, and the
    # note renders byte-identically to the retry's.
    assert len(server.bodies) == 3
    assert _wire_images(server.bodies[2]) == [GOOD]
    cut = len(server.bodies[1]["messages"]) - 1
    assert server.bodies[2]["messages"][:cut] == server.bodies[1]["messages"][:cut]


@pytest.mark.asyncio
async def test_a_refusal_is_retried_only_once():
    calls = []

    def always_refuse(request: httpx.Request) -> httpx.Response:
        # Refuses whichever image is first on the wire, every time: the retry
        # (GOOD replaced) is refused again, naming BAD, and must surface.
        calls.append(request)
        body = json.loads(request.content)
        index = _index_of(body, _wire_images(body)[0])
        return httpx.Response(
            400, json={"error": {"message": DEEPSEEK_REFUSAL.format(index=index)}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(always_refuse)) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        with pytest.raises(ProviderError, match="unsupported image"):
            await _run(client, ChatRequest(model=_spec(), messages=_history()))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_an_unrelated_400_is_not_retried():
    calls = []

    def refuse(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "invalid tool schema"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        with pytest.raises(ProviderError, match="invalid tool schema"):
            await _run(client, ChatRequest(model=_spec(), messages=_history()))
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_an_unresolvable_path_with_several_valid_images_re_raises():
    """No index to follow and nothing locally damaged: guessing would drop the
    wrong image, so the error surfaces and the session's degrade owns it."""
    calls = []

    def refuse(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "unsupported image"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        with pytest.raises(ProviderError):
            await _run(client, ChatRequest(model=_spec(), messages=_history()))
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_anthropic_path_syntax_names_the_image_too():
    bodies = []

    def serve(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        text = json.dumps(body)
        if BAD in text:
            # Locate BAD by the dotted path Anthropic uses.
            for m_index, message in enumerate(body["messages"]):
                for c_index, part in enumerate(message["content"]):
                    for i_index, inner in enumerate(part.get("content") or []):
                        if isinstance(inner, dict) and inner.get("source", {}).get("data") == BAD:
                            path = f"messages.{m_index}.content.{c_index}.content.{i_index}"
                            return httpx.Response(
                                400,
                                json={
                                    "type": "error",
                                    "error": {
                                        "type": "invalid_request_error",
                                        "message": f"{path}.image.source.base64: invalid image",
                                    },
                                },
                            )
            raise AssertionError("BAD not where the fixture expects it")
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        payload = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, content=payload.encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
        client = AnthropicClient("https://api.anthropic.com", http_client=http)
        request = ChatRequest(model=_spec("anthropic", "claude-sonnet-4-5"), messages=_history())
        events = await _run(client, request)

    assert any(isinstance(event, StreamEndEvent) for event in events)
    assert len(bodies) == 2
    assert BAD not in json.dumps(bodies[1]) and GOOD in json.dumps(bodies[1])
    assert REJECTED_IMAGE_PLACEHOLDER in json.dumps(bodies[1])


@pytest.mark.asyncio
async def test_without_a_path_a_locally_damaged_png_is_the_culprit():
    """A provider that names no index still gets the damaged PNG replaced:
    the local structure check finds it, and the valid image is kept."""
    raw = base64.b64decode(BAD)
    damaged = base64.b64encode(raw[:-12]).decode("ascii")  # IEND stripped
    history = _history()
    history[4] = history[4].model_copy(
        update={"content": [TextContent(text="Image b.png"), ImageContent(data=damaged)]}
    )
    bodies = []

    def serve(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if damaged in _wire_images(body):
            return httpx.Response(400, json={"error": {"message": "unsupported image"}})
        return httpx.Response(200, content=_sse_ok())

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        await _run(client, ChatRequest(model=_spec(), messages=history))
    assert len(bodies) == 2 and _wire_images(bodies[1]) == [GOOD]
