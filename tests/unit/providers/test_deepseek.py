"""Native DeepSeek contract: live inventory, explicit metadata and stable replay.

The real /models response captured 2026-09-12 contains ids/object/owner ONLY.
Richer fixtures exercise precedence if that endpoint later supplies recognized
compatibility fields, not a claim that it advertises those capabilities today.
"""

import asyncio
import json
from dataclasses import asdict

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    TextContent,
    ToolCall,
)
from local_operator.model import configure, discovery
from local_operator.model.registry import deepseek_models
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.failover import ProviderError
from local_operator.providers.replay import credential_scope


@pytest.fixture(autouse=True)
def isolated_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    configure.invalidate_model_info_cache()
    yield
    configure.invalidate_model_info_cache()


def spec(effort="high"):
    return ModelSpec(
        provider="deepseek",
        model_id="deepseek-flash",
        context_window=1_000_000,
        max_output_tokens=393_216,
        supports_images=True,
        supports_prompt_cache=True,
        reasoning_efforts=("none", "low", "high", "max"),
        reasoning_effort=effort,
    )


def request(messages=None, effort="high", **kwargs):
    return ChatRequest(
        model=spec(effort),
        messages=messages or [Message.user("hello")],
        system_blocks=["Stable system", "Stable context"],
        prompt_cache_key="synthetic-deepseek",
        **kwargs,
    )


def sse(delta=None, finish=None):
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]})
        + "\n\n"
    )


def test_authoritative_inventory_cache_and_offline(tmp_path):
    calls = []

    def serve(req):
        calls.append(req.url.path)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "deepseek-flash", "object": "model", "owned_by": "deepseek"},
                    {"id": "deepseek-v4-pro", "object": "model", "owned_by": "deepseek"},
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        rows, status = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path / "live"
        )
        assert status == "ok"
        assert {r.id for r in rows} == {"deepseek-flash", "deepseek-v4-pro"}
        assert next(r for r in rows if r.id == "deepseek-flash").supports_images is True
        _, status = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path / "live"
        )
        assert status == "cached"
        assert len(calls) == 1
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client:
        rows, status = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path / "cold"
        )
        assert status == "static"
        assert {r.id for r in rows} == set(deepseek_models)


def test_successful_empty_inventory_does_not_resurrect_static_models(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": []}))
    ) as client:
        rows, status = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path
        )
    assert rows == []
    assert status == "empty"


@pytest.mark.parametrize("value", [True, False, None])
def test_live_capabilities_override_complete_rows(monkeypatch, value):
    payload = {"id": "deepseek-flash"}
    if value is not None:
        payload.update(
            supports_images=value,
            supports_tools=value,
            reasoning=value,
            supports_prompt_cache=value,
        )
    row = discovery._row_from_openai_entry(payload, "deepseek")
    assert row is not None
    rows = discovery.merge_models(deepseek_models, [row], include_static_only=False)
    monkeypatch.setattr(discovery, "available_models", lambda *a, **k: (rows, "ok"))
    cfg = configure.configure_model("deepseek", "deepseek-flash")
    expected = value is not False
    assert cfg.spec.supports_images is expected
    assert cfg.spec.supports_tools is expected
    assert cfg.spec.reasoning is expected
    assert cfg.spec.supports_prompt_cache is expected
    assert cfg.spec.reasoning_efforts == (() if value is False else ("none", "low", "high", "max"))


def test_live_limits_zero_prices_and_empty_effort_survive_cache(monkeypatch, tmp_path):
    payload = {
        "id": "deepseek-flash",
        "context_length": 200_000,
        "max_output_tokens": 4096,
        "pricing": {
            "prompt": "0",
            "completion": "0",
            "input_cache_read": "0",
            "input_cache_write": "0",
        },
        "reasoning": {"supported_efforts": []},
        "supports_prompt_cache": False,
    }
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": [payload]}))
    ) as client:
        first, _ = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path / "listing"
        )
        rows, status = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path / "listing"
        )
    assert status == "cached"
    assert asdict(rows[0]) == asdict(first[0])
    monkeypatch.setattr(discovery, "available_models", lambda *a, **k: (rows, "cached"))
    cfg = configure.configure_model("deepseek", "deepseek-flash")
    assert cfg.spec.context_window == 200_000
    assert cfg.spec.max_output_tokens == 4096
    assert cfg.info.input_price == cfg.info.output_price == cfg.info.cache_reads_price == 0
    assert cfg.spec.reasoning_efforts == ()
    assert cfg.spec.reasoning_effort is None
    assert cfg.spec.supports_prompt_cache is False


@pytest.mark.parametrize(
    "model,images",
    [("deepseek-flash", True), ("deepseek-v4-pro", False), ("deepseek-v4-flash", True)],
)
def test_documented_fallbacks_and_provider_effort(monkeypatch, model, images):
    monkeypatch.setattr(discovery, "available_models", lambda *a, **k: ([], "static"))
    cfg = configure.configure_model("deepseek", model)
    assert cfg.spec.context_window == 1_000_000
    assert cfg.spec.max_output_tokens == 393_216
    assert cfg.spec.supports_images is images
    assert cfg.spec.reasoning_efforts == ("none", "low", "high", "max")
    assert cfg.spec.reasoning_effort == "high"
    assert "peak price estimate" in cfg.info.description
    other = configure.build_model_spec(
        "openrouter", "deepseek/deepseek-flash", info=deepseek_models[model]
    )
    assert other.reasoning_efforts != cfg.spec.reasoning_efforts


@pytest.mark.parametrize(
    "effort,budget", [("none", 8192), ("low", 65536), ("high", 65536), ("max", 131072)]
)
def test_native_effort_and_budget(effort, budget):
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    body = client._build_body(request(effort=effort))
    assert body["thinking"] == {"type": "disabled" if effort == "none" else "enabled"}
    assert body.get("reasoning_effort") == (None if effort == "none" else effort)
    assert body["max_tokens"] == budget
    assert client._build_body(request(effort=effort, max_tokens=24))["max_tokens"] == 24
    assert client._build_body(request(effort=effort, max_tokens=500_000))["max_tokens"] == 393_216


def test_implicit_cache_prefix_is_unchanged_and_tool_images_are_user_only():
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    history = [Message.user("first"), Message.assistant("first answer"), Message.user("second")]
    first = client._build_body(request(history))
    calls = [
        ToolCall(id="a", name="inspect", arguments={}),
        ToolCall(id="b", name="inspect", arguments={}),
    ]
    history += [Message.assistant(tool_calls=calls)]
    for call in calls:
        history.append(
            Message(
                role="tool",
                tool_call_id=call.id,
                content=[TextContent(text="found"), ImageContent(data="YWJj")],
            )
        )
    second = client._build_body(request(history))
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert [m["role"] for m in second["messages"][-4:]] == ["assistant", "tool", "tool", "user"]
    assert all(isinstance(m["content"], str) for m in second["messages"] if m["role"] == "tool")
    assert len([b for b in second["messages"][-1]["content"] if b["type"] == "image_url"]) == 2
    history += [Message.assistant("seen"), Message.user("third")]
    third = client._build_body(request(history))
    assert third["messages"][: len(second["messages"])] == second["messages"]
    assert "cache_control" not in json.dumps(third)
    assert "prompt_cache_key" not in third
    assert all(isinstance(m["content"], str) for m in third["messages"] if m["role"] == "system")


@pytest.mark.asyncio
async def test_native_reasoning_only_replay_is_scoped():
    response = (
        sse({"reasoning_content": "opaque native thought"})
        + sse(finish="stop")
        + "data: [DONE]\n\n"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=response))
    ) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        events = [e async for e in client.stream(request(), "fixture")]
        end = next(e for e in events if isinstance(e, StreamEndEvent))
        native = Message.assistant(provider_payload=end.provider_payload)
        native = Message.model_validate_json(native.model_dump_json())
        req = request([Message.user("hello"), native, Message.user("continue")])
        body = client._build_body(req, scope=credential_scope("fixture", None))
        assert body["messages"][3]["reasoning_content"] == "opaque native thought"
        changed = client._build_body(req, scope="other-account")
        assert not any(m["role"] == "assistant" for m in changed["messages"])
        req.model = req.model.model_copy(update={"model_id": "deepseek-v4-pro"})
        assert not any(
            m["role"] == "assistant"
            for m in client._build_body(req, scope=credential_scope("fixture", None))["messages"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "",
        sse({"content": "partial"}, "stop"),
        sse({"content": "partial"}) + "data: [DONE]\n\n",
        "data: {bad}\n\n",
        "data: []\n\n",
        sse(finish="unknown") + "data: [DONE]\n\n",
        sse(finish="insufficient_system_resource") + "data: [DONE]\n\n",
    ],
)
async def test_invalid_native_terminal_is_retryable_failure(payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))
    ) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        with pytest.raises(ProviderError) as caught:
            async for event in client.stream(request(), "fixture"):
                assert not isinstance(event, StreamEndEvent)
        assert caught.value.retryable


@pytest.mark.asyncio
async def test_cancellation_does_not_become_truncated_stream_failure():
    class CancelStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse({"content": "partial"}).encode()
            raise asyncio.CancelledError

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=CancelStream()))
    ) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        with pytest.raises(asyncio.CancelledError):
            async for event in client.stream(request(), "fixture"):
                assert not isinstance(event, StreamEndEvent)


@pytest.mark.asyncio
async def test_other_compat_routes_keep_terminal_tolerance():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=sse({"content": "ok"}, "stop"))
        )
    ) as http:
        client = OpenAICompatClient("https://other.invalid/v1", http_client=http)
        req = request()
        req.model = req.model.model_copy(update={"provider": "openrouter"})
        events = [e async for e in client.stream(req, "fixture")]
        assert isinstance(events[-1], StreamEndEvent)
