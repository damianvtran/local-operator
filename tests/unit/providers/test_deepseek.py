"""Native DeepSeek contract: live inventory, explicit metadata and stable replay.

The real /models response captured 2026-09-12 contains ids/object/owner ONLY.
Richer fixtures exercise precedence if that endpoint later supplies recognized
compatibility fields, not a claim that it advertises those capabilities today.
"""

import asyncio
import base64
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
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCall,
)
from local_operator.model import catalogue, configure, discovery
from local_operator.model.registry import deepseek_models
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.context import (
    ContextBinding,
    ContextTokenTracker,
    measure_request,
)
from local_operator.providers.controller import ProviderController
from local_operator.providers.failover import ProviderError
from local_operator.providers.replay import (
    REASONING_ECHO_PLACEHOLDER,
    credential_scope,
    native_payload,
)

#: The endpoint the chat body builder derives from the client's own base URL,
#: spelled out because ``native_payload`` records it in the replay provenance
#: and a mismatch there silently drops the recorded reasoning.
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

#: A minimal PNG payload for the frame turns above, so the frame prune has
#: something to drop. Base64 of the same 8-byte header the other suites use.
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode("ascii")


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


def echo_spec(effort="high"):
    """The same route WITH the capability ``build_model_spec`` derives for it.

    Set by hand rather than resolved through ``build_model_spec`` so the wire
    contract can be pinned without a model catalogue, a cache directory or a
    live listing; the derivation itself is pinned in ``tests/unit/model``.
    """
    return spec(effort).model_copy(update={"requires_reasoning_echo": True})


def native_turn(*, text="", reasoning=None, tool_calls=(), scope=None, model=None):
    """An assistant turn carrying real recorded native state, as resume loads it.

    The fingerprint and provenance fields are computed exactly as the wire
    parser writes them, so a mismatch in the fixture would drop the reasoning
    for real instead of failing an assertion here.
    """
    active = model or echo_spec()
    scope = scope if scope is not None else credential_scope("fixture")
    calls = [{"id": call.id, "name": call.name, "args": call.arguments} for call in tool_calls]
    items = [{"reasoning_content": reasoning}] if reasoning is not None else []
    return Message.assistant(
        text,
        tool_calls=list(tool_calls),
        provider_payload=native_payload(
            active, DEEPSEEK_ENDPOINT, "openai-chat", items, text, calls, scope
        ),
    )


def assistant_entries(body):
    return [m for m in body["messages"] if m.get("role") == "assistant"]


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
    """The ladder is the ask for a request that names none, and the provider's
    own published defaults are the point of it.

    ``none/low/high/max`` map to DeepSeek's server-side 8K/64K/64K/128K output
    defaults. A request that names no ask now carries the harness's policy bound,
    and this ladder is what that bound sits ABOVE -- it is a provider-native ask,
    not a second policy, so it stays in force for exactly the request that named
    nothing. An earlier revision of this test accepted the opposite: because the
    contract fills ``max_tokens``, its ``request.max_tokens or ladder`` spelling
    took the left branch on every rung, so all four asked the same number and
    ``max`` silently stopped buying its 128K on a shipped provider (review m1 /
    QA round 1, Q3). The ladder is not dead, and the "escape hatch" that used to
    be offered for reaching it (``max_tokens=0``) is gone on purpose: it did not
    mean "no cap" on any wire that requires one, and on a capped model it fell
    back to the advertised capability.

    An explicit small ask and an explicit ask above the advertised maximum behave
    as before either way.
    """
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


@pytest.mark.parametrize(
    "bad", [[{}], [None], [{"name": "missing id"}], [None, {"name": "missing id"}]]
)
def test_malformed_native_inventory_uses_stale_or_static(tmp_path, bad):
    payload = {"data": [{"id": "deepseek-flash"}]}
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        first, _ = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path / "warm"
        )
        payload = {"data": bad}
        warm, status = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path / "warm", ttl_s=-1
        )
        assert status == "stale"
        assert [row.id for row in warm] == [row.id for row in first] == ["deepseek-flash"]
        cold, status = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path / "cold"
        )
        assert status == "static"
        assert {row.id for row in cold} == set(deepseek_models)


def test_untrusted_historic_empty_cache_is_refetched_only_for_deepseek(tmp_path):
    # Exact old capture reproduced from data:[null]: no raw provenance remains.
    key = discovery._cache_key("deepseek")
    catalogue.cached_listing(key, lambda: {"capture": 1, "models": []}, cache_dir=tmp_path)
    calls = []

    def serve(req):
        calls.append(req.url.path)
        return httpx.Response(200, json={"data": [{"id": "deepseek-flash"}]})

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        rows, status = discovery.available_models(
            "deepseek", api_key="fixture", client=client, cache_dir=tmp_path
        )
    assert status == "ok"
    assert [row.id for row in rows] == ["deepseek-flash"]
    assert calls == ["/v1/models"]
    stored = catalogue.peek_listing(key, cache_dir=tmp_path).payload
    assert stored is not None
    assert stored["capture"] == 2
    # The same historical stamp remains valid for an untouched native route.
    catalogue.cached_listing(
        discovery._cache_key("mistral"),
        lambda: {"capture": 1, "models": [{"id": "qa-mistral"}]},
        cache_dir=tmp_path,
    )
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        other, status = discovery.available_models(
            "mistral", api_key="fixture", client=client, cache_dir=tmp_path
        )
    assert status == "cached"
    assert "qa-mistral" in {row.id for row in other}
    assert calls == ["/v1/models"]


@pytest.mark.asyncio
async def test_native_sse_assembles_multiline_bom_crlf_and_split_utf8():
    payload = (
        "\ufeff: heartbeat\r\nevent: message\r\nid: 7\r\n"
        'data: {"choices": [\r\n'
        'data: {"index": 0, "delta": {"content": "café"}, "finish_reason": null}]}\r\n\r\n'
        + sse(finish="stop")
        + "data: [DONE]\r\n\r\n"
    ).encode()

    class SplitBytes(httpx.AsyncByteStream):
        async def __aiter__(self):
            # Every possible byte boundary, including UTF-8 and CR/LF pairs.
            for byte in payload:
                yield bytes([byte])

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=SplitBytes()))
    ) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        events = [event async for event in client.stream(request(), "fixture")]
    assert "".join(e.delta for e in events if isinstance(e, StreamTextDelta)) == "café"
    assert isinstance(events[-1], StreamEndEvent)
    assert events[-1].stop_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["data: [DONE]", "data: [DONE]\n", "data: [DONE]\r\n"])
async def test_native_done_requires_blank_event_terminator(tail):
    payload = sse({"content": "answer"}, "stop") + tail
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))
    ) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        with pytest.raises(ProviderError, match="before \\[DONE\\]"):
            async for event in client.stream(request(), "fixture"):
                assert not isinstance(event, StreamEndEvent)


@pytest.mark.asyncio
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
async def test_native_sse_rejects_nonstandard_json_constants(constant):
    payload = (
        'data: {"unused": '
        + constant
        + ', "choices": []}\n\n'
        + sse(finish="stop")
        + "data: [DONE]\n\n"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))
    ) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        with pytest.raises(ProviderError) as caught:
            async for event in client.stream(request(), "fixture"):
                assert not isinstance(event, StreamEndEvent)
        assert caught.value.retryable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delta,finish",
    [
        ({"content": "late text"}, None),
        ({"reasoning_content": "late thought"}, None),
        (
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "late",
                        "type": "function",
                        "function": {"name": "write", "arguments": "{}"},
                    }
                ]
            },
            "tool_calls",
        ),
        ({}, "length"),
    ],
)
async def test_native_finish_seals_text_reasoning_and_executable_tools(delta, finish):
    payload = sse({"content": "safe"}, "stop") + sse(delta, finish) + "data: [DONE]\n\n"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))
    ) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        with pytest.raises(ProviderError, match="after its finish reason") as caught:
            async for event in client.stream(request(), "fixture"):
                assert not isinstance(event, (StreamEndEvent, StreamToolCallDelta))
                if isinstance(event, StreamTextDelta):
                    assert event.delta == "safe"
        assert caught.value.retryable


@pytest.mark.asyncio
async def test_native_finish_allows_usage_only_trailer():
    payload = (
        sse({"content": "safe"}, "stop")
        + 'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}\n\n'
        + "data: [DONE]\n\n"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))
    ) as http:
        client = OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
        events = [event async for event in client.stream(request(), "fixture")]
    assert isinstance(events[-1], StreamEndEvent)
    assert events[-1].usage is not None
    assert events[-1].usage.input_tokens == 10


@pytest.mark.parametrize("state", ["fresh", "stale", "empty", "cold", "historic-invalid"])
def test_first_frame_uses_authoritative_native_cache_without_fetching(monkeypatch, tmp_path, state):
    cache_dir = tmp_path / "listing"
    key = discovery._cache_key("deepseek")
    if state == "historic-invalid":
        catalogue.cached_listing(key, lambda: {"capture": 1, "models": []}, cache_dir=cache_dir)
    elif state != "cold":
        payload = {
            "data": (
                []
                if state == "empty"
                else [
                    {"id": "deepseek-flash", "context_length": 200_000},
                    {"id": "deepseek-v4-pro"},
                ]
            )
        }
        with httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
        ) as client:
            discovery.available_models(
                "deepseek", api_key="fixture", client=client, cache_dir=cache_dir
            )
        if state == "stale":
            path = catalogue._cache_path(key, cache_dir)
            raw = json.loads(path.read_text())
            raw["fetched_at"] -= 2 * 86400
            path.write_text(json.dumps(raw))

    def no_network(*args, **kwargs):
        raise AssertionError("first frame must only peek at cache")

    monkeypatch.setattr(discovery, "fetch_models", no_network)
    rows, status = discovery.cached_available_models("deepseek", cache_dir=cache_dir)
    expected = (
        set(deepseek_models)
        if state in {"cold", "historic-invalid"}
        else set() if state == "empty" else {"deepseek-flash", "deepseek-v4-pro"}
    )
    assert {row.id for row in rows} == expected
    assert status == ("static" if state in {"cold", "historic-invalid"} else "cached")
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("deepseek", {"type": "api_key", "key": "synthetic-firstframe-only"})
    try:
        controller = ProviderController(store)
        entries = [
            entry
            for entry in controller.initial_catalogue(cache_dir=cache_dir)
            if entry.provider == "deepseek"
        ]
        assert {entry.model_id for entry in entries} == expected
        assert all(entry.connected and not entry.aggregated for entry in entries)
        if state in {"fresh", "stale"}:
            assert (
                next(
                    entry for entry in entries if entry.model_id == "deepseek-flash"
                ).context_window
                == 200_000
            )
    finally:
        store.close()


def test_native_tool_text_uses_real_newlines():
    tool = Message(
        role="tool",
        tool_call_id="sample",
        content=[TextContent(text="alpha"), TextContent(text="beta"), ImageContent(data="YWJj")],
    )
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    body = client._build_body(request([tool]))
    assert next(m for m in body["messages"] if m["role"] == "tool")["content"] == "alpha\nbeta"


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


# ---------------------------------------------------------------------------
# The thinking-mode reasoning echo (ModelSpec.requires_reasoning_echo)
# ---------------------------------------------------------------------------
#
# DeepSeek's thinking mode refuses requests that do not carry the reasoning
# back -- "The `reasoning_content` in the thinking mode must be passed back to the
# API" -- and the refusal is recorded in the operator's sessions. What is
# MEASURED is the failure and its repair on the shapes the app sends: the
# request rebuilt from a failing transcript is refused, and the same body with a
# non-blank echo on every assistant turn is accepted, while the exact
# server-side rule is not fully characterised (see the field's own comment on
# ``ModelSpec.requires_reasoning_echo``, including the shapes that are accepted
# WITHOUT the echo). These tests pin the harness-side policy -- every blank
# assistant turn gets the echo -- rather than a claimed rule of the API, because
# the live 400 needs a real key and real money (the manual recipe lives in
# ``scripts/deepseek_reasoning_echo_probe.py``).


def test_thinking_route_echoes_reasoning_on_every_assistant_turn():
    """Every assistant turn leaves the builder with a non-blank echo."""
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    scope = credential_scope("fixture")
    call = ToolCall(id="call_a", name="inspect", arguments={"path": "a"})
    history = [
        Message.user("inspect a, then summarise"),
        # Reasoned and recorded: the real text must be what goes back.
        native_turn(reasoning="read a first", tool_calls=[call]),
        Message(role="tool", tool_call_id="call_a", content=[TextContent(text="found")]),
        # Recorded under ANOTHER credential scope: replay is refused, so this
        # turn has nothing to echo (the second way users meet this 400).
        native_turn(
            text="a is present",
            reasoning="scope-bound thought",
            scope=credential_scope("another-account"),
        ),
        Message.user("now summarise in one line"),
        # A plain-text turn the model produced no reasoning for at all.
        native_turn(text="a is present."),
    ]

    body = client._build_body(
        ChatRequest(model=echo_spec(), messages=history, system_blocks=["Stable"]), scope=scope
    )
    entries = assistant_entries(body)

    assert [entry.get("reasoning_content") for entry in entries] == [
        "read a first",
        REASONING_ECHO_PLACEHOLDER,
        REASONING_ECHO_PLACEHOLDER,
    ]
    # The placeholder is a sentence, not a blank: blank is what 400s.
    assert all(str(entry["reasoning_content"]).strip() for entry in entries)
    # And it stops at the assistant turns: a user or tool entry carrying the
    # field would be a key the validator has no meaning for.
    assert all(
        "reasoning_content" not in entry
        for entry in body["messages"]
        if entry.get("role") != "assistant"
    )


def test_thinking_route_echoes_tool_call_and_truncated_turns_and_drops_empty_ones():
    """The echo covers every RENDERED assistant turn, and only those."""
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    scope = credential_scope("fixture")
    call = ToolCall(id="call_b", name="inspect", arguments={"path": "b"})
    history = [
        Message.user("go"),
        # Errored before a single token: the body drops this turn entirely, so
        # there is nothing to echo (and nothing referencing it downstream).
        Message.assistant("", stop_reason="error"),
        Message.assistant("", stop_reason="aborted"),
        # Tool-call turn with no reasoning recorded.
        native_turn(tool_calls=[call]),
        Message(role="tool", tool_call_id="call_b", content=[TextContent(text="found")]),
        # Truncated mid-answer: replay refuses a ``length`` turn by contract.
        Message.assistant("half a sen", stop_reason="length"),
    ]

    body = client._build_body(
        ChatRequest(model=echo_spec(), messages=history, system_blocks=["Stable"]), scope=scope
    )
    entries = assistant_entries(body)

    assert len(entries) == 2
    assert all(entry["reasoning_content"] == REASONING_ECHO_PLACEHOLDER for entry in entries)
    # The empty error/abort turns stayed dropped rather than gaining an echo.
    assert not any(entry.get("content") == "" for entry in entries)


def test_thinking_route_counts_the_echo_as_the_input_it_is():
    """The placeholders are billed input, so the calibration counts them.

    They are added by the BODY, after the replay count, and the API
    concatenates them into the context it reads -- a calibration that ignored
    them would under-report every request on this route.
    """
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    scope = credential_scope("fixture")
    history = [
        Message.user("go"),
        native_turn(tool_calls=[ToolCall(id="call_c", name="inspect", arguments={})]),
        Message(role="tool", tool_call_id="call_c", content=[TextContent(text="found")]),
        native_turn(text="done"),
    ]
    req = ChatRequest(model=echo_spec(), messages=history, system_blocks=["Stable"])
    req.context_binding = ContextBinding(ContextTokenTracker(), measure_request(req))
    client._build_body(req, scope=scope)

    # Two blank turns at ``max(1, len // 4)`` each -- the one ruler for both the
    # replayed reasoning and the echo ``bind_native_context`` adds here.
    assert req.context_binding.measured.native_tokens == 2 * (len(REASONING_ECHO_PLACEHOLDER) // 4)


@pytest.mark.parametrize("provider", ["openrouter", "openai", "kimi", "xai"])
def test_routes_without_the_capability_are_byte_identical(provider, monkeypatch):
    """A spec WITHOUT the capability keeps the old body, byte for byte.

    This is the regression guard. The FIELD decides, and it only ever arrives by
    derivation: a hand-built spec (every test double and embedder), a local
    server, and a model from another family must all get exactly the body they
    got before the capability existed, on the same history DeepSeek fills in.
    """
    client = OpenAICompatClient(f"https://{provider}.invalid/v1")
    scope = credential_scope("fixture")
    history = [
        Message.user("go"),
        native_turn(tool_calls=[ToolCall(id="call_d", name="inspect", arguments={})]),
        Message(role="tool", tool_call_id="call_d", content=[TextContent(text="found")]),
        native_turn(text="done"),
    ]
    req = ChatRequest(
        model=spec().model_copy(update={"provider": provider, "model_id": "some/model"}),
        messages=history,
        system_blocks=["Stable"],
    )

    body = client._build_body(req, scope=scope)

    assert all("reasoning_content" not in entry for entry in assistant_entries(body))
    # And the capability, spelled out, is what drives it.
    capable = client._build_body(
        req.model_copy(
            update={"model": req.model.model_copy(update={"requires_reasoning_echo": True})}
        ),
        scope=scope,
    )
    assert all(str(entry["reasoning_content"]).strip() for entry in assistant_entries(capable))


@pytest.mark.parametrize(
    "provider,model_id",
    [
        ("openrouter", "openai/gpt-5.2"),
        ("openrouter", "anthropic/claude-opus-4.6"),
    ],
)
def test_a_derived_unrelated_aggregator_route_keeps_the_old_body(provider, model_id):
    """The DERIVED spec, not a hand-flipped field, keeps the old body.

    The sibling test above pins the body builder's field gating by flipping
    ``requires_reasoning_echo`` by hand, which leaves the DERIVER's own
    contribution to an unrelated route uncovered: nothing asserted that
    ``build_model_spec`` on a live aggregator row for another family yields the
    body that route got before the capability existed. On a route whose every
    other field is decided by the same registry, that is the composition the
    regression would actually arrive through.
    """
    client = OpenAICompatClient(f"https://{provider}.invalid/v1")
    scope = credential_scope("fixture")
    history = [
        Message.user("go"),
        native_turn(tool_calls=[ToolCall(id="call_e", name="inspect", arguments={})]),
        Message(role="tool", tool_call_id="call_e", content=[TextContent(text="found")]),
        native_turn(text="done"),
    ]
    derived = configure.build_model_spec(provider, model_id)
    assert derived.requires_reasoning_echo is False, "this family must not gain the field"
    req = ChatRequest(model=derived, messages=history, system_blocks=["Stable"])

    body = client._build_body(req, scope=scope)

    assert all("reasoning_content" not in entry for entry in assistant_entries(body))
    # Sensitive rather than vacuous: the SAME request on the SAME derived history
    # does gain the key once the field is set, so a deriver that flipped an
    # unrelated family would fail here instead of passing quietly.
    capable = client._build_body(
        req.model_copy(
            update={"model": derived.model_copy(update={"requires_reasoning_echo": True})}
        ),
        scope=scope,
    )
    assert all(str(entry["reasoning_content"]).strip() for entry in assistant_entries(capable))


def test_thinking_off_still_echoes_and_needs_no_special_case():
    """``thinking: disabled`` accepts the echo too, so the rule stays total.

    Measured live: with thinking disabled the API is lenient about the echo --
    it accepts a body with it, without it, and with real reasoning. Applying the
    echo unconditionally therefore keeps ONE invariant for this route (every
    assistant turn echoes) rather than one that depends on the effort setting,
    and it is what makes the bounded recovery's retry body legal as it stands.
    """
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    scope = credential_scope("fixture")
    history = [Message.user("go"), native_turn(text="done")]
    body = client._build_body(
        ChatRequest(model=echo_spec(effort="none"), messages=history, system_blocks=["Stable"]),
        scope=scope,
    )

    assert body["thinking"] == {"type": "disabled"}
    assert assistant_entries(body)[0]["reasoning_content"] == REASONING_ECHO_PLACEHOLDER


def test_the_compat_reasoning_key_is_echoed_as_the_real_thing():
    """A reply whose thought arrived under the compat ``reasoning`` key echoes
    THAT text, not the placeholder.

    Some OpenAI-shaped proxies return reasoning under ``reasoning`` instead of
    ``reasoning_content``, and the harness stores whichever key the wire used, so
    a fill that only looked at ``reasoning_content`` would answer "the harness
    lost this" over a thought it is holding.
    """
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    scope = credential_scope("fixture")
    # Built the way the wire parser stores a compat reply: the item carries
    # ``reasoning`` and no ``reasoning_content`` at all.
    compat = Message.assistant(
        "considered it",
        provider_payload=native_payload(
            echo_spec(),
            DEEPSEEK_ENDPOINT,
            "openai-chat",
            [{"reasoning": "compat recorded thought"}],
            "considered it",
            [],
            scope,
        ),
    )

    body = client._build_body(
        ChatRequest(model=echo_spec(), messages=[Message.user("go"), compat], system_blocks=["S"]),
        scope=scope,
    )

    assert assistant_entries(body)[0]["reasoning_content"] == "compat recorded thought"


def test_a_blank_echo_is_treated_as_missing():
    """A stored empty or whitespace reasoning value still takes the placeholder.

    The harness's own native state can hold a blank value where a reply carried
    no reasoning, and the fix's rule is about what the provider READS: a body
    that relies on a blank being accepted is relying on leniency no replica has
    promised (see ``REASONING_ECHO_PLACEHOLDER``), so blank is filled in like an
    absent key rather than passed through.
    """
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    scope = credential_scope("fixture")
    history = [
        Message.user("go"),
        native_turn(text="blank value", reasoning="", scope=scope),
        native_turn(text="whitespace value", reasoning="   ", scope=scope),
    ]

    body = client._build_body(
        ChatRequest(model=echo_spec(), messages=history, system_blocks=["Stable"]), scope=scope
    )

    assert [entry["reasoning_content"] for entry in assistant_entries(body)] == [
        REASONING_ECHO_PLACEHOLDER,
        REASONING_ECHO_PLACEHOLDER,
    ]


@pytest.mark.asyncio
async def test_the_echo_is_replayed_after_a_compaction_rebuild():
    """A conversation continued across a compaction boundary still echoes.

    Compaction is the one place the harness REBUILDS a prefix: the tool-output
    prune and the frame prune rewrite messages in place, and a summarising pass
    replaces the summarised prefix wholesale. The wire contract is about the
    body that goes out AFTER that, so both strategies are driven here and the
    body built from the result is checked turn by turn -- with the reasoning
    that survived the rebuild echoed as the REAL thing, and a turn whose native
    state no longer binds echoed as the placeholder.

    Compaction was a live suspect for the blank echo that kills a session, and
    it is innocent by construction (neither strategy removes or blanks an
    ASSISTANT turn); this pins that, so a later compaction change that starts
    rebuilding assistant turns brings the refusal straight back.
    """
    from local_operator.compaction.pass_ import run_compaction_pass
    from local_operator.compaction.pruning import prune_stale_frames, prune_tool_outputs
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.compaction.tokens import estimate_messages_tokens

    client = OpenAICompatClient("https://api.deepseek.com/v1")
    scope = credential_scope("fixture")
    now = 10_000_000

    def frame(text: str) -> Message:
        """An observation turn carrying an image: what the frame prune counts."""
        return Message.user(text, [ImageContent(data=PNG, mime_type="image/png")])

    keep = ToolCall(id="call_keep", name="inspect", arguments={"path": "a"})
    stale = ToolCall(id="call_stale", name="inspect", arguments={"path": "b"})
    bulk = "content " * 900
    history = [
        # The summarised prefix: bulk far past the cut, including one assistant
        # turn the rebuild does not keep.
        frame("frame 0 " + "state " * 120),
        native_turn(reasoning="summarised away", tool_calls=[keep]),
        Message(role="tool", tool_call_id="call_keep", content=[TextContent(text=bulk)]),
        # The kept tail: one turn whose native state still binds, one whose state
        # was recorded under ANOTHER credential scope (so it cannot be replayed),
        # and the user message the request continues from.
        frame("frame 1 " + "state " * 120),
        native_turn(text="kept finding", reasoning="read it first", tool_calls=[stale]),
        Message(role="tool", tool_call_id="call_stale", content=[TextContent(text="found b")]),
        native_turn(
            text="stale finding",
            reasoning="scope-bound thought",
            scope=credential_scope("another-account"),
        ),
        Message.user("please continue"),
    ]

    # Both prune strategies, in the order a session runs them, then the pass.
    pruned, _ = prune_tool_outputs(history, now_ms=now, last_activity_ms=now)
    pruned, dropped = prune_stale_frames(pruned, keep_recent_frames=1)
    assert dropped, "the fixture must actually compress a frame"
    summarizer_calls: list[str] = []

    async def summarize(prompt: str) -> str:
        summarizer_calls.append(prompt)
        return "Everything so far."

    post_prune = estimate_messages_tokens(pruned)
    result = await run_compaction_pass(
        pruned,
        model=ModelSpec(
            provider="deepseek", model_id="deepseek-flash", context_window=int(post_prune / 0.9)
        ),
        settings=CompactionSettings(
            strategy="context-full", keep_recent_tokens=300, keep_recent_frames=1
        ),
        summarize=summarize,
        now_ms=now,
        last_activity_ms=now,
    )
    assert result.ran is True and summarizer_calls, "the fixture must actually rebuild"
    assert len(result.messages) < len(history)

    body = client._build_body(
        ChatRequest(model=echo_spec(), messages=result.messages, system_blocks=["Stable"]),
        scope=scope,
    )
    entries = assistant_entries(body)
    rendered = [entry.get("reasoning_content") for entry in entries]
    assert entries, "the rebuilt prefix must still render the conversation's turns"
    # The contract: nothing left blank, on the body that continues the rebuild.
    assert all(str(value).strip() for value in rendered)
    # The reasoning that survived the rebuild is echoed as ITSELF...
    assert "read it first" in rendered
    # ...and the turn whose native state no longer binds takes the placeholder.
    assert REASONING_ECHO_PLACEHOLDER in rendered


def test_an_unchanged_conversation_counts_only_the_reasoning_it_replays():
    """The calibration must not double-count a conversation the echo left alone.

    Every assistant turn here already carries its recorded reasoning, so the
    builder fills NOTHING: ``native_context_tokens`` is the replayed reasoning
    alone. If the echo's placeholder term were added unconditionally -- or the
    replayed bytes were counted twice -- this is the measurement that would
    drift, and it is the one the context window is sized from.
    """
    client = OpenAICompatClient("https://api.deepseek.com/v1")
    scope = credential_scope("fixture")
    history = [
        Message.user("go"),
        native_turn(text="one", reasoning="first thought", scope=scope),
        Message.user("again"),
        native_turn(text="two", reasoning="second thought, longer", scope=scope),
    ]
    req = ChatRequest(model=echo_spec(), messages=history, system_blocks=["Stable"])
    req.context_binding = ContextBinding(ContextTokenTracker(), measure_request(req))
    body = client._build_body(req, scope=scope)

    expected = len("first thought") // 4 + len("second thought, longer") // 4
    assert expected > 0
    assert req.context_binding.measured.native_tokens == expected
    assert REASONING_ECHO_PLACEHOLDER not in [
        entry.get("reasoning_content") for entry in assistant_entries(body)
    ]


def test_private_reasoning_is_replayed_as_provider_state_never_as_content():
    """The echo is provider STATE; it must not become a message anyone reads.

    Replaying reasoning is a different act from showing it (see
    ``StreamReasoningDelta``'s docstring): it rides the wire as
    ``reasoning_content`` and must never appear in a message's ``content``, in
    the transcript the model is shown, or in what the user is handed -- which is
    the one thing a naive "put it back in the history" fix would get wrong.
    """
    from local_operator.model.configure import build_model_spec
    from local_operator.providers.clients import OpenAICompatClient
    from local_operator.providers.replay import credential_scope, native_payload

    spec = build_model_spec("deepseek", "deepseek-flash")
    scope = credential_scope("fixture")
    private = "the operator's private reasoning"
    endpoint = f"{spec.base_url}/chat/completions"
    reasoned = Message.assistant(
        "here is the answer",
        provider_payload=native_payload(
            spec,
            endpoint,
            "openai-chat",
            [{"reasoning_content": private}],
            "here is the answer",
            [],
            scope,
        ),
    )
    client = OpenAICompatClient(spec.base_url or "https://api.deepseek.com/v1")
    body = client._build_body(
        ChatRequest(model=spec, system_blocks=["sys"], messages=[Message.user("go"), reasoned]),
        scope=scope,
    )

    assistant = [entry for entry in body["messages"] if entry.get("role") == "assistant"][0]
    assert assistant["reasoning_content"] == private
    # Content carries the answer and only the answer. Nothing a later turn (or a
    # UI) reads as prose has the reasoning welded into it, and the harness
    # message the transcript holds never gained it either.
    assert private not in json.dumps(assistant.get("content") or "")
    assert private not in json.dumps(body["messages"][:-1])
    assert private not in repr(reasoned.text)
    assert private not in json.dumps(reasoned.content, default=str)
