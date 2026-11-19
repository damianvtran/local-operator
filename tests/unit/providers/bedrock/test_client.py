"""T4: BedrockAnthropicClient over a fake httpx transport speaking event-stream.

Covers the body transform, the three extracted hooks (with a first-party
byte-identity regression), streaming of text / tool_use / thinking, usage incl.
cache counters and the invocationMetrics fill, mid-stream exception frames and
the HTTP error table.
"""

from __future__ import annotations

import asyncio
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
    StreamUsageEvent,
)
from local_operator.providers.bedrock import client as bclient
from local_operator.providers.bedrock.client import (
    BedrockAnthropicClient,
    transform_body,
)
from local_operator.providers.clients import AnthropicClient, _encode_json_body
from local_operator.providers.failover import ProviderError

from . import eventstream as es

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
KEY = "bedrock://env-bearer?region=us-west-2"
TOKEN = "test-bearer-value"
#: Built, not spelled: the expected header for the env bearer token.
EXPECTED_AUTH = "Bear" + "er " + TOKEN


@pytest.fixture(autouse=True)
def bearer_env(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)
    monkeypatch.delenv("LOP_BEDROCK_ENDPOINT_URL", raising=False)


def _request(model: str = MODEL, **overrides: Any) -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(provider="bedrock", model_id=model),
        system_blocks=["stable system", "volatile tail"],
        messages=[Message.user("hi")],
        **overrides,
    )


def _client(handler) -> BedrockAnthropicClient:
    return BedrockAnthropicClient(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def _eventstream(events: list[dict[str, Any]], *, extra: bytes = b"") -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "application/vnd.amazon.eventstream"},
        content=es.stream_bytes(events) + extra,
    )


def _aws_error(status: int, error_type: str, message: str) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"x-amzn-ErrorType": f"{error_type}:http://internal.amazon.com/coral/"},
        json={"message": message},
    )


async def _collect(client, request=None, key=KEY) -> list[Any]:
    return [event async for event in client.stream(request or _request(), key)]


# -- body transform -------------------------------------------------------------


def test_transform_pops_first_party_keys_and_moves_betas() -> None:
    marker_1h = {"type": "ephemeral", "ttl": "1h"}
    body = {
        "model": MODEL,
        "stream": True,
        "speed": "fast",
        "diagnostics": {"previous_message_id": None},
        "future_first_party_key": 1,
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": dict(marker_1h)}],
            }
        ],
        "system": [{"type": "text", "text": "s", "cache_control": dict(marker_1h)}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "tools": [{"name": "bash", "input_schema": {}, "cache_control": {"type": "ephemeral"}}],
    }
    headers = {
        "anthropic-beta": "oauth-2025-04-20,effort-2025-11-24,interleaved-thinking-2025-05-14"
    }
    out = transform_body(body, headers)
    for dropped in ("model", "stream", "speed", "diagnostics", "future_first_party_key"):
        assert dropped not in out
    assert out["anthropic_version"] == "bedrock-2023-05-31"
    assert out["anthropic_beta"] == ["effort-2025-11-24", "interleaved-thinking-2025-05-14"]
    assert out["system"][0]["cache_control"] == marker_1h
    assert out["messages"][0]["content"][0]["cache_control"] == marker_1h
    assert out["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert out["thinking"] == {"type": "adaptive"}
    assert out["output_config"] == {"effort": "high"}


def test_no_beta_header_means_no_anthropic_beta_key() -> None:
    assert "anthropic_beta" not in transform_body({"max_tokens": 1}, {})


# -- request on the wire --------------------------------------------------------


@pytest.mark.asyncio
async def test_request_url_headers_and_body_on_the_wire() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["raw_path"] = request.url.raw_path.decode()
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return _eventstream(es.anthropic_events(usage={"input_tokens": 7}))

    await _collect(_client(handler))
    assert seen["raw_path"] == (
        "/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/invoke-with-response-stream"
    )
    assert seen["url"].startswith("https://bedrock-runtime.us-west-2.amazonaws.com/")
    headers = seen["headers"]
    assert headers["authorization"] == EXPECTED_AUTH
    assert headers["x-amzn-bedrock-accept"] == "application/json"
    assert headers["content-type"] == "application/json"
    assert "x-api-key" not in headers and "anthropic-version" not in headers
    body = seen["body"]
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert "model" not in body and "stream" not in body
    # The shared body builder ran: system blocks in Anthropic's block form.
    assert [block["text"] for block in body["system"]] == ["stable system", "volatile tail"]
    assert body["messages"][0]["role"] == "user" and body["max_tokens"] > 0


@pytest.mark.asyncio
async def test_endpoint_override_routes_to_the_fake_server(monkeypatch) -> None:
    monkeypatch.setenv("LOP_BEDROCK_ENDPOINT_URL", "http://127.0.0.1:9")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return _eventstream(es.anthropic_events(usage={"input_tokens": 1}))

    await _collect(_client(handler))
    assert seen["url"].startswith("http://127.0.0.1:9/model/")


# -- streaming ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_usage_and_cache_counters_stream_through() -> None:
    usage = {
        "input_tokens": 11,
        "cache_read_input_tokens": 1200,
        "cache_creation_input_tokens": 300,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 300},
    }

    def handler(request):
        return _eventstream(es.anthropic_events(text="hello", usage=usage, output_tokens=4))

    events = await _collect(_client(handler))
    assert "".join(e.delta for e in events if isinstance(e, StreamTextDelta)) == "hello"
    final = next(e for e in events if isinstance(e, StreamUsageEvent)).usage
    assert (final.input_tokens, final.output_tokens) == (11, 4)
    assert (final.cache_read_tokens, final.cache_write_tokens) == (1200, 300)
    assert final.cache_write_1h_tokens == 300
    assert final.context_tokens == 11 + 1200 + 300
    end = events[-1]
    assert isinstance(end, StreamEndEvent) and end.stop_reason == "stop"


@pytest.mark.asyncio
async def test_invocation_metrics_fill_unreported_usage() -> None:
    metrics = {
        "inputTokenCount": 21,
        "outputTokenCount": 9,
        "cacheReadInputTokenCount": 100,
        "cacheWriteInputTokenCount": 50,
    }

    def handler(request):
        return _eventstream(es.anthropic_events(usage=None, output_tokens=None, metrics=metrics))

    events = await _collect(_client(handler))
    final = next(e for e in events if isinstance(e, StreamUsageEvent)).usage
    assert (final.input_tokens, final.output_tokens) == (21, 9)
    assert (final.cache_read_tokens, final.cache_write_tokens) == (100, 50)
    assert final.context_tokens == 171


@pytest.mark.asyncio
async def test_metrics_never_override_reported_usage() -> None:
    metrics = {"inputTokenCount": 999, "outputTokenCount": 999}

    def handler(request):
        return _eventstream(
            es.anthropic_events(usage={"input_tokens": 5}, output_tokens=2, metrics=metrics)
        )

    final = next(
        e for e in await _collect(_client(handler)) if isinstance(e, StreamUsageEvent)
    ).usage
    assert (final.input_tokens, final.output_tokens) == (5, 2)


@pytest.mark.asyncio
async def test_tool_use_and_thinking_carry_through() -> None:
    thinking = [
        {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "ponder"}},
        {"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": "sig=="}},
        {"type": "content_block_stop"},
    ]
    tool = [
        {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}},
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"cmd":'},
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '"ls"}'},
        },
        {"type": "content_block_stop"},
    ]

    def handler(request):
        return _eventstream(
            es.anthropic_events(
                usage={"input_tokens": 3}, extra_blocks=[thinking, tool], stop_reason="tool_use"
            )
        )

    events = await _collect(_client(handler))
    calls = [e for e in events if isinstance(e, StreamToolCallDelta)]
    assert calls[0].id == "toolu_1" and calls[0].name == "bash"
    assert "".join(c.argument_delta or "" for c in calls) == '{"cmd":"ls"}'
    end = events[-1]
    assert end.stop_reason == "toolUse"
    payload = json.dumps(end.provider_payload, default=str)
    assert "ponder" in payload and "sig==" in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame_type", "kind", "retryable"),
    [
        ("throttlingException", "quota", True),
        ("modelStreamErrorException", "transient", True),
        ("internalServerException", "transient", True),
        ("serviceUnavailableException", "transient", True),
        ("modelTimeoutException", "timeout", True),
        ("validationException", "request", False),
    ],
)
async def test_mid_stream_exception_frames_raise(frame_type, kind, retryable) -> None:
    def handler(request):
        events = es.anthropic_events(usage={"input_tokens": 1})[:3]
        return _eventstream(events, extra=es.exception(frame_type, "boom"))

    with pytest.raises(ProviderError) as caught:
        await _collect(_client(handler))
    assert caught.value.kind == kind and caught.value.retryable is retryable


# -- HTTP error table -------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_type", "message", "kind", "auth_error", "retryable"),
    [
        (429, "ThrottlingException", "Too many requests", "quota", False, True),
        (
            403,
            "AccessDeniedException",
            "You don't have access to the model with the specified model ID.",
            "request",
            False,
            False,
        ),
        (
            403,
            "AccessDeniedException",
            "User is not authorized to perform: bedrock:InvokeModelWithResponseStream",
            "auth",
            True,
            False,
        ),
        # R1/Q1: an IAM deny naming ONE model / inference-profile ARN is
        # model-scoped: the credential still works for other models.
        (
            403,
            "AccessDeniedException",
            "User: arn:aws:iam::123456789012:user/qa is not authorized to perform: "
            "bedrock:InvokeModelWithResponseStream on resource: "
            "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-opus-5-5",
            "request",
            False,
            False,
        ),
        (
            403,
            "AccessDeniedException",
            "User: arn:aws:sts::123456789012:assumed-role/worker/x is not authorized to "
            "perform: bedrock:InvokeModelWithResponseStream on resource: "
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-x",
            "request",
            False,
            False,
        ),
        (
            400,
            "ValidationException",
            "Input is too long for requested model.",
            "request",
            False,
            False,
        ),
        (
            403,
            "ExpiredTokenException",
            "The security token included in the request is expired",
            "auth",
            True,
            False,
        ),
        (
            403,
            "UnrecognizedClientException",
            "The security token included in the request is invalid.",
            "auth",
            True,
            False,
        ),
        (404, "ResourceNotFoundException", "Model not found", "request", False, False),
        (503, "ServiceUnavailableException", "busy", "transient", False, True),
    ],
)
async def test_http_error_table(status, error_type, message, kind, auth_error, retryable) -> None:
    def handler(request):
        return _aws_error(status, error_type, message)

    with pytest.raises(ProviderError) as caught:
        await _collect(_client(handler))
    error = caught.value
    assert (error.kind, error.auth_error, error.retryable) == (kind, auth_error, retryable)
    if error_type == "ResourceNotFoundException":
        assert MODEL in str(error) and "us-west-2" in str(error)
    if error_type == "ValidationException":
        # Verbatim, so incidents.py's overflow wording match sees it.
        assert "Input is too long" in str(error)


@pytest.mark.asyncio
async def test_missing_credentials_fail_before_any_request(monkeypatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
    sent = []

    def handler(request):
        sent.append(request)
        return _eventstream([])

    with pytest.raises(ProviderError) as caught:
        await _collect(_client(handler))
    assert not sent
    assert caught.value.kind == "auth" and caught.value.request_sent is False


# -- 1h cache TTL gate ----------------------------------------------------------


def test_1h_ttl_kept_only_for_known_bedrock_models(monkeypatch) -> None:
    monkeypatch.setattr(AnthropicClient, "_cache_ttl_for", lambda self, *a, **k: "1h")
    client = BedrockAnthropicClient(region="us-east-1")
    assert client._cache_ttl_for(_request(), {}) == "1h"
    assert client._cache_ttl_for(_request("us.anthropic.claude-3-haiku-20240307-v1:0"), {}) is None


# -- first-party regression of the hook extraction ------------------------------


@pytest.mark.asyncio
async def test_first_party_request_bytes_and_headers_unchanged() -> None:
    """The extracted hooks send exactly what ``json=`` sent before."""
    seen: dict[str, Any] = {}

    def handler(request):
        seen["content"] = request.content
        seen["headers"] = dict(request.headers)
        seen["url"] = str(request.url)
        body = (
            'event: message_start\ndata: {"type":"message_start","message":'
            '{"usage":{"input_tokens":1}}}\n\n'
            'event: message_delta\ndata: {"type":"message_delta","delta":'
            '{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
        )
        return httpx.Response(200, content=body.encode())

    client = AnthropicClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    request = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-haiku-4-5"),
        system_blocks=["stable", "tail \u00e9 \u2713"],
        messages=[Message.user("hi")],
    )
    events = [e async for e in client.stream(request, "sk-ant-test")]
    end = events[-1]
    assert isinstance(end, StreamEndEvent) and end.stop_reason == "stop"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    body = json.loads(seen["content"])
    expected = httpx.Request("POST", seen["url"], json=body).content
    assert seen["content"] == expected == _encode_json_body(body)
    assert seen["headers"]["x-api-key"] == "sk-ant-test"
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["headers"]["anthropic-version"]
    assert body["model"] == "claude-haiku-4-5" and body["stream"] is True


def test_region_sentinel_is_never_the_first_party_url() -> None:
    from local_operator.providers.clients import ANTHROPIC_API_URL

    assert bclient.region_sentinel("us-east-1") != ANTHROPIC_API_URL
    assert BedrockAnthropicClient(region="us-east-1")._base_url == "bedrock-runtime:us-east-1"


# -- round-1 remediation: rotation, recovery hint, no-credential preflight ---------

_IAM_MODEL_DENY = (
    "User: arn:aws:iam::123456789012:user/qa is not authorized to perform: "
    "bedrock:InvokeModelWithResponseStream on resource: "
    "arn:aws:bedrock:us-east-1::foundation-model/x with an explicit deny"
)


def test_model_scoped_deny_is_not_rotation_eligible_but_bare_deny_is() -> None:
    """R1/Q1: rotation (and so the 60 s credential block) must skip a per-model deny."""
    from local_operator.providers.bedrock.errors import provider_error_for
    from local_operator.providers.failover import ProviderError, is_rotation_eligible

    scoped = provider_error_for("AccessDeniedException", _IAM_MODEL_DENY, 403)
    console = provider_error_for(
        "AccessDeniedException",
        "You don't have access to the model with the specified model ID.",
        403,
    )
    bare = provider_error_for(
        "AccessDeniedException",
        "User is not authorized to perform: bedrock:InvokeModelWithResponseStream",
        403,
    )
    assert not is_rotation_eligible(scoped)
    assert not is_rotation_eligible(console)
    assert is_rotation_eligible(bare)
    # First-party control: a 403 with no explicit kind still derives auth and rotates.
    assert is_rotation_eligible(ProviderError(403, "forbidden"))


def test_next_turn_on_allowed_model_is_not_blocked_after_model_scoped_deny(
    tmp_path, monkeypatch
) -> None:
    """R1/Q1 through the real AuthStore: the deny leaves no credential block behind.

    Mirrors the failover driver: it calls ``rotate_sibling`` only when
    ``is_rotation_eligible``. The bare (credential-wide) deny is the control
    that DOES block, so the assertion is not vacuous.
    """
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.bedrock.errors import provider_error_for
    from local_operator.providers.failover import is_rotation_eligible

    # The env tier would outrank the stored row; this test is about the row.
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")

    def run(message: str) -> list[tuple[int, str]]:
        base = tmp_path / str(abs(hash(message)))
        base.mkdir()
        store = AuthStore(base / "auth.db", config_dir=base)
        try:
            row = store.upsert_credential("bedrock", {"type": "api_key", "key": "fake-bedrock-key"})
            err = provider_error_for("AccessDeniedException", message, 403)
            if is_rotation_eligible(err):
                store.rotate_sibling("bedrock", None, err, api_key=row.data["key"])
            resolved = asyncio.run(store.get_api_key("bedrock"))
            blocks = store._conn.execute(
                "SELECT credential_id, block_scope FROM auth_credential_blocks"
            ).fetchall()
            if not blocks:
                # A later turn on an allowed model resolves the stored row itself.
                assert resolved == row.data["key"]
            return blocks
        finally:
            store.close()

    assert run(_IAM_MODEL_DENY) == []
    assert run("User is not authorized to perform: bedrock:InvokeModel") != []


def test_bedrock_auth_recovery_hint_names_no_phantom_key_variable() -> None:
    """Q3: no ``credential update <BEDROCK_API_KEY>`` -- nothing reads that name."""
    from local_operator.providers.failover import auth_recovery_hint

    hint = auth_recovery_hint("bedrock")
    assert "BEDROCK_API_KEY" not in hint and "credential update" not in hint
    assert "login bedrock" in hint
    # Other providers keep the key-replacement remedy.
    assert "credential update <OPENAI_API_KEY>" in auth_recovery_hint("openai")


def test_preflight_without_credentials_names_bedrock_env_switches(
    tmp_path, monkeypatch, capsys
) -> None:
    """Q4: the no-credential error names AWS_BEARER_TOKEN_BEDROCK and the ambient switch."""
    import os

    from local_operator import cli

    for name in list(os.environ):
        if name.startswith(("AWS_", "LOP_BEDROCK")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli._preflight_api_key("bedrock", tmp_path / ".local-operator") == 1
    err = capsys.readouterr().err
    assert "local-operator login bedrock" in err
    assert "AWS_BEARER_TOKEN_BEDROCK" in err
    assert "LOP_BEDROCK_AUTH=ambient" in err
