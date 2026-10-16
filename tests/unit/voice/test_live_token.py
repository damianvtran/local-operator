"""R2/R7: the Live token's locked body, the mint, and the secrets that never log.

No network: every mint here runs over ``httpx.MockTransport``, and the
transport asserts it is only ever asked for the auth-token path.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from local_operator.voice import live_config, live_token

NOW = datetime(2026, 10, 16, 12, 0, 0, tzinfo=timezone.utc)
API_KEY = "AIzaSy-TEST-not-a-real-key-0123456789abcdef"
MINTED = "auth_tokens/eph-TEST-token-9f8e7d6c5b4a"


def test_the_body_locks_exactly_two_tools_and_no_gate_tool() -> None:
    body = live_config.token_request_body(NOW)
    names = live_config.tool_names(body)
    assert names == ["relay_to_session", "voice_control"]
    # R7: nothing that could approve, deny or answer a card.
    forbidden = ("gate", "approv", "answer", "deny", "allow", "confirm")
    assert not any(word in name for name in names for word in forbidden)


def test_the_body_shape_is_the_locked_setup() -> None:
    body = live_config.token_request_body(NOW)
    assert set(body) == {"uses", "expireTime", "newSessionExpireTime", "bidiGenerateContentSetup"}
    # No fieldMask: the token's setup is the WHOLE effective setup (API reference).
    assert "fieldMask" not in body
    assert body["uses"] == 1
    assert body["expireTime"] == "2026-10-16T12:30:00Z"
    assert body["newSessionExpireTime"] == "2026-10-16T12:01:00Z"
    setup = body["bidiGenerateContentSetup"]
    assert setup["model"] == "models/gemini-3.8-live"
    assert setup["generationConfig"] == {"responseModalities": ["AUDIO"]}
    assert setup["systemInstruction"] == {"parts": [{"text": live_config.SYSTEM_INSTRUCTION}]}
    assert setup["realtimeInputConfig"] == {"automaticActivityDetection": {"disabled": True}}
    assert setup["sessionResumption"] == {}
    assert setup["contextWindowCompression"] == {"slidingWindow": {}}
    assert setup["inputAudioTranscription"] == {} and setup["outputAudioTranscription"] == {}
    (tool,) = setup["tools"]
    relay, control = tool["functionDeclarations"]
    assert relay["behavior"] == control["behavior"] == "NON_BLOCKING"
    assert relay["parameters"]["required"] == ["text"]
    assert control["parameters"]["properties"]["action"]["enum"] == [
        "stop_speaking",
        "repeat_last",
        "louder",
        "softer",
    ]
    # Plain JSON, so it is exactly what goes on the wire.
    assert json.loads(json.dumps(body)) == body


def test_the_instruction_never_lets_the_model_answer() -> None:
    assert live_config.SYSTEM_INSTRUCTION == (
        "You are the ears of another agent. For each take call exactly one tool. "
        "Never answer, summarise or advise."
    )


def _client(handler) -> httpx.AsyncClient:
    def guarded(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1beta/auth_tokens"), request.url
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(guarded))


@pytest.mark.asyncio
async def test_a_mint_sends_the_key_in_a_header_and_the_locked_body() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["query"] = str(request.url.query)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "name": MINTED,
                "expireTime": "2026-10-16T12:30:00Z",
                "newSessionExpireTime": "2026-10-16T12:01:00Z",
            },
        )

    async with _client(handler) as client:
        minted = await live_token.mint(API_KEY, now=NOW, client=client)
    assert seen["key"] == API_KEY
    assert API_KEY not in seen["query"], "the key must never ride the URL"
    assert seen["body"] == live_config.token_request_body(NOW)
    assert minted.token == MINTED
    assert minted.model == "gemini-3.8-live"
    assert minted.expires_at == "2026-10-16T12:30:00Z"
    assert MINTED not in repr(minted)


@pytest.mark.asyncio
async def test_neither_secret_reaches_any_log_record(caplog: pytest.LogCaptureFixture) -> None:
    """A mint, a refusal that echoes the key, and a transport failure, at DEBUG."""
    caplog.set_level(logging.DEBUG)

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": MINTED})

    def echoing_refusal(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": f"API key not valid: {API_KEY}", "status": "INVALID"}},
        )

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connect failed for {request.url} key={API_KEY}")

    async with _client(ok) as client:
        await live_token.mint(API_KEY, now=NOW, client=client)
    errors: list[str] = []
    for handler in (echoing_refusal, down):
        async with _client(handler) as client:
            with pytest.raises(live_token.LiveTokenError) as raised:
                await live_token.mint(API_KEY, now=NOW, client=client)
        errors.append(str(raised.value))
        # Nothing chained: an httpx exception's repr carries the request.
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None or raised.value.__suppress_context__
        # The refusal is surfaced by the route as `live_failed`; log it the way
        # the route does, so this test also covers that line.
        logging.getLogger("local_operator.server.routes.desktop_voice").info(
            "voice live token refused: %s", raised.value
        )
    assert "HTTP 400" in errors[0] and "API key not valid" in errors[0]
    assert errors[1] == "Google could not be reached (ConnectError)."
    for text in [record.getMessage() for record in caplog.records] + errors:
        assert API_KEY not in text
        assert MINTED not in text


@pytest.mark.asyncio
async def test_an_answer_without_a_token_is_a_refusal() -> None:
    async with _client(lambda request: httpx.Response(200, json={})) as client:
        with pytest.raises(live_token.LiveTokenError, match="without a token"):
            await live_token.mint(API_KEY, now=NOW, client=client)
