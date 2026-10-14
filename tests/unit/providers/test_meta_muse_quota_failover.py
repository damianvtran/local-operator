"""Muse's recoverable failures must descend the configured cascade.

THE BUG THIS PINS (evidence session 5ebdcd0d1ec8): a Muse subscription that ran
out mid-turn answers HTTP 429; ``meta_muse_client`` forced ``fail_closed=True``
on EVERY wrapped error, so ``failover.py``'s early raise (``if exc.fail_closed``)
ended the turn before the user's configured ``retry.fallbackChains`` were
consulted. ``fail_closed`` exists to stop a SILENT BILLABLE Muse PAYG fallback
and to stop replaying local resource failures; a quota error is neither -- the
request never ran, and descending to a target the user CONFIGURED is the user's
explicit choice. The genuine non-recoverables keep it: the billing-attribution
refusal, the missing-meter refusal, and the local output envelope
(``StreamOutputEnvelopeExceeded``).

These drive the REAL engine (``stream_with_failover``) with the REAL
``MetaMuseClient`` over an ``httpx.MockTransport``, so the adapter's
classification and the cascade's descent are exercised together. No live Meta
traffic, no credentials, no sockets.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.harness.types import ChatRequest, StreamEndEvent, StreamTextDelta
from local_operator.model.configure import build_model_spec
from local_operator.providers.clients import client_for_spec
from local_operator.providers.failover import ProviderError, stream_with_failover
from tests.unit.providers.test_failover import FakeAuth, ScriptedClient

pytestmark = pytest.mark.asyncio

KEY = "LLM|1234|fixture_subscription_key"
#: The wait the incident advertised (3h8m) -- far beyond any loop-back ceiling,
#: so nothing may sleep on it inside the turn.
INCIDENT_RETRY_AFTER = "11328"


def _request() -> ChatRequest:
    return ChatRequest(model=build_model_spec("meta-muse", "muse-wonder-1.0"))


def _sse(*frames: dict[str, Any]) -> str:
    return "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)


def _http_error(status: int, *, retry_after: str | None = None) -> httpx.MockTransport:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.MockTransport(
        lambda _request: httpx.Response(
            status, headers=headers, json={"error": {"message": "quota exceeded"}}
        )
    )


def _fallback_events() -> list[Any]:
    return [StreamTextDelta(delta="fallback"), StreamEndEvent(stop_reason="stop")]


async def _drive(
    transport: httpx.MockTransport,
    *,
    fallback: ScriptedClient | None = None,
) -> tuple[list[str], list[Any] | ProviderError]:
    """One engine call; returns the targets visited and events OR the raise."""
    seen: list[str] = []
    http = httpx.AsyncClient(transport=transport)

    async def client_for(spec: Any) -> Any:
        seen.append(f"{spec.provider}/{spec.model_id}")
        if spec.provider == "meta-muse":
            return client_for_spec(spec, http_client=http)
        return fallback or ScriptedClient(_fallback_events())

    settings = {"retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["anthropic/claude-x"]}}}
    auth = FakeAuth({"meta-muse": [KEY], "anthropic": ["k2"]})
    outcome: list[Any] | ProviderError
    try:
        outcome = [
            event async for event in stream_with_failover(_request(), auth, settings, client_for)
        ]
    except ProviderError as error:
        outcome = error
    finally:
        await http.aclose()
    return seen, outcome


def _deltas(outcome: list[Any] | ProviderError) -> list[str]:
    assert isinstance(outcome, list), f"expected a served stream, got {outcome!r}"
    return [event.delta for event in outcome if isinstance(event, StreamTextDelta)]


async def test_a_muse_quota_429_descends_to_the_configured_fallback() -> None:
    """The incident shape: 429 + a 3h8m Retry-After must still fail over NOW."""
    seen, outcome = await _drive(_http_error(429, retry_after=INCIDENT_RETRY_AFTER))
    assert seen[0] == "meta-muse/muse-wonder-1.0"
    assert seen[-1] == "anthropic/claude-x", seen
    assert "fallback" in _deltas(outcome)


@pytest.mark.parametrize("status", [402, 429])
async def test_subscription_402_and_a_bare_429_also_descend(status: int) -> None:
    """A 402 is quota-class by classification (billing), and a bare 429 is the
    short-throttle shape -- both must reach the chain rather than end the turn."""
    seen, outcome = await _drive(_http_error(status))
    assert seen[-1] == "anthropic/claude-x", seen
    assert "fallback" in _deltas(outcome)


async def test_a_wordless_mid_stream_failure_descends_after_the_fix() -> None:
    """No content reached the caller, so the attempt is replayable: the walk
    must move on instead of failing closed on an unknown provider fault."""
    frames = _sse({"type": "response.failed", "response": {"error": {"message": "boom"}}})
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, content=frames))
    seen, outcome = await _drive(transport)
    assert seen[-1] == "anthropic/claude-x", seen
    assert "fallback" in _deltas(outcome)


async def test_a_mid_stream_failure_after_output_never_replays() -> None:
    """CONTROL (must hold before and after the fix): once a delta has been
    forwarded, the attempt cannot be retried anywhere -- one delta, no replay,
    the fallback never asked."""
    frames = _sse(
        {"type": "response.output_text.delta", "delta": "partial"},
        {"type": "response.failed", "response": {"error": {"message": "boom"}}},
    )
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, content=frames))
    seen: list[str] = []
    yielded: list[Any] = []
    http = httpx.AsyncClient(transport=transport)

    async def client_for(spec: Any) -> Any:
        seen.append(f"{spec.provider}/{spec.model_id}")
        if spec.provider == "meta-muse":
            return client_for_spec(spec, http_client=http)
        return ScriptedClient(_fallback_events())

    settings = {"retry": {"baseDelayMs": 1, "fallbackChains": {"default": ["anthropic/claude-x"]}}}
    auth = FakeAuth({"meta-muse": [KEY], "anthropic": ["k2"]})
    with pytest.raises(ProviderError):
        async for event in stream_with_failover(_request(), auth, settings, client_for):
            yielded.append(event)
    await http.aclose()
    assert [event.delta for event in yielded if isinstance(event, StreamTextDelta)] == [
        "partial"
    ], "the forwarded delta must appear exactly once"
    assert seen == ["meta-muse/muse-wonder-1.0"], seen


async def test_a_dead_sign_in_is_auth_and_still_descends() -> None:
    """A response-side 401 is kind='auth' (never quota) and must not fail
    closed: the configured fallback is the user's own answer to a dead key."""
    http = httpx.AsyncClient(transport=_http_error(401))
    client = client_for_spec(build_model_spec("meta-muse", "muse-wonder-1.0"), http_client=http)
    with pytest.raises(ProviderError) as caught:
        _ = [event async for event in client.stream(_request(), KEY)]
    assert caught.value.kind == "auth"
    assert not caught.value.fail_closed
    await http.aclose()

    seen, outcome = await _drive(_http_error(401))
    assert seen[-1] == "anthropic/claude-x", seen
    assert "fallback" in _deltas(outcome)


async def test_a_refusal_end_is_the_providers_answer_not_a_provider_fault() -> None:
    """A content refusal must reach the caller as a refusal end (the loop
    renders it), not as a fail-closed transport error -- the failover engine
    must never re-ask refused content on another route."""
    frames = _sse(
        {
            "type": "response.subscription_usage",
            "subscription": {
                "window": {"used_percent": 7, "resets_at": 1788431188, "window_duration_mins": 300},
                "weekly": {"used_percent": 2, "resets_at": 1788739200},
            },
        },
        {
            "type": "response.incomplete",
            "response": {"incomplete_details": {"reason": "content_filter"}},
        },
    )
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, content=frames))
    http = httpx.AsyncClient(transport=transport)
    client = client_for_spec(build_model_spec("meta-muse", "muse-wonder-1.0"), http_client=http)
    events = [event async for event in client.stream(_request(), KEY)]
    await http.aclose()
    assert isinstance(events[-1], StreamEndEvent)
    assert events[-1].stop_reason == "refusal", events[-1]


async def test_all_targets_exhausted_raises_one_error_and_stops() -> None:
    """Both legs down: exactly one ProviderError, no loop, no hours-long wait."""
    fallback = ScriptedClient(ProviderError(500, "fallback down", retryable=True))
    seen, outcome = await _drive(
        _http_error(429, retry_after=INCIDENT_RETRY_AFTER), fallback=fallback
    )
    assert isinstance(outcome, ProviderError)
    assert seen[-1] == "anthropic/claude-x", seen
    # The walk visited each target a bounded number of times; nothing stalled.
    assert len(seen) <= 4, seen
