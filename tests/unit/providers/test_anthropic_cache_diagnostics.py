"""Anthropic cache diagnostics: body shaping, stream parse, chain ownership,
fail-closed latch, and the ledger columns that keep the verdict.

The API compares a request with the one that produced ``previous_message_id``
and says on ``message_start`` WHERE the prompt prefix diverged
(https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics).
These tests pin the four properties that make that telemetry trustworthy:
the chain is per-conversation, only turns join it, a refusal costs one retry
for the whole process, and the verdict reaches ``calls`` unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
import time
from typing import Any

import httpx
import pytest

from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.store import AnalyticsStore
from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamStartEvent,
    Usage,
)
from local_operator.providers import clients
from local_operator.providers.clients import AnthropicClient


@pytest.fixture(autouse=True)
def _reset_latch():
    clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] = False
    yield
    clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] = False


def _sse(message: dict[str, Any], *, cache_write: int = 0) -> bytes:
    events = [
        (
            "message_start",
            {
                "message": {
                    "usage": {
                        "input_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": cache_write,
                    },
                    **message,
                }
            },
        ),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        ("message_stop", {}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps({'type': name, **payload})}\n\n"
        for name, payload in events
    ).encode()


def _request(**overrides: Any) -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-haiku-4-5"),
        system_blocks=["stable"],
        messages=[Message.user("hi")],
        **overrides,
    )


def _client(handler) -> AnthropicClient:
    return AnthropicClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def _collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


def _start(events: list[Any]) -> StreamStartEvent:
    return next(event for event in events if isinstance(event, StreamStartEvent))


# -- body shaping ------------------------------------------------------------


def test_an_undiagnosed_request_sends_no_diagnostics_key() -> None:
    """Default OFF: helpers, scripts and other conversations' bodies are unchanged."""
    assert "diagnostics" not in AnthropicClient()._build_body(_request())


def test_the_first_turn_opts_in_with_an_explicit_null() -> None:
    body = AnthropicClient()._build_body(_request(cache_diagnostics=True))
    assert body["diagnostics"] == {"previous_message_id": None}
    assert json.dumps(body["diagnostics"]) == '{"previous_message_id": null}'


def test_a_later_turn_names_the_previous_response() -> None:
    body = AnthropicClient()._build_body(
        _request(cache_diagnostics=True, cache_diagnostics_previous_id="msg_prev")
    )
    assert body["diagnostics"] == {"previous_message_id": "msg_prev"}


def test_a_non_first_party_endpoint_never_sends_it() -> None:
    """Bedrock/Vertex/relays do not offer the feature; an unknown key 400s there."""
    client = AnthropicClient(base_url="https://relay.example.com")
    assert "diagnostics" not in client._build_body(_request(cache_diagnostics=True))


def test_the_latch_suppresses_the_key() -> None:
    clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] = True
    assert "diagnostics" not in AnthropicClient()._build_body(_request(cache_diagnostics=True))


# -- stream parse --------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("diagnostics", "previous", "expected"),
    [
        (None, None, ("baseline", 0)),
        (None, "msg_prev", ("unchanged", 0)),
        ({"cache_miss_reason": None}, "msg_prev", ("pending", 0)),
        (
            {
                "cache_miss_reason": {
                    "type": "system_changed",
                    "cache_missed_input_tokens": 4321,
                }
            },
            "msg_prev",
            ("system_changed", 4321),
        ),
        (
            {"cache_miss_reason": {"type": "previous_message_not_found"}},
            "msg_prev",
            ("previous_message_not_found", 0),
        ),
        ("garbage", "msg_prev", ("unavailable", 0)),
        ({"cache_miss_reason": {"type": ""}}, "msg_prev", ("unavailable", 0)),
    ],
)
async def test_message_start_diagnostics_reach_the_start_event(
    diagnostics: Any, previous: str | None, expected: tuple[str, int]
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse({"id": "msg_new", "diagnostics": diagnostics}),
            headers={"content-type": "text/event-stream"},
        )

    events = await _collect(
        _client(handler).stream(
            _request(cache_diagnostics=True, cache_diagnostics_previous_id=previous), "sk-test"
        )
    )
    start = _start(events)
    assert start.response_id == "msg_new"
    assert (start.cache_miss_type, start.cache_missed_tokens) == expected


@pytest.mark.asyncio
async def test_an_undiagnosed_request_reports_no_verdict() -> None:
    """``None``, never a fabricated ``baseline``, when we did not opt in."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_sse({"id": "msg_x"}), headers={"content-type": "text/event-stream"}
        )

    start = _start(await _collect(_client(handler).stream(_request(), "sk-test")))
    assert start.cache_miss_type is None


# -- fail closed ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_field_is_retried_once_without_it_and_latched_off(caplog) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "diagnostics" in body:
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "diagnostics: Extra inputs are not permitted",
                    },
                },
            )
        return httpx.Response(
            200, content=_sse({"id": "msg_ok"}), headers={"content-type": "text/event-stream"}
        )

    client = _client(handler)
    with caplog.at_level(logging.INFO, logger="local_operator.providers.clients"):
        events = await _collect(client.stream(_request(cache_diagnostics=True), "sk-test"))
    assert [("diagnostics" in body) for body in bodies] == [True, False]
    assert _start(events).cache_miss_type is None, "the retried call was not diagnosed"
    assert clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] is True
    assert sum("cache diagnostics refused" in r.message for r in caplog.records) == 1

    # Latched: the next request never carries the key, so no second round trip.
    await _collect(client.stream(_request(cache_diagnostics=True), "sk-test"))
    assert [("diagnostics" in body) for body in bodies] == [True, False, False]


@pytest.mark.asyncio
async def test_an_unrelated_400_is_not_retried() -> None:
    """Status alone never triggers the retry: an ordinary 400 is a request defect."""
    from local_operator.providers.failover import ProviderError

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            400,
            json={"error": {"type": "invalid_request_error", "message": "max_tokens: too big"}},
        )

    with pytest.raises(ProviderError):
        await _collect(_client(handler).stream(_request(cache_diagnostics=True), "sk-test"))
    assert len(calls) == 1
    assert clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] is False


# -- the conversation chain (SessionStreamFn) ------------------------------------


def _anthropic_stream_fn(tmp_path, monkeypatch, session_id: str, bodies: list[dict[str, Any]]):
    from local_operator import analytics
    from local_operator.model.configure import create_stream_fn
    from local_operator.providers.auth_store import AuthStore

    monkeypatch.setattr(analytics, "record_call", lambda snapshot: None)
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        counter["n"] += 1
        return httpx.Response(
            200,
            content=_sse({"id": f"msg_{session_id}_{counter['n']}", "diagnostics": None}),
            headers={"content-type": "text/event-stream"},
        )

    store = AuthStore(tmp_path / f"auth-{session_id}.db")
    store.upsert_credential("anthropic", {"key": "sk-ant-test", "source": "login"})
    stream = create_stream_fn(store, {}, session_id=session_id)
    return stream, store, handler


@pytest.mark.asyncio
async def test_turns_chain_per_conversation_and_helpers_stay_out(tmp_path, monkeypatch) -> None:
    """Turn N names turn N-1's id; a fork starts its own chain; an aside is
    never diagnosed and never moves the chain."""
    parent_bodies: list[dict[str, Any]] = []
    parent, store, handler = _anthropic_stream_fn(tmp_path, monkeypatch, "p", parent_bodies)
    await parent._http.aclose()
    parent._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parent._transport.http = parent._http
    spec = ModelSpec(provider="anthropic", model_id="claude-haiku-4-5")

    def turn(**extra: Any) -> ChatRequest:
        return ChatRequest(model=spec, messages=[Message.user("hi")], **extra)

    try:
        await _collect(parent(turn(), None))
        await _collect(parent(turn(purpose="aside"), None))
        await _collect(parent(turn(), None))

        child = parent.fork("c")
        await _collect(child(turn(), None))
        await _collect(parent(turn(), None))
        await child.close()
    finally:
        await parent.close()
        store.close()

    diag = [body.get("diagnostics", "absent") for body in parent_bodies]
    assert diag == [
        {"previous_message_id": None},  # parent turn 1 opts in
        "absent",  # the aside is never diagnosed
        {"previous_message_id": "msg_p_1"},  # turn 2 skips the aside's response
        {"previous_message_id": None},  # the fork starts its OWN chain
        {"previous_message_id": "msg_p_3"},  # parent unaffected by the child's id
    ]


# -- ledger ------------------------------------------------------------------------


def _snap(**overrides: Any) -> CallSnapshot:
    return CallSnapshot(
        ts_ms=int(time.time() * 1000),
        session_id="s1",
        provider="anthropic",
        model_id="claude",
        input_tokens=10,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=30_000,
        reasoning_tokens=0,
        context_tokens=30_010,
        component_chars={"conversation": 10},
        ok=True,
        cost_micro=0,
        cost_known=True,
        priced=True,
        **overrides,
    )


def test_the_verdict_is_recorded_and_an_older_ledger_gains_the_columns(tmp_path) -> None:
    db = tmp_path / "pre-diag.db"
    store = AnalyticsStore(db)
    assert store.record_batch([_snap()]) == 1
    conn = store._connect()
    assert conn is not None
    conn.execute("ALTER TABLE calls DROP COLUMN cache_miss_type")
    conn.execute("ALTER TABLE calls DROP COLUMN cache_missed_tokens")
    store.close()

    # Re-open: the ALTER path, not the CREATE, is what must put them back.
    store = AnalyticsStore(db)
    snap = dataclasses.replace(
        _snap(), cache_miss_type="system_changed", cache_missed_tokens=29_000
    )
    assert store.record_batch([snap]) == 1
    store.close()
    rows = (
        sqlite3.connect(str(db))
        .execute("SELECT cache_miss_type, cache_missed_tokens FROM calls ORDER BY id")
        .fetchall()
    )
    assert rows == [("", 0), ("system_changed", 29_000)], "old rows read as undiagnosed"


# -- the break log ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("miss_type", "write", "logged"),
    [
        ("system_changed", 30_000, True),
        ("system_changed", 5_000, False),  # small write: the "rare, low impact" case
        ("unchanged", 30_000, False),  # TTL expiry, not a break
        ("previous_message_not_found", 30_000, False),
        ("", 30_000, False),
    ],
)
def test_a_costly_named_break_logs_one_line(caplog, miss_type, write, logged) -> None:
    from unittest.mock import MagicMock

    from local_operator.model.configure import SessionStreamFn

    stream = SessionStreamFn(MagicMock(), {}, "sess-1")
    usage = Usage(input_tokens=10, cache_read_tokens=0, cache_write_tokens=write)
    with caplog.at_level(logging.INFO, logger="local_operator.model.configure"):
        stream._log_cache_break(usage, miss_type, 1234)
    lines = [r.getMessage() for r in caplog.records if "[prompt cache break]" in r.getMessage()]
    assert len(lines) == (1 if logged else 0)
    if logged:
        assert "session=sess-1" in lines[0] and "type=system_changed" in lines[0]
        assert "missed_tokens=1234" in lines[0]
