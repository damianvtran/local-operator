"""Recording is wired into the ONE place every provider call funnels through.

``SessionStreamFn._record_stream`` wraps the provider stream: it forwards every
event unchanged (so a turn is byte-for-byte what it was) and, only after the
stream is fully consumed, records the call's usage. These tests prove the
forwarding is transparent, the authoritative counts are captured, a failed
stream is still recorded (it cost input tokens), and analytics failures never
propagate into the turn.
"""

from __future__ import annotations

import asyncio
import sqlite3

from local_operator.analytics.recorder import reset_recorder_for_test
from local_operator.analytics.store import AnalyticsStore
from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamUsageEvent,
    TextContent,
    ToolResult,
    Usage,
)
from local_operator.model.configure import SessionStreamFn


async def _noop(tool_call_id: str, *_args: object) -> ToolResult:
    return ToolResult(tool_call_id=tool_call_id, tool_name="stub", content=[])


def _fn(session_id="sess-1"):
    fn = object.__new__(SessionStreamFn)
    fn._session_id = session_id
    return fn


def _request():
    block0 = (
        "Persona.\n\n## User's custom instructions\n\n<user_instructions>terse</user_instructions>"
    )
    return ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-opus-5"),
        system_blocks=[block0, "## Available tools\ntools", "env", "<skills>k</skills>"],
        messages=[Message(role="user", content=[TextContent(text="hi " * 50)])],
        tools=[
            AgentTool(
                name="bash",
                description="run",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
                execute=_noop,
            )
        ],
    )


async def _drain(fn, request, events):
    async def stream():
        for ev in events:
            yield ev

    out = []
    async for ev in fn._record_stream(request, stream()):
        out.append(ev)
    return out


def test_forwards_events_unchanged(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    reset_recorder_for_test(store)
    fn = _fn()
    usage = Usage(input_tokens=1000, output_tokens=200, context_tokens=1000)
    events = [
        StreamTextDelta(delta="hello"),
        StreamUsageEvent(usage=usage),
        StreamEndEvent(stop_reason="stop", usage=usage),
    ]
    out = asyncio.run(_drain(fn, _request(), events))
    assert [type(e).__name__ for e in out] == [
        "StreamTextDelta",
        "StreamUsageEvent",
        "StreamEndEvent",
    ]


def test_records_authoritative_counts(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-abc")
    usage = Usage(
        input_tokens=1000,
        output_tokens=300,
        cache_read_tokens=8000,
        cache_write_tokens=500,
        reasoning_tokens=120,
        context_tokens=9500,
    )
    asyncio.run(_drain(fn, _request(), [StreamEndEvent(stop_reason="stop", usage=usage)]))
    rec.flush_for_test()
    agg = store.aggregate()
    assert agg.calls == 1
    assert agg.ok_calls == 1
    assert agg.input_tokens == 1000
    assert agg.output_tokens == 300
    assert agg.reasoning_tokens == 120
    assert agg.generation_tokens == 180
    assert agg.context_tokens == 9500
    assert "anthropic" in agg.by_provider
    assert "sess-abc" in agg.by_session
    # The component split summed to the authoritative context total.
    assert sum(agg.components.values()) == 9500
    # System prompt and custom instructions were both attributed nonzero.
    assert agg.components["system_prompt"] > 0
    assert agg.components["custom_instructions"] > 0


def test_records_cost_for_a_priced_model(tmp_path):
    # A model with a real registry price records a positive, known cost.
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-cost")
    request = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-sonnet-4-5"),
        system_blocks=["persona", "tools", "env", "skills"],
        messages=[Message(role="user", content=[TextContent(text="hi " * 50)])],
    )
    usage = Usage(
        input_tokens=5000,
        output_tokens=1000,
        cache_read_tokens=90_000,
        cache_write_tokens=5000,
        context_tokens=100_000,
    )
    asyncio.run(_drain(fn, request, [StreamEndEvent(stop_reason="stop", usage=usage)]))
    rec.flush_for_test()
    agg = store.aggregate()
    assert agg.calls == 1
    # Cost is computed through the shared cost_for_usage, so it is positive and
    # known; the exact figure depends on the live price table, so assert the
    # invariants rather than a brittle dollar amount.
    assert agg.cost_is_known is True
    assert agg.cost_is_partial is False
    assert agg.cost_micro > 0
    assert agg.by_provider["anthropic"].cost_micro == agg.cost_micro


def test_failed_stream_still_recorded(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn()
    # An error end that still carried a usage (input was billed).
    end = StreamEndEvent(
        stop_reason="error", usage=Usage(input_tokens=500, context_tokens=500), error="boom"
    )
    asyncio.run(_drain(fn, _request(), [end]))
    rec.flush_for_test()
    agg = store.aggregate()
    assert agg.calls == 1
    assert agg.ok_calls == 0


def test_missing_usage_is_recorded_as_unknown_spend_and_incomplete(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn()
    # Missing usage must not make failed requests invisible or invent free
    # successful work. Old rows retain their counts; new diagnostics are exact.
    asyncio.run(_drain(fn, _request(), [StreamTextDelta(delta="x")]))
    rec.flush_for_test()
    assert store.aggregate().calls == 1
    assert store.aggregate().ok_calls == 0
    with sqlite3.connect(tmp_path / "a.db") as connection:
        row = connection.execute(
            "SELECT request_id, purpose, duration_ms, ttft_ms, outcome, "
            "usage_reported, cost_known FROM calls"
        ).fetchone()
    assert row[0]
    assert row[1] == "turn"
    assert row[2] >= row[3] >= 0
    assert row[4:] == ("incomplete", 0, 0)


def test_context_fallback_from_input(tmp_path):
    # A provider that omits an explicit context size: the recorder falls back
    # to input + cache_read so the component split still has a denominator.
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn()
    usage = Usage(input_tokens=400, output_tokens=50, cache_read_tokens=100, context_tokens=None)
    asyncio.run(_drain(fn, _request(), [StreamEndEvent(stop_reason="stop", usage=usage)]))
    rec.flush_for_test()
    agg = store.aggregate()
    assert agg.calls == 1
    assert sum(agg.components.values()) == 500  # 400 + 100


def test_context_fallback_uses_serving_provider_cache_convention(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    usage = Usage(
        input_tokens=400,
        cache_read_tokens=300,
        cache_write_tokens=50,
        provider="openai",
        context_tokens=None,
    )
    asyncio.run(_drain(_fn(), _request(), [StreamEndEvent(stop_reason="stop", usage=usage)]))
    rec.flush_for_test()
    assert sum(store.aggregate().components.values()) == 400
    rec.close()
    store.close()


def test_analytics_failure_never_breaks_turn(tmp_path, monkeypatch):
    # If recording raises, the stream must still complete normally.
    store = AnalyticsStore(tmp_path / "a.db")
    reset_recorder_for_test(store)
    fn = _fn()

    import local_operator.analytics as analytics_pkg

    def _boom(_snapshot):
        raise RuntimeError("recorder exploded")

    monkeypatch.setattr(analytics_pkg, "record_call", _boom)
    usage = Usage(input_tokens=10, context_tokens=10)
    out = asyncio.run(_drain(fn, _request(), [StreamEndEvent(stop_reason="stop", usage=usage)]))
    # The turn's event still came through; the analytics failure was swallowed.
    assert [type(e).__name__ for e in out] == ["StreamEndEvent"]


def test_records_serving_model_not_session_primary(tmp_path):
    """A primary→xai failover must land under xai/grok-4.6, not anthropic.

    ``stream_with_failover`` rewrites the on-the-wire request but used to leave
    the recorder reading the ORIGINAL ChatRequest. After Anthropic failed over
    to Grok every successful call was stored as ``anthropic/claude-opus-4-8``
    and priced at Opus rates — which is why By provider showed only anthropic.
    The failover layer now stamps the serving spec onto Usage; this is the
    contract the recorder must honour.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-failover")
    request = _request()  # anthropic/claude-opus-5
    usage = Usage(
        input_tokens=1000,
        output_tokens=200,
        context_tokens=1000,
        provider="xai",
        model_id="grok-4.6",
    )
    asyncio.run(_drain(fn, request, [StreamEndEvent(stop_reason="stop", usage=usage)]))
    rec.flush_for_test()
    agg = store.aggregate()
    assert agg.calls == 1
    assert "xai" in agg.by_provider
    assert "anthropic" not in agg.by_provider
    rec.close()
    store.close()


def test_records_primary_success_under_primary(tmp_path):
    """A call that never failed over must still attribute to the session primary.

    The serving-spec stamp is how failover is honest; it must not invent a
    fallback on a primary success (or an isolated naming call that never
    walked the chain).
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn()
    usage = Usage(
        input_tokens=100,
        output_tokens=20,
        context_tokens=100,
        provider="anthropic",
        model_id="claude-opus-5",
    )
    asyncio.run(_drain(fn, _request(), [StreamEndEvent(stop_reason="stop", usage=usage)]))
    rec.flush_for_test()
    assert set(store.aggregate().by_provider) == {"anthropic"}
    rec.close()
    store.close()


def test_canonicalizes_login_flavour_to_storage_id(tmp_path):
    """``xai-oauth`` spend must roll up under ``xai``, not a second row.

    The login flavour is the same billable vendor as the API-key id. Splitting
    them in By provider was the other half of "I am on grok and analytics
    still shows only anthropic" once the serving-spec stamp landed.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn()
    request = ChatRequest(
        model=ModelSpec(provider="xai-oauth", model_id="grok-4.6"),
        system_blocks=["persona", "tools", "env", "skills"],
        messages=[Message(role="user", content=[TextContent(text="hi " * 50)])],
    )
    usage = Usage(input_tokens=100, output_tokens=20, context_tokens=100)
    asyncio.run(_drain(fn, request, [StreamEndEvent(stop_reason="stop", usage=usage)]))
    rec.flush_for_test()
    assert set(store.aggregate().by_provider) == {"xai"}
    rec.close()
    store.close()


def test_provider_reported_usd_cost_survives_into_the_ledger(tmp_path):
    """OpenRouter's ``usage.cost`` must become ``cost_micro``, not a table estimate.

    ``CallSnapshot`` used to drop ``usd_cost``, so ``price_snapshot`` always
    re-estimated from the registry. A reported $0.0075 on a model whose table
    price is wildly different must store 7500 micro-USD.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn()
    request = ChatRequest(
        model=ModelSpec(provider="openrouter", model_id="some/unpriced-sku"),
        system_blocks=["persona", "tools", "env", "skills"],
        messages=[Message(role="user", content=[TextContent(text="hi " * 50)])],
    )
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        context_tokens=1_000_000,
        usd_cost=0.0075,
    )
    asyncio.run(_drain(fn, request, [StreamEndEvent(stop_reason="stop", usage=usage)]))
    rec.flush_for_test()
    agg = store.aggregate()
    assert agg.calls == 1
    assert agg.cost_is_known is True
    assert agg.cost_micro == 7500
    rec.close()
    store.close()
