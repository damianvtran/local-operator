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


def test_no_usage_records_nothing(tmp_path):
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn()
    # A stream that never reported usage: nothing to attribute, nothing stored.
    asyncio.run(_drain(fn, _request(), [StreamTextDelta(delta="x")]))
    rec.flush_for_test()
    assert store.aggregate().calls == 0


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
