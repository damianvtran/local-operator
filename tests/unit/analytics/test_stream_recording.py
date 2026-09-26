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
    StreamReasoningDelta,
    StreamTextDelta,
    StreamUsageEvent,
    TextContent,
    ToolResult,
    Usage,
    decode_window_of,
)
from local_operator.model.configure import SessionStreamFn


async def _noop(tool_call_id: str, *_args: object) -> ToolResult:
    return ToolResult(tool_call_id=tool_call_id, tool_name="stub", content=[])


def _fn(session_id="sess-1"):
    # These recorder tests bypass __init__; preserve an ordinary root stream's
    # default so they exercise recording without inventing fork activity.
    fn = object.__new__(SessionStreamFn)
    fn._session_id = session_id
    fn._counts_as_child_request = False
    return fn


def test_recorder_fixture_defaults_to_root_stream() -> None:
    assert _fn()._counts_as_child_request is False


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


def test_records_first_reasoning_ms_against_the_stream_start(tmp_path):
    """The reasoning wait is recorded next to ``ttft_ms``, on the same origin.

    Both are stamped by the wrapper off one monotonic clock started where the
    stream is entered, so the reasoning gap is a subtraction -- which is the
    whole point of recording the second number: before it existed, the operator's
    own ledger could not see the wait they were complaining about (the fragment
    never reached the harness, so nothing timed it).
    """
    import sqlite3

    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-r")
    usage = Usage(input_tokens=10, output_tokens=5, context_tokens=10)
    asyncio.run(
        _drain(
            fn,
            _request(),
            [
                StreamReasoningDelta(delta="weighing the options"),
                StreamTextDelta(delta="answer"),
                StreamEndEvent(stop_reason="stop", usage=usage),
            ],
        )
    )
    rec.flush_for_test()
    row = (
        sqlite3.connect(tmp_path / "a.db")
        .execute("SELECT ttft_ms, first_reasoning_ms FROM calls")
        .fetchone()
    )
    assert row[0] >= 0, "ttft_ms keeps its meaning: the first text delta"
    assert row[1] >= 0
    # The fragment was streamed BEFORE the text, and both are stamped from one
    # monotonic clock, so the reasoning instant cannot follow the text one.
    assert row[1] <= row[0]


def test_a_turn_that_never_reasoned_records_minus_one(tmp_path):
    """No reasoning is a SAMPLE, not a zero: ``-1``, matching its neighbours.

    A 0 ms would claim the model reasoned instantly, which is not what happened
    and would drag every mean toward a phase that never ran.
    """
    import sqlite3

    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-n")
    usage = Usage(input_tokens=10, output_tokens=5, context_tokens=10)
    asyncio.run(
        _drain(
            fn,
            _request(),
            [StreamTextDelta(delta="answer"), StreamEndEvent(stop_reason="stop", usage=usage)],
        )
    )
    rec.flush_for_test()
    row = (
        sqlite3.connect(tmp_path / "a.db")
        .execute("SELECT ttft_ms, first_reasoning_ms FROM calls")
        .fetchone()
    )
    assert row[0] >= 0
    assert row[1] == -1


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


# ---------------------------------------------------------------------------
# The decode window: measured AT THE SEAM, which is the only place it exists
# ---------------------------------------------------------------------------
#
# These drive the real ``_record_stream`` and read the rate back out of the
# store. They exist because every other decode test in the suite sets
# ``CallSnapshot.decode_*`` BY HAND — so deleting the measurement from
# ``configure.py`` entirely left the whole suite green (review round 1, MAJOR 1,
# demonstrated by forcing ``_record_usage`` to receive zeros and observing an
# identical pass count). A metric whose only test is a fixture asserting itself
# has no regression guard at all.


def _decode_of(store, session_id: str = "sess-1"):
    """The recorded row's decode measures, read back through the report."""
    report = store.session_report(session_id)
    return report.aggregate, report.recent[0] if report.recent else None


def _stream_with_deltas(deltas, *, output_tokens: int, stop_reason: str = "stop"):
    usage = Usage(input_tokens=100, output_tokens=output_tokens, context_tokens=100)
    return [
        *deltas,
        StreamUsageEvent(usage=usage),
        StreamEndEvent(stop_reason=stop_reason, usage=usage),
    ]


def test_seam_measures_a_window_across_text_deltas(tmp_path):
    """A text stream produces a window, one contributing call, and its tokens."""
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-1")
    deltas = [StreamTextDelta(delta=f"chunk {i} ") for i in range(6)]
    asyncio.run(_drain(fn, _request(), _stream_with_deltas(deltas, output_tokens=240)))
    rec.flush_for_test()

    agg, row = _decode_of(store)
    assert agg.decode_calls == 1
    assert agg.decode_us > 0
    # Numerator and denominator cover the SAME calls: the contributing call's own
    # output, not every call's.
    assert agg.decode_tokens == 240
    assert agg.decode_tps is not None and agg.decode_tps > 0
    assert row is not None and row.output_tokens == 240
    rec.close()
    store.close()


def test_seam_measures_a_reasoning_only_stream(tmp_path):
    """Reasoning IS output: a thinking model's window must open on it.

    ``ttft_ms`` stays -1 here because it is stamped only on text/tool-call
    deltas — which is exactly why a window derived as ``duration - ttft`` was
    unusable (design §8a), and why this test also pins the reasoning-only case
    that derivation could never see.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-1")
    deltas = [StreamReasoningDelta(delta=f"think {i} ") for i in range(5)]
    asyncio.run(_drain(fn, _request(), _stream_with_deltas(deltas, output_tokens=180)))
    rec.flush_for_test()

    agg, row = _decode_of(store)
    assert agg.decode_calls == 1 and agg.decode_us > 0 and agg.decode_tokens == 180
    # ``None`` here, not ``-1``: the report's projection maps the column's "no
    # sample" sentinel through ``NULLIF(..., -1)``, so unknown reaches a reader as
    # NULL. Either spelling means the same thing — no text delta ever arrived —
    # and that fact is what this test is about.
    assert row is not None and row.ttft_ms is None
    rec.close()
    store.close()


def test_seam_excludes_a_single_delta_call(tmp_path):
    """One delta has no measurable window, so it is EXCLUDED and COUNTED.

    This is the one-frame population the benchmark measured: its implied rate is
    absurd because the whole answer lands in the trailing chunk. The exclusion is
    the design's ``output_deltas >= 2`` guard, and it must show up as zero
    contributions rather than as a very fast call.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-1")
    asyncio.run(
        _drain(
            fn,
            _request(),
            _stream_with_deltas([StreamTextDelta(delta="the whole answer")], output_tokens=900),
        )
    )
    rec.flush_for_test()

    agg, _ = _decode_of(store)
    assert agg.calls == 1 and agg.output_tokens == 900
    assert agg.decode_calls == 0 and agg.decode_us == 0 and agg.decode_tokens == 0
    assert agg.decode_tps is None  # unknown, never a very fast number
    rec.close()
    store.close()


def test_seam_excludes_a_call_that_generated_nothing(tmp_path):
    """No output tokens means no rate to report, however many deltas arrived."""
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-1")
    deltas = [StreamTextDelta(delta="x") for _ in range(4)]
    asyncio.run(_drain(fn, _request(), _stream_with_deltas(deltas, output_tokens=0)))
    rec.flush_for_test()

    agg, _ = _decode_of(store)
    assert agg.decode_calls == 0 and agg.decode_tps is None
    rec.close()
    store.close()


def test_seam_records_a_window_for_an_aborted_stream(tmp_path):
    """A failure mid-stream still records what was measured, and does not re-raise.

    ``ok`` is not part of the eligibility predicate on purpose (design risk 8):
    dropping exactly the slow calls would bias every rate upward. The usage event
    has to arrive BEFORE the failure for there to be a rate at all — a stream that
    dies before reporting usage has output_tokens == 0, which the predicate
    excludes for the honest reason that nothing generated is known.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-1")
    usage = Usage(input_tokens=100, output_tokens=120, context_tokens=100)

    async def stream():
        yield StreamTextDelta(delta="one ")
        yield StreamTextDelta(delta="two ")
        yield StreamUsageEvent(usage=usage)
        raise RuntimeError("provider died mid-stream")

    events = []

    async def drain():
        async for event in fn._record_stream(_request(), stream()):
            events.append(event)

    raised = False
    try:
        asyncio.run(drain())
    except RuntimeError:
        raised = True
    assert raised, "the wrapper must not swallow the provider's own failure"
    assert len(events) == 3, "the events that DID arrive were forwarded"
    rec.flush_for_test()

    agg, row = _decode_of(store)
    assert agg.calls == 1 and agg.ok_calls == 0
    assert agg.decode_calls == 1 and agg.decode_us > 0 and agg.decode_tokens == 120
    assert row is not None and row.ok is False
    rec.close()
    store.close()


def test_a_broken_analytics_call_cannot_reach_the_stream(tmp_path, monkeypatch):
    """The never-raise contract, at the seam rather than at the recorder."""
    from local_operator.analytics import recorder as recorder_mod

    store = AnalyticsStore(tmp_path / "a.db")
    reset_recorder_for_test(store)
    fn = _fn("sess-1")

    def explode(*_args, **_kwargs):
        raise RuntimeError("analytics is broken")

    monkeypatch.setattr(recorder_mod, "record_call", explode)
    events = asyncio.run(
        _drain(fn, _request(), _stream_with_deltas([StreamTextDelta(delta="a")], output_tokens=5))
    )
    assert [type(e).__name__ for e in events] == [
        "StreamTextDelta",
        "StreamUsageEvent",
        "StreamEndEvent",
    ]
    store.close()


def test_the_seam_reads_the_clock_once_per_output_delta(tmp_path, monkeypatch):
    """The hot-path bound, asserted STRUCTURALLY rather than by timing.

    A timing assertion is a bet on machine load; this counts the calls instead,
    and isolates the per-delta reads by draining two streams that differ ONLY in
    how many output deltas they carry — so the wrapper's own housekeeping (the
    ``started_at`` and ``duration_ms`` reads, one each per drain) cancels, and
    what remains is the per-delta cost.

    Counted through a PROXY on the wrapper's own module attribute rather than by
    patching ``time.monotonic`` globally: the event loop itself reads the clock
    to schedule (``loop.time()``), so a global patch counts asyncio's scheduling
    reads too and the measurement flapped by a read or two between runs. The
    proxy replaces only what ``configure`` sees, and delegates everything else.
    That is the load-independent half of the design's §12.2.
    """
    import time as time_mod

    from local_operator.model import configure as configure_mod

    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    calls = {"n": 0}

    class _CountingTime:
        """The ``time`` module as ``configure`` sees it, with one read counted."""

        def monotonic(self) -> float:
            calls["n"] += 1
            return time_mod.monotonic()

        def __getattr__(self, name: str):
            return getattr(time_mod, name)

    monkeypatch.setattr(configure_mod, "time", _CountingTime())

    def drain_with(deltas: int) -> int:
        calls["n"] = 0
        fn = _fn(f"sess-{deltas}")
        events = _stream_with_deltas(
            [StreamTextDelta(delta="x") for _ in range(deltas)], output_tokens=10 + deltas
        )
        asyncio.run(_drain(fn, _request(), events))
        rec.flush_for_test()
        return calls["n"]

    without = drain_with(0)
    with_one = drain_with(1)
    with_500 = drain_with(500)
    # One read per output delta, plus ONE more on the first delta: that one is
    # the PRE-EXISTING ``ttft_ms`` stamp, which the base revision pays too, so it
    # is not part of what this change adds. Both assertions are stated because
    # they say different things — the first that the per-delta cost is exactly
    # one read, the second that the wrapper's own housekeeping is constant.
    assert with_500 - without == 501, (
        "500 per-delta reads plus the first delta's pre-existing TTFT read; "
        f"got {with_500 - without}"
    )
    assert (
        with_500 - with_one == 499
    ), f"499 further deltas must cost 499 reads; got {with_500 - with_one}"
    rec.close()
    store.close()


def test_the_relayed_window_agrees_with_the_ledger_when_usage_arrives_early(tmp_path):
    """The one invariant §9.3 states: the band and the ledger agree about a call.

    A provider may send its usage frame BEFORE its final deltas — ``text → usage →
    tool_call_delta…`` is ordinary, because tool-call argument deltas count as
    output — so a stamp applied only at the usage event closes the window too early
    and would have the band contradict the ledger (measured at 3.6x on this shape).
    The seam re-stamps in its ``finally`` with the complete window; this is the
    assertion that keeps it.
    """
    store = AnalyticsStore(tmp_path / "a.db")
    rec = reset_recorder_for_test(store)
    fn = _fn("sess-1")
    relayed = Usage(input_tokens=100, output_tokens=240, context_tokens=100)
    events = [
        StreamTextDelta(delta="first "),
        StreamUsageEvent(usage=relayed),  # usage BEFORE the last delta
        StreamTextDelta(delta="second "),
        StreamTextDelta(delta="third "),
        StreamEndEvent(stop_reason="stop", usage=relayed),
    ]
    asyncio.run(_drain(fn, _request(), events))
    rec.flush_for_test()

    agg, _row = _decode_of(store)
    window = decode_window_of(relayed)
    assert agg.decode_calls == 1
    assert window is not None, "the final deltas arrived; the window must be complete"
    assert window == (agg.decode_us, agg.decode_tokens), (
        f"the relay carries {window} while the ledger recorded "
        f"{(agg.decode_us, agg.decode_tokens)} — the band would contradict the ledger"
    )
    rec.close()
    store.close()
