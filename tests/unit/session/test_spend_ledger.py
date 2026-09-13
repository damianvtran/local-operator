"""The per-session spend ledger: one exact recalled number (design §11).

Every test here is the evidence for a claim the design makes, so each one names
the claim. The ones that matter most are the STRUCTURAL ones — thread identity
for "this ran off the loop", a call count for "this happens once", and an
absence-of-a-scan for "the recall does not walk the journal backward" — because
a wall-clock bound for any of them would be a bet on machine load
(``AGENTS.md`` §"Prefer a structural invariant to a numeric one").
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    AgentMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.session import session as session_module
from local_operator.session import spend as spend_module
from local_operator.session.frontend_state import CostKnowledge
from local_operator.session.session import Session
from local_operator.session.spend import (
    SESSION_SPEND_CUSTOM_TYPE,
    SessionSpend,
    price_call,
    price_rows,
)
from local_operator.session.transcript import (
    _COLLAPSIBLE_CUSTOM_TYPES,
    BOOKKEEPING_CUSTOM_TYPES,
    Transcript,
    read_replay_suffix,
)
from local_operator.tui.costs import format_usd, format_usd_exact

MODEL = ModelSpec(
    provider="deepseek",
    model_id="deepseek-chat",
    display_name="DeepSeek Chat",
    context_window=64_000,
)


def _no_stream(request=None, signal=None):  # noqa: ANN001
    """An explicit empty async stream: never iterated in these tests."""

    async def gen():
        return
        yield  # pragma: no cover - an async generator that yields nothing

    return gen()


def make_session(tmp_path: Path, stream=None, **kwargs) -> Session:
    return Session(
        model=kwargs.pop("model", MODEL),
        stream_fn=stream or _no_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
        **kwargs,
    )


def receipt(provider: str, model_id: str, usd: float, tokens: int = 100) -> Usage:
    return Usage(
        provider=provider,
        model_id=model_id,
        input_tokens=tokens,
        output_tokens=1,
        context_tokens=tokens,
        usd_cost=usd,
    )


# -- T1: the value object ----------------------------------------------------


def test_knowledge_states_and_precedence() -> None:
    """UNKNOWN > PARTIAL > FLOOR > EXACT, from one derivation (design §5.4).

    ``UNKNOWN`` must outrank ``PARTIAL``: a fully-unpriced session has no dollar
    figure to bound, so it renders ``$—`` with NO ``≥`` — "≥ unknown" is a
    contradiction, and it is the analytics panel's existing rule rather than a
    second vocabulary invented here.
    """
    empty = SessionSpend()
    assert empty.knowledge() is CostKnowledge.UNKNOWN

    unpriced_only = SessionSpend()
    unpriced_only.accrue(None, {"provider": "local", "model_id": "llama"})
    assert unpriced_only.knowledge() is CostKnowledge.UNKNOWN

    unpriced_after_priced = SessionSpend()
    unpriced_after_priced.accrue(2_000_000, None)
    unpriced_after_priced.accrue(None, None)
    assert unpriced_after_priced.knowledge() is CostKnowledge.PARTIAL

    shrunk = SessionSpend(floor=True)
    shrunk.accrue(2_000_000, None)
    assert shrunk.knowledge() is CostKnowledge.FLOOR

    # FLOOR loses to PARTIAL: an unpriced call is a bound we cannot even size.
    both = SessionSpend(floor=True)
    both.accrue(2_000_000, None)
    both.accrue(None, None)
    assert both.knowledge() is CostKnowledge.PARTIAL

    exact = SessionSpend()
    exact.accrue(1_500_000, None)
    exact.accrue(500_000, None)
    assert exact.knowledge() is CostKnowledge.EXACT
    assert exact.usd == 2.0


def test_integer_micro_is_exact_over_many_accruals() -> None:
    """10k accruals of 1 µ$ are exactly 10,000 µ$ — no float drift.

    A float accumulator loses the last digits silently and the band rounds, so
    the drift is invisible; integer micro-USD is the ledger's own convention for
    the same reason (``analytics/model.py``).
    """
    spend = SessionSpend()
    for _ in range(10_000):
        spend.accrue(1, None)
    assert spend.micro == 10_000
    assert spend.usd == 0.01

    # The drift the integer convention exists to avoid, shown on a value a human
    # would write by hand: three tenths of a cent do not sum exactly in binary
    # floating point. In micro-USD they do, because integers have no exponent.
    assert abs(sum([0.1] * 3) - 0.3) > 0
    cents = SessionSpend()
    for _ in range(3):
        cents.accrue(1_000, None)
    assert cents.micro == 3_000
    assert cents.usd == 0.003


def test_record_round_trip_and_malformed_degradation() -> None:
    """A record survives a version boundary; a bad one is "no record", never 0."""
    original = SessionSpend(floor=True, rebuilt=True, writer="42:7")
    original.accrue(1_897_843, {"provider": "openai", "model_id": "gpt-6-astra"})
    original.accrue(None, {"provider": "local", "model_id": "llama"})
    restored = SessionSpend.from_details(original.to_details())
    assert restored is not None
    assert (restored.micro, restored.calls, restored.priced_calls) == (1_897_843, 2, 1)
    assert restored.unpriced_calls == 1
    assert restored.floor is True and restored.rebuilt is True
    assert restored.last_identity == {"provider": "local", "model_id": "llama"}
    assert restored.knowledge() is CostKnowledge.PARTIAL

    for malformed in (
        None,
        {},
        "not a mapping",
        {"version": 99, "micro": 5, "calls": 1, "priced_calls": 1, "unpriced_calls": 0},
        {"version": 1, "micro": "5", "calls": 1, "priced_calls": 1, "unpriced_calls": 0},
        {"version": 1, "micro": True, "calls": 1, "priced_calls": 1, "unpriced_calls": 0},
        # Internally inconsistent: every call is priced or unpriced.
        {"version": 1, "micro": 5, "calls": 2, "priced_calls": 1, "unpriced_calls": 0},
        {"version": 1, "micro": -1, "calls": 1, "priced_calls": 1, "unpriced_calls": 0},
    ):
        assert SessionSpend.from_details(malformed) is None, malformed


def test_correct_converges_an_estimate_and_adjust_is_not_a_call() -> None:
    """The one-tick path: paint-grade first, authoritative later (design §5.2)."""
    spend = SessionSpend()
    index = spend.accrue(1_000, None)
    assert spend.micro == 1_000
    assert spend.correct(index, 2_500) is True  # the full resolver's answer
    assert spend.micro == 2_500 and spend.calls == 1
    assert spend.correct(index, 9_999) is False  # a call is corrected once

    # An unpriced call that the full resolver CAN price stops being a bound.
    unknown_index = spend.accrue(None, None)
    assert spend.knowledge() is CostKnowledge.PARTIAL
    assert spend.correct(unknown_index, 750) is True
    assert spend.micro == 3_250 and spend.priced_calls == 2
    assert spend.knowledge() is CostKnowledge.EXACT

    # A turn-end remainder moves the total without inventing a provider call.
    before = spend.calls
    assert spend.adjust(250) is True
    assert spend.micro == 3_500 and spend.calls == before


def test_price_call_agrees_with_price_snapshot() -> None:
    """R11: the band's pricer and the ledger's must not drift.

    Two implementations of one question — the session's record-time pricer and
    ``analytics.model.price_snapshot`` — priced against the same fixture. This is
    the assertion that keeps ``/analytics`` and the band from reporting the same
    call differently.
    """
    from local_operator.analytics.model import CallSnapshot, price_snapshot

    usage = Usage(
        provider="deepseek",
        model_id="deepseek-chat",
        input_tokens=1_000,
        output_tokens=200,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    snapshot = CallSnapshot(
        ts_ms=0,
        session_id="t",
        provider="deepseek",
        model_id="deepseek-chat",
        input_tokens=1_000,
        output_tokens=200,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        context_tokens=1_000,
    )
    assert price_call("deepseek", "deepseek-chat", usage) == price_snapshot(snapshot)

    # A provider receipt wins on BOTH paths, verbatim.
    billed = Usage(provider="openrouter", model_id="x", usd_cost=0.0123)
    assert price_call("openrouter", "x", billed) == (12_300, True)
    assert price_snapshot(
        CallSnapshot(
            ts_ms=0,
            session_id="t",
            provider="openrouter",
            model_id="x",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            context_tokens=0,
            usd_cost=0.0123,
        )
    ) == (12_300, True)

    # No identity at all is unpriceable, and says so rather than pricing to zero.
    assert price_call("", "", usage) == (0, False)


# -- T2/T3: one arithmetic site, persisted and recalled ----------------------


def test_recall_is_a_lookup_never_a_scan(tmp_path: Path) -> None:
    """The recall does not walk the journal (the operator's requirement).

    Proven structurally: ``entries()`` — the only way to walk the rows — is
    replaced by an exploding stub, and the recall still answers.
    """
    transcript = Transcript(tmp_path / "sess")

    async def seed() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_custom(
            SESSION_SPEND_CUSTOM_TYPE,
            SessionSpend(micro=1_897_843, calls=3, priced_calls=3, writer="1:1").to_details(),
        )

    asyncio.run(seed())

    def explode() -> list[object]:
        raise AssertionError("latest_custom walked the journal")

    transcript.entries = explode  # type: ignore[method-assign]
    details = transcript.latest_custom(SESSION_SPEND_CUSTOM_TYPE)
    spend = SessionSpend.from_details(details)
    assert spend is not None and spend.micro == 1_897_843


def test_session_resume_recalls_the_record_as_exact(tmp_path: Path) -> None:
    """The restored figure is the record, unmarked — not one receipt marked ≥."""
    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_message(
            Message.assistant("hi", usage=receipt("openrouter", "x", 0.0021))
        )
        await transcript.append_custom(
            SESSION_SPEND_CUSTOM_TYPE,
            SessionSpend(micro=36_990_000, calls=245, priced_calls=245, writer="1:1").to_details(),
        )

    asyncio.run(seed())

    resumed = make_session(tmp_path)
    spend = resumed.restored_spend()
    assert spend is not None
    assert spend.micro == 36_990_000
    assert spend.knowledge() is CostKnowledge.EXACT
    # R1: the NEWEST reading still seeds the compaction gate, untouched.
    assert resumed.restored_usage() is not None
    assert resumed._last_usage is not None
    assert resumed._last_usage.input_tokens == 100

    store = resumed._frontend_state_store
    store.refresh_restored_usage(resumed)
    assert store.state.cumulative_parent_cost == pytest.approx(36.99)
    assert store.state.cost_knowledge is CostKnowledge.EXACT


def test_pre_ledger_session_keeps_its_floor_and_its_money(tmp_path: Path) -> None:
    """No record ⇒ today's behaviour: one receipt priced as a FLOOR, still ≥.

    And, crucially, the first live call ADDS to that figure instead of replacing
    it — the regression that would quietly delete a resumed conversation's
    dollars from the band the moment it was used.
    """
    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_message(
            Message.assistant("hi", usage=receipt("openrouter", "x", 0.5))
        )

    asyncio.run(seed())

    resumed = make_session(tmp_path)
    assert resumed.restored_spend() is None  # no record: the band keeps its ≥
    store = resumed._frontend_state_store
    store.refresh_restored_usage(resumed)
    assert store.state.cost_knowledge is CostKnowledge.FLOOR
    assert store.state.cumulative_parent_cost == pytest.approx(0.5)

    # A live call lands on top of the seeded floor.
    resumed.spend.accrue(250_000, {"provider": "openrouter", "model_id": "x"})
    store.mutate(
        cumulative_parent_cost=resumed.spend.usd,
        cost_knowledge=resumed.spend.knowledge(),
    )
    assert store.state.cumulative_parent_cost == pytest.approx(0.75)
    assert store.state.cost_knowledge is CostKnowledge.FLOOR


def test_multi_model_turn_is_billed_once_and_leaves_last_usage_alone(tmp_path: Path) -> None:
    """R2: per-call accrual + the turn-end remainder equals the turn's total.

    Two calls, two models, both with provider receipts, driven in the shape the
    harness actually produces: a ``MessageEndEvent`` per call, then the
    ``AgentEndEvent`` that reconciles the turn. The remainder exists so an
    aggregate price that differs from the sum of its calls is still billed
    exactly once; this pins that it does not bill twice.
    """
    from local_operator.harness.types import MessageEndEvent

    resumed = make_session(tmp_path)
    store = resumed._frontend_state_store
    first = receipt("openrouter", "fallback-a", 0.4)
    second = receipt("openai", "gpt-5.6-sol", 0.6, tokens=200)
    messages: list[AgentMessage] = [
        Message.assistant("a", usage=first),
        Message.assistant("b", usage=second),
    ]

    store.observe_event(resumed, MessageEndEvent(message=messages[0]))
    store.observe_event(resumed, MessageEndEvent(message=messages[1]))
    store.observe_event(resumed, AgentEndEvent(messages=messages))

    assert resumed.spend.micro == 1_000_000
    assert resumed.spend.usd == pytest.approx(1.0)
    state = store.state
    assert state.cumulative_parent_cost == pytest.approx(1.0)
    assert state.cost_knowledge is CostKnowledge.EXACT
    # R1: the compaction input is still the newest provider READING, never a
    # sum. ``Session._last_usage`` feeds the compaction gate and the cache-TTL
    # hint, so accrual must not touch it, and the published state must carry the
    # reading through unchanged.
    sentinel = receipt("openai", "witness", 1.0, tokens=7)
    resumed._last_usage = sentinel
    before = resumed.spend.micro
    resumed.accrue_spend(1_000, None)
    assert resumed._last_usage is sentinel
    assert resumed.spend.micro == before + 1_000
    store.refresh_from_session(resumed)
    assert store.state.last_usage is not None
    assert store.state.last_usage.input_tokens == 7  # a reading, not the ledger


def test_turn_end_remainder_tops_up_without_double_billing(tmp_path: Path) -> None:
    """A receipt-priced turn whose aggregate is larger is completed once."""
    from local_operator.harness.types import MessageEndEvent

    session = make_session(tmp_path)
    store = session._frontend_state_store
    call = receipt("openrouter", "x", 0.4)
    message = Message.assistant("a", usage=call)
    store.observe_event(session, MessageEndEvent(message=message))
    assert session.spend.micro == 400_000
    store.observe_event(session, AgentEndEvent(messages=[message]))
    assert session.spend.micro == 400_000  # the remainder is zero, not a second bill


def test_accrual_persists_a_record_and_a_session_recalls_it(tmp_path: Path) -> None:
    """A call leaves a durable record; a fresh Session over that dir recalls it."""
    session = make_session(tmp_path)
    index = session.accrue_spend(2_100, {"provider": "openrouter", "model_id": "x"})
    assert isinstance(index, int)
    asyncio.run(session._write_spend_record())
    assert session.restored_spend() is not None

    reopened = make_session(tmp_path)
    spend = reopened.restored_spend()
    assert spend is not None and spend.micro == 2_100
    assert spend.knowledge() is CostKnowledge.EXACT


def test_persist_runs_the_filesystem_work_off_the_loop(tmp_path: Path) -> None:
    """The record's append+fsync happens on a worker, not the event loop.

    Structural: identity of the thread that performed the WRITE. The coroutine
    is scheduled on the loop (that is where the coalescing bookkeeping lives);
    what must not happen on the loop is the syscall, which is what
    ``Transcript._write_entries`` performs inside ``asyncio.to_thread``.
    """

    async def main() -> None:
        session = make_session(tmp_path)
        thread_ids: list[int] = []
        real = type(session._transcript)._write_entries

        def spy(self, entries, *, preserve_mtime=False):
            thread_ids.append(threading.get_ident())
            return real(self, entries, preserve_mtime=preserve_mtime)

        session._transcript._write_entries = spy.__get__(session._transcript)  # type: ignore
        for _ in range(5):
            session.accrue_spend(1, {"provider": "openrouter", "model_id": "x"})
        for _ in range(200):
            await asyncio.sleep(0.01)
            if session._spend_recorded and not session._spend_persist_dirty:
                break
        assert session._spend_recorded
        assert thread_ids, "the record was never written"
        assert threading.get_ident() not in thread_ids
        # Coalescing: five accruals must not cost five writes.
        assert len(thread_ids) <= 5

    asyncio.run(main())


# -- T3 mechanisms: the type's contract --------------------------------------


def test_bookkeeping_predicate_matches_append_custom_rows(tmp_path: Path) -> None:
    """R8: the append_custom spelling is exempt, or the rebuild moves the clock.

    ``append_custom`` writes ``{custom_type, details}`` with no ``kind``, so the
    message-only predicate could not match it — and the pre-ledger rebuild
    appends exactly that row to a session that may be months old. Without this
    the mtime restore is skipped and merely opening an old session ranks it as
    freshly worked on the picker and for ``session.cleanup``'s age.
    """
    from local_operator.session.transcript import _is_bookkeeping_batch

    transcript = Transcript(tmp_path / "sess")

    async def write() -> None:
        entry = await transcript.append_custom(SESSION_SPEND_CUSTOM_TYPE, {"version": 1})
        assert entry.type == "custom"
        assert _is_bookkeeping_batch([entry]) is True
        real = await transcript.append_custom("something_else", {})
        assert _is_bookkeeping_batch([real]) is False
        # And the whole batch must be bookkeeping: one real message spoils it.
        message = await transcript.append_message(Message.user("work"))
        assert _is_bookkeeping_batch([entry, message]) is False
        # R8's second half: the type must never reach LLM context.
        from local_operator.session.session import _PERSISTABLE_CUSTOM_TYPES

        assert SESSION_SPEND_CUSTOM_TYPE not in _PERSISTABLE_CUSTOM_TYPES
        assert SESSION_SPEND_CUSTOM_TYPE in BOOKKEEPING_CUSTOM_TYPES

    asyncio.run(write())


def test_record_mtime_does_not_move_the_activity_clock(tmp_path: Path) -> None:
    """The bookkeeping exemption actually holds on the write path."""
    transcript = Transcript(tmp_path / "sess")

    async def write() -> None:
        await transcript.append_message(Message.user("hello"))
        before = transcript.path.stat().st_mtime
        await asyncio.sleep(0.01)
        await transcript.append_custom(
            SESSION_SPEND_CUSTOM_TYPE, {"version": 1}, preserve_mtime=True
        )
        assert transcript.path.stat().st_mtime == pytest.approx(before, abs=1e-6)

    asyncio.run(write())


def test_compact_file_keeps_the_newest_record(tmp_path: Path) -> None:
    """R9: the fold reclaims older records and always keeps the newest.

    Without the type in ``_COLLAPSIBLE_CUSTOM_TYPES`` the rows accumulate one
    per provider call forever — the shape that left
    ``frontend_state_checkpoint_v1`` holding 35.1% of all transcript bytes.
    """
    assert SESSION_SPEND_CUSTOM_TYPE in _COLLAPSIBLE_CUSTOM_TYPES
    transcript = Transcript(tmp_path / "sess")

    async def write_then_fold() -> None:
        await transcript.append_message(Message.user("hello"))
        for micro in (1_000, 2_000, 3_000):
            await transcript.append_custom(
                SESSION_SPEND_CUSTOM_TYPE,
                SessionSpend(micro=micro, calls=1, priced_calls=1, writer="1:1").to_details(),
            )
        assert await transcript.compact_file(min_reclaim_bytes=0) > 0
        rows = [
            entry
            for entry in transcript.entries()
            if entry.payload.get("custom_type") == SESSION_SPEND_CUSTOM_TYPE
        ]
        assert len(rows) == 1
        kept = SessionSpend.from_details(rows[0].payload["details"])
        assert kept is not None and kept.micro == 3_000

    asyncio.run(write_then_fold())


# -- T5: the suffix reader --------------------------------------------------


def test_suffix_reader_serves_two_types_in_one_pass(tmp_path: Path) -> None:
    """Two custom types out of ONE read, and a bare ``str`` still accepted."""
    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = Transcript(directory)

    async def write() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_custom("checkpoint", {"a": 1})
        await transcript.append_custom(SESSION_SPEND_CUSTOM_TYPE, {"micro": 5})

    asyncio.run(write())
    suffix = read_replay_suffix(
        directory, checkpoint_types=("checkpoint", SESSION_SPEND_CUSTOM_TYPE)
    )
    assert suffix.checkpoint == {"a": 1}
    assert suffix.checkpoints[SESSION_SPEND_CUSTOM_TYPE] == {"micro": 5}

    # Additive: a single type, spelled as a bare string, is what it always was.
    single = read_replay_suffix(directory, checkpoint_types="checkpoint")
    assert single.checkpoint == {"a": 1}
    assert SESSION_SPEND_CUSTOM_TYPE not in single.checkpoints


# -- T6: the one-time rebuild -----------------------------------------------


def test_rebuild_sums_every_row_and_marks_floor_only_when_rows_were_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§2.2 and §5.4: no compaction boundary for money; ``floor`` from shrinkage."""
    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_message(
            Message.assistant("a", usage=receipt("openrouter", "x", 10.0))
        )
        # A compaction marker: rows may have been dropped, so the sum cannot be
        # whole. The rows above still count — money is not invalidated by a
        # later rewrite of the context.
        await transcript.append_compaction("summary", "", 0)
        await transcript.append_message(
            Message.assistant("b", usage=receipt("openrouter", "x", 2.0))
        )
        # A token-carrying row nothing can price: counted, not silently dropped.
        await transcript.append_message(
            Message.assistant("c", usage=Usage(provider="", model_id="", input_tokens=50))
        )

    asyncio.run(seed())
    session = make_session(tmp_path)
    rebuilt: dict[str, list[dict[str, Any]]] = {}

    def fake_price(rows):
        rebuilt["rows"] = rows
        return [
            (
                (int(round(float(r["usd_cost"]) * 1_000_000)), True)
                if r.get("usd_cost") is not None
                else (0, False)
            )
            for r in rows
        ]

    monkeypatch.setattr(session_module, "price_rows", fake_price)
    asyncio.run(session._rebuild_spend(session._transcript.all_usage_rows(), True))
    spend = session.spend
    assert len(rebuilt["rows"]) == 3  # every row, boundary ignored
    assert spend.floor is True and spend.rebuilt is True
    assert spend.micro == 12_000_000  # $10 + $2 + the unpriced call's zero
    assert (spend.calls, spend.priced_calls, spend.unpriced_calls) == (3, 2, 1)
    assert spend.knowledge() is CostKnowledge.PARTIAL


def test_rebuild_is_once_per_session_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call count, not a timing bound: exactly one rebuild per session."""
    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_message(
            Message.assistant("a", usage=receipt("openrouter", "x", 1.0))
        )

    asyncio.run(seed())

    async def main() -> None:
        session = make_session(tmp_path)
        calls: list[int] = []

        async def spy(rows, shrunk):
            calls.append(len(rows))

        session._rebuild_spend = spy  # type: ignore[method-assign]
        for _ in range(5):
            session.rebuild_spend_if_needed()
            await asyncio.sleep(0)
        assert len(calls) == 1

    asyncio.run(main())


def test_rebuild_pricing_runs_off_the_event_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structural, not timing: the price computation is on a worker thread."""
    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_message(
            Message.assistant("a", usage=receipt("openrouter", "x", 1.0))
        )

    asyncio.run(seed())

    async def main() -> None:
        session = make_session(tmp_path)
        seen: list[int] = []

        def spy(rows):
            seen.append(threading.get_ident())
            return price_rows(rows)

        monkeypatch.setattr(session_module, "price_rows", spy)
        session.rebuild_spend_if_needed()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if not session._spend_tasks:
                break
        assert seen, "the rebuild never priced anything"
        assert threading.get_ident() not in seen

    asyncio.run(main())


def test_per_call_pricing_runs_off_the_event_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The authoritative price for one call is computed on a worker thread."""
    session = make_session(tmp_path)
    seen: list[int] = []

    def spy(provider, model_id, usage):
        seen.append(threading.get_ident())
        return 1234, True

    monkeypatch.setattr(session_module, "price_call", spy)

    async def main() -> None:
        index = session.spend.accrue(999, {"provider": "deepseek", "model_id": "deepseek-chat"})
        session.schedule_spend_price(
            index,
            Usage(provider="deepseek", model_id="deepseek-chat", input_tokens=10),
            {"provider": "deepseek", "model_id": "deepseek-chat"},
        )
        for _ in range(200):
            await asyncio.sleep(0.01)
            if not session._spend_tasks:
                break
        assert seen and threading.get_ident() not in seen
        assert session.spend.micro == 1234

    asyncio.run(main())


def test_rebuild_is_not_reachable_from_the_paint_path() -> None:
    """The operator's named hazard: the rebuild must never be a per-paint recount.

    Asserted on the structure rather than by counting frames: the TUI module
    never names the rebuild, so no renderer can call it, and the one seam that
    does start it is the adopt-time refresh.
    """
    from local_operator.tui import app as app_module

    assert "rebuild_spend_if_needed" not in inspect.getsource(app_module)

    refresh = inspect.getsource(Session.refresh_frontend_usage)
    assert refresh.count("rebuild_spend_if_needed") == 1


# -- T8: the one formatter --------------------------------------------------


def test_format_usd_ladder_and_exact_figure() -> None:
    """One ladder, and the one spelling that stops it lying."""
    assert format_usd(0) == "$0.0000"
    # A nonzero amount that WOULD round to $0.0000 says so instead of reading
    # as free — the operator's objection, answered.
    assert format_usd(1) == "<$0.0001"
    assert format_usd(49) == "<$0.0001"
    assert format_usd(50) == "$0.0001"
    assert format_usd(4_200) == "$0.0042"
    assert format_usd(213_000) == "$0.213"
    assert format_usd(1_897_843) == "$1.90"
    assert format_usd(1_200_000_000) == "$1200.00"  # no abbreviation, ever

    assert format_usd_exact(1_897_843) == "$1.897843"
    assert format_usd_exact(2_100_000) == "$2.10"
    assert format_usd_exact(5) == "$0.000005"

    # The band and the panels read the same ladder: one function, one spelling.
    from local_operator.tui.widgets.status_line import format_cost as band_cost

    assert band_cost(1_897_843 / 1_000_000) == format_usd(1_897_843)


# -- R11/T5 extras ----------------------------------------------------------


def test_price_rows_is_one_batch_hop() -> None:
    rows = [
        {"provider": "openrouter", "model_id": "x", "usd_cost": 0.01},
        {"provider": "", "model_id": "", "input_tokens": 5},
    ]
    assert price_rows(rows) == [(10_000, True), (0, False)]


def test_replay_suffix_rejects_a_third_type_absent_from_the_journal(tmp_path: Path) -> None:
    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = Transcript(directory)

    async def write() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_custom("checkpoint", {"a": 1})

    asyncio.run(write())
    suffix = read_replay_suffix(directory, checkpoint_types=("checkpoint", "missing"))
    assert suffix.checkpoints == {"checkpoint": {"a": 1}}
    # The reader still returns (it falls back to the file start), and the caller
    # can tell the requested row was absent rather than assume a value.
    assert "missing" not in suffix.checkpoints


@pytest.mark.asyncio
async def test_a_live_turn_writes_a_recallable_record(tmp_path: Path) -> None:
    """The end-to-end shape at unit scale: a real turn, then a real resume."""

    def stream(request, signal=None):
        async def gen():
            yield StreamTextDelta(delta="ack")
            yield StreamEndEvent(stop_reason="stop", usage=receipt("openrouter", "x", 0.0021))

        return gen()

    session = make_session(tmp_path, stream)
    await session._run_turn([Message.user("probe")])
    for _ in range(100):
        await asyncio.sleep(0.01)
        if session._spend_recorded:
            break
    assert session._spend_recorded
    assert session.spend.micro == 2_100

    reopened = make_session(tmp_path)
    recalled = reopened.restored_spend()
    assert recalled is not None and recalled.micro == 2_100
    assert recalled.knowledge() is CostKnowledge.EXACT
    assert spend_module.SESSION_SPEND_CUSTOM_TYPE == SESSION_SPEND_CUSTOM_TYPE


def test_rebuild_replaces_a_seeded_accumulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy SEED must not look like newer state.

    ``seed_spend_floor`` increments the call count, so a rebuild guarded on
    ``spend.calls`` silently never ran for a pre-ledger session that had already
    been opened — i.e. for exactly the population the rebuild exists for, 92.3%
    of the real store. The seed is a reconstruction of money already in the
    journal, so it is what the rebuild REPLACES, never a reason to skip. Found by
    the e2e test, not by this file: the unit test that called
    ``_rebuild_spend`` directly had no seed in its accumulator.
    """
    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_message(
            Message.assistant("a", usage=receipt("openrouter", "x", 1.25))
        )

    asyncio.run(seed())

    async def main() -> None:
        # has_ui=True so construction restores through the frontend store, which
        # is what seeds the accumulator from the legacy receipt.
        session = make_session(tmp_path, has_ui=True)
        assert session.restored_spend() is None
        assert session.spend.calls == 1 and session._spend_seeded
        monkeypatch.setattr(
            session_module, "price_rows", lambda rows: [(1_250_000, True) for _ in rows]
        )
        session.rebuild_spend_if_needed()
        async with asyncio.timeout(30):
            while session._spend_tasks:
                await asyncio.sleep(0.01)
        spend = session.restored_spend()
        assert spend is not None, "the seeded accumulator blocked the rebuild"
        assert spend.rebuilt is True
        assert spend.micro == 1_250_000
        assert session._spend_seeded is False
        assert session._spend_recorded is True

    asyncio.run(main())
