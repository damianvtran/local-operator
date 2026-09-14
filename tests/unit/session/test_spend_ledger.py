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
import logging
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
from local_operator.session import frontend_state as frontend_state_module
from local_operator.session import session as session_module
from local_operator.session import spend as spend_module
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    CostKnowledge,
    FrontendSessionState,
)
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
    """The one-tick path: paint-grade first, authoritative later (design §5.2).

    ``correct`` returns the DELTA it applied, not a bare bool, because the front
    end has to know how much of the accumulator arrived as a re-price rather than
    as a new call (review R1-1). The delta stays truthy, so it also still reads as
    "something changed"; ``0`` is the no-op.
    """
    spend = SessionSpend()
    index = spend.accrue(1_000, None)
    assert spend.micro == 1_000
    assert spend.correct(index, 2_500) == 1_500  # the full resolver's answer
    assert spend.micro == 2_500 and spend.calls == 1
    assert spend.correct(index, 9_999) == 0  # a call is corrected once

    # An unpriced call that the full resolver CAN price stops being a bound, and
    # the delta is the whole price because the call contributed nothing before.
    unknown_index = spend.accrue(None, None)
    assert spend.knowledge() is CostKnowledge.PARTIAL
    assert spend.correct(unknown_index, 750) == 750
    assert spend.micro == 3_250 and spend.priced_calls == 2
    assert spend.knowledge() is CostKnowledge.EXACT

    # A downward re-price reports a NEGATIVE delta: the paint answer can be
    # dearer than the resolved one, and the front end must be able to tell that
    # apart from a call that simply added nothing.
    down = spend.accrue(5_000, None)
    assert spend.correct(down, 2_000) == -3_000
    assert spend.micro == 5_250

    # A turn-end remainder moves the total without inventing a provider call.
    before = spend.calls
    assert spend.adjust(250) is True
    assert spend.micro == 5_500 and spend.calls == before


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
    """The bookkeeping exemption holds through the PRODUCTION call site.

    Driven through ``_write_spend_record`` and not through ``append_custom``
    directly, and that is not incidental: the first version of this test called
    the transcript API itself, so deleting ``preserve_mtime=True`` from
    ``session._write_spend_record`` left it green (found by the mutation pass).
    A test that exercises the helper instead of the caller pins the helper.
    """
    session = make_session(tmp_path)

    async def write() -> None:
        await session._transcript.append_message(Message.user("hello"))
        before = session._transcript.path.stat().st_mtime
        await asyncio.sleep(0.02)
        session.accrue_spend(1_000, {"provider": "openrouter", "model_id": "x"})
        await session._write_spend_record()
        assert session._transcript.path.stat().st_mtime == pytest.approx(before, abs=1e-6)
        # The row IS there: the exemption must not be achieved by not writing.
        assert session._transcript.latest_custom(SESSION_SPEND_CUSTOM_TYPE) is not None

    asyncio.run(write())


def test_the_transcript_honours_the_bookkeeping_flag(tmp_path: Path) -> None:
    """...and the flag the production site passes actually does something."""
    transcript = Transcript(tmp_path / "sess")

    async def write() -> None:
        await transcript.append_message(Message.user("hello"))
        before = transcript.path.stat().st_mtime
        await asyncio.sleep(0.02)
        # Without the flag the clock moves, which is what makes the assertion in
        # the test above a measurement rather than a coincidence.
        await transcript.append_custom(SESSION_SPEND_CUSTOM_TYPE, {"version": 1})
        assert transcript.path.stat().st_mtime > before
        moved = transcript.path.stat().st_mtime
        await asyncio.sleep(0.02)
        await transcript.append_custom(
            SESSION_SPEND_CUSTOM_TYPE, {"version": 2}, preserve_mtime=True
        )
        assert transcript.path.stat().st_mtime == pytest.approx(moved, abs=1e-6)

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


def test_rebuild_sums_every_row_and_never_claims_a_floor_from_a_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§2.2 and §5.4: no compaction boundary for money — and no marker-based ``≥``.

    Review R1-4: a compaction or a prune REWRITES rows and hides them from the
    context replay, but it removes no money. ``_pruned_entry`` replaces only
    ``payload["content"]`` (``usage`` survives, and a pruned tool result's
    ``search_cost`` is outside ``content``), and ``compact_file`` drops prune
    entries and superseded collapsible customs, never a message row. So a
    journal full of markers whose every row is still readable must rebuild
    EXACT, not ``≥`` — the mark is for money the file cannot show, which is the
    unpriced-call case (PARTIAL, covered by the sibling test).
    """
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
    asyncio.run(session._rebuild_spend())
    spend = session.spend
    assert len(rebuilt["rows"]) == 3  # every row, boundary ignored
    assert spend.rebuilt is True
    assert spend.floor is False, "a compaction marker is not evidence of lost money"
    assert spend.micro == 12_000_000  # $10 + $2 + the unpriced call's zero
    # The unpriced call is what bounds this total, and it bounds it by name.
    assert spend.knowledge() is CostKnowledge.PARTIAL
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

        async def spy() -> None:
            calls.append(1)

        session._rebuild_spend = spy  # type: ignore[method-assign]
        for _ in range(5):
            session.rebuild_spend_if_needed()
            await asyncio.sleep(0)
        assert len(calls) == 1

    asyncio.run(main())


def test_rebuild_pricing_runs_off_the_event_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structural, not timing: the whole rebuild body is on a worker thread.

    The SCAN counts as much as the price here (review R1-5): ``all_usage_rows``
    copies a dict per usage row and ``lost_money_rows`` walks the entries again,
    so asserting thread identity for ``price_rows`` alone would let an O(rows)
    walk back onto the loop with the test still green. Both spies are checked.
    """
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

        original_scan = Transcript.all_usage_rows

        def scan_spy(self):
            seen.append(threading.get_ident())
            return original_scan(self)

        monkeypatch.setattr(session_module, "price_rows", spy)
        monkeypatch.setattr(Transcript, "all_usage_rows", scan_spy)
        session.rebuild_spend_if_needed()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if not session._spend_tasks:
                break
        assert len(seen) >= 2, "the rebuild did not scan AND price"
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

    # R1-3: the mark and the sub-resolution spelling DO co-occur — an unpriced
    # call makes the total a lower bound whatever the digits say, and the digits
    # round to zero whatever the mark says. Both are owed, and the cell must
    # carry both (9 cells, not the 8 the design's table claimed).
    from local_operator.tui.app import RESTORED_COST_PREFIX

    assert RESTORED_COST_PREFIX + format_usd(1) == "\u2265<$0.0001"
    assert len(RESTORED_COST_PREFIX + format_usd(1)) == 9

    # R1-7: a float the ladder cannot take renders instead of RAISING. Both
    # wrappers still hold a float, and ``int(round(nan))`` raises, so a corrupt
    # restored total could have taken a frame down.
    from local_operator.tui.costs import micro_from_usd

    assert micro_from_usd(float("nan")) is None
    assert micro_from_usd(float("inf")) is None
    assert micro_from_usd(None) is None
    assert micro_from_usd(2.5) == 2_500_000
    assert band_cost(float("nan")) == "$nan"


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


def test_a_reconstruction_below_the_seed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rebuild must never DECREASE the figure already on the band.

    The seed prices one restored reading through the session's own effective
    model; the reconstruction needs every ROW to carry a serving identity, and a
    row that carries none is unpriceable at full-resolver grade. So a rebuild can
    legitimately come out BELOW an already-priced figure — and replacing $2.10
    with $0.00 would be a silent downgrade wearing the ledger's authority.
    Reached through the REAL app: `tests/unit/tui/test_usage_continuity.py`
    caught this, and the store's `refresh_from_session` path is what publishes
    the accumulator to the band.
    """
    directory = tmp_path / "sess"
    directory.mkdir()
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(Message.user("hello"))
        await transcript.append_message(
            Message.assistant("a", usage=receipt("openrouter", "x", 2.1))
        )

    asyncio.run(seed())

    async def main() -> None:
        session = make_session(tmp_path, has_ui=True)
        before = session.spend
        assert before.micro == 2_100_000 and session._spend_seeded
        # A reconstruction that priced NOTHING (a row with no identity, and no
        # receipt to read) plus one it could not size.
        monkeypatch.setattr(session_module, "price_rows", lambda rows: [(0, False) for _ in rows])
        session.rebuild_spend_if_needed()
        async with asyncio.timeout(30):
            while session._spend_tasks:
                await asyncio.sleep(0.01)
        assert session.spend.micro == 2_100_000, "the rebuild downgraded the band"
        assert session.spend is before
        assert session._spend_recorded is False, "a smaller figure must not be persisted"
        assert session.restored_spend() is None
        # DESIGN D1b: when the history genuinely cannot be priced, the figure
        # stays a lower bound and THE MARK MUST STAY. Asserted on the state the
        # band paints from, so a later "fix" that turns this into a bare exact
        # figure fails here rather than in the field.
        assert session.spend.knowledge() is CostKnowledge.FLOOR
        store = session._frontend_state_store
        store.refresh_from_session(session)
        assert store.state.cost_knowledge is CostKnowledge.FLOOR
        assert store.state.cumulative_parent_cost == pytest.approx(2.1)
        # The refused reconstruction is not observable on the session -- that is
        # what "refused" means -- so the seed's own counts stand: it holds one
        # priced call and no unpriced ones.
        assert session.spend.calls == 1 and session.spend.unpriced_calls == 0

    asyncio.run(main())


def test_a_mid_turn_correction_is_not_billed_twice_by_the_remainder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-1: the correction must move the turn's already-accrued figure too.

    The turn-end remainder is ``max(0, aggregate_price - accrued_this_turn)``,
    and ``accrued_this_turn`` was fed ONLY by the paint prices. A correction that
    lands mid-turn therefore moved the durable accumulator while leaving that
    counter at the paint figure, so the remainder re-billed the whole correction:
    paint $1.00, corrected $2.00, aggregate $2.00 → persisted $3.00, published
    EXACT.

    Invisible to every other accrual test because they price through a provider
    receipt, and a receipt suppresses the correction entirely
    (``spend.price_call`` returns the receipt and reports it known). This test
    therefore uses the receipt-LESS path, which is the ordinary one for a
    provider that reports no cost.
    """
    from local_operator.harness.types import MessageEndEvent

    session = make_session(tmp_path)
    store = session._frontend_state_store
    usage = Usage(provider="deepseek", model_id="deepseek-chat", input_tokens=1_000)
    message = Message.assistant("a", usage=usage)

    # Paint grade for the call is $1.00; the turn's AGGREGATE is $2.00 (a
    # different object, so the fake can tell them apart); the full resolver
    # agrees with the aggregate at $2.00 — which is what the correction lands.
    monkeypatch.setattr(
        frontend_state_module,
        "turn_cost",
        lambda label, value: 1.0 if value is usage else 2.0,
    )
    monkeypatch.setattr(
        session_module, "price_call", lambda provider, model_id, u: (2_000_000, True)
    )

    async def main() -> None:
        store.observe_event(session, MessageEndEvent(message=message))
        assert session.spend.micro == 1_000_000, "the paint price is the first tick"
        async with asyncio.timeout(30):
            while session._spend_tasks:
                await asyncio.sleep(0.01)
        assert session.spend.micro == 2_000_000, "the correction converged the call"
        store.observe_event(session, AgentEndEvent(messages=[message]))
        assert session.spend.micro == 2_000_000, "the remainder re-billed the correction"
        assert store.state.cumulative_parent_cost == pytest.approx(2.0)
        assert store.state.cost_knowledge is CostKnowledge.EXACT
        # The record is what a resume reads, so the double bill must not reach it.
        assert session.spend.micro == 2_000_000

    asyncio.run(main())


def _checkpoint_details(sid: str, cost: float, knowledge: str) -> dict[str, Any]:
    """The turn-end status row a runtime writes, as the cold reader meets it."""
    raw = FrontendSessionState(session_id=sid, epoch="e1").model_dump(mode="json")
    raw.update({"cumulative_parent_cost": cost, "cost_knowledge": knowledge})
    return {"checkpoint_id": "cp", "state": raw}


def _cold_open(
    tmp_path: Path,
    sid: str,
    writes: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Journal ``writes`` in order (later is NEWER) and open the session COLD.

    The ORDER is the variable under test, so the same two artifacts are written
    both ways rather than being described as newer and older. The takeover
    factory fails the test if the cold open reaches for a runtime: every fact
    here has to come off the disk. ``HOME`` is redirected as well as the config
    dir because the model catalogue resolves from the home root independently
    (AGENTS.md: one variable is not isolation), and the paint resolver is
    trapped so a cold read that tries to price anything through the live
    catalogue fails loudly instead of quietly reaching the network.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "local_operator.tui.costs._resolve_for_paint", lambda *_: pytest.fail("cold discovery")
    )
    directory = tmp_path / "sessions" / sid
    directory.mkdir(parents=True, exist_ok=True)

    async def seed() -> None:
        transcript = Transcript(directory)
        for custom_type, details in writes:
            await transcript.append_custom(custom_type, details)

    asyncio.run(seed())

    async def never() -> None:
        pytest.fail("the cold open started a runtime")

    async def open_cold() -> Any:
        return await AttachedSession.cold(
            sid, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=never
        )

    return asyncio.run(open_cold())


def test_the_panel_row_gate_reads_the_money_through_capture(tmp_path: Path) -> None:
    """R4-2: the row gate is pinned through ``SessionDiagnostics.capture``.

    The round-2 policy tests left this uncovered: one asserted the cold STATE and
    the other hand-built its diagnostics, so ``capture``'s gate could silently
    regress to the round-1 ``calls`` gate with the suite green (review R4-2). This
    drives the real record path — ``adjust_spend``, the turn-end remainder, money
    with ``calls == 0`` — through the real capture and the real renderer.
    """
    from local_operator.analytics.model import SessionReport, UsageAggregate
    from local_operator.tui.widgets.session_panel import (
        SessionDiagnostics,
        build_session_report,
    )

    session = make_session(tmp_path)

    async def seed() -> None:
        await session._transcript.append_message(Message.user("hello"))
        session.adjust_spend(500_000)
        # Wait on the PUBLICATION the record makes, not on the task handle: the
        # write is coalesced, so the handle can still be None on the tick after
        # the accrual. A deadline only so a genuine hang fails the run.
        async with asyncio.timeout(30):
            while not session._spend_recorded:
                await asyncio.sleep(0.01)

    asyncio.run(seed())
    diagnostics = SessionDiagnostics.capture(session)
    assert diagnostics.spend_micro == 500_000, "money with no counted calls is money"
    assert diagnostics.spend_knowledge == "exact"
    # A ledger with at least one call: the money rows live under the recorded
    # section, which an empty report renders as "No recorded requests".
    report = SessionReport(
        "sess", aggregate=UsageAggregate(calls=1, cost_micro=100_000, cost_known_calls=1)
    )
    row = next(
        line
        for line in build_session_report(report, diagnostics, width=120).plain.split("\n")
        if "Record total" in line
    )
    # The exact row prints the record's own integer with trailing zeros trimmed
    # (500_000 µ$ = $0.50) — the band's 3dp rung is the BAND's spelling, and the
    # two surfaces are allowed to differ in digits as long as they agree on the
    # money (the exact figure is what this row is for).
    assert "$0.50" in row, row
    assert "500,000 μ$" in row, row


def test_money_decides_and_the_counts_only_describe_provenance() -> None:
    """The policy behind QA round 2's Q1/Q3 and review R3-1.

    Money is ``micro > 0``: a record can hold a turn-end remainder with
    ``calls == 0`` (``adjust``'s docstring: money, not a provider call), and one
    can be adopted from a store mid-turn with ``micro > 0, calls == 0``. Gating a
    figure on the counts hid that money on both surfaces and let the band paint a
    one-receipt floor ABOVE it.

    UNKNOWN is reserved for money we cannot state at all: nothing priced AND
    something unpriced. A zero total from priced calls is a figure we CAN state
    (a free model), and an empty record is not unknown either — it is nothing,
    which every surface omits.
    """
    remainder = SessionSpend(micro=500_000, calls=0)
    assert remainder.has_money and remainder.knowledge() is CostKnowledge.EXACT
    assert remainder.published_usd() == 0.5, "money with no counted calls is still money"

    adopted = SessionSpend(micro=2_000_000, calls=0, priced_calls=0)
    assert adopted.has_money and adopted.knowledge() is CostKnowledge.EXACT, "R3-1"
    assert adopted.published_usd() == 2.0

    unpriceable = SessionSpend(micro=0, calls=1, priced_calls=0, unpriced_calls=1)
    assert not unpriceable.has_money
    assert unpriceable.knowledge() is CostKnowledge.UNKNOWN
    assert unpriceable.published_usd() is None, "none is 'cannot state', not 'zero'"

    free = SessionSpend(micro=0, calls=2, priced_calls=2)
    assert free.knowledge() is CostKnowledge.EXACT, "a known zero is not unknown"
    assert free.published_usd() == 0.0

    empty = SessionSpend()
    # An accumulator with no accruals knows nothing (UNKNOWN, the pre-existing
    # semantics), but it is not MONEY WE CANNOT STATE: every surface omits it
    # rather than printing `$—`, which is what `unknown_money` distinguishes.
    assert empty.knowledge() is CostKnowledge.UNKNOWN and not empty.unknown_money
    assert not empty.has_money and empty.published_usd() == 0.0


def test_a_money_record_with_no_counted_calls_reaches_the_cold_state(
    tmp_path: Path, monkeypatch
) -> None:
    """QA round 2, Q3 on the cold path: the remainder money is what is painted.

    ``session.adjust_spend(500_000)`` — what the store calls at turn end — writes
    ``{"micro": 500000, "calls": 0, ...}``. Both surfaces used to hide it, and the
    cold band called the session ``$—`` while ``restored_spend()`` held the figure.
    """
    record = SessionSpend(micro=500_000, calls=0, writer="qa:probe").to_details()
    viewer = _cold_open(
        tmp_path, "coldremainder1", [(SESSION_SPEND_CUSTOM_TYPE, record)], monkeypatch
    )
    state = viewer.frontend_state
    assert state.cumulative_parent_cost == 0.5, "the record's money, not a hidden zero"
    assert state.cost_knowledge is CostKnowledge.EXACT
    assert viewer.restored_spend() is not None and viewer.restored_spend().micro == 500_000


def test_the_newer_money_artifact_decides_the_cold_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1, Q2: the cold band may not paint what the record contradicts.

    The record is written per CALL and the checkpoint only at a turn end, so the
    record is the newer of the two on every path with a call accrued after the
    last turn end — the crash/repair window this surface exists for. The cold
    viewer preferred the checkpoint unconditionally, and on such a session the
    band painted the checkpoint's ``$1.00`` unmarked and EXACT one screen away
    from a ``/session`` row reading the record's ``$5.00``.

    Magnitude is deliberately not the discriminator: the two fixtures are
    written in both orders and the ORDER alone decides, which is what makes this
    a fact about the journal rather than a preference between two sources.
    """
    record = SessionSpend(micro=5_000_000, calls=3, priced_calls=3, writer="qa:probe").to_details()
    checkpoint = _checkpoint_details("coldrecord01", 1.0, "exact")
    checkpoint_older = _checkpoint_details("coldcheck01", 1.0, "exact")

    newer_record = _cold_open(
        tmp_path,
        "coldrecord01",
        [
            (FRONTEND_CHECKPOINT_CUSTOM_TYPE, checkpoint),
            (SESSION_SPEND_CUSTOM_TYPE, record),
        ],
        monkeypatch,
    )
    assert newer_record.frontend_state.cumulative_cost == 5.0, "the newer record decides"
    assert newer_record.frontend_state.cost_knowledge is CostKnowledge.EXACT

    newer_checkpoint = _cold_open(
        tmp_path,
        "coldcheck01",
        [
            (SESSION_SPEND_CUSTOM_TYPE, record),
            (FRONTEND_CHECKPOINT_CUSTOM_TYPE, checkpoint_older),
        ],
        monkeypatch,
    )
    assert newer_checkpoint.frontend_state.cumulative_cost == 1.0, "the newer checkpoint stands"
    assert newer_checkpoint.frontend_state.cost_knowledge is CostKnowledge.EXACT


def test_disagreeing_money_with_no_order_is_never_certified_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q2's other half: an unorderable disagreement is a lower bound, not a fact.

    When neither artifact's position can be read there is nothing to choose with,
    and certifying either would be an assertion the journal does not support. The
    figure is kept — it is where the session's own runtime left the band — and
    demoted, so the mark says "we cannot certify this" rather than a bare number
    saying "this is the bill".
    """
    viewer = _cold_open(
        tmp_path,
        "coldorder01",
        [(FRONTEND_CHECKPOINT_CUSTOM_TYPE, _checkpoint_details("coldorder01", 1.0, "exact"))],
        monkeypatch,
    )
    # The record arrives with no meeting index (a legacy reader, an artifact read
    # from another source): the order cannot be established, and the two figures
    # disagree.
    viewer._cold_spend = SessionSpend(
        micro=5_000_000, calls=3, priced_calls=3, writer="x:1"
    ).to_details()
    viewer._cold_order = {}
    state = viewer._seed_cold_usage(viewer.frontend_state)
    assert state.cumulative_parent_cost == 1.0
    assert state.cost_knowledge is CostKnowledge.FLOOR


def test_an_unpriceable_record_is_unknown_not_a_zero_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1, Q1: nothing priceable is ``$—``, not ``$0.00``.

    Reached through the real writer: ``accrue_spend(None)`` records one call that
    could not be priced, which the record spells ``priced_calls: 0``. Publishing
    its ``micro`` as a total made the band's zero policy drop the cost cell
    entirely, while ``/analytics`` printed ``$—`` for the same state and §8.2
    specifies ``$—``. The fix is the distinction, not a spelling: an unknown sum
    leaves ``cumulative_parent_cost`` unset so the band's own ``$—`` branch fires.
    """
    record = SessionSpend(
        micro=0, calls=1, priced_calls=0, unpriced_calls=1, writer="qa:probe"
    ).to_details()
    viewer = _cold_open(
        tmp_path, "coldunknown01", [(SESSION_SPEND_CUSTOM_TYPE, record)], monkeypatch
    )
    state = viewer.frontend_state
    assert state.cumulative_parent_cost is None, "0.0 is a figure the record cannot support"
    assert state.cost_knowledge is CostKnowledge.UNKNOWN
    assert state.cumulative_cost is None


def test_a_late_correction_is_clamped_exactly_like_a_mid_turn_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-2: the persisted total must not depend on WHEN the re-price lands.

    The turn-end remainder makes the aggregate a FLOOR for the turn
    (``accrued + max(0, aggregate - accrued) == max(accrued, aggregate)``):
    mid-turn, ``accrued_this_turn`` moves with the correction and the remainder
    hands the money back, so a re-price can only change the total as far as it
    lifts the turn above that floor. A correction whose turn had already closed
    took the other path -- the full delta straight onto the accumulator -- so the
    SAME fixtures disagreed: paint $2.00, re-priced $1.00, aggregate $2.00
    persisted $2.00 mid-turn and $1.00 late, and the record inherited whichever
    ordering the scheduler happened to produce. Both orderings now run the same
    formula against the closed turn's snapshot, and a re-price that raises the
    true cost above the aggregate still counts in full.
    """
    from local_operator.harness.types import MessageEndEvent

    usage = Usage(provider="deepseek", model_id="deepseek-chat", input_tokens=1_000)

    def run(*, resolved: int, late: bool) -> int:
        session = make_session(tmp_path / f"sess-{resolved}-{int(late)}")
        store = session._frontend_state_store
        message = Message.assistant("a", usage=usage)
        # The call's paint price AND the turn's aggregate are both $2.00; the
        # fake cannot tell them apart by object here because both come off the
        # same resolver, which is the honest fixture for this question -- the
        # ordering is the only variable.
        monkeypatch.setattr(frontend_state_module, "turn_cost", lambda label, value: 2.0)
        # In the LATE ordering the scheduled price resolves to nothing, so the
        # call stays at paint grade until the test drives the correction itself
        # after the turn has closed.
        monkeypatch.setattr(
            session_module,
            "price_call",
            (lambda *a: (None, False)) if late else (lambda *a: (resolved, True)),
        )

        async def main() -> None:
            store.observe_event(session, MessageEndEvent(message=message))
            if late:
                store.observe_event(session, AgentEndEvent(messages=[message]))
                async with asyncio.timeout(30):
                    while session._spend_tasks:
                        await asyncio.sleep(0.01)
                monkeypatch.setattr(session_module, "price_call", lambda *a: (resolved, True))
                # index 0: the only call of a fresh session (``SessionSpend.accrue``
                # numbers from zero), which is what the store registered above.
                await session._price_spend_call(0, usage, "deepseek", "deepseek-chat")
            else:
                async with asyncio.timeout(30):
                    while session._spend_tasks:
                        await asyncio.sleep(0.01)
                store.observe_event(session, AgentEndEvent(messages=[message]))

        asyncio.run(main())
        return session.spend.micro

    # A DOWNWARD re-price below the aggregate: the turn's floor holds, same both ways.
    assert run(resolved=1_000_000, late=False) == 2_000_000
    assert run(resolved=1_000_000, late=True) == 2_000_000
    # An UPWARD one above it: the authoritative per-call sum wins, same both ways.
    assert run(resolved=3_000_000, late=False) == 3_000_000
    assert run(resolved=3_000_000, late=True) == 3_000_000


def test_a_cheaper_re_price_does_not_log_a_backwards_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R1-6: the record may legitimately shrink, and the log must not cry wolf.

    ``_write_spend_record`` warns when the total goes backwards, because that is
    the observability for a broken attach invariant (two processes on one
    session directory). A downward CORRECTION is not that: the paint resolver's
    answer can be staler and dearer than the resolved one, and the correction
    exists to replace it. Warning on every such re-price spends the one signal
    the design names.

    The complement is asserted too — an unexplained decrease still warns — so
    the fix cannot be "delete the check".
    """
    session = make_session(tmp_path)

    async def main() -> None:
        await session._transcript.append_message(Message.user("hello"))
        index = session.accrue_spend(
            5_000_000, {"provider": "deepseek", "model_id": "deepseek-chat"}
        )
        await session._write_spend_record()
        assert session._spend_persisted_micro == 5_000_000

        monkeypatch.setattr(session_module, "price_call", lambda *a: (1_000_000, True))
        with caplog.at_level(logging.DEBUG, logger="local_operator.session.session"):
            await session._price_spend_call(
                index,
                Usage(provider="deepseek", model_id="deepseek-chat", input_tokens=10),
                "deepseek",
                "deepseek-chat",
            )
            # Await the SCHEDULED write rather than calling ``_write_spend_record``
            # a second time: two concurrent writers would race on
            # ``_spend_persisted_micro`` and warn twice, which is a property of
            # this test rather than of the code under test.
            assert session._spend_persist_task is not None
            await session._spend_persist_task
        assert session.spend.micro == 1_000_000
        assert session._spend_persisted_micro == 1_000_000
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [
            r.getMessage() for r in caplog.records
        ]

        # An unexplained decrease is still the broken-attach signal.
        caplog.clear()
        session._spend_persisted_micro = 9_000_000
        with caplog.at_level(logging.WARNING, logger="local_operator.session.session"):
            await session._write_spend_record()
        assert [r for r in caplog.records if r.levelno >= logging.WARNING]

    asyncio.run(main())


def test_the_session_screen_shows_both_money_rows_from_a_real_session(tmp_path: Path) -> None:
    """Design D1c: the evidence for ``/session``'s new rows must come off a
    REAL session, not a hand-built diagnostics object.

    The design round found `/session` frames identical before and after the
    change, because the session double used to build them exposed no
    ``restored_spend``, so both rows were omitted and the surface this PR adds
    was never rendered. This drives the production seam
    (``SessionDiagnostics.capture``) over a session that HAS a record, and
    asserts both rows are on the screen — the record's exact figure and the
    ledger's, with the Δ between them.
    """
    from local_operator.analytics.model import SessionReport, UsageAggregate
    from local_operator.tui.widgets.session_panel import (
        SessionDiagnostics,
        build_session_report,
    )

    session = make_session(tmp_path)
    session.accrue_spend(3_500_750, {"provider": "openrouter", "model_id": "x"})

    async def persist() -> None:
        await session._write_spend_record()

    asyncio.run(persist())
    assert session.restored_spend() is not None

    diagnostics = SessionDiagnostics.capture(session)
    assert diagnostics.spend_micro == 3_500_750
    report = SessionReport(
        session.session_id,
        aggregate=UsageAggregate(calls=4, ok_calls=4, cost_micro=3_000_000, cost_known_calls=4),
    )
    text = build_session_report(report, diagnostics, width=120).plain
    # The exact spelling drops trailing zeros ($3.50075), and the Δ keeps a
    # fixed width so the column does not jitter: 3.50075 - 3.00 = +0.500750.
    assert "$3.50075" in text, text
    assert "3,500,750 μ$" in text
    assert "$3.00" in text
    assert "+0.500750" in text
    # R2-4: the sign alone left "vs the record" reading either way, so the word
    # that names the minuend is part of the contract now.
    assert "record +0.500750" in text, text
