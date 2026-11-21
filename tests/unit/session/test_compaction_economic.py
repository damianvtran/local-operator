"""Turn-end economic compaction wired into a real Session.

Real plan, real pass, real marker; only the provider is scripted (the idle
suite's ``Provider``) and the route's prices are pinned, because the scripted
model id is in no price table and an unpriced route fails closed by design.

Numbers are scaled like the idle suite (G_min 6k instead of 60k) and the
marker allowance is scaled to 1k so a ~25k test context has the same shape a
~250k production one does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import store as ask_store
from local_operator.compaction import economics
from local_operator.compaction.api import CompactionSettings
from local_operator.compaction.cutpoint import elision_notice_text
from local_operator.compaction.economics import (
    COMPACTION_ECONOMIC_DECLINED_TYPE,
    COMPACTION_ECONOMIC_PASS_TYPE,
)
from local_operator.harness.types import CustomMessage
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.unit.session.test_compaction_idle import (
    BIG_MODEL,
    WAKE_TOKENS_TEXT,
    Provider,
    custom_rows,
    markers,
    reported,
)
from tests.unit.session.test_quiet_acceptance import quiet_stream

GROWTH = 6_000
#: Cheap read, 2x write, input = read: an OpenAI-like ratio keeps delta*
#: small at test scale (production's 40x ratio is covered by the pure suite).
PRICES = (1e-6, 2e-6, 1e-6)


def pin_workload_tariff(monkeypatch):
    """Real resolver arithmetic; small configured output bound for scale tests.

    The scripted provider emits a tiny summary. Production's full generation
    bound is tested separately, without this workload fixture.
    """
    from local_operator.compaction import api
    from local_operator.model import configure
    from local_operator.model.registry import ModelInfo

    original = configure.resolve_model_info
    info = ModelInfo(
        description="Synthetic economic tariff, not a provider billing claim",
        id=BIG_MODEL.model_id,
        name="economic workload",
        input_price=1,
        output_price=1,
        cache_reads_price=1,
        cache_writes_price=2,
    )
    monkeypatch.setattr(
        configure,
        "resolve_model_info",
        lambda provider, model: info if model == BIG_MODEL.model_id else original(provider, model),
    )
    monkeypatch.setattr(api, "MAX_SUMMARY_TOKENS", 1000)
    monkeypatch.setattr(economics, "ECONOMIC_ARCHIVE_ALLOWANCE_TOKENS", 1000)


def make(directory: Path, provider: Any, monkeypatch, *, high_wake: bool = True, **over: Any):
    pin_workload_tariff(monkeypatch)
    config = directory.parent / "config"
    config.mkdir(exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setattr(Session, "_is_high_wake", lambda self: high_wake)
    shape: dict[str, Any] = {
        "keep_recent_tokens": 400,
        "idle_min_growth_tokens": GROWTH,
        "idle_enabled": False,
    }
    shape.update(over)
    return Session(
        model=BIG_MODEL,
        stream_fn=provider,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(**shape),
    )


def wake(text: str) -> CustomMessage:
    return CustomMessage(custom_type="wake_prompt", attribution="user", details={"text": text})


async def grow(session: Session, tokens: int) -> None:
    """Reach the target or the first new pass, which resets usage by design.

    Waiting for a pre-pass high-water mark after a successful compaction can
    loop forever: every cycle shrinks before the requested target is visible.
    Callers still assert the marker/decline and the actual economic outcome.
    """
    before = session._transcript.latest_entry("compaction")
    before_id = before.id if before else None
    index = 0
    while reported(session) < tokens:
        await session._prompt_messages([wake(f"setup wake {index}")])
        index += 1
        marker = session._transcript.latest_entry("compaction")
        if marker is not None and marker.id != before_id:
            return


@pytest.mark.asyncio
async def test_economic_pass_fires_at_a_wake_turn_end_below_threshold(tmp_path, monkeypatch):
    directory = tmp_path / "s"
    provider = Provider()
    session = make(directory, provider, monkeypatch)
    # Honest request/output pricing delays this pass: the 21,039-token check
    # has gain 17,028 < delta* 17,672 in this fixture. Allow the next G_min
    # regrowth/recheck, still well below the ordinary 400k size threshold.
    await grow(session, 5 * GROWTH)
    found = markers(directory)
    assert found, "no economic pass at any wake turn end"
    assert found[0].payload.get("trigger") == "economic"
    receipt, *_rest = custom_rows(directory, COMPACTION_ECONOMIC_PASS_TYPE)
    assert receipt["marker_id"] == found[0].id
    assert receipt["tokens_after"] < receipt["tokens_before"]
    # Below the size trigger by construction: 400k on this 1M window.
    assert receipt["context_tokens"] < 400_000
    # No auto-continue: the pass did not buy a model turn.
    turns = len(provider.turn_requests)
    assert provider.summaries == 1
    await session.dispose()
    assert len(provider.turn_requests) == turns


@pytest.mark.asyncio
async def test_quiet_no_reply_turn_end_reaches_the_economic_gate(tmp_path, monkeypatch):
    """Unverified item 1 of the design: a quiet turn reaches ``_maybe_compact``."""
    calls: list[str] = []
    original = Session._maybe_economic_compact

    async def spy(self):
        calls.append("economic")
        return await original(self)

    monkeypatch.setattr(Session, "_maybe_economic_compact", spy)
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    stream = quiet_stream()
    session = Session(
        model=BIG_MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "q"),
        system_blocks_provider=lambda: ["stable"],
    )
    await session._prompt_messages(
        [CustomMessage(custom_type="peer_message", attribution="user", details={"text": "fyi"})]
    )
    assert len(stream.requests) == 1
    assert session._quiet_accepted_generation is not None, "the turn did not end quiet"
    assert calls == ["economic"]
    await session.dispose()


@pytest.mark.asyncio
async def test_no_back_to_back_passes_and_restart_keeps_the_baseline(tmp_path, monkeypatch):
    directory = tmp_path / "s"
    provider = Provider()
    session = make(directory, provider, monkeypatch)
    provider.reply = WAKE_TOKENS_TEXT  # ~800 tokens per wake, << G_min
    log: list[tuple[int, int]] = []
    for index in range(60):
        if index and index % 15 == 0:
            await session.dispose()
            session = make(directory, provider, monkeypatch)  # restart: transcript only
        before = len(markers(directory))
        await session._prompt_messages([wake(f"wake {index}")])
        log.append((reported(session), len(markers(directory)) - before))
    await session.dispose()
    passes = [i for i, (_c, fired) in enumerate(log) if fired]
    assert len(passes) >= 2, log
    for first, second in zip(passes, passes[1:]):
        assert second - first > 1, log
    receipts = custom_rows(directory, COMPACTION_ECONOMIC_PASS_TYPE)
    assert len(receipts) == len(passes)
    for previous, nxt in zip(receipts, receipts[1:]):
        # >= G_min of growth past the previous pass's after-size, across restarts
        assert nxt["context_tokens"] - previous["tokens_after"] >= GROWTH, receipts


@pytest.mark.asyncio
async def test_failed_exact_recheck_leaves_context_and_journals(tmp_path, monkeypatch):
    directory = tmp_path / "s"
    provider = Provider()
    session = make(directory, provider, monkeypatch)

    # An exact residual far above rho * C: the cheap gate says due, the
    # re-check must refuse and change nothing.
    async def huge(self, plan, **_kw):
        return 10**9

    monkeypatch.setattr(Session, "_economic_exact_residual", huge)
    armed: list[bool] = []
    original_arm = Session._arm_idle_compaction
    monkeypatch.setattr(
        Session, "_arm_idle_compaction", lambda self: (armed.append(True), original_arm(self))
    )
    await grow(session, 3 * GROWTH)
    assert not markers(directory)
    assert provider.summaries == 0
    rows = custom_rows(directory, COMPACTION_ECONOMIC_DECLINED_TYPE)
    assert rows and rows[0]["why"] == "recheck:residual_ratio"
    # A decline suppresses re-planning until G_min more growth.
    declined_at = rows[0]["context_tokens"]
    later = [row["context_tokens"] for row in rows[1:]]
    assert all(value - declined_at >= GROWTH for value in later), rows
    # Idle arming still runs whenever the economic pass declines.
    assert armed
    await session.dispose()


@pytest.mark.asyncio
async def test_unpriced_route_never_compacts(tmp_path, monkeypatch):

    async def unpriced(self, *args):
        return None

    monkeypatch.setattr(Session, "_economic_prices", unpriced)
    directory = tmp_path / "s"
    session = make(directory, Provider(), monkeypatch)
    await grow(session, 3 * GROWTH)
    assert not markers(directory)
    await session.dispose()


@pytest.mark.asyncio
async def test_high_wake_cap_fails_closed_without_memory_checkpoint(tmp_path, monkeypatch):
    settings = CompactionSettings(keep_recent_tokens=400)
    high = make(tmp_path / "h", Provider(), monkeypatch, high_wake=True)
    assert high._preserved_turns_cap(settings) == 100_000  # no acknowledged checkpoint
    await high.dispose()
    low = make(tmp_path / "l", Provider(), monkeypatch, high_wake=False)
    assert low._preserved_turns_cap(settings) == 100_000  # 400k // 4, unchanged
    await low.dispose()


def test_high_wake_rate_threshold_counts_the_trailing_six_hours():
    now = 10 * 3600 * 1000
    recent = [now - i * 60_000 for i in range(economics.HIGH_WAKE_MIN_REQUESTS)]
    assert economics.is_high_wake_rate(recent, now_ms=now)
    assert not economics.is_high_wake_rate(recent[:-1], now_ms=now)
    stale = [now - economics.HIGH_WAKE_WINDOW_MS - 1] + recent[:-1]
    assert not economics.is_high_wake_rate(stale, now_ms=now)


@pytest.mark.asyncio
async def test_economic_pass_carries_open_asks_and_decisions(tmp_path, monkeypatch):
    from local_operator.memory import MemoryIdentity, MemoryStore

    directory = tmp_path / "s"
    provider = Provider()
    session = make(directory, provider, monkeypatch)
    now = ask_store.now_ms()
    ask_dir = ask_store.session_dir(session._config_dir, session._session_id)
    assert ask_store.append_event(
        ask_dir,
        {
            "ask_id": "a-econ1",
            "at": now,
            "created_at": now,
            "expires_at": now + 86_400_000,
            "kind": "queued",
            "questions": [{"id": "q-1", "question": "Ship the Q4 deck?", "multi": False}],
        },
    )
    MemoryStore(
        session._config_dir, MemoryIdentity(session._session_id, "", ""), origin="device-test"
    ).put(
        "session",
        {
            "title": "route billing via ops",
            "body": "decision: route billing questions via #ops",
            "tags": ["decision"],
            "created_at": now / 1000,
        },
    )
    await grow(session, 5 * GROWTH)
    found = [m for m in markers(directory) if m.payload.get("trigger") == "economic"]
    assert found
    payload = found[-1].payload
    assert any(a.get("question") == "Ship the Q4 deck?" for a in payload.get("carried_asks", []))
    assert any("#ops" in str(d) for d in payload.get("carried_decisions", []))
    await session.dispose()


def test_elision_notice_points_at_peek_recall():
    text = elision_notice_text(3, 0)
    assert text is not None and "sessions(op='peek'" in text


def test_preserved_block_sheds_harness_chrome_but_not_lookalikes():
    from local_operator.compaction.cutpoint import cap_preserved_user_turns
    from local_operator.harness.chrome_prompts import CONNECTIVITY_CONTINUATION_PROMPT
    from local_operator.harness.rows import harness_chrome_prompts

    recovery = next(p for p in harness_chrome_prompts() if p.startswith("Harness recovery notice"))
    turns = [
        {"id": "u1", "text": "please draft the memo"},
        {"id": "u2", "text": recovery},
        {"id": "u3", "text": CONNECTIVITY_CONTINUATION_PROMPT},
        {"id": "u4", "text": "Harness recovery notice: my own words about it"},
    ]
    capped = cap_preserved_user_turns(turns, cap=10_000)
    kept = [turn["id"] for turn in capped.turns]
    assert kept == ["u1", "u4"]
    assert capped.injections_dropped == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "google"])
@pytest.mark.parametrize("context", [100_000, 300_000])
async def test_real_resolver_prices_one_actual_prompt_tier(
    tmp_path, monkeypatch, provider, context
):
    """R3: inclusive and disjoint cache buckets must select the same tier."""
    from local_operator.harness.types import Usage
    from local_operator.model import configure
    from local_operator.model.registry import ModelInfo

    info = ModelInfo(
        description="Synthetic economic tariff, not a provider billing claim",
        id="tiered",
        name="tiered fixture",
        input_price=4,
        output_price=8,
        cache_reads_price=0.2,
        cache_writes_price=5,
        long_context_threshold=200_000,
        long_context_input_price=8,
        long_context_output_price=16,
        long_context_cache_reads_price=0.4,
        long_context_cache_writes_price=10,
    )
    monkeypatch.setattr(configure, "resolve_model_info", lambda *_: info)
    session = Session(
        model=BIG_MODEL.model_copy(update={"provider": provider, "model_id": "tiered"}),
        stream_fn=Provider(),
        tools=[],
        system_blocks_provider=lambda: ["stable"],
        transcript=Transcript(tmp_path / "tariff"),
    )
    session._idle_cache_route = (provider, "tiered")
    session._idle_cache_ttl_s = 3600
    session._last_usage = Usage(context_tokens=context)
    factor = 2 if context > 200_000 else 1
    expected = (0.2 * factor, (8 if provider == "anthropic" else 5) * factor, 4 * factor)
    prices = await session._economic_prices(context)
    assert prices is not None
    assert tuple(x * 1e6 for x in prices) == pytest.approx(expected)
    # Same Session crosses the tariff boundary: no stale size-insensitive cache.
    other = 300_000 if context == 100_000 else 100_000
    actual = await session._economic_prices(other)
    assert actual is not None
    assert actual[2] * 1e6 == (8 if other == 300_000 else 4)
    await session.dispose()


@pytest.mark.asyncio
async def test_unknown_service_tier_fails_closed_and_stamp_changes_reprice(tmp_path, monkeypatch):
    from local_operator.harness.types import Usage
    from local_operator.model import configure
    from local_operator.model.registry import ModelInfo

    monkeypatch.setattr(
        configure,
        "resolve_model_info",
        lambda *_: ModelInfo(
            description="Synthetic economic tariff, not a provider billing claim",
            id="gpt-6.1",
            name="synthetic tier",
            input_price=4,
            output_price=8,
            cache_reads_price=0.2,
            cache_writes_price=5,
        ),
    )
    session = Session(
        model=BIG_MODEL.model_copy(
            update={
                "provider": "openai",
                "model_id": "gpt-6.1",
            }
        ),
        stream_fn=Provider(),
        tools=[],
        system_blocks_provider=lambda: ["stable"],
        transcript=Transcript(tmp_path / "tier"),
    )
    session._last_usage = Usage(openai_service_tier="unknown")
    assert await session._economic_prices(100_000) is None
    # Published first-party write TTL is absent, so the real resolver refuses
    # even with a known service tier; never invent a write tariff.
    session._last_usage = Usage(openai_service_tier="default", openai_long_context=True)
    assert await session._economic_prices(100_000) is None
    await session.dispose()


@pytest.mark.asyncio
async def test_r1_prices_real_summary_span_template_previous_summary_and_output(
    tmp_path, monkeypatch
):
    from local_operator.compaction import api
    from local_operator.compaction.marker import build_compaction_marker
    from local_operator.compaction.tokens import count_text_tokens
    from local_operator.harness.types import Message
    from local_operator.model import configure
    from local_operator.model.registry import ModelInfo
    from local_operator.session.session import _CompactionPlan

    # Full production generation bound, unlike the small-scale workload tests.
    assert api.MAX_SUMMARY_TOKENS == 16384
    monkeypatch.setattr(
        configure,
        "resolve_model_info",
        lambda *_: ModelInfo(
            description="Synthetic economic tariff, not a provider billing claim",
            id=BIG_MODEL.model_id,
            name="R1 fixture",
            input_price=4,
            output_price=8,
            cache_reads_price=0.2,
            cache_writes_price=5,
        ),
    )
    provider = Provider()
    session = Session(
        model=BIG_MODEL,
        stream_fn=provider,
        tools=[],
        system_blocks_provider=lambda: ["stable"],
        transcript=Transcript(tmp_path / "cost"),
    )
    # Retained operator text is still sent to the summarizer. The old formula
    # priced only 68k removed tokens instead of the 150k summarized span.
    messages = [Message.user("constraint " * 150_000)]
    previous = "Earlier decision: keep routing via ops."
    marker = session._convert_to_llm([build_compaction_marker(previous)])
    history = marker + messages
    plan = _CompactionPlan(
        api,
        CompactionSettings(),
        "context-full",
        history,
        len(history),
        160_000,
        api.estimate_messages_tokens(history),
    )
    cost = await session._economic_summary_cost(plan)
    assert cost is not None
    prior, span = api.split_leading_marker(history)
    prompt = api.build_compaction_prompt(span, previous_summary=prior)
    plain_input = count_text_tokens(api.SUMMARIZATION_SYSTEM_PROMPT) + api.estimate_messages_tokens(
        [Message.user(prompt)]
    )
    lower_bound = plain_input * 4e-6 + api.MAX_SUMMARY_TOKENS * 8e-6
    assert cost >= lower_bound
    assert cost > (160_000 - 92_000) * 8e-6  # .544 avoided rewrite
    decision = economics.economic_compaction_due(
        enabled=True,
        strategy="context-full",
        is_subagent=False,
        priced=True,
        context_tokens=160_000,
        min_context_tokens=0,
        projected_residual=92_000,
        baseline_after_tokens=0,
        declined_at_tokens=None,
        growth_per_request=525,
        p_read=0.2e-6,
        p_write=8e-6,
        cold_event_rate=0,
        cold_next=True,
        pass_cost=cost,
        min_gain_tokens=60_000,
        max_residual_ratio=0.6,
        low_yield_streak=0,
        t_eff=400_000,
    )
    assert not decision.due and decision.reason == "below_eoq"

    # Exercise the production gate with the same counterexample, not only
    # the pure predicate. No provider call or marker may be bought.
    async def planned(**_kwargs):
        return plan

    async def residual(*_args, **_kwargs):
        return 92_000

    monkeypatch.setattr(session, "_idle_gate_tokens", lambda: 160_000)
    monkeypatch.setattr(session, "_is_high_wake", lambda: True)
    monkeypatch.setattr(session, "_economic_cold_next", lambda _gap: True)
    monkeypatch.setattr(session, "_plan_compaction", planned)
    monkeypatch.setattr(session, "_economic_exact_residual", residual)
    session._idle_cache_ttl_s = 3600
    assert not await session._maybe_economic_compact()
    declines = custom_rows(tmp_path / "cost", COMPACTION_ECONOMIC_DECLINED_TYPE)
    assert declines[-1]["why"] == "recheck:below_eoq"
    assert not markers(tmp_path / "cost") and provider.summaries == 0
    await session.dispose()


@pytest.mark.asyncio
async def test_r2_actual_archive_rejects_the_32k_counterexample(tmp_path, monkeypatch):
    from local_operator.compaction import api, snapcompact
    from local_operator.compaction.tokens import count_text_tokens
    from local_operator.harness.types import Message
    from local_operator.session.session import _CompactionPlan

    monkeypatch.setattr(snapcompact, "render_frame", lambda *_args, **_kw: b"")
    session = Session(
        model=BIG_MODEL.model_copy(update={"supports_images": True}),
        stream_fn=Provider(),
        tools=[],
        system_blocks_provider=lambda: ["stable"],
        transcript=Transcript(tmp_path / "archive"),
    )
    user = Message.user(" ".join(f"rule{i}: constraint{i*i}" for i in range(6500)))
    assistant = Message.assistant(" ".join(f"item{i}: value{i*i}" for i in range(13000)))
    history: list[Message] = [user, assistant]
    session._context.messages = list(history)
    before = api.estimate_messages_tokens(history)
    plan = _CompactionPlan(api, CompactionSettings(), "snapcompact", history, 2, before, before)
    projected = await session._economic_exact_residual(
        plan, prefix_tokens=0, provider_context=before
    )
    archive_metadata = plan.economic_summary[0][1]
    assert archive_metadata is not None
    archive = snapcompact.Archive.model_validate(archive_metadata["snapcompact"])
    actual_archive = snapcompact.estimate_archive_tokens(
        archive
    ) + snapcompact.replayed_frame_count(archive) * (
        snapcompact.frame_token_estimate_for(BIG_MODEL.provider, BIG_MODEL.model_id)
        - snapcompact.FRAME_TOKEN_ESTIMATE
    )
    preserved = 0
    for turn in api.extract_preserved_user_turns(history, {user.id}):
        text = turn["text"]
        assert isinstance(text, str)
        preserved += count_text_tokens(text)
    assert actual_archive > 32_000
    assert projected >= actual_archive + preserved
    assert projected / before > 0.6
    # Same archive bytes are handed to the commit half; no second raster pass.
    observed = {}

    async def finish(plan, **kw):
        observed.update(kw)
        return object()

    monkeypatch.setattr(session, "_finish_compaction", finish)
    await session._run_compaction(plan, reason="economic")
    assert observed["summarized"] is plan.economic_summary[0]
    await session.dispose()


@pytest.mark.asyncio
async def test_snapcompact_economic_failure_cannot_buy_llm_fallback(tmp_path, monkeypatch):
    from local_operator.compaction import api, snapcompact
    from local_operator.harness.types import Message

    provider = Provider()
    session = Session(
        model=BIG_MODEL,
        stream_fn=provider,
        tools=[],
        system_blocks_provider=lambda: ["stable"],
        transcript=Transcript(tmp_path / "fallback"),
    )

    def fail(*_args, **_kw):
        raise RuntimeError("synthetic archive failure")

    monkeypatch.setattr(snapcompact, "compact_to_archive", fail)
    with pytest.raises(RuntimeError, match="synthetic archive failure"):
        await session._produce_summary(
            api, [Message.user("retain this")], "snapcompact", allow_llm_fallback=False
        )
    assert provider.summaries == 0
    # Mandatory size/manual behavior still has its existing fallback.
    summary, preserve = await session._produce_summary(
        api, [Message.user("retain this")], "snapcompact"
    )
    assert summary and preserve is None and provider.summaries == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_tariff_crossing_declines_before_paying_for_summary(tmp_path, monkeypatch):
    from local_operator.model import configure
    from local_operator.model.registry import ModelInfo

    provider = Provider()
    session = make(tmp_path / "crossing", provider, monkeypatch, enabled=False)
    await grow(session, 3 * GROWTH)
    monkeypatch.setattr(
        configure,
        "resolve_model_info",
        lambda *_: ModelInfo(
            id=BIG_MODEL.model_id,
            name="crossing fixture",
            description="Synthetic tiers",
            input_price=1,
            output_price=1,
            cache_reads_price=1,
            cache_writes_price=2,
            long_context_threshold=20_000,
            long_context_input_price=2,
            long_context_output_price=2,
            long_context_cache_reads_price=2,
            long_context_cache_writes_price=4,
        ),
    )
    session._compaction_settings = CompactionSettings(
        keep_recent_tokens=400,
        idle_min_growth_tokens=GROWTH,
        idle_enabled=False,
    )
    await session._prompt_messages([wake("tariff crossing negative control")])
    assert provider.summaries == 0 and not markers(tmp_path / "crossing")
    declines = custom_rows(tmp_path / "crossing", COMPACTION_ECONOMIC_DECLINED_TYPE)
    assert declines[-1]["why"] == "recheck:tariff_change"
    await session.dispose()
