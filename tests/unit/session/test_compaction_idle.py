"""Idle / pre-wake compaction wired into a real Session.

Real compaction throughout (plan, summarizer request, marker, rebuilt
context); only the provider is scripted. The timer is not waited on: a test
asserts what was ARMED (handle + delay), then fires the pass the way the
timer would, by awaiting ``_run_idle_compaction`` with the armed generation.

Numbers are the production design scaled 1/10 so the real tokenizer stays
fast: idle threshold 26k (prod 200k region), growth G = 6k (prod 60k),
+800 tokens per wake (prod +8k).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.compaction import api as compaction_api
from local_operator.compaction.api import CompactionSettings
from local_operator.compaction.idle import (
    COMPACTION_IDLE_DECLINED_TYPE,
    COMPACTION_IDLE_PASS_TYPE,
)
from local_operator.harness.types import (
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.harness.wake import build_wake_schedule
from local_operator.session.session import Session, _CompactionPlan
from local_operator.session.transcript import Transcript

#: Text-only, so the strategy is context-full and the receipt's after-size is
#: a plain local measurement (snapcompact clamps it; see _idle_growth_baseline).
BIG_MODEL = ModelSpec(
    provider="anthropic",
    model_id="opus-like",
    context_window=1_000_000,
    supports_images=False,
)
KEEP_RECENT = 400
IDLE_THRESHOLD = 26_000
GROWTH = 6_000
#: One wake's growth: the assistant reply, ~800 cl100k tokens.
WAKE_TOKENS_TEXT = "wake material " * 400


def settings(**overrides: Any) -> CompactionSettings:
    base: dict[str, Any] = {
        "keep_recent_tokens": KEEP_RECENT,
        "idle_threshold_tokens": IDLE_THRESHOLD,
        "idle_min_growth_tokens": GROWTH,
    }
    base.update(overrides)
    return CompactionSettings(**base)


def accept_latest_user(request: Any) -> str:
    """Parked advisor that anchors at the newest user turn, confidently."""
    text = request.messages[-1].text
    ids = [line.split(" ", 2)[1] for line in text.splitlines() if line.startswith("- ")]
    user_ids = [
        line.split(" ", 2)[1]
        for line in text.splitlines()
        if line.startswith("- ") and "(user)" in line
    ]
    anchor = (user_ids or ids)[-1]
    return (
        "```json\n"
        + json.dumps(
            {
                "preserve_from": anchor,
                "compact_now": True,
                "confidence": 0.9,
                "reason": "previous work finished",
            }
        )
        + "\n```"
    )


def anchor_keeping(tokens: int) -> Callable[[Any], str]:
    """Parked advisor that keeps ~``tokens`` of tail verbatim ("open work").

    Walks the request's history backwards to the newest CANDIDATE whose tail
    reaches ``tokens``, so the sim lands near its production shape
    (260k -> ~150k, here 26k -> ~13k) instead of folding everything.
    """
    from local_operator.compaction.tokens import estimate_tokens

    def advise(request: Any) -> str:
        text = request.messages[-1].text
        candidates = {line.split(" ", 2)[1] for line in text.splitlines() if line.startswith("- ")}
        tail = 0
        anchor = None
        for message in reversed(list(request.messages[:-1])):
            tail += estimate_tokens(message) if isinstance(message, Message) else 0
            if getattr(message, "id", None) in candidates:
                anchor = message.id
                if tail >= tokens:
                    break
        payload = {"preserve_from": anchor, "compact_now": True, "confidence": 0.9, "reason": "ok"}
        return "```json\n" + json.dumps(payload) + "\n```"

    return advise


def decline(_request: Any) -> str:
    return '```json\n{"preserve_from": "x", "compact_now": false, "confidence": 0.9}\n```'


class Provider:
    """Scripted wire: turns report usage, advisor/summarizer answer by purpose."""

    def __init__(
        self,
        advice: Callable[[Any], str] = accept_latest_user,
        *,
        cache_write_1h: bool = False,
    ) -> None:
        self.advice = advice
        self.cache_write_1h = cache_write_1h
        self.advisor_calls = 0
        self.summaries = 0
        self.turn_requests: list[Any] = []
        self.advisor_gate: asyncio.Event | None = None
        self.summary_gate: asyncio.Event | None = None
        #: Growth comes from ASSISTANT output, as in real sessions: user turns
        #: survive a pass verbatim, so user-borne growth could never be folded.
        self.reply = "finished history " * 1500

    def __call__(self, request: Any, signal: Any):
        purpose = getattr(request, "purpose", None)

        async def gen():
            if purpose == "compaction_advisor":
                self.advisor_calls += 1
                if self.advisor_gate is not None:
                    await self.advisor_gate.wait()
                yield StreamTextDelta(delta=self.advice(request))
                yield StreamEndEvent(stop_reason="stop")
                return
            if purpose == "compaction":
                self.summaries += 1
                if self.summary_gate is not None:
                    await self.summary_gate.wait()
                yield StreamTextDelta(delta="Summary: earlier work is complete.")
                yield StreamEndEvent(stop_reason="stop")
                return
            self.turn_requests.append(request)
            context = compaction_api.estimate_messages_tokens(list(request.messages))
            usage = Usage(
                context_tokens=context,
                cache_write_1h_tokens=100 if self.cache_write_1h else 0,
                cache_write_5m_tokens=0 if self.cache_write_1h else 100,
            )
            yield StreamTextDelta(delta=self.reply)
            yield StreamEndEvent(stop_reason="stop", usage=usage)

        return gen()


def make_session(directory: Path, provider: Provider, **overrides: Any) -> Session:
    return Session(
        model=BIG_MODEL,
        stream_fn=provider,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=settings(**overrides),
    )


async def grow_to(session: Session, tokens: int) -> None:
    """Prompt until the provider-reported context passes ``tokens``."""
    index = 0
    while True:
        usage = session._last_usage
        if usage is not None and (usage.context_tokens or 0) >= tokens:
            return
        await session.prompt(f"setup task {index}")
        index += 1


def reported(session: Session) -> int:
    """The provider-reported context size the idle arm reads (0 when none)."""
    usage = session._last_usage
    return int(usage.context_tokens or 0) if usage is not None else 0


def custom_rows(directory: Path, custom_type: str) -> list[dict[str, Any]]:
    return [
        dict(entry.payload.get("details", {}))
        for entry in Transcript(directory).entries()
        if entry.type == "custom" and entry.payload.get("custom_type") == custom_type
    ]


def markers(directory: Path) -> list[Any]:
    return [entry for entry in Transcript(directory).entries() if entry.type == "compaction"]


def _marker_count(session: Session) -> int:
    return sum(1 for entry in session._transcript.entries() if entry.type == "compaction")


async def fire(session: Session) -> bool:
    """Fire the armed timer's pass now; True only when a pass COMMITTED.

    ``False`` both when nothing was armed and when the armed pass declined.
    """
    if session._idle_timer is None:
        return False
    session._idle_timer.cancel()
    session._idle_timer = None
    before = _marker_count(session)
    await session._run_idle_compaction(session._generation)
    return _marker_count(session) > before


@pytest.mark.asyncio
async def test_idle_pass_writes_trigger_and_receipt(tmp_path):
    provider = Provider()
    session = make_session(tmp_path / "s", provider)
    await grow_to(session, IDLE_THRESHOLD + 1)
    provider.reply = WAKE_TOKENS_TEXT
    assert session._idle_timer is not None
    assert await fire(session)
    found = markers(tmp_path / "s")
    assert len(found) == 1 and found[0].payload.get("trigger") == "idle"
    (receipt,) = custom_rows(tmp_path / "s", COMPACTION_IDLE_PASS_TYPE)
    assert receipt["marker_id"] == found[0].id
    assert receipt["tokens_after"] < receipt["tokens_before"]
    assert provider.advisor_calls == 1 and provider.summaries == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_size_and_manual_markers_carry_no_trigger(tmp_path):
    session = make_session(tmp_path / "s", Provider())
    await grow_to(session, IDLE_THRESHOLD + 1)
    outcome = await session.compact_now()
    assert outcome.ran
    assert "trigger" not in markers(tmp_path / "s")[0].payload
    await session.dispose()


@pytest.mark.asyncio
async def test_multi_wake_simulation_never_thrashes(tmp_path):
    """20 wakes of +800 tokens from ~26k, rebuilding the Session each time.

    The production scenario scaled 1/10: from 260k a pass lands near 150k
    (here: keep_recent 12k + summary), the idle line is 200k (20k) and G is
    60k (6k), so the next pass is due only once BOTH lines are cleared.
    """
    directory = tmp_path / "s"
    provider = Provider(advice=anchor_keeping(12_000))
    # keep_recent 3k -> _advisor_floor_cap 15k (prod: 20k -> 100k), so the
    # advisor's 12k window is honoured rather than clamped.
    shape = {"idle_threshold_tokens": 20_000, "keep_recent_tokens": 3_000}
    session = make_session(directory, provider, **shape)
    await grow_to(session, IDLE_THRESHOLD + 1)
    provider.reply = WAKE_TOKENS_TEXT
    log: list[tuple[int, int, bool]] = []
    for wake in range(20):
        if wake:
            await session.dispose()
            session = make_session(directory, provider, **shape)  # restart: transcript only
            await session.prompt(f"wake {wake}")
        context = reported(session)
        passed = await fire(session)
        log.append((wake, context, passed))
    await session.dispose()

    passes = [entry for entry in log if entry[2]]
    assert len(passes) >= 2, log
    # Never two consecutive wakes compacting.
    for (_, _, first), (_, _, second) in zip(log, log[1:]):
        assert not (first and second), log
    # Every pass preceded by >= G growth past the previous pass's after-size.
    receipts = custom_rows(directory, COMPACTION_IDLE_PASS_TYPE)
    assert len(receipts) == len(passes)
    previous_after = 0
    for (_, context, _), receipt in zip(passes, receipts):
        assert context - previous_after >= GROWTH, (context, previous_after, log)
        previous_after = receipt["tokens_after"]


@pytest.mark.asyncio
async def test_decline_suppresses_the_advisor_until_g_more_growth(tmp_path):
    directory = tmp_path / "s"
    provider = Provider(advice=decline)
    session = make_session(directory, provider)
    await grow_to(session, IDLE_THRESHOLD + 1)
    provider.reply = WAKE_TOKENS_TEXT
    assert session._idle_timer is not None
    assert not await fire(session)
    assert provider.advisor_calls == 1
    (row,) = custom_rows(directory, COMPACTION_IDLE_DECLINED_TYPE)
    declined_at = row["context_tokens"]
    assert row["why"] == "advisor_declined" and not markers(directory)
    # Small growth: nothing armed, advisor not asked again.
    await session.prompt("small")
    assert session._idle_timer is None
    assert provider.advisor_calls == 1
    # Survives a restart.
    await session.dispose()
    session = make_session(directory, provider)
    while reported(session) < declined_at + GROWTH:
        await session.prompt("more")
        if reported(session) < declined_at + GROWTH:
            assert session._idle_timer is None
    assert session._idle_timer is not None
    await session.dispose()


@pytest.mark.asyncio
async def test_one_hour_cache_and_a_wake_inside_it_arms_nothing(tmp_path):
    provider = Provider(cache_write_1h=True)
    session = make_session(tmp_path / "s", provider)
    await grow_to(session, IDLE_THRESHOLD + 1)
    provider.reply = WAKE_TOKENS_TEXT
    assert session._idle_cache_ttl_s == 3600.0
    # Arm with a wake due in 30 minutes: inside the 1h cache, so warm.
    session._cancel_idle_compaction()
    now = session._wake._now()
    built = build_wake_schedule({"message": "check back", "in": "30m"}, [], now)
    schedule = built.get("schedule")
    assert schedule is not None, built
    await session._wake.update([schedule])
    assert any(row.next_due_at <= now + 31 * 60_000 for row in session._wake.armed_schedules)
    session._arm_idle_compaction()
    assert session._idle_timer is None
    await session.dispose()


@pytest.mark.asyncio
async def test_five_minute_cache_arms_at_expiry_minus_lead(tmp_path):
    session = make_session(tmp_path / "s", Provider())
    await grow_to(session, IDLE_THRESHOLD + 1)
    assert session._idle_cache_ttl_s == 300.0
    assert session._idle_timer is not None
    delay = session._idle_timer.when() - asyncio.get_running_loop().time()
    # F = anchor + 300 s - (30 s advisor budget + 30 s margin).
    assert 235.0 <= delay <= 240.5, delay
    await session.dispose()
    assert session._idle_timer is None


@pytest.mark.asyncio
async def test_wake_during_advisor_aborts_the_pass(tmp_path):
    directory = tmp_path / "s"
    provider = Provider()
    session = make_session(directory, provider)
    await grow_to(session, IDLE_THRESHOLD + 1)
    provider.reply = WAKE_TOKENS_TEXT
    session._cancel_idle_compaction()
    provider.advisor_gate = asyncio.Event()
    task = asyncio.ensure_future(session._run_idle_compaction(session._generation))
    session._idle_task = task
    while provider.advisor_calls == 0:
        await asyncio.sleep(0)
    await session.prompt("the wake arrives")
    provider.advisor_gate.set()
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled() or task.done()
    assert not markers(directory)
    # Nothing was decided, so nothing is journaled as a decline.
    assert not custom_rows(directory, COMPACTION_IDLE_DECLINED_TYPE)
    await session.dispose()


@pytest.mark.asyncio
async def test_wake_during_commit_runs_after_on_the_compacted_context(tmp_path):
    directory = tmp_path / "s"
    provider = Provider()
    session = make_session(directory, provider)
    await grow_to(session, IDLE_THRESHOLD + 1)
    provider.reply = WAKE_TOKENS_TEXT
    session._cancel_idle_compaction()
    provider.summary_gate = asyncio.Event()
    task = asyncio.ensure_future(session._run_idle_compaction(session._generation))
    session._idle_task = task
    while provider.summaries == 0:
        await asyncio.sleep(0)
    assert session._idle_committing and session._turn_lock.locked()
    before = len(provider.turn_requests)
    # A WAKE delivery (``_prompt_messages``) queues on the turn lock; a typed
    # ``prompt()`` is refused with TurnInFlight while ``_compacting``, exactly
    # as during a manual /compact.
    wake = asyncio.ensure_future(session._prompt_messages([Message.user("the wake arrives")]))
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(provider.turn_requests) == before, "the wake must queue on the lock"
    provider.summary_gate.set()
    await task
    await wake
    assert len(markers(directory)) == 1
    wake_request = provider.turn_requests[before]
    first = wake_request.messages[0]
    assert "Summary: earlier work is complete." in (getattr(first, "text", "") or str(first))
    await session.dispose()


@pytest.mark.asyncio
async def test_subagent_and_disabled_sessions_never_arm(tmp_path):
    session = make_session(tmp_path / "a", Provider(), idle_enabled=False)
    await grow_to(session, IDLE_THRESHOLD + 1)
    assert session._idle_timer is None
    await session.dispose()
    off = make_session(tmp_path / "b", Provider(), idle_advisor=False)
    await grow_to(off, IDLE_THRESHOLD + 1)
    assert off._idle_timer is None
    await off.dispose()
    child = Session(
        model=BIG_MODEL,
        stream_fn=Provider(),
        tools=[],
        transcript=Transcript(tmp_path / "c"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=settings(),
        parent_session_id="parent-1",
    )
    await grow_to(child, IDLE_THRESHOLD + 1)
    assert child._idle_timer is None
    await child.dispose()


@pytest.mark.asyncio
async def test_parked_hint_replaces_the_task_floor_only_when_parked(tmp_path, monkeypatch):
    """Narrowing below the task floor is reachable ONLY through ``parked_hint``."""
    from local_operator.compaction.advisor import CompactionHint
    from local_operator.compaction.tokens import estimate_tokens

    session = make_session(tmp_path / "s", Provider(), keep_recent_tokens=3_000)
    await grow_to(session, IDLE_THRESHOLD + 1)
    session._cancel_idle_compaction()
    # A wide live-task floor, as on a session whose last genuine user turn is
    # far back: the ordinary plan must honour it.
    monkeypatch.setattr(compaction_api, "task_boundary_floor", lambda *a, **k: 14_000)
    ordinary = await session._plan_compaction(respect_threshold=False)
    history = list(session._context.messages)
    last = history[-1]
    hint = CompactionHint(
        preserve_from_id=last.id,
        preserve_tokens=estimate_tokens(last),
        compact_now=True,
        confidence=0.9,
        reason="finished",
    )
    parked = await session._plan_compaction(respect_threshold=False, parked_hint=hint)
    assert isinstance(ordinary, _CompactionPlan) and isinstance(parked, _CompactionPlan)
    assert parked.cut > ordinary.cut, (parked.cut, ordinary.cut)
    # keep_recent stays the floor even for a parked hint: the tail kept is at
    # least keep_recent_tokens.
    kept = sum(estimate_tokens(m) for m in parked.llm_history[parked.cut :])
    assert kept >= 3_000, kept
    assert parked.advisor_hint is hint
    await session.dispose()


@pytest.fixture(autouse=True)
def _no_title_requests(monkeypatch):
    # Naming would add provider calls the counters above do not expect.
    monkeypatch.setattr(Session, "_maybe_spawn_title", lambda self, *a, **k: None, raising=False)
