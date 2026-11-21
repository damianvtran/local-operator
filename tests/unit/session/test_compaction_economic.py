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
from local_operator.memory_capture import session_tail
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


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    async def prices(self):
        return PRICES

    monkeypatch.setattr(Session, "_economic_prices", prices)
    monkeypatch.setattr(economics, "ECONOMIC_ARCHIVE_ALLOWANCE_TOKENS", 1_000)
    # The idle pass is not under test; keep its timer out of the way.
    yield


def make(directory: Path, provider: Any, monkeypatch, *, high_wake: bool = True, **over: Any):
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
    index = 0
    while reported(session) < tokens:
        await session._prompt_messages([wake(f"setup wake {index}")])
        index += 1


@pytest.mark.asyncio
async def test_economic_pass_fires_at_a_wake_turn_end_below_threshold(tmp_path, monkeypatch):
    directory = tmp_path / "s"
    provider = Provider()
    session = make(directory, provider, monkeypatch)
    await grow(session, 3 * GROWTH)
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
    async def unpriced(self):
        return None

    monkeypatch.setattr(Session, "_economic_prices", unpriced)
    directory = tmp_path / "s"
    session = make(directory, Provider(), monkeypatch)
    await grow(session, 3 * GROWTH)
    assert not markers(directory)
    await session.dispose()


@pytest.mark.asyncio
async def test_high_wake_cap_tightens_only_high_wake_sessions(tmp_path, monkeypatch):
    settings = CompactionSettings(keep_recent_tokens=400)
    high = make(tmp_path / "h", Provider(), monkeypatch, high_wake=True)
    assert high._preserved_turns_cap(settings) == economics.PRESERVED_HIGH_WAKE_CAP
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
async def test_high_wake_eviction_hint_lists_dropped_operator_turns_once(tmp_path, monkeypatch):
    monkeypatch.setattr(economics, "PRESERVED_HIGH_WAKE_CAP", 60)

    async def no_economic(self):
        return False

    # One pass, the manual one, so every eviction is this marker's own.
    monkeypatch.setattr(Session, "_maybe_economic_compact", no_economic)
    directory = tmp_path / "s"
    provider = Provider()
    # keep_recent below the tiny cap so the cap (not keep_recent) binds.
    session = make(directory, provider, monkeypatch, keep_recent_tokens=40)
    for index in range(8):
        await session.prompt(f"standing instruction {index}: always route billing via ops " * 3)
    outcome = await session.compact_now()
    assert outcome.ran
    marker = markers(directory)[-1]
    assert marker.payload.get("preserved_turns_dropped", 0) > 0
    assert session._eviction_hint is not None and session._eviction_hint[0] == marker.id
    tail = await session_tail(session)
    assert f'compaction="{marker.id}" evicted=' in tail
    assert "standing instruction 0" in tail
    assert "memory(op='store'" in tail and "sessions(op='peek'" in tail
    # Shown for the first turn after the pass, then dropped.
    await session.prompt("next")
    assert session._eviction_hint is None
    assert "evicted=" not in await session_tail(session)
    await session.dispose()


@pytest.mark.asyncio
async def test_low_wake_pass_arms_no_eviction_hint(tmp_path, monkeypatch):
    directory = tmp_path / "s"
    session = make(directory, Provider(), monkeypatch, high_wake=False, keep_recent_tokens=40)
    for index in range(4):
        await session.prompt(f"instruction {index} " * 20)
    assert (await session.compact_now()).ran
    assert session._eviction_hint is None
    await session.dispose()


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
    await grow(session, 3 * GROWTH)
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
