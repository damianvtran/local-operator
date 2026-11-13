"""Exercise cache namespace changes through real Session turns and route bridge.

The scripted wire follows failover's ordering: outbound edges precede usage,
recovery follows it. No provider calls, live sessions, or real-time waits.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.compaction import api as compaction_api
from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.providers.failover import FallbackTarget
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

PRIMARY = ModelSpec(provider="anthropic", model_id="synthetic-primary", context_window=1_000_000)
FALLBACK = ModelSpec(provider="openai", model_id="synthetic-fallback", context_window=1_000_000)
NOW = 1_800_000_000.0


class ClockWire:
    """A wire whose serving stamp need not match the selected/display model."""

    def __init__(self) -> None:
        self.handler: Any = None
        self.model = PRIMARY
        self.write_1h = True
        self.edge: str | None = None
        self.context = 300_000
        self.omit_usage = False

    def set_route_handler(self, handler: Any) -> None:
        self.handler = handler

    def __call__(self, request: Any, signal: Any) -> Any:
        async def generate():
            if self.edge == "fallback":
                await self.handler(
                    FallbackTarget(f"{self.model.provider}/{self.model.model_id}"),
                    "synthetic fallback",
                )
            yield StreamTextDelta(delta="synthetic answer")
            yield StreamEndEvent(
                stop_reason="stop",
                usage=(
                    Usage(
                        provider=self.model.provider,
                        model_id=self.model.model_id,
                        context_tokens=self.context,
                        cache_write_1h_tokens=100 if self.write_1h else 0,
                        cache_read_tokens=100 if not self.write_1h else 0,
                    )
                    if not self.omit_usage
                    else None
                ),
            )
            if self.edge == "recovery":
                await self.handler(None, "synthetic primary recovered")

        return generate()


@pytest.fixture
def clock_session(tmp_path, monkeypatch):
    import local_operator.session.session as module

    clock = [NOW]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    wire = ClockWire()
    session = Session(
        model=PRIMARY,
        stream_fn=wire,
        tools=[],
        transcript=Transcript(tmp_path / "synthetic-route-clock"),
        system_blocks_provider=lambda: ["synthetic cache namespace"],
        compaction_settings=compaction_api.CompactionSettings(),
    )
    monkeypatch.setattr(session, "_spec_for_route", lambda selector, effort: wire.model)
    return session, wire, clock


def fire_delay(session: Session) -> float:
    """Exact deadline arithmetic, independent of event-loop execution speed."""
    # Every caller has reported usage; a missing figure must fail loudly here
    # rather than be defaulted, since the missing-usage case asserts no timer.
    gate_tokens = session._idle_gate_tokens()
    assert gate_tokens is not None, "fire_delay needs reported usage"
    decision, expiry, _ = session._idle_inputs(
        session._compaction_settings, compaction_api, gate_tokens
    )
    assert decision.due, decision.reason
    return (compaction_api.idle_fire_at_ms(expiry, 30.0) - int(NOW * 1000)) / 1000


@pytest.mark.asyncio
async def test_fallback_replaces_one_hour_clock_with_serving_provider_default(clock_session):
    session, wire, clock = clock_session
    try:
        await session.prompt("first primary request")
        assert fire_delay(session) == 3540.0
        old_timer = session._idle_timer
        assert old_timer is not None
        wire.model, wire.write_1h, wire.edge = FALLBACK, False, "fallback"
        await session.prompt("request automatically falls back")
        delay = fire_delay(session)
        print(
            f"selected={session.model.provider} serving={session.effective_model.provider} "
            f"delay_s={delay}"
        )
        assert delay == 240.0
        assert session._idle_timer is not None and old_timer.cancelled()
        armed_delay = session._idle_timer.when() - asyncio.get_running_loop().time()
        print(f"armed_timer_delay_s={armed_delay:.3f}")
        assert 239.0 < armed_delay <= 240.0
        assert session._idle_cache_ttl_s is None
        assert session._idle_cache_anchor_ms == int(clock[0] * 1000)
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("write_1h, expected_delay", [(True, 3540.0), (False, 240.0)])
async def test_recovery_establishes_the_primary_clock(clock_session, write_1h, expected_delay):
    session, wire, clock = clock_session
    try:
        wire.model, wire.write_1h, wire.edge = FALLBACK, False, "fallback"
        await session.prompt("fallback request")
        assert fire_delay(session) == 240.0
        wire.model, wire.write_1h, wire.edge = PRIMARY, write_1h, "recovery"
        await session.prompt("primary recovers after usage")
        assert session.effective_model.provider == "anthropic"
        assert session._idle_cache_ttl_s == (3600.0 if write_1h else None)
        assert session._idle_cache_anchor_ms == int(clock[0] * 1000)
        assert fire_delay(session) == expected_delay
        # A late/duplicate route notification must not erase usage already
        # adopted from that serving route.
        await wire.handler(None, "synthetic duplicate recovery")
        assert fire_delay(session) == expected_delay
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_same_route_read_only_turns_preserve_one_hour_lifetime(clock_session):
    session, wire, clock = clock_session
    try:
        await session.prompt("write one hour prefix")
        wire.write_1h = False
        for index in range(3):
            clock[0] += 10
            await session.prompt(f"read-only wake {index}")
            assert session._idle_cache_ttl_s == 3600.0
            assert session._idle_cache_anchor_ms == int(clock[0] * 1000)
            assert fire_delay(session) == 3540.0 + (index + 1) * 10
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_route_edge_invalidates_old_timer_and_queued_callback(clock_session):
    session, wire, _clock = clock_session
    try:
        await session.prompt("warm primary")
        old_timer = session._idle_timer
        old_cache_generation = session._idle_cache_generation
        old_turn_generation = session._generation
        assert old_timer is not None
        wire.model = FALLBACK
        await wire.handler(FallbackTarget("openai/synthetic-fallback"), "synthetic edge")
        assert old_timer.cancelled() and session._idle_timer is None
        assert session._idle_cache_ttl_s is None and session._idle_cache_anchor_ms == 0
        session._arm_idle_compaction()
        assert session._idle_timer is None  # no usage in the new cache namespace yet
        wire.write_1h, wire.edge = False, None
        await session.prompt("establish fallback clock")
        new_timer = session._idle_timer
        assert new_timer is not None
        # Even within the SAME turn generation a queued old-route callback is
        # stale. Deliberately use the current turn, isolating the cache guard.
        session._on_idle_timer(session._generation, old_cache_generation)
        assert session._idle_timer is new_timer and session._idle_task is None
        session._on_idle_timer(old_turn_generation, session._idle_cache_generation)
        assert session._idle_task is None and session._idle_timer is new_timer
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_missing_usage_does_not_reanchor_previous_route_receipt(clock_session):
    session, wire, clock = clock_session
    try:
        await session.prompt("old route usage")
        clock[0] += 10
        wire.model, wire.edge, wire.omit_usage = FALLBACK, "fallback", True
        await session.prompt("new route reports no cache usage")
        assert session._idle_cache_ttl_s is None
        assert session._idle_cache_anchor_ms == 0
        assert session._idle_timer is None
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_same_provider_new_model_does_not_inherit_write_lifetime(clock_session):
    session, wire, _clock = clock_session
    try:
        await session.prompt("primary cache")
        wire.model = PRIMARY.model_copy(update={"model_id": "synthetic-other-model"})
        wire.write_1h, wire.edge = False, "fallback"
        await session.prompt("same provider different cache namespace")
        assert session._idle_cache_ttl_s is None
        assert fire_delay(session) == 240.0
    finally:
        await session.dispose()
