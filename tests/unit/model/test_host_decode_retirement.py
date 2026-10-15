"""OpenRouter host affinity: retire a host that decodes slowly (not only one that misses cache).

WHY THIS FILE EXISTS. The harness pins a conversation to the upstream host that
served its last turn (``order=[host]``) and used to let go of it ONLY on a
prompt-cache miss. A host that caches fine but decodes at a fifth of its peers'
speed was therefore never left: one measured run (DeepSeek v4.1 Flash via
OpenRouter) sat 63 consecutive requests on a ~20 tok/s host while two other hosts
served the same model at 120-170 tok/s, and that one host consumed 93% of the
run's request time. ``SessionStreamFn._score_host_decode_rate`` folds the decode
rate the stream funnel already measures into per-host strikes.

Zero provider spend, zero network: every case drives the REAL ``_record_stream``
funnel (the seam that reads ``served_provider`` and closes the decode window)
with a scripted stream and a controllable clock. The fixture
``tests/fixtures/provider_host_decode/task065_turns.json`` is the measured
per-turn shape; its ``decode_s`` is the audit's request window minus each host's
fitted non-decode intercept, an APPROXIMATION of the harness's first-to-last-
delta window (see the fixture's ``_provenance``).
"""

from __future__ import annotations

import json
import time as _real_time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamStartEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.model import configure as configure_mod
from local_operator.model.configure import (
    OPENROUTER_RETIRE_BELOW_TPS,
    SessionStreamFn,
    _openrouter_retire_below_tps,
)

MODEL = "deepseek/deepseek-v4.1-flash"
FIXTURE = Path(__file__).parents[2] / "fixtures" / "provider_host_decode" / "task065_turns.json"

#: The fitted decode rates from the audit, by host (tok/s). Only used to REPLAY
#: the measured run through a router that honours ``provider.ignore``.
_FITTED_TPS = {"DigitalOcean": 172.0, "OpenInference": 20.3, "Morph": 130.0}


class FakeClock:
    """A monotonic clock the scripted stream advances, so decode windows are exact."""

    def __init__(self, start: float = 10_000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _TimeProxy:
    """``configure.time`` with ``monotonic`` replaced and everything else real."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.monotonic()

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(configure_mod, "time", _TimeProxy(fake))
    return fake


def _stream(
    settings: dict[str, Any] | None = None,
) -> tuple[SessionStreamFn, list[tuple[str, str]]]:
    """A stream over a mocked auth store, plus the notices it raised."""
    from unittest.mock import MagicMock

    stream = SessionStreamFn(MagicMock(), settings if settings is not None else {}, "s-speed")
    notices: list[tuple[str, str]] = []

    def collect(text: str, kind: str) -> None:
        notices.append((text, kind))

    stream.set_notice_handler(collect)
    return stream, notices


def _request(
    model_id: str = MODEL, provider: str = "openrouter", purpose: str = "turn"
) -> ChatRequest:
    spec = ModelSpec(provider=provider, model_id=model_id, supports_prompt_cache=True)
    return ChatRequest(model=spec, messages=[Message.user("hi")], purpose=purpose)


async def _serve(
    stream: SessionStreamFn,
    clock: FakeClock,
    *,
    host: str,
    tokens: int,
    tps: float,
    deltas: int = 40,
    request: ChatRequest | None = None,
    idle_s: float = 5.0,
    queue_s: float = 8.0,
) -> ChatRequest:
    """One turn through the real funnel: ``tokens`` at ``tps``, behind ``queue_s``.

    The queue/prefill wait happens BEFORE the first delta, so it must never count
    toward the decode rate -- the property that makes a slow-to-start host not
    look slow-to-decode.
    """
    clock.advance(idle_s)
    stamped = stream._apply_cache_affinity(request or _request())

    async def scripted() -> AsyncIterator[StreamEvent]:
        yield StreamStartEvent(response_id="r")
        clock.advance(queue_s)
        window = tokens / tps
        step = window / max(deltas - 1, 1)
        for index in range(deltas):
            yield StreamTextDelta(delta="x")
            if index < deltas - 1:
                clock.advance(step)
        yield StreamEndEvent(
            stop_reason="stop",
            served_provider=host,
            usage=Usage(input_tokens=100, output_tokens=tokens),
        )

    assert [event async for event in stream._record_stream(stamped, scripted())]
    return stamped


def _slow(tokens: int = 1700) -> dict[str, Any]:
    """The measured slow host's shape: ~1.7K-token outputs at ~20 tok/s."""
    return {"tokens": tokens, "tps": 20.3}


# ---------------------------------------------------------------------------
# The measured shape: base never retires, head retires after N turns
# ---------------------------------------------------------------------------


def _replay(settings: dict[str, Any]) -> dict[str, Any]:
    """Replay task 065's per-turn shape through the real funnel.

    The router serves the host the measured run saw, UNLESS that host is in the
    request's ``provider_avoid`` (``provider.ignore`` on the wire), in which case
    it serves DigitalOcean -- the host that held the run's first 27 turns. That
    routing assumption is the replay's one modelled fact; everything else is the
    harness's own state machine.
    """
    turns = json.loads(FIXTURE.read_text())["turns"]
    fake = FakeClock()
    original = configure_mod.time
    configure_mod.time = _TimeProxy(fake)  # type: ignore[assignment]
    try:
        import asyncio

        stream, notices = _stream(settings)

        async def run() -> dict[str, Any]:
            served_log: list[tuple[int, str]] = []
            first_ignored_at: int | None = None
            for row in turns:
                request = stream._apply_cache_affinity(_request())
                host = row["host"]
                if host in request.provider_avoid:
                    host = "DigitalOcean"
                    if first_ignored_at is None:
                        first_ignored_at = row["turn"]
                tps = _FITTED_TPS[host]
                await _serve(
                    stream,
                    fake,
                    host=host,
                    tokens=row["output_tokens"],
                    tps=tps,
                    request=request,
                    idle_s=1.0,
                )
                served_log.append((row["turn"], host))
            return {
                "served": served_log,
                "retired": set(stream._provider_retired.get(MODEL, set())),
                "first_ignored_at": first_ignored_at,
            }

        result = asyncio.run(run())
        result["notices"] = notices
        return result
    finally:
        configure_mod.time = original  # type: ignore[assignment]


def test_the_measured_run_retires_the_slow_host_after_three_slow_turns() -> None:
    result = _replay({})

    oi_turns = [turn for turn, host in result["served"] if host == "OpenInference"]
    # Head: the slow host is retired on its 3rd QUALIFYING slow turn. Turn 29 is a
    # 100-token turn and neither strikes nor clears, so the strikes land on turns
    # 28, 30 and 31 -- and the very next request carries the ignore.
    assert result["retired"] == {"OpenInference"}
    assert oi_turns == [28, 29, 30, 31], oi_turns
    assert result["first_ignored_at"] == 32
    # Only the slow host is ever named: the fast hosts are left alone.
    assert not result["retired"] & {"DigitalOcean", "Morph"}
    # One notice, naming the host, the measured rate, the floor and the cache cost.
    assert len(result["notices"]) == 1
    text, kind = result["notices"][0]
    assert kind == "info"
    assert "OpenInference" in text and "tok/s" in text and "floor 25" in text
    assert "cold prompt cache" in text


def test_with_the_check_off_the_same_run_never_retires() -> None:
    """Base behaviour, selected by the documented off switch: 63 slow requests."""
    result = _replay({"providers": {"openrouter": {"retire_below_tps": 0}}})

    assert result["retired"] == set()
    assert result["notices"] == []
    assert sum(1 for _, host in result["served"] if host == "OpenInference") == 63


# ---------------------------------------------------------------------------
# Single-turn mechanics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_and_prefill_time_is_not_charged_to_decode(clock: FakeClock) -> None:
    """A host that is slow to START but decodes fast must not be retired.

    90 s of queue before the first delta, then 1,700 tokens at 150 tok/s: the
    request window is ~101 s (an apparent 17 tok/s on a naive tokens/duration
    estimate) while the decode window is ~11 s.
    """
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    for _ in range(5):
        await _serve(stream, clock, host="Queued", tokens=1700, tps=150.0, queue_s=90.0)

    assert stream._provider_retired.get(MODEL) in (None, set())
    assert stream._provider_host_rate[MODEL]["Queued"] == pytest.approx(150.0, rel=0.05)


@pytest.mark.asyncio
async def test_short_outputs_never_strike(clock: FakeClock) -> None:
    """Below 500 output tokens the rate is burst quantisation, not decode speed."""
    stream, notices = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    for _ in range(40):
        await _serve(stream, clock, host="Slow", tokens=499, tps=5.0)

    assert not stream._provider_slow_strikes.get(MODEL)
    assert not stream._provider_retired.get(MODEL)
    assert notices == []


@pytest.mark.asyncio
async def test_a_short_turn_neither_strikes_nor_forgives(clock: FakeClock) -> None:
    """A 100-token tool-call turn between two slow reasoning turns is no evidence
    either way: it must not reset the run, or the measured host (whose turns
    interleave 100-token and 1,800-token replies) could never reach three."""
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    await _serve(stream, clock, host="Slow", **_slow())
    await _serve(stream, clock, host="Slow", tokens=100, tps=150.0)
    await _serve(stream, clock, host="Slow", **_slow())
    assert stream._provider_slow_strikes[MODEL]["Slow"] == 2
    await _serve(stream, clock, host="Slow", **_slow())

    assert stream._provider_retired[MODEL] == {"Slow"}


@pytest.mark.asyncio
async def test_a_single_delta_has_no_window_and_never_strikes(clock: FakeClock) -> None:
    """``_decode_measure``'s rule: one delta is a burst, not a rate."""
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    for _ in range(5):
        await _serve(stream, clock, host="Slow", tokens=1700, tps=20.0, deltas=1)

    assert not stream._provider_slow_strikes.get(MODEL)
    assert not stream._provider_retired.get(MODEL)


# ---------------------------------------------------------------------------
# Negative controls: when retiring would be wrong
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_intermittent_slow_turn_below_n_never_retires(clock: FakeClock) -> None:
    """Slow, slow, FAST, slow, slow: consecutive means consecutive."""
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    for tps in (20.0, 20.0, 90.0, 20.0, 20.0):
        await _serve(stream, clock, host="Mixed", tokens=1700, tps=tps)

    assert stream._provider_slow_strikes[MODEL]["Mixed"] == 2
    assert not stream._provider_retired.get(MODEL)


@pytest.mark.asyncio
async def test_slow_turns_separated_by_idle_gaps_do_not_pair(clock: FakeClock) -> None:
    """Host load moves over minutes-to-hours: three slow turns a day apart are
    three first readings, not a pattern."""
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    gap = SessionStreamFn.HOST_SLOW_STRIKE_MAX_GAP_S + 60
    for _ in range(6):
        await _serve(stream, clock, host="Slow", idle_s=gap, **_slow())

    assert stream._provider_slow_strikes[MODEL]["Slow"] == 1
    assert not stream._provider_retired.get(MODEL)


@pytest.mark.asyncio
async def test_a_single_host_is_released_never_ignored(clock: FakeClock) -> None:
    """Only one host has ever served: `ignore` could leave "all providers have
    been ignored" (a 404 the retry ladder cannot fix). The pin is released --
    OpenRouter's own routing is free again -- and nothing reaches `ignore`."""
    stream, notices = _stream()
    for _ in range(8):
        await _serve(stream, clock, host="Only", **_slow())

    assert not stream._provider_retired.get(MODEL)
    assert stream._provider_slow_released[MODEL] == {"Only"}
    stamped = stream._apply_cache_affinity(_request())
    assert stamped.provider_avoid == []
    assert stamped.provider_affinity is None  # the pin is gone, not re-acquired
    assert len(notices) == 1 and "no longer pinning" in notices[0][0]


@pytest.mark.asyncio
async def test_all_hosts_slow_retires_none(clock: FakeClock) -> None:
    """A slow MODEL, not a slow host: every move buys a cold prefix and no speed.
    Both hosts measured slow means neither is ever retired -- the last viable
    host always stays."""
    stream, _ = _stream()
    for host in ("A", "B", "A", "B", "A", "B", "A", "B", "A", "B"):
        await _serve(stream, clock, host=host, **_slow())

    assert not stream._provider_retired.get(MODEL)
    assert stream._apply_cache_affinity(_request()).provider_avoid == []


@pytest.mark.asyncio
async def test_a_release_is_promoted_only_when_a_fast_host_is_measured(clock: FakeClock) -> None:
    """The reversible release becomes a hard `ignore` the moment another host is
    MEASURED at or above the floor -- and not before."""
    stream, notices = _stream()
    for _ in range(3):
        await _serve(stream, clock, host="Slow", **_slow())
    assert stream._provider_slow_released[MODEL] == {"Slow"}
    assert not stream._provider_retired.get(MODEL)

    # A different host serving SLOWLY is not proof of a faster route...
    await _serve(stream, clock, host="AlsoSlow", **_slow())
    assert not stream._provider_retired.get(MODEL)

    # ...a different host measured fast is.
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    assert stream._provider_retired[MODEL] == {"Slow"}
    assert stream._apply_cache_affinity(_request()).provider_avoid == ["Slow"]
    assert any("routing away from Slow" in text for text, _ in notices)


@pytest.mark.asyncio
async def test_a_released_host_that_recovers_is_forgiven(clock: FakeClock) -> None:
    stream, _ = _stream()
    for _ in range(3):
        await _serve(stream, clock, host="Slow", **_slow())
    assert stream._provider_slow_released[MODEL] == {"Slow"}

    await _serve(stream, clock, host="Slow", tokens=1700, tps=120.0)
    assert not stream._provider_slow_released[MODEL]
    # ...and the pin is re-acquired from that very turn.
    assert stream._provider_affinity[MODEL] == "Slow"


@pytest.mark.asyncio
async def test_the_shared_retirement_cap_bounds_speed_retirements(clock: FakeClock) -> None:
    stream, _ = _stream()
    stream._provider_retired[MODEL] = {"X1", "X2", "X3"}
    assert len(stream._provider_retired[MODEL]) == SessionStreamFn.MAX_RETIRED_PROVIDERS
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    for _ in range(4):
        await _serve(stream, clock, host="Slow", **_slow())

    assert stream._provider_retired[MODEL] == {"X1", "X2", "X3"}
    assert "Slow" not in stream._provider_retired[MODEL]


@pytest.mark.asyncio
async def test_a_non_turn_request_never_strikes(clock: FakeClock) -> None:
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    for purpose in ("compaction", "compaction_advisor"):
        for _ in range(5):
            await _serve(stream, clock, host="Slow", request=_request(purpose=purpose), **_slow())

    assert not stream._provider_slow_strikes.get(MODEL)
    assert not stream._provider_retired.get(MODEL)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "request_overrides", "why"),
    [
        ({"providers": {"openrouter": {"provider_affinity": False}}}, {}, "affinity off"),
        ({"providers": {"openrouter": {"order": ["Wafer"]}}}, {}, "the user pinned a host"),
        ({"providers": {"openrouter": {"ignore": ["Wafer"]}}}, {}, "the user ignores a host"),
        ({"providers": {"openrouter": {"sort": "throughput"}}}, {}, "the user sorts"),
        ({}, {"provider": "radient"}, "radient aggregates but was never measured"),
        ({}, {"provider": "deepseek"}, "a direct provider has no host to retire"),
    ],
)
async def test_nothing_is_recorded_when_affinity_does_not_apply(
    clock: FakeClock, settings: dict[str, Any], request_overrides: dict[str, Any], why: str
) -> None:
    """Zero effect when the pin cannot apply: no strike, no rate, no retirement."""
    stream, notices = _stream(settings)
    for host in ("Fast", "Slow", "Slow", "Slow", "Slow"):
        tps = 150.0 if host == "Fast" else 20.3
        await _serve(
            stream,
            clock,
            host=host,
            tokens=1700,
            tps=tps,
            request=_request(**request_overrides),
        )

    assert not stream._provider_retired, why
    assert not stream._provider_slow_strikes, why
    assert not stream._provider_host_rate, why
    assert not stream._provider_slow_released, why
    assert notices == [], why


# ---------------------------------------------------------------------------
# Interaction with the cache machinery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_speed_retirement_survives_a_compaction_but_a_cache_one_does_not(
    clock: FakeClock,
) -> None:
    """Cache evidence is about a prefix a compaction replaces; decode speed is not.
    Forgetting the speed retirement would send the conversation back to the slow
    host for three more turns after every compaction."""
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    for _ in range(3):
        await _serve(stream, clock, host="Slow", **_slow())
    stream._provider_retired[MODEL].add("CacheDead")  # a cache-strike retirement
    assert stream._provider_retired[MODEL] == {"Slow", "CacheDead"}

    stream._apply_cache_affinity(_request(purpose="compaction"))

    assert stream._provider_retired[MODEL] == {"Slow"}


@pytest.mark.asyncio
async def test_a_speed_only_retirement_does_not_redo_the_clear_on_every_compaction(
    clock: FakeClock,
) -> None:
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    for _ in range(3):
        await _serve(stream, clock, host="Slow", **_slow())

    before = stream._provider_retired
    stream._apply_cache_affinity(_request(purpose="compaction"))
    assert stream._provider_retired is before  # the early return, not a rebuild


@pytest.mark.asyncio
async def test_the_cache_and_speed_strike_streams_do_not_pair(clock: FakeClock) -> None:
    """One cache miss plus two slow turns is not three of anything."""
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    stream._provider_strikes.setdefault(MODEL, {})["Slow"] = 1
    for _ in range(2):
        await _serve(stream, clock, host="Slow", **_slow())

    assert not stream._provider_retired.get(MODEL)


@pytest.mark.asyncio
async def test_a_retirement_removes_the_pin_and_names_the_host_in_the_next_request(
    clock: FakeClock,
) -> None:
    stream, _ = _stream()
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    for _ in range(3):
        await _serve(stream, clock, host="Slow", **_slow())

    stamped = stream._apply_cache_affinity(_request())
    assert stamped.provider_avoid == ["Slow"]
    assert stamped.provider_affinity != "Slow"
    # The next host to serve is re-pinned as usual.
    await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
    assert stream._apply_cache_affinity(_request()).provider_affinity == "Fast"


@pytest.mark.asyncio
async def test_a_fork_with_lineage_inherits_speed_state_as_copies(clock: FakeClock) -> None:
    parent, _ = _stream()
    await _serve(parent, clock, host="Fast", tokens=1700, tps=150.0)
    for _ in range(3):
        await _serve(parent, clock, host="Slow", **_slow())

    child = parent.fork("s-child", cache_lineage_id="s-speed")
    try:
        assert child._provider_retired[MODEL] == {"Slow"}
        assert child._provider_retired_slow[MODEL] == {"Slow"}
        assert child._provider_host_rate[MODEL]["Fast"] == pytest.approx(150.0, rel=0.05)
        child._provider_retired[MODEL].add("Other")
        child._provider_host_rate[MODEL]["Fast"] = 1.0
        assert parent._provider_retired[MODEL] == {"Slow"}
        assert parent._provider_host_rate[MODEL]["Fast"] == pytest.approx(150.0, rel=0.05)
    finally:
        child._transport.owners -= 1


@pytest.mark.asyncio
async def test_a_fresh_delegated_fork_inherits_nothing(clock: FakeClock) -> None:
    parent, _ = _stream()
    await _serve(parent, clock, host="Fast", tokens=1700, tps=150.0)
    for _ in range(3):
        await _serve(parent, clock, host="Slow", **_slow())

    child = parent.fork("s-child")
    try:
        assert child._provider_retired == {}
        assert child._provider_host_rate == {}
    finally:
        child._transport.owners -= 1


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_floor_is_configurable(clock: FakeClock) -> None:
    """A 60 tok/s host is healthy at the default and retired under a 100 floor."""
    for floor, retired in ((None, False), (100, True)):
        providers = (
            {} if floor is None else {"providers": {"openrouter": {"retire_below_tps": floor}}}
        )
        stream, _ = _stream(providers)
        await _serve(stream, clock, host="Fast", tokens=1700, tps=150.0)
        for _ in range(3):
            await _serve(stream, clock, host="Mid", tokens=1700, tps=60.0)
        assert bool(stream._provider_retired.get(MODEL)) is retired, floor


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (None, OPENROUTER_RETIRE_BELOW_TPS),
        ({}, OPENROUTER_RETIRE_BELOW_TPS),
        ({"providers": {}}, OPENROUTER_RETIRE_BELOW_TPS),
        ({"providers": {"openrouter": {}}}, OPENROUTER_RETIRE_BELOW_TPS),
        ({"providers": {"openrouter": {"retire_below_tps": 0}}}, 0.0),
        ({"providers": {"openrouter": {"retire_below_tps": 40}}}, 40.0),
        ({"providers": {"openrouter": {"retire_below_tps": 12.5}}}, 12.5),
        # Malformed values fall back to the default: "disabled" would be a session
        # quietly staying on a slow host, which is the failure this exists to end.
        ({"providers": {"openrouter": {"retire_below_tps": -1}}}, OPENROUTER_RETIRE_BELOW_TPS),
        ({"providers": {"openrouter": {"retire_below_tps": True}}}, OPENROUTER_RETIRE_BELOW_TPS),
        ({"providers": {"openrouter": {"retire_below_tps": "fast"}}}, OPENROUTER_RETIRE_BELOW_TPS),
        (
            {"providers": {"openrouter": {"retire_below_tps": float("nan")}}},
            OPENROUTER_RETIRE_BELOW_TPS,
        ),
        ({"providers": "nope"}, OPENROUTER_RETIRE_BELOW_TPS),
    ],
)
def test_the_floor_resolver(settings: Any, expected: float) -> None:
    assert _openrouter_retire_below_tps(settings) == expected


def test_the_new_key_does_not_make_the_default_config_emit_a_provider_object() -> None:
    """It is a HARNESS switch: the wire-preference resolver must not read it, or
    the shipped config would start sending a `provider` object on every request
    and defeat OpenRouter's own sticky routing."""
    from local_operator.model.configure import _openrouter_provider_preferences

    assert (
        _openrouter_provider_preferences({"providers": {"openrouter": {"retire_below_tps": 99}}})
        is None
    )


def test_the_default_config_matches_the_consumer_constant() -> None:
    from local_operator.config import DEFAULT_CONFIG

    assert (
        DEFAULT_CONFIG.values["providers"]["openrouter"]["retire_below_tps"]
        == OPENROUTER_RETIRE_BELOW_TPS
    )


def test_the_constants_state_their_relationships() -> None:
    """The policy's shape, pinned so a tweak to one number is a conscious one."""
    assert SessionStreamFn.HOST_SLOW_STRIKES_TO_RETIRE >= SessionStreamFn.PROVIDER_STRIKES_TO_RETIRE
    # The floor must sit between the measured slow host and the measured fast ones.
    assert _FITTED_TPS["OpenInference"] < OPENROUTER_RETIRE_BELOW_TPS < _FITTED_TPS["Morph"]
    # A qualifying turn at the slow rate must be long enough for decode to dominate.
    assert SessionStreamFn.HOST_SLOW_MIN_OUTPUT_TOKENS / _FITTED_TPS["OpenInference"] >= 20
