"""Pure rules of idle / pre-wake compaction (``compaction/idle.py``).

Table tests at the three window sizes the registry actually spans (1M, 200k,
64k), the cache-TTL helpers, and the parked relaxations of the advisor's
prompt and validator. The session wiring is ``tests/unit/session/
test_compaction_idle.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.compaction import idle
from local_operator.compaction.advisor import build_advisor_prompt, validate_hint
from local_operator.compaction.thresholds import (
    DEFAULT_IDLE_MIN_GROWTH_TOKENS,
    DEFAULT_IDLE_THRESHOLD_TOKENS,
    CompactionSettings,
    resolve_threshold_tokens,
)
from local_operator.harness.types import Message, Usage
from local_operator.session.transcript import BOOKKEEPING_CUSTOM_TYPES

NOW = 1_000_000_000
TTL_5M_MS = 300_000


def t_eff(window: int) -> int:
    return resolve_threshold_tokens(window, CompactionSettings())


def due(**overrides: Any) -> idle.IdleDecision:
    window = overrides.pop("window", 1_000_000)
    base: dict[str, Any] = dict(
        enabled=True,
        advisor_enabled=True,
        strategy="auto",
        is_subagent=False,
        context_tokens=260_000,
        t_eff=t_eff(window),
        idle_threshold_tokens=DEFAULT_IDLE_THRESHOLD_TOKENS,
        idle_min_growth_tokens=DEFAULT_IDLE_MIN_GROWTH_TOKENS,
        baseline_after_tokens=150_000,
        declined_at_tokens=None,
        next_wake_due_ms=None,
        cache_expiry_ms=NOW + TTL_5M_MS,
        now_ms=NOW,
        advisor_timeout_s=30.0,
    )
    base.update(overrides)
    return idle.idle_compaction_due(**base)


@pytest.mark.parametrize(
    ("window", "threshold", "growth"),
    [
        # 1M: T_eff = min(0.8 * 1M, 400k) = 400k -> min(200k, 240k), min(60k, 100k).
        (1_000_000, 200_000, 60_000),
        # 200k: T_eff = 160k -> min(200k, 96k), min(60k, 40k).
        (200_000, 96_000, 40_000),
        # 64k: T_eff = 51.2k -> min(200k, 30.7k), min(60k, 12.8k).
        (64_000, 30_720, 12_800),
    ],
)
def test_threshold_and_growth_scale_with_the_window(window, threshold, growth):
    assert idle.idle_threshold_for(t_eff(window), DEFAULT_IDLE_THRESHOLD_TOKENS) == threshold
    assert idle.idle_growth_for(t_eff(window), DEFAULT_IDLE_MIN_GROWTH_TOKENS) == growth


@pytest.mark.parametrize(
    ("window", "context", "baseline", "declined", "reason"),
    [
        (1_000_000, 199_999, 0, None, "below_idle_threshold"),
        (1_000_000, 200_000, 0, None, "due"),
        (1_000_000, 260_000, 150_000, None, "due"),
        (1_000_000, 260_000, 200_001, None, "insufficient_growth"),
        (1_000_000, 260_000, 150_000, 210_000, "insufficient_growth"),
        (1_000_000, 270_000, 150_000, 210_000, "due"),
        (1_000_000, 260_000, None, None, "no_growth_baseline"),
        (200_000, 95_000, 0, None, "below_idle_threshold"),
        (200_000, 120_000, 80_000, None, "due"),
        (200_000, 120_000, 81_000, None, "insufficient_growth"),
        (64_000, 30_000, 0, None, "below_idle_threshold"),
        (64_000, 40_000, 27_200, None, "due"),
        (64_000, 40_000, 27_201, None, "insufficient_growth"),
    ],
)
def test_predicate_table(window, context, baseline, declined, reason):
    decision = due(
        window=window,
        context_tokens=context,
        baseline_after_tokens=baseline,
        declined_at_tokens=declined,
    )
    assert decision.reason == reason
    assert decision.due is (reason == "due")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"enabled": False}, "disabled"),
        ({"strategy": "off"}, "disabled"),
        ({"advisor_enabled": False}, "advisor_off"),
        ({"is_subagent": True}, "subagent"),
    ],
)
def test_switches_refuse_first(overrides, reason):
    assert due(**overrides).reason == reason


def test_wake_inside_the_ttl_keeps_the_cache_warm():
    near = due(next_wake_due_ms=NOW + 120_000)
    assert near.reason == "wake_keeps_cache_warm" and near.wake_pending
    far = due(next_wake_due_ms=NOW + 3_600_000)
    assert far.due and far.wake_pending
    assert not due().wake_pending


def test_cold_cache_is_never_read():
    # The advisor's worst case must finish before E.
    assert due(now_ms=NOW + TTL_5M_MS - 30_000).due
    assert due(now_ms=NOW + TTL_5M_MS - 29_999).reason == "cache_cold"


def test_fire_time_leads_the_expiry_by_advisor_budget_plus_margin():
    assert idle.idle_fire_at_ms(NOW + TTL_5M_MS, 30.0) == NOW + TTL_5M_MS - 60_000
    # A timer that fires a little late still passes the cold-cache gate.
    late = idle.idle_fire_at_ms(NOW + TTL_5M_MS, 30.0) + 500
    assert due(now_ms=late).due


def test_growth_gate_never_collapses_to_zero():
    assert idle.idle_growth_for(400_000, 0) == 1
    assert idle.idle_growth_for(0, 60_000) == 60_000


def test_cache_ttl_from_usage():
    assert idle.cache_ttl_from_usage(Usage(cache_write_1h_tokens=10)) == 3600.0
    assert idle.cache_ttl_from_usage(Usage(cache_write_5m_tokens=10)) == 300.0
    assert idle.cache_ttl_from_usage(Usage(cache_read_tokens=10)) is None
    assert idle.cache_ttl_from_usage(None) is None
    assert idle.fallback_cache_ttl_s("anthropic") == 300.0
    assert idle.fallback_cache_ttl_s("some-new-provider") == 300.0


def test_receipt_types_are_bookkeeping_literals():
    # transcript.py lists the literals without importing compaction.
    assert idle.COMPACTION_IDLE_PASS_TYPE in BOOKKEEPING_CUSTOM_TYPES
    assert idle.COMPACTION_IDLE_DECLINED_TYPE in BOOKKEEPING_CUSTOM_TYPES


def _history() -> list[Message]:
    filler = "older finished work " * 400
    messages: list[Message] = [Message.user("start the original task: " + filler)]
    for index in range(6):
        messages.append(Message.assistant(f"done step {index} " + filler))
        messages.append(Message.user(f"next step {index}"))
    messages.append(Message.assistant("all finished; waiting for the next wake"))
    return messages


def test_prompt_is_byte_identical_when_not_parked():
    history = _history()
    plain = build_advisor_prompt(history, context_tokens=1, threshold_tokens=2)
    assert plain == build_advisor_prompt(
        history, context_tokens=1, threshold_tokens=2, parked=False
    )
    assert idle.PARKED_ADVISOR_NOTE not in plain
    parked = build_advisor_prompt(
        history, context_tokens=1, threshold_tokens=2, parked=True, wake_pending=True
    )
    assert idle.PARKED_ADVISOR_NOTE in parked and "A wake is scheduled" in parked
    # Append-only within the turn: the candidate list and the closing line remain.
    assert parked.endswith(plain.split("\n\n")[-1])


def _validate(history, payload, *, parked, keep_recent=50):
    # genuine_user_ids = every user message: the task floor then spans from
    # the LAST genuine user turn, which is the widen-only rule's floor.
    return validate_hint(
        payload,
        history,
        genuine_user_ids={m.id for m in history if m.role == "user"},
        min_confidence=0.6,
        keep_recent_tokens=keep_recent,
        floor_cap=10**9,
        parked=parked,
    )


def test_parked_may_narrow_below_the_task_floor_but_not_unparked():
    history = _history()
    # Anchor on the final assistant message: narrower than the task floor
    # (which spans from the last user turn), wider than keep_recent.
    anchor = history[-1].id
    payload = {"preserve_from": anchor, "compact_now": True, "confidence": 0.9, "reason": "idle"}
    assert _validate(history, payload, parked=False, keep_recent=1) is None
    hint = _validate(history, payload, parked=True, keep_recent=1)
    assert hint is not None and hint.preserve_from_id == anchor


def test_parked_requires_compact_now_and_high_confidence():
    history = _history()
    anchor = history[-2].id
    base = {"preserve_from": anchor, "compact_now": True, "confidence": 0.9, "reason": "ok"}
    assert _validate(history, base, parked=True, keep_recent=1) is not None
    assert _validate(history, {**base, "compact_now": False}, parked=True, keep_recent=1) is None
    assert _validate(history, {**base, "confidence": 0.7}, parked=True, keep_recent=1) is None
    # 0.7 clears the ordinary floor (0.6) when not parked.
    assert _validate(history, {**base, "confidence": 0.7}, parked=False, keep_recent=1) is not None


def test_parked_still_respects_keep_recent():
    history = _history()
    payload = {
        "preserve_from": history[-1].id,
        "compact_now": True,
        "confidence": 0.9,
        "reason": "x",
    }
    assert _validate(history, payload, parked=True, keep_recent=10**6) is None
