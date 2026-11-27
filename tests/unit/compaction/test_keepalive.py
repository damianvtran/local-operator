"""The cache keep-alive's pure rules: when a turn end arms, when a fire pings.

Table-driven over ``compaction.keepalive`` exactly as the session calls it, so
every skip reason in the design is one row here (mode off, subagent, 5m cache,
a wake that keeps the cache warm anyway, a late fire, the consecutive and daily
caps, a changed prefix, a different route, a request that touched no cache).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.analytics.store import AnalyticsStore
from local_operator.compaction import keepalive
from local_operator.compaction.api import CompactionSettings
from local_operator.harness.wake_types import WakeSchedule
from local_operator.providers.anthropic_cache_ttl import cache_role
from local_operator.wakes.store import (
    is_cache_keepalive_row,
    is_internal_wake_row,
    scheduled_rows,
)

ANCHOR = 1_800_000_000_000
HOUR_MS = 3_600_000


def _arm(**overrides: Any) -> keepalive.KeepaliveDecision:
    args: dict[str, Any] = {
        "mode": "chief_of_staff",
        "covered": True,
        "is_subagent": False,
        "ttl_s": 3600.0,
        "anchor_ms": ANCHOR,
        "cache_read_tokens": 150_000,
        "cache_write_1h_tokens": 400,
        "route_matches": True,
        "other_wake_due_ms": None,
        "remaining_pings": 4,
    }
    args.update(overrides)
    return keepalive.arm_decision(**args)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({}, "armed"),
        ({"mode": "off"}, "mode_off"),
        ({"is_subagent": True}, "subagent"),
        ({"covered": False}, "not_covered"),
        ({"ttl_s": 300.0}, "not_1h_cache"),
        ({"ttl_s": None}, "not_1h_cache"),
        ({"anchor_ms": 0}, "no_cache_anchor"),
        ({"cache_read_tokens": 0, "cache_write_1h_tokens": 0}, "no_cache_activity"),
        ({"route_matches": False}, "route_mismatch"),
        ({"other_wake_due_ms": ANCHOR + HOUR_MS - 1}, "wake_keeps_cache_warm"),
        ({"remaining_pings": 0}, "consecutive_cap"),
    ],
)
def test_arm_decision_table(overrides: dict[str, Any], reason: str) -> None:
    decision = _arm(**overrides)
    assert decision.reason == reason
    assert decision.ok is (reason == "armed")


def test_a_cold_request_that_wrote_the_whole_prefix_is_worth_keeping_warm() -> None:
    """read=0 with a full 1h write is the case the feature exists for."""
    assert _arm(cache_read_tokens=0, cache_write_1h_tokens=200_000).ok
    # Read-only (the entry's lifetime was written earlier) arms too.
    assert _arm(cache_read_tokens=150_000, cache_write_1h_tokens=0).ok


def test_a_wake_due_after_expiry_does_not_suppress_the_keepalive() -> None:
    assert _arm(other_wake_due_ms=ANCHOR + HOUR_MS).ok


def _fire(**overrides: Any) -> keepalive.KeepaliveDecision:
    args: dict[str, Any] = {
        "now_ms": ANCHOR + 55 * 60_000,
        "anchor_ms": ANCHOR,
        "attempt": 0,
        "max_consecutive": 4,
        "daily_count": 0,
        "daily_max": 24,
        "stored_digest": "abc",
        "current_digest": "abc",
        "route_matches": True,
        "busy": False,
    }
    args.update(overrides)
    return keepalive.fire_decision(**args)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({}, "ping"),
        ({"now_ms": ANCHOR + HOUR_MS}, "expired"),
        ({"now_ms": ANCHOR + 3 * HOUR_MS}, "expired"),
        ({"attempt": 4}, "consecutive_cap"),
        ({"busy": True}, "busy"),
        ({"route_matches": False}, "route_mismatch"),
        ({"daily_count": None}, "ledger_unreadable"),
        ({"daily_count": 24}, "daily_cap"),
        ({"current_digest": "def"}, "prefix_changed"),
        ({"current_digest": None}, "prefix_changed"),
        ({"stored_digest": "", "current_digest": ""}, "prefix_changed"),
    ],
)
def test_fire_decision_table(overrides: dict[str, Any], reason: str) -> None:
    decision = _fire(**overrides)
    assert decision.reason == reason
    assert decision.ok is (reason == "ping")


def test_the_fourth_ping_is_the_last() -> None:
    assert _fire(attempt=3).ok
    assert not _fire(attempt=4).ok
    assert _fire(daily_count=23).ok


def _tool(name: str, description: str = "d") -> Any:
    # The digest reads the three wire fields only, so a stand-in carries just those.
    return SimpleNamespace(name=name, description=description, parameters={"type": "object"})


def test_prefix_digest_tracks_system_tools_and_history_tail() -> None:
    base = keepalive.prefix_digest(["a", "b"], [_tool("bash")], "m1")
    assert base == keepalive.prefix_digest(["a", "b"], [_tool("bash")], "m1")
    assert base != keepalive.prefix_digest(["a", "c"], [_tool("bash")], "m1")
    assert base != keepalive.prefix_digest(["a", "b"], [_tool("bash"), _tool("read")], "m1")
    assert base != keepalive.prefix_digest(["a", "b"], [_tool("bash", "other")], "m1")
    assert base != keepalive.prefix_digest(["a", "b"], [_tool("bash")], "m2")


def test_the_ping_is_marked_at_the_main_threads_ttl() -> None:
    assert cache_role("cache_keepalive", delegated=False) == "main"
    assert cache_role("compaction_advisor", delegated=False) == "auxiliary"


def test_settings_defaults() -> None:
    settings = CompactionSettings()
    assert settings.keepalive_mode == "chief_of_staff"
    assert settings.keepalive_max_consecutive == 4
    assert settings.keepalive_daily_max == 24
    assert CompactionSettings(keepalive_mode="always_on").keepalive_mode == "always_on"
    with pytest.raises(ValueError):
        CompactionSettings.model_validate({"keepalive_mode": "sometimes"})
    with pytest.raises(ValueError):
        CompactionSettings(keepalive_max_consecutive=-1)


def test_the_row_is_internal_on_every_human_surface() -> None:
    row = WakeSchedule(
        id=keepalive.row_id(ANCHOR),
        message="prompt-cache keep-alive",
        next_due_at=ANCHOR + 3_300_000,
        kind="cache_keepalive",
        hidden=True,
        cache_anchor_ms=ANCHOR,
        prefix_digest="abc",
    )
    visible = WakeSchedule(id="w1", message="remind me", next_due_at=ANCHOR)
    dumped = row.model_dump()
    assert is_cache_keepalive_row(row) and is_cache_keepalive_row(dumped)
    assert is_internal_wake_row(row) and is_internal_wake_row(dumped)
    # ``scheduled_rows`` is the one filter the wake tool's list, the CLI, the
    # desktop route and feed, the mobile projection and the picker read.
    assert scheduled_rows([row, visible]) == [visible]
    assert scheduled_rows([dumped]) == []
    # The persisted shape round-trips through the model (``extra="forbid"``).
    assert WakeSchedule.model_validate(dumped) == row


def test_count_purpose_calls_reads_the_ledger(tmp_path) -> None:
    from local_operator.analytics.model import CallSnapshot

    store = AnalyticsStore(tmp_path / "analytics.db")

    def snap(session: str, purpose: str, ts: int) -> CallSnapshot:
        return CallSnapshot(
            ts_ms=ts,
            session_id=session,
            provider="anthropic",
            model_id="claude-opus-5-5",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            context_tokens=1,
            purpose=purpose,
        )

    store.record_batch(
        [
            snap("s1", "cache_keepalive", ANCHOR),
            snap("s1", "cache_keepalive", ANCHOR + 10),
            snap("s1", "turn", ANCHOR + 20),
            snap("s2", "cache_keepalive", ANCHOR + 30),
            snap("s1", "cache_keepalive", ANCHOR - 86_400_001),
        ]
    )
    try:
        assert store.count_purpose_calls("s1", "cache_keepalive", ANCHOR - 86_400_000) == 2
        assert store.count_purpose_calls("s3", "cache_keepalive", 0) == 0
    finally:
        store.close()


def test_count_purpose_calls_answers_none_when_unreadable(tmp_path) -> None:
    store = AnalyticsStore(tmp_path / "analytics.db")
    store._read_connection = lambda: None  # type: ignore[method-assign]
    try:
        assert store.count_purpose_calls("s1", "cache_keepalive", 0) is None
    finally:
        store.close()
