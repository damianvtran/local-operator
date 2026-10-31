"""Frozen-clock account policy contracts; no stores, credentials or probes."""

import asyncio
import math
from datetime import UTC, datetime

import httpx
import pytest

from local_operator.providers.account_policy import (
    ResetCandidate,
    effective_account_selection_policy,
    reset_candidate,
    reset_ranked_ids,
)
from local_operator.providers.usage import (
    UsageAmount,
    UsageLimit,
    UsageReport,
    fetch_usage,
)

NOW = 1_800_000_000_000


@pytest.mark.parametrize(
    ("retry", "expected"),
    [
        ({}, "reset-soonest"),
        ({"accountSelectionPolicy": "reset-soonest"}, "reset-soonest"),
        ({"accountSelectionPolicy": "headroom"}, "headroom"),
        ({"accountSelectionPolicy": "fair"}, "fair"),
        ({"account_selection_policy": "headroom"}, "headroom"),
        ({"usageAwareAccountPick": True}, "headroom"),
        ({"usageAwareAccountPick": False}, "fair"),
        ({"usage_aware_account_pick": False}, "fair"),
        ({"usageAwareAccountPick": "false"}, "reset-soonest"),
        ({"usageAwareAccountPick": 1}, "reset-soonest"),
        ({"usageAwareAccountPick": None}, "reset-soonest"),
        ({"accountSelectionPolicy": "fair", "usageAwareAccountPick": True}, "fair"),
        ({"accountSelectionPolicy": "bad", "usageAwareAccountPick": True}, "reset-soonest"),
        ({"accountSelectionPolicy": None}, "reset-soonest"),
        ({"accountSelectionPolicy": 1}, "reset-soonest"),
    ],
)
def test_central_presence_and_strict_legacy_resolution(retry, expected):
    assert effective_account_selection_policy({"retry": retry}) == expected


@pytest.mark.parametrize("settings", [None, {}, {"retry": None}, {"retry": []}])
def test_malformed_or_absent_retry_is_default(settings):
    assert effective_account_selection_policy(settings) == "reset-soonest"


def limit(window, offset, used=0.5, *, tier="", paid=False):
    return UsageLimit(
        id=f"synthetic:{window}:{tier}",
        label="Not a period identifier",
        window=window,
        amount=UsageAmount(used_fraction=used),
        resets_at_ms=NOW + offset if offset is not None else None,
        tier=tier,
        shared=not tier,
        paid_fallback=paid,
    )


def report(*limits, age=0):
    return UsageReport("anthropic", fetched_at=NOW - age, limits=list(limits))


def deadline_candidate(account_id: int, field: str, deadline: int) -> ResetCandidate:
    # Keep parameterized boundary coverage while naming the actual dataclass
    # field: a generic **dict could also bind the unrelated `state` parameter.
    assert field in ("weekly_ms", "five_hour_ms")
    if field == "weekly_ms":
        return ResetCandidate(account_id, weekly_ms=deadline)
    return ResetCandidate(account_id, five_hour_ms=deadline)


@pytest.mark.parametrize("delta,in_bucket", [(59_999, True), (60_000, True), (60_001, False)])
@pytest.mark.parametrize("field", ["weekly_ms", "five_hour_ms"])
def test_inclusive_fixed_minimum_tolerance(delta, in_bucket, field):
    rows = [deadline_candidate(1, field, NOW + 1), deadline_candidate(2, field, NOW + 1 + delta)]
    firsts = {reset_ranked_ids(rows, f"session-{i}")[0] for i in range(100)}
    assert firsts == ({1, 2} if in_bucket else {1})


@pytest.mark.parametrize("field", ["weekly_ms", "five_hour_ms"])
def test_no_transitive_neighbour_chaining(field):
    rows = [
        deadline_candidate(i + 1, field, NOW + offset)
        for i, offset in enumerate((0, 59_000, 118_000))
    ]
    for i in range(100):
        assert reset_ranked_ids(rows, str(i))[-1] == 3


def test_weekly_primary_five_hour_secondary_then_hash():
    rows = [
        ResetCandidate(1, "healthy", NOW + 100_000, NOW + 900_000),
        ResetCandidate(2, "healthy", NOW + 150_000, NOW + 100_000),
        ResetCandidate(3, "healthy", NOW + 170_000, NOW + 1),
    ]
    assert reset_ranked_ids(rows, "stable") == [2, 1, 3]
    assert reset_ranked_ids(list(reversed(rows)), "stable") == [2, 1, 3]


def test_unknown_neutral_spread_reserve_last_depleted_absent():
    rows = [
        ResetCandidate(1, "healthy", NOW + 1),
        ResetCandidate(2),
        ResetCandidate(3, "reserve", NOW),
        ResetCandidate(4, "depleted", NOW),
    ]
    assert {reset_ranked_ids(rows, str(i))[0] for i in range(100)} == {1, 2}
    assert all(reset_ranked_ids(rows, str(i))[-1] == 3 for i in range(100))


def test_shared_and_matching_family_caps_all_gate_eligibility():
    candidate = reset_candidate(
        1,
        report(limit("5 hour", 10_000), limit("7 day", 20_000, 1, tier="sonnet")),
        "claude-sonnet-5",
        NOW,
    )
    assert candidate.state == "depleted"
    assert reset_ranked_ids([candidate, ResetCandidate(2)], "x") == [2]
    other = reset_candidate(
        1,
        report(limit("5 hour", 10_000), limit("7 day", 20_000, 1, tier="sonnet")),
        "claude-opus-4",
        NOW,
    )
    assert other.state == "healthy"
    assert other.weekly_ms is None


def test_earliest_relevant_weekly_refill_not_recovery_horizon():
    candidate = reset_candidate(
        1,
        report(limit("7 day", 900_000), limit("7 day", 100_000, tier="sonnet")),
        "claude-sonnet-5",
        NOW,
    )
    assert candidate.weekly_ms == NOW + 100_000
    spent = reset_candidate(
        1,
        report(limit("5 hour", -1, 1), limit("7 day", 100_000, 1)),
        "m",
        NOW,
    )
    assert spent.state == "depleted"  # one elapsed cap does not reopen the other


@pytest.mark.parametrize("offset", [None, -1, 0, 8 * 24 * 60 * 60_000])
def test_missing_past_and_far_future_deadlines_do_not_rank(offset):
    candidate = reset_candidate(1, report(limit("7 day", offset)), "m", NOW)
    assert candidate.weekly_ms is None


@pytest.mark.parametrize("value", ["bad", math.inf, math.nan, True])
def test_corrupt_reset_never_manufactures_eligibility(value):
    window = limit("7 day", None, 1)
    window.resets_at_ms = value
    candidate = reset_candidate(1, report(window), "m", NOW)
    assert candidate.state == "depleted"
    assert candidate.weekly_ms is None


def test_stale_forecast_unknown_but_conservative_depletion_retained():
    assert (
        reset_candidate(1, report(limit("7 day", 100_000), age=1_800_001), "m", NOW).state
        == "unknown"
    )
    spent = reset_candidate(1, report(limit("7 day", 100_000, 1), age=1_800_001), "m", NOW)
    assert spent.state == "depleted"
    assert spent.weekly_ms is None
    assert reset_candidate(1, report(limit("7 day", 100_000), age=-1), "m", NOW) == ResetCandidate(
        1
    )


def test_five_hour_fallback_and_no_fabricated_paid_period():
    rows = [
        reset_candidate(i, report(limit("5 hour", offset)), "m", NOW)
        for i, offset in [(1, 800_000), (2, 100_000)]
    ]
    assert reset_ranked_ids(rows, "x") == [2, 1]
    paid = reset_candidate(1, report(limit("lifetime", None, paid=True)), "m", NOW)
    assert paid.weekly_ms is None and paid.five_hour_ms is None


def test_a_fresh_depletion_states_when_every_spent_window_reopens():
    """2026-10-28: the broker's ``quota_blocked`` remainder is the LATEST spent reset.

    Both spent windows must reopen before the account can serve, so the earliest
    of them would be a time the account can still refuse; a window with headroom
    does not gate the reopening at all.
    """
    both = reset_candidate(
        1,
        report(limit("5 hour", 600_000, 1), limit("7 day", 900_000, 1), limit("7 day", 50_000)),
        "m",
        NOW,
    )
    assert both.state == "depleted"
    assert both.reopens_at_ms == NOW + 900_000


def test_a_reopening_is_never_stated_without_fresh_proof():
    """Stale evidence, a spent window with no reset, or any non-depleted verdict: None."""
    stale = reset_candidate(1, report(limit("5 hour", 600_000, 1), age=1_800_001), "m", NOW)
    assert stale.state == "depleted" and stale.reopens_at_ms is None
    unstated = reset_candidate(1, report(limit("5 hour", None, 1)), "m", NOW)
    assert unstated.state == "depleted" and unstated.reopens_at_ms is None
    healthy = reset_candidate(1, report(limit("5 hour", 600_000, 0.2)), "m", NOW)
    assert healthy.state == "healthy" and healthy.reopens_at_ms is None
    assert reset_candidate(1, None, "m", NOW).reopens_at_ms is None


def test_a_garbled_far_future_reset_never_becomes_a_proven_reopening():
    """#3108 review m2: the ranking path's sanity window bounds the reopening too.

    A 5-hour window that claims to reset two years out is garbled evidence. The
    verdict stays ``depleted`` (the account is spent), but no reopening is proven,
    so the owner cannot turn the garbage into its retry figure. The tolerance edge
    still counts, and an unrecognised window proves nothing.
    """
    garbled = reset_candidate(1, report(limit("5 hour", 2 * 365 * 86_400_000, 1)), "m", NOW)
    assert garbled.state == "depleted" and garbled.reopens_at_ms is None
    edge = 5 * 3_600_000 + 60_000
    assert reset_candidate(1, report(limit("5 hour", edge, 1)), "m", NOW).reopens_at_ms == (
        NOW + edge
    )
    assert reset_candidate(1, report(limit("5 hour", edge + 1, 1)), "m", NOW).reopens_at_ms is None
    odd = reset_candidate(1, report(limit("lifetime", 600_000, 1)), "m", NOW)
    assert odd.state == "depleted" and odd.reopens_at_ms is None


def _parsed_verified_ultra(*, weekly_reset_ms: int, age_ms: int) -> UsageReport:
    """The live verified-Ultra payload (2026-10-30), run through the real parser.

    Synthetic values on the measured shape: an idle ``five_hour`` bucket
    (0.0, every other field null) and an idle ``session`` entry, beside a 100%
    weekly window. Parsing it here — rather than hand-building limits — is the
    point: the defect was the parser turning the idle bucket into a measured
    window with no reset, which made a stale report "unknown".
    """
    reset_iso = datetime.fromtimestamp(weekly_reset_ms / 1000, UTC).isoformat()
    payload = {
        "five_hour": {
            "utilization": 0.0,
            "resets_at": None,
            "limit_dollars": None,
            "used_dollars": None,
            "remaining_dollars": None,
            "locked_reason": None,
        },
        "seven_day": {"utilization": 100.0, "resets_at": reset_iso},
        "limits": [
            {
                "kind": "session",
                "group": "g",
                "percent": 0,
                "severity": "normal",
                "resets_at": None,
                "scope": None,
                "is_active": False,
            },
            {
                "kind": "weekly_all",
                "group": "g",
                "percent": 100,
                "severity": "critical",
                "resets_at": reset_iso,
                "scope": None,
                "is_active": True,
            },
        ],
    }

    async def fetch() -> UsageReport | None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_usage(client, "anthropic", access_token="tok")

    parsed = asyncio.run(fetch())
    assert parsed is not None
    parsed.fetched_at = NOW - age_ms
    return parsed


@pytest.mark.parametrize("age_ms", [60_000, 3 * 3_600_000])
def test_verified_ultra_weekly_exhaustion_binds_fresh_and_stale(age_ms):
    """2026-10-30: a verified-Ultra account at 100% weekly must stay depleted.

    The idle five-hour bucket used to parse as a 0% window with no reset; a
    stale report (3h old) with any reset-less measured window is "unknown", so
    failover could re-pick a weekly-exhausted account. With the bucket dropped
    the weekly window binds: depleted, no five-hour deadline, and (fresh) the
    reopening is the weekly reset.
    """
    weekly_reset = NOW + 2 * 86_400_000
    report = _parsed_verified_ultra(weekly_reset_ms=weekly_reset, age_ms=age_ms)
    assert [limit.id for limit in report.limits] == ["anthropic:7d"]
    candidate = reset_candidate(1, report, "claude-opus", NOW)
    assert candidate.state == "depleted"
    assert candidate.five_hour_ms is None
    fresh = age_ms <= 1_800_000
    assert candidate.reopens_at_ms == (weekly_reset if fresh else None)
