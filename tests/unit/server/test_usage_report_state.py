"""``usage_report_state`` precedence, as the desktop quota catalogue sees it.

The desktop route (``desktop_catalogues.py``) and the paired phone label
accounts with this ONE rule; the phone side is pinned in
``tests/unit/mobile/test_paid_usage.py``. This table pins the order the
desktop relies on, so a reordering in the shared helper fails here too
(review round 1 on PR #3322, R4).
"""

import pytest

from local_operator.providers.usage import (
    UsageAmount,
    UsageLimit,
    UsageReport,
    usage_report_state,
)

_LIMIT = [UsageLimit("5h", "5 hour", UsageAmount(used_fraction=0.2, unit="percent"), "5h")]


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (UsageReport(provider="anthropic", limits=_LIMIT), "available"),
        (UsageReport(provider="anthropic", limits=_LIMIT, consecutive_failures=2), "partial"),
        # unavailable outranks partial; no limits at all is unavailable too.
        (
            UsageReport(
                provider="anthropic",
                limits=_LIMIT,
                usage_unavailable=True,
                consecutive_failures=2,
            ),
            "unavailable",
        ),
        (UsageReport(provider="anthropic"), "unavailable"),
        # reauth outranks everything: only a new login fixes it.
        (
            UsageReport(
                provider="anthropic",
                limits=_LIMIT,
                usage_unavailable=True,
                consecutive_failures=2,
                credential_invalid=True,
            ),
            "reauth_required",
        ),
    ],
    ids=["available", "partial", "unavailable-over-partial", "no-limits", "reauth-wins"],
)
def test_usage_report_state_precedence(report, expected):
    assert usage_report_state(report) == expected
