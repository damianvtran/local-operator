"""Drive the REAL `r` path end to end and print what the header does.

Not a unit test: this presses the actual key on the actual app with a controller
that behaves like the operator's did — the healthy accounts advance their
``fetched_at`` on every fetch, the stuck kimi account keeps returning the SAME
report object with its 169-minute-old stamp (what
``ProviderController._mark_account_failure`` does when a forced re-probe misses
again). Prints the header age before and after each press.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())  # run from the repo root

from local_operator.providers.usage import (  # noqa: E402
    UsageAmount,
    UsageLimit,
    UsageReport,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.usage_panel import UsagePanel  # noqa: E402
from tests.unit.tui.test_app_pilot import (  # noqa: E402
    FakeProviderController,
    FakeSession,
    _factory,
)

NOW_MS = 1_788_400_000_000.0
MINUTE = 60_000.0


def _limit(limit_id: str, label: str, percent: float) -> UsageLimit:
    return UsageLimit(
        id=limit_id,
        label=label,
        amount=UsageAmount(
            used=percent,
            limit=100.0,
            remaining=100.0 - percent,
            used_fraction=percent / 100.0,
            unit="percent",
        ),
        window=label,
        shared=True,
    )


#: The stuck account, created ONCE and handed back by identity on every fetch —
#: that object identity is the bug's mechanism, not an approximation of it.
STUCK = UsageReport(
    provider="kimi",
    identity="cred:8",
    fetched_at=int(NOW_MS - 169 * MINUTE),
    limits=[_limit("kimi:7d", "7 day", 64.0)],
)
STUCK.consecutive_failures = 1


class _MixedController(FakeProviderController):
    """Healthy accounts refresh; the kimi account never advances its stamp."""

    def __init__(self) -> None:
        super().__init__()
        #: Simulated wall clock. The test advances it between presses; a healthy
        #: probe lands AT it, which is what a real refresh does.
        self.now = NOW_MS
        self.clock = NOW_MS - 1.8 * MINUTE

    def _set(self) -> list[UsageReport]:
        healthy = [
            UsageReport(
                provider="anthropic",
                identity=f"acct{i}@example.com",
                fetched_at=int(self.clock),
                limits=[_limit(f"anthropic:5h:{i}", "5 hour", 20.0 + i)],
            )
            for i in range(5)
        ]
        # `_mark_account_failure` returns the PREVIOUS object; its bumped streak
        # is the only thing that changes.
        STUCK.consecutive_failures += 1
        return healthy + [STUCK]

    def cached_usage_reports(self, provider=None):
        return self._set()

    async def fetch_usage(self, provider_ids=None, *, force_refresh: bool = False):
        self.usage_calls.append((provider_ids, force_refresh))
        # The healthy probes land at the current simulated instant.
        self.clock = self.now
        return self._set()


async def main() -> None:
    ctrl = _MixedController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("slash")
        for key in "usage":
            await pilot.press(key)
        await pilot.press("enter")
        for _ in range(8):
            await pilot.pause()
        panel = app.query_one(UsagePanel)
        panel.set_clock(NOW_MS)
        panel._repaint()
        await pilot.pause()

        def header() -> str:
            return panel.render_lines_for_test()[0].strip()

        def notes() -> list[str]:
            return [r.strip() for r in panel.render_lines_for_test() if "last known" in r]

        print(f"after open      : {header()!r}  fetched_ms={panel.fetched_ms:.0f}")
        for press in (1, 2, 3):
            ctrl.now = NOW_MS + press * 5 * MINUTE
            panel.set_clock(NOW_MS + press * 5 * MINUTE)
            await pilot.press("r")
            for _ in range(10):
                await pilot.pause()
            print(
                f"after r #{press}      : {header()!r}  "
                f"fetched_ms={panel.fetched_ms:.0f}  forced={ctrl.usage_calls[-1][1]}"
            )
        panel.action_scroll_end()
        for _ in range(4):
            await pilot.pause()
        print(f"stuck account   : {notes()}")
        print(f"fetch calls     : {ctrl.usage_calls}")
        print(f"stuck fetched_at: {(NOW_MS - STUCK.fetched_at) / MINUTE:.0f} min old (unchanged)")


asyncio.run(main())
