"""Measure what the ``/analytics`` session table costs to paint and to move in.

Usage: python scripts/analytics_collapse_probe.py [LEDGER] [WIDTHxHEIGHT]

Written for the collapsible-session-rows change and kept because the numbers it
prints are the only ones that can answer "is the screen still slow?". The user's
report was "my screen is freezing on /analytics" — a symptom that a unit test
cannot see and that a single wall-clock sample mis-attributes, because three
different costs hide inside it:

* the STORE READ, which already runs on a worker thread (``_open_analytics_worker``)
  and so delays the screen appearing without freezing anything;
* the FIRST PAINT, which is on the event loop and is dominated by how many rows
  ``build_report`` composes into the one body ``Static``;
* every subsequent REBUILD — and the one that actually froze the terminal is
  ``on_resize``, because a drag emits a storm of resize events and each one
  used to re-render the whole report.

So each is timed separately, against a real ledger rather than a fixture: the
shape that hurts (hundreds of roots, a handful of them with dozens of subagent
children) does not exist in any test fixture and cannot be conjured by scaling
one up. Point it at a COPY of a real ledger; opening a store runs migrations and
may build an index, so this must never be aimed at a live ``analytics.db``.

The pilot runs headless with every ``CMUX_*`` identifier cleared and its own
config dir (``scripts.probe_isolation``): a headless app that inherits
``CMUX_WORKSPACE_ID`` renames the operator's real cmux workspaces.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import (see module
# docstring): inherited CMUX ids have previously been acted on by a headless app.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import time  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def _timed(label: str, body) -> float:
    """Run ``body`` and print how long it took, in the pilot's own frame."""
    start = time.perf_counter()
    await body()
    elapsed = time.perf_counter() - start
    print(f"{label:<34} {elapsed:>7.3f}s")
    return elapsed


async def main() -> None:
    ledger = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/an-probe.db")
    size = sys.argv[2] if len(sys.argv) > 2 else "110x40"
    width, height = (int(part) for part in size.split("x"))

    store = AnalyticsStore(ledger)
    start = time.perf_counter()
    aggregate = store.aggregate()
    daily = store.daily_series(30)
    monthly = store.monthly_series(12)
    window_totals = store.series_totals(daily_days=30)
    read = time.perf_counter() - start
    print(
        f"{'store read (worker thread)':<34} {read:>7.3f}s  "
        f"sessions={len(aggregate.by_session)}"
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        screen = AnalyticsScreen(
            aggregate, daily=daily, monthly=monthly, window_totals=window_totals
        )

        async def _push() -> None:
            await app.push_screen(screen)
            await pilot.pause()

        await _timed("push_screen + first paint", _push)

        # The freeze the user reported. A terminal resize emits a burst of these,
        # so the per-event cost is what decides whether a drag is smooth or wedged.
        for index, term_width in enumerate((width - 2, width - 4, width - 4)):

            async def _resize(w=term_width) -> None:
                await pilot.resize_terminal(w, height)
                await pilot.pause()

            # The third resize repeats the second width on purpose: a rebuild that
            # is skipped when the card width band is unchanged shows up here and
            # nowhere else.
            await _timed(f"resize #{index + 1} -> {term_width}", _resize)

        async def _toggle() -> None:
            await pilot.press("t")
            await pilot.pause()

        await _timed("metric toggle (full rebuild)", _toggle)

        async def _arrow() -> None:
            await pilot.press("down")
            await pilot.pause()

        await _timed("one down press", _arrow)

        # Size of the thing being painted, recomposed off the timed path (the
        # widget does not expose what it was handed). Lines and spans are the two
        # quantities the paint cost tracks: Rich walks every span of every line.
        report = screen._report_lines()
        lines = sum(block.plain.count("\n") + 1 for block in report)
        spans = sum(len(block.spans) for block in report)
        print(f"{'body lines':<34} {lines:>8}")
        print(f"{'style spans':<34} {spans:>8}")

        await pilot.press("escape")
        await pilot.pause()


if __name__ == "__main__":
    asyncio.run(main())
