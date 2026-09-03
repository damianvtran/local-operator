"""Capture the `/usage` panel in the operator's reported state.

Drives the REAL ``OperatorApp`` (the lightweight ``_PanelHost`` in the test file
declares no ``CSS_PATH``, so a still taken from it shows none of the card's
padding, fill, or placement — see AGENTS.md "Visual validation").

State mirrors the bug report exactly: five healthy Anthropic accounts refreshed
minutes ago plus one `kimi cred:8` account in per-account backoff whose
``fetched_at`` is 169 minutes old. The header age is computed through the app's
own ``_usage_data_fetched_ms`` so the frame shows what the shipped code does,
not what the script decides.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        docs/evidence/usage-freshness/usage_shot.py out.svg [WxH] [end] [target]

Run from the worktree root.

A third argument ``end`` scrolls the body to the bottom first, which is where
the stuck kimi block and its ``last known`` note live in a set this tall. A
fourth sets the panel's target, i.e. captures the scoped ``/usage <provider>``
path — the one whose title has to give up cells before the stale count does.
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
from local_operator.tui.widgets.usage_panel import UsagePanel  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

# A fixed clock so before/after frames differ only by the code under test.
NOW_MS = 1_788_400_000_000.0
MINUTE = 60_000.0


def _percent(limit_id: str, label: str, percent: float, resets_in_h: float) -> UsageLimit:
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
        resets_at_ms=int(NOW_MS + resets_in_h * 3600 * 1000),
    )


def _reports() -> list[UsageReport]:
    """The operator's set: five fresh Anthropic logins plus one stuck Kimi."""
    fresh_ms = int(NOW_MS - 1.8 * MINUTE)
    identities = [
        ("damian@gominerva.com", 27.0, 28.0),
        ("damian@radienthq.com", 12.0, 45.0),
        ("damian@pergamonhq.com", 33.0, 21.0),
        ("damianvtran@gmail.com", 8.0, 61.0),
        ("damian@local-operator.com", 55.0, 39.0),
    ]
    reports = [
        UsageReport(
            provider="anthropic",
            identity=identity,
            fetched_at=fresh_ms,
            limits=[
                _percent(f"anthropic:5h:{index}", "5 hour", five_hour, 3.5),
                _percent(f"anthropic:7d:{index}", "7 day", seven_day, 96.0),
            ],
        )
        for index, (identity, five_hour, seven_day) in enumerate(identities)
    ]
    # The stuck account: one consecutive miss, still inside its backoff, so the
    # merge serves last-good and its `fetched_at` never advances.
    kimi = UsageReport(
        provider="kimi",
        identity="cred:8",
        fetched_at=int(NOW_MS - 169 * MINUTE),
        limits=[_percent("kimi:7d", "7 day", 64.0, 40.0)],
    )
    kimi.consecutive_failures = 1
    kimi.next_probe_at_ms = int(NOW_MS + 10_000)
    reports.append(kimi)
    return reports


async def main() -> None:
    out = sys.argv[1]
    width, _, height = (sys.argv[2] if len(sys.argv) > 2 else "100x30").partition("x")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(int(width), int(height))) as pilot:
        # Let the status band finish adopting the session before capturing.
        # It settles from `connecting…` to the model name a few ticks after
        # boot, and a frame taken during that window differs from the settled
        # one in chrome that has nothing to do with the panel — which reads as
        # an unstable capture when comparing before/after.
        for _ in range(30):
            await pilot.pause()
        panel = app.query_one(UsagePanel)
        reports = _reports()
        # Pin the panel's clock so the rendered ages are the ones the operator
        # saw rather than "now minus a fixture stamp from 2026".
        panel.set_clock(NOW_MS)
        panel.display = True
        if len(sys.argv) > 4:
            panel._target = sys.argv[4]
        # The app's own header computation — this is the line under test.
        header_ms = app._usage_data_fetched_ms(reports)
        panel.show_reports(reports, now_ms=header_ms)
        for _ in range(4):
            await pilot.pause()
        if len(sys.argv) > 3 and sys.argv[3] == "end":
            panel.action_scroll_end()
            for _ in range(4):
                await pilot.pause()
        rows = panel.render_lines_for_test()
        print(f"header now_ms   : {header_ms:.0f}")
        print(f"title age shown : {rows[0].strip()}")
        newest = max(r.fetched_at for r in reports)
        oldest = min(r.fetched_at for r in reports)
        print(f"newest report   : {(NOW_MS - newest) / MINUTE:.1f} min old")
        print(f"stalest report  : {(NOW_MS - oldest) / MINUTE:.1f} min old")
        # Read the note off the FULL body, not the windowed view: in a set this
        # tall the stuck block can be scrolled out of the frame while still
        # being the thing under test.
        from local_operator.tui.widgets.usage_panel import build_usage_body

        body = build_usage_body(reports, 72, NOW_MS)
        for line in body.lines:
            if "last known" in line.plain or "unavailable" in line.plain:
                print(f"account note    : {line.plain.strip()}")
        print(f"virtual/actual  : {app.screen.virtual_size} / {app.screen.size}")
        print(f"screen scrollbar: {app.screen.show_vertical_scrollbar}")
        app.save_screenshot(out)
    print(f"wrote {out}")


asyncio.run(main())
