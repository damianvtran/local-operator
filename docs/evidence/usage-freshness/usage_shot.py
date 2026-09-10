"""Capture the `/usage` panel in the operator's reported post-`r` frame.

Drives the REAL ``OperatorApp`` (the lightweight ``_PanelHost`` in the test file
declares no ``CSS_PATH``, so a still taken from it shows none of the card's
padding, fill, or placement — see AGENTS.md "Visual validation"), and exports
through ``scripts.visual_capture.save_capture`` rather than ``save_screenshot``
so the frame is a native-cell measurement, not a Rich presentation.

The seed reproduces the reported defect exactly: pressing ``r`` during a
rate-limit storm zeroes every streak and re-probes all accounts in one burst.
Accounts that miss in that burst keep the previous round's seconds-old
``fetched_at`` and acquire ``consecutive_failures=1``. Under the OLD predicate
the failure streak alone fired the note, so the title read ``just now · 5
stale`` over rows reading ``last known just now`` — a number confirmed seconds
ago called stale, and the self-contradictory ``last known just now``. One row is
additionally latched (``usage_unavailable=True``) with sub-minute-old numbers to
show the ``usage unavailable — last known just now`` → ``usage unavailable``
change. The header age is computed through the app's own
``_usage_data_fetched_ms`` so the frame shows what the shipped code does, not
what the script decides.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        docs/evidence/usage-freshness/usage_shot.py out.svg [WxH]

Run from the worktree root.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())  # run from the repo root

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()  # BEFORE app imports: isolate HOME, config and caches

from local_operator.providers.usage import (  # noqa: E402
    UsageAmount,
    UsageLimit,
    UsageReport,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.usage_panel import (  # noqa: E402
    UsagePanel,
    build_usage_body,
)
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
    """The post-`r` burst: one healthy sibling plus five rows that missed.

    Four rows are fresh-but-missed (``consecutive_failures=1``, ``fetched_at``
    seconds old — exactly what ``_mark_account_failure`` leaves after a forced
    round 429s). One is latched (``usage_unavailable=True``) with sub-minute-old
    numbers. Under the old predicate all five render a note and the title counts
    ``5 stale``; under the new one the four missed rows are clean and the latched
    row says ``usage unavailable`` without the contradictory ``just now`` age.
    """
    fresh_ms = int(NOW_MS - 5_000)  # sibling confirmed 5s ago: the title's stamp
    missed_ms = int(NOW_MS - 10_000)  # rows confirmed 10s ago, then missed the burst

    sibling = UsageReport(
        provider="anthropic",
        identity="newest@x",
        fetched_at=fresh_ms,
        limits=[_percent("anthropic:5h:0", "5 hour", 12.0, 3.5)],
    )

    missed_identities = [
        ("damian@gominerva.com", 27.0),
        ("damian@radienthq.com", 41.0),
        ("damian@pergamonhq.com", 33.0),
        ("damianvtran@gmail.com", 8.0),
    ]
    missed = [
        UsageReport(
            provider="anthropic",
            identity=identity,
            fetched_at=missed_ms,
            limits=[_percent(f"anthropic:7d:{i}", "7 day", pct, 96.0)],
        )
        for i, (identity, pct) in enumerate(missed_identities)
    ]
    for report in missed:
        report.consecutive_failures = 1  # the forced-round miss; fetched_at kept

    # The latched row: usage_unavailable with sub-minute-old numbers.
    latched = UsageReport(
        provider="kimi",
        identity="cred:8",
        fetched_at=int(NOW_MS - 30_000),
        limits=[_percent("kimi:7d", "7 day", 64.0, 40.0)],
    )
    latched.usage_unavailable = True
    latched.consecutive_failures = 5

    return [sibling, *missed, latched]


async def main() -> None:
    out = sys.argv[1]
    width, _, height = (sys.argv[2] if len(sys.argv) > 2 else "100x34").partition("x")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(int(width), int(height))) as pilot:
        # Let the status band settle from `connecting…` to the model name so a
        # frame taken during that window does not differ from the settled one in
        # chrome that has nothing to do with the panel.
        for _ in range(30):
            await pilot.pause()
        panel = app.query_one(UsagePanel)
        reports = _reports()
        panel.set_clock(NOW_MS)
        panel.display = True
        # The app's own header computation — this is the line under test.
        header_ms = app._usage_data_fetched_ms(reports)
        panel.show_reports(reports, now_ms=header_ms)
        for _ in range(4):
            await pilot.pause()
        if len(sys.argv) > 3 and sys.argv[3] == "end":
            # Scroll to the bottom so the latched kimi block and its note are in
            # the frame; in a set this tall it sits below the fold otherwise.
            panel.action_scroll_end()
            for _ in range(4):
                await pilot.pause()

        rows = panel.render_lines_for_test()
        print(f"header now_ms   : {header_ms:.0f}")
        print(f"title           : {rows[0].strip()}")
        stale, expired = panel._flagged_account_counts()
        print(f"flagged counts  : stale={stale} expired={expired}")
        # Read the notes off the FULL body, not the windowed view.
        body = build_usage_body(reports, 72, NOW_MS, header_ms)
        for line in body.lines:
            if "last known" in line.plain or "unavailable" in line.plain:
                print(f"account note    : {line.plain.strip()}")
        print(f"virtual/actual  : {app.screen.virtual_size} / {app.screen.size}")
        save_capture(app, out)
    print(f"wrote {out}")


asyncio.run(main())
