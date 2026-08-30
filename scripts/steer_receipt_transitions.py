"""Every height change a steer receipt makes over its life, per width.

A receipt row is restated at most twice: ``queued`` -> ``deferred`` when the
turn ends without draining, and either of those -> ``sent`` when the engine
takes the message. Each restate that changes the row count reflows everything
below it. This tabulates all three transitions so the fix can be judged against
the whole set rather than the one transition issue #160 names.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/steer_receipt_transitions.py [lo] [hi]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.tui.app import (  # noqa: E402
    DEFERRED_SENT_STEER_NOTICE,
    DEFERRED_STEER_NOTICE,
    QUEUED_STEER_NOTICE,
    SENT_STEER_NOTICE,
    OperatorApp,
)
from local_operator.tui.widgets.transcript import NoticeBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def _heights(cols: int) -> dict[str, int]:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(cols, 24)) as pilot:
        await pilot.pause()
        blocks = {
            "queued": NoticeBlock(QUEUED_STEER_NOTICE, "note"),
            "sent": NoticeBlock(SENT_STEER_NOTICE, "success"),
            "deferred": NoticeBlock(DEFERRED_STEER_NOTICE, "note"),
            "deferred_sent": NoticeBlock(DEFERRED_SENT_STEER_NOTICE, "success"),
        }
        for block in blocks.values():
            app._append_block(block)
        await pilot.pause()
        await pilot.pause()
        return {name: block.virtual_region.height for name, block in blocks.items()}


async def main() -> None:
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    print("cols  q  s  d ds   q->s (same turn)  q->d (turn end)  d->ds (cross turn)")
    bands: dict[str, list[int]] = {"q->s": [], "q->d": [], "d->ds": []}
    for cols in range(lo, hi + 1):
        h = await _heights(cols)
        q, s, d, ds = h["queued"], h["sent"], h["deferred"], h["deferred_sent"]
        cells = []
        for key, a, b in (("q->s", q, s), ("q->d", q, d), ("d->ds", d, ds)):
            if a == b:
                cells.append(" " * 16)
                continue
            bands[key].append(cols)
            verb = "shrink" if b < a else "GROW  "
            cells.append(f"{verb} {a}->{b}      "[:16])
        print(f"{cols:4}  {q}  {s}  {d}  {ds}   {cells[0]} {cells[1]} {cells[2]}")
    print()
    for key, cols_list in bands.items():
        print(f"{key}: reflows at {cols_list or 'no width'}")


asyncio.run(main())
