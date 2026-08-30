"""Find a cross-turn settled string that does not reflow the row (#160 D1/D2/D5).

The round-1 candidate was chosen by CHARACTER COUNT and was wrong: 27 characters
is identical to the current string and moves no wrap point. Rendered height is
the only thing that matters, and it is a step function of the wrap points, so
candidates are measured in a mounted block at every width instead of counted.

A candidate PASSES when its rendered height equals ``DEFERRED_STEER_NOTICE``'s
at every width in the range: that is exactly the condition under which the
cross-turn settle rewrites the row without moving anything below it.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/steer_receipt_candidates.py [lo] [hi]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.tui.app import DEFERRED_STEER_NOTICE, OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

CANDIDATES = [
    "sent with your message — the agent has it now",
    "sent — the agent has it with your message",
    "sent — it went with the message you just sent",
    "sent — it rode along with your next message",
    "sent with your message — the agent has it",
    "sent — the agent has it, with your message",
    "sent with your last message — the agent has it",
    "sent — the agent has it now",  # the current shared string, as control
]


async def _heights(cols: int, texts: list[str]) -> list[int]:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(cols, 24)) as pilot:
        await pilot.pause()
        blocks = [NoticeBlock(text, "note") for text in texts]
        for block in blocks:
            app._append_block(block)
        await pilot.pause()
        await pilot.pause()
        return [block.virtual_region.height for block in blocks]


async def main() -> None:
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    texts = [DEFERRED_STEER_NOTICE, *CANDIDATES]
    mismatches: dict[str, list[int]] = {c: [] for c in CANDIDATES}
    for cols in range(lo, hi + 1):
        heights = await _heights(cols, texts)
        deferred, rest = heights[0], heights[1:]
        for candidate, height in zip(CANDIDATES, rest):
            if height != deferred:
                mismatches[candidate].append(cols)
    for candidate in CANDIDATES:
        bad = mismatches[candidate]
        verdict = "MATCHES deferred at every width" if not bad else f"differs at {bad}"
        print(f"len={len(candidate):3} {candidate!r}\n      {verdict}\n")


asyncio.run(main())
