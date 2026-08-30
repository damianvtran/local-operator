"""Find a cross-turn settled string that does not reflow the row (#160 D1/D2/D5).

The round-1 candidate was chosen by CHARACTER COUNT and was wrong: 27 characters
is identical to the current string and moves no wrap point. Rendered height is
the only thing that matters, and it is a step function of the wrap points, so
candidates are measured in a mounted block at every width instead of counted.

A candidate PASSES when its rendered height equals ``DEFERRED_STEER_NOTICE``'s
at every width in the range: that is exactly the condition under which the
cross-turn settle rewrites the row without moving anything below it.

Coupled to ``tests.unit.tui.test_app_pilot`` for its ``FakeSession``/``_factory``
(review round 1, F5): nothing runs these in CI, so a rename there breaks them
silently -- which is exactly when someone editing the receipt copy needs them.
If the import fails, that pair is what moved.

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
    # The shipped string, and the alternatives design review round 1 (D1/D2)
    # measured against it. `your next message` is FALSE on every turn the user
    # did not start -- an idle wake, a peer `lop send`, or a background job
    # result each open their own turn and drain the queue -- so the row named a
    # message that need not exist. `that next message` drops the
    # authorship claim without asserting a position the reader must verify.
    "sent — it rode along with that next message",  # shipped
    # Rejected by measurement, and the reason is instructive: the binding
    # constraint is WORD SHAPE, not length. These are 43 characters like the
    # shipped string, but their tails break differently and so reflow in the
    # narrow band. `the message below` was design round 1's suggestion.
    "sent — it rode along with the message below",
    "sent — it went along with the message below",
    "sent — it rode along with your next message",
    "sent — it rode along with your last message",
    "sent — it rode along with the next message",
    "sent — it went out with the message below it",
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
