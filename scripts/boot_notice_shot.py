"""Capture a system notice under the boot splash, at any terminal width.

Usage::

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/boot_notice_shot.py out.svg [WIDTHxHEIGHT] [STATE]

States:

``skew``       one long build-skew notice — the shape the operator reported.
``stack``      three notices of different lengths — the case that showed the
               per-row centring as a visible zigzag of four ragged left edges.
``short``      one short notice, the shape the block was first measured on.
``populated``  the same notice with the splash retired by a conversation
               block: the OFF-splash rendering, which this change must leave
               alone.

The boot notice is the one transcript block that used to carry its own text
alignment (``NoticeBlock._build`` centred every row on its own width), so a
frame is the only way to judge it: the column it lands in is a function of the
sentence's length, which no unit assertion about a single width can show. Pair
a ``stack`` frame with a ``skew`` frame to see both defects at once — the
zigzag between rows, and the block's own one-cell drift off the card.

Isolates HOME and the config dir before importing the app (``isolate_capture``)
so the frame never depends on the developer's own providers or MCP roster, and
so a capture cannot touch a live session.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()  # BEFORE the app imports: it reads HOME/config at import time.

from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402

#: The operator's reported notice, verbatim in shape: two stamps and a clause.
#: Long enough to wrap at every width the splash is used at, which is what
#: makes it the frame that shows a continuation row's alignment.
SKEW = (
    "this session is running 0.51.0@ad6db35 \u2192 0.51.5@ad6db35 \u2014 it will "
    "switch to the new version when it is next idle."
)
SHORT = "MCP cloudflare failed: needs authorization"
STACK = [
    SHORT,
    "MCP linear failed: connection refused after three attempts",
    "MCP notion failed: the server exited before it answered initialize",
]

STATES: dict[str, list[str]] = {
    "skew": [SKEW],
    "stack": STACK,
    "short": [SHORT],
    "populated": [SKEW],
}


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"
    state = sys.argv[3] if len(sys.argv) > 3 else "skew"
    width, height = (int(part) for part in size.split("x"))

    from tests.unit.tui.test_boot_layout import _make_app, _settle

    app = _make_app()
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        await _settle(pilot)
        if state == "populated":
            # Retire the splash FIRST, so the notice is appended to a real
            # conversation rather than to the empty state: this is the
            # full-width spine rendering, not the boot column.
            app._append_block(UserBlock("what changed in the last release?"))
            await _settle(pilot)
        for body in STATES[state]:
            app._system_notice(body, "error")
        await _settle(pilot)
        save_capture(app, out)
        print(f"state={state} size={width}x{height} -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
