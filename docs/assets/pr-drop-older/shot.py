"""Capture the status band over a bounded resume, for visual validation.

Run from the repo root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        docs/assets/pr-drop-older/shot.py OUT.svg [COLSxROWS] [--plain]

Seeds a 150-message conversation (50 turns) so the `/resume` replay bounds the
transcript: the state where the deferred-history segment used to paint
`▾ N older` at rest. `--plain` seeds an empty session instead — the ordinary
band that must stay byte-for-byte identical.

Also writes OUT.json beside the SVG with the band's own plain text and the
screen geometry numbers AGENTS.md asks for (virtual vs actual size, scrollbar).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import TranscriptView  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_resume_render import _history  # noqa: E402


async def main() -> None:
    out = Path(sys.argv[1])
    size = (100, 30)
    plain = "--plain" in sys.argv
    positional = [a for a in sys.argv[2:] if not a.startswith("--")]
    if positional:
        cols, rows = positional[0].split("x")
        size = (int(cols), int(rows))

    session = FakeSession()
    if not plain:
        session._history = _history(50)  # 150 messages > RESUME_RENDER_MESSAGES
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        # Wait for the resume replay worker the way the paging suite does.
        for _ in range(50):
            await pilot.pause()
            if len(app.query_one(TranscriptView).blocks()) >= 1:
                break
        await pilot.pause()
        await pilot.pause()
        probe = {
            "size": list(app.screen.size),
            "virtual_size": list(app.screen.virtual_size),
            "show_vertical_scrollbar": app.screen.show_vertical_scrollbar,
            "resume_pending_head": len(app._resume_pending_head),
            "band_text": app._status.render_text(size[0]).plain,
        }
        app.save_screenshot(str(out))
        out.with_suffix(".json").write_text(json.dumps(probe, indent=2) + "\n")
        print(json.dumps(probe))


asyncio.run(main())
