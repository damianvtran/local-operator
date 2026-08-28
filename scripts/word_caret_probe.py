"""Feed raw terminal bytes into the real composer and report what happened.

Used to reproduce GitHub issue #370's escape-prefixed defect and to demonstrate
the fix: the byte sequences here are the literal ones the named emulators write
for option+arrow, so a claim about "terminal X" is checked against X's actual
encoding rather than against a key NAME someone assumed it produces.

Run with the worktree pinned on PYTHONPATH so it exercises THIS source:

    cd <worktree>
    PYTHONPATH=$PWD env -u NO_COLOR TERM=xterm-256color \
        ~/local-operator/.venv/bin/python scripts/word_caret_probe.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual import events  # noqa: E402
from textual._xterm_parser import XTermParser  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor, StopRequested  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: One row per terminal encoding of option+arrow. Same table as the test.
ENCODINGS: list[tuple[str, str, str]] = [
    ("\x1b[1;3D", "CSI-modifier alt", "Ghostty / kitty / WezTerm / iTerm2 CSI"),
    ("\x1bb", "readline meta-b", "iTerm2 default preset / Terminal.app meta"),
    ("\x1b\x1b[D", "Esc-prefixed", "iTerm2 'Esc+' / Terminal.app 'Esc+'"),
]

SAMPLE = "alpha beta gamma delta"


async def _feed_bytes(app: OperatorApp, raw: str) -> None:
    """Parse ``raw`` exactly as the driver would and inject the key events."""
    parser = XTermParser()
    parsed = list(parser.feed(raw)) + list(parser.feed(""))
    driver = app._driver
    assert driver is not None
    # Sent WITHOUT yielding between events, as the real driver does: one parse
    # pass emits the chord's `escape` and `left` together and posts both before
    # the loop is pumped.
    for event in parsed:
        if isinstance(event, events.Key):
            event.set_sender(app)
            driver.send_message(event)
    await asyncio.sleep(0)


async def main() -> None:
    for raw, label, terminals in ENCODINGS:
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        stops: list[StopRequested] = []
        async with app.run_test(size=(100, 24)) as pilot:
            for _ in range(200):
                if app._session is not None:
                    break
                await pilot.pause()
                await asyncio.sleep(0.01)
            editor = app.query_one(Editor)
            editor.focus()
            editor.text = SAMPLE
            editor.move_cursor((0, len(SAMPLE)))
            await pilot.pause()

            original = app.post_message

            def _spy(message, _orig=original, _stops=stops):  # type: ignore[no-untyped-def]
                if isinstance(message, StopRequested):
                    _stops.append(message)
                return _orig(message)

            app.post_message = _spy  # type: ignore[method-assign]

            await _feed_bytes(app, raw)
            # Long enough for any deferral window to expire either way.
            for _ in range(30):
                await pilot.pause()
                await asyncio.sleep(0.01)

            column = editor.cursor_location[1]
            print(
                f"{label:18} {raw!r:14} caret_col={column:2d} "
                f"stop_requested={bool(stops)}  [{terminals}]"
            )


if __name__ == "__main__":
    asyncio.run(main())
