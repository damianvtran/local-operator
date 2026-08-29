"""Consequence of D6: all three notices are now TOAST_FAILURE_MS, which
`Toast` reads as ACTIONABLE. Actionable cards hold the single slot against
every courtesy notice for their full duration.

The vague variant names no remedy, so it now blocks a copy receipt for 10 s on
the strength of a sentence the user can do nothing with. Measure it, and
capture what the splash card at 30 cols now covers.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from textual import events  # noqa: E402

from local_operator.clipboard import ClipboardContents  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets import editor as editor_module  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.toast import (  # noqa: E402
    TOAST_DEFAULT_MS,
    TOAST_FAILURE_MS,
    Toast,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

OUT = Path(sys.argv[1])
F = {"TOAST_DEFAULT_MS": TOAST_DEFAULT_MS, "TOAST_FAILURE_MS": TOAST_FAILURE_MS}
editor_module.read_clipboard = lambda *a, **k: ClipboardContents()


async def boot(pilot, app):
    await pilot.pause()
    for _ in range(400):
        if app._session is not None:
            break
        await pilot.pause()
    ed = app.query_one(Editor)
    ed.focus()
    await pilot.pause()
    return ed


async def blocking():
    """Empty paste raises the vague notice; the user then copies a selection.
    Does the receipt appear, or defer behind a card naming no action?"""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await boot(pilot, app)
        toast = app.query_one(Toast)
        app.post_message(events.Paste(""))
        for _ in range(80):
            await pilot.pause()
        notice = {"showing": toast.message, "actionable": toast._actionable}
        # a copy receipt arrives while it is up
        app._put_on_clipboard("hello world", object())
        await pilot.pause()
        await pilot.pause()
        F["blocking"] = {
            "paste_notice": notice,
            "after_copy": {
                "showing": toast.message,
                "deferred": None if toast._deferred is None else str(toast._deferred[0]),
                "receipt_visible": toast.message.startswith("copied"),
            },
        }
        app.save_screenshot(str(OUT / "new9-vague-notice-blocks-receipt.svg"))


async def narrow_overlap():
    """What the card covers at 30 and 40 columns, per variant."""
    rows = {}
    for w, h in [(30, 18), (40, 20), (60, 24)]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(w, h)) as pilot:
            await boot(pilot, app)
            toast = app.query_one(Toast)
            app.post_message(events.Paste(""))
            for _ in range(80):
                await pilot.pause()
            r = toast.region
            rows[f"{w}x{h}"] = {
                "card_rows": r.height,
                "card_region": [r.x, r.y, r.width, r.height],
                "screen": list(app.screen.size),
                "pct_of_height": round(100 * r.height / app.screen.size.height),
                "virtual_vs_size": [
                    list(app.screen.virtual_size),
                    list(app.screen.size),
                ],
                "vscrollbar": bool(app.screen.show_vertical_scrollbar),
            }
    F["narrow"] = rows


async def main():
    await blocking()
    await narrow_overlap()
    print(json.dumps(F, indent=2))
    (OUT / "actionable.json").write_text(json.dumps(F, indent=2))


asyncio.run(main())
