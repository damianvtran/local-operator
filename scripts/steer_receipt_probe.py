"""Measure the settle reflow of a DEFERRED steer receipt (issue #160, D1/D2).

Drives the real ``OperatorApp`` (so the stylesheet and the transcript's real
gap/anchor accounting apply), puts one steer row into the DEFERRED state, and
reports the geometry either side of the delivery that settles it:

  * the receipt block's row count,
  * the transcript's ``virtual_size`` and ``scroll_offset``,
  * how many viewport rows change between the two frames.

Both scroll states are measured because they fail differently: pinned to the
bottom the offset auto-corrects (only the tail moves), while scrolled up the
offset stays put and the same offset shows different text.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/steer_receipt_probe.py [cols ...]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import SteeringDelivered, TurnEnded  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    NoticeBlock,
    TranscriptView,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


class _Streaming(FakeSession):
    """Mid-turn, so a submit is STEERED rather than prompted."""

    @property
    def is_streaming(self) -> bool:
        return True


async def _submit(pilot: Any, app: OperatorApp, text: str) -> None:
    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _rows(app: OperatorApp) -> list[str]:
    return [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]


def _receipt(app: OperatorApp) -> NoticeBlock:
    held = app._deferred_steer_notices or app._queued_steer_notices
    return held[0]


def _geometry(app: OperatorApp, block: NoticeBlock) -> dict[str, Any]:
    view = app.query_one(TranscriptView)
    return {
        "text": block._text,
        "h": block.virtual_region.height,
        "virtual_size": view.virtual_size.height,
        "scroll_offset": view.scroll_offset.y,
        "max_scroll_y": view.max_scroll_y,
    }


async def _measure(cols: int, *, scrolled_up: bool, shot: str | None = None) -> None:
    app = OperatorApp(lambda: _factory(_Streaming()))
    async with app.run_test(size=(cols, 24)) as pilot:
        await pilot.pause()
        # Enough history that scrolling up has somewhere to go. Plain rows,
        # not steers: only ONE receipt may be in flight or the settle would
        # move several rows at once and overstate the magnitude.
        for index in range(30):
            app._append_block(NoticeBlock(f"history row {index}", "info"))
        await pilot.pause()
        await _submit(pilot, app, "and use the staging credentials")
        for index in range(30, 44):
            app._append_block(NoticeBlock(f"history row {index}", "info"))
        await pilot.pause()
        app.post_message(TurnEnded(True, None))
        await pilot.pause()
        await pilot.pause()

        view = app.query_one(TranscriptView)
        if scrolled_up:
            view.scroll_to(y=25, animate=False, immediate=True)
            await pilot.pause()
            await pilot.pause()

        block = _receipt(app)
        before = _geometry(app, block)
        before_rows = _rows(app)
        if shot:
            app.save_screenshot(f"{shot}-before.svg")

        app.post_message(SteeringDelivered(1))
        await pilot.pause()
        await pilot.pause()

        after = _geometry(app, block)
        after_rows = _rows(app)
        if shot:
            app.save_screenshot(f"{shot}-after.svg")

        changed = sum(1 for a, b in zip(before_rows, after_rows) if a != b)
        state = "scrolled up" if scrolled_up else "pinned to bottom"
        print(f"--- {cols} cols, {state}")
        for label, snap in (("before", before), ("after", after)):
            print(
                f"  {label:6} h={snap['h']} "
                f"virtual_size={snap['virtual_size']} "
                f"scroll_offset={snap['scroll_offset']} "
                f"max_scroll_y={snap['max_scroll_y']} "
                f"{snap['text']!r}"
            )
        print(f"  viewport rows changed: {changed}/{len(before_rows)}")


async def main() -> None:
    widths = [int(arg) for arg in sys.argv[1:] if arg.isdigit()] or [52, 53]
    shot_base = next((arg for arg in sys.argv[1:] if not arg.isdigit()), None)
    for cols in widths:
        for scrolled_up in (False, True):
            shot = None
            if shot_base:
                suffix = "scrolled" if scrolled_up else "pinned"
                shot = f"{shot_base}-{cols}-{suffix}"
            await _measure(cols, scrolled_up=scrolled_up, shot=shot)


asyncio.run(main())
