"""Capture the `/links` picker over a populated transcript, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/link_shot.py OUT.svg [COLSxROWS] [SHAPE]

``SHAPE`` selects the conversation the card is opened over, because the three
things that can look wrong about this card are each a different shape:

    default  (default)  markdown links, bare URLs and an autolink together —
                        the ordinary frame, and the one that shows the rows sit
                        under the title with the footer below them
    long                one very long URL, for the row TRUNCATION. The row is cut
                        and the whole URL is what opens, so the frame is the only
                        place the cut is visible
    many                more links than the card shows at once, for the window
                        and its `showing 3–14 of 24` counter row
    parens              one markdown link whose URL contains BALANCED parentheses,
                        the shape review round 1 found listed twice with the
                        truncated target under the cursor

Driven through the real command — typed into the editor and submitted — rather
than by pushing the screen directly: the extraction walk and the app's own
hand-off are as much a part of what is being looked at as the card is.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.link_picker import LinkPickerScreen  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

MIXED_ANSWER = """The migration is in `core-svc`, and the rollback is beside it.

[the pull request](https://gitlab.com/minervaai/core-svc/-/merge_requests/412)
is the place to start, and the run that produced it is
https://gitlab.com/minervaai/core-svc/-/pipelines/88213 — the logs there are
what the numbers below come from.

Read <https://docs.gitlab.com/ee/ci/yaml/> before changing the pipeline.
"""


def _answer(text: str) -> AssistantBlock:
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def _seed(app: OperatorApp, shape: str) -> None:
    if shape == "long":
        app._append_block(UserBlock("where is that report?", fold_width=100))
        app._append_block(
            _answer(
                "Here it is: https://example.com/reports/2026/09/"
                + "very-long-segment/" * 4
                + "index.html\n\nand the summary is https://example.com/short"
            )
        )
        return

    if shape == "parens":
        app._append_block(UserBlock("which one is the Foo case?", fold_width=100))
        app._append_block(
            _answer(
                "See [Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar)) for the case,"
                " and **https://a.test/bold** for the emphasis form."
            )
        )
        return

    if shape == "many":
        app._append_block(UserBlock("collect the links for the review", fold_width=100))
        for index in range(1, 25):
            app._append_block(_answer(f"Source {index}: https://example.test/source/{index}"))
        return

    # One prompt and one answer, so the before-frame (the same conversation on
    # `main`, where `/links` is an unknown command) differs from this one by the
    # card and nothing else.
    app._append_block(UserBlock("how do I roll this out?", fold_width=100))
    app._append_block(_answer(MIXED_ANSWER))


async def _run_links(pilot, app: OperatorApp) -> LinkPickerScreen | None:
    """Type ``/links`` into the real editor and submit it."""
    editor = app.query_one(Editor)
    editor.text = "/links"
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()
    screen = app.screen
    return screen if isinstance(screen, LinkPickerScreen) else None


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    shape = sys.argv[3] if len(sys.argv) > 3 else "default"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        _seed(app, shape)
        await pilot.pause()
        screen = await _run_links(pilot, app)
        save_capture(app, out)

        # The numbers behind the pixels: a still shows the symptom, these show
        # the cause. A scrollable Screen is always a bug in this app, and the
        # card's own box is what the row budget and the truncation read from.
        # `_body` and its parent are typed optional; a capture script reads
        # them for the printed digest rather than for the frame, so a missing
        # one is reported as `None` instead of raising.
        body = getattr(screen, "_body", None)
        card = getattr(body, "parent", None)
        box = getattr(card, "region", None)
        print(
            f"{Path(out).name}: size={tuple(app.screen.size)} "
            f"virtual={tuple(app.screen.virtual_size)} "
            f"vscroll={app.screen.show_vertical_scrollbar} "
            f"screen={type(app.screen).__name__} "
            f"drawable={None if screen is None else screen.is_drawable()} "
            f"budget={None if screen is None else screen._row_budget()} "
            f"card_w={None if screen is None else screen._card_width()} "
            f"card_box={None if box is None else (box.width, box.height)} "
            f"sel={None if screen is None else screen._selected}"
        )


asyncio.run(main())
