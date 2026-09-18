"""Does the `/links` card answer a pointer? Hover frame + click frame.

    LO_ROOT=<tree> .venv/bin/python /tmp/lo-design-1193/hover.py OUTDIR
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ.get("LO_ROOT", "/Users/damian/lo-wt/open-links-r3")).resolve()
sys.path.insert(0, str(ROOT))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from textual import events  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.link_picker import LinkPickerScreen  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

ANSWER = "Sources: https://one.test/a, https://two.test/b and https://three.test/c."


class _Opener:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def __call__(self, url: str) -> bool:
        self.urls.append(url)
        return True


def _mouse(body, kind, *, screen_x, screen_y, button=0, delta_x=0, delta_y=0):
    return kind(
        widget=body,
        x=screen_x - body.region.x,
        y=screen_y - body.region.y,
        delta_x=delta_x,
        delta_y=delta_y,
        button=button,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=screen_x,
        screen_y=screen_y,
    )


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    opener = _Opener()
    app = OperatorApp(lambda: _factory(FakeSession()))
    report: dict = {"root": str(ROOT), "link_picker": None}
    with patch("local_operator.mcp.auth.open_browser_quietly", opener):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(60):
                await pilot.pause()
                if app._session is not None:
                    break
            app._append_block(UserBlock("collect them", fold_width=100))
            block = AssistantBlock()
            block.update_text(ANSWER)
            block.finalize_text()
            app._append_block(block)
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.text = "/links"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LinkPickerScreen)
            body = screen._body
            region = body.region
            report["body_region"] = [region.x, region.y, region.width, region.height]
            report["rows"] = screen.render_lines_for_test()
            report["pointer_before"] = str(screen.styles.pointer)
            report["hovered_attr_before"] = hasattr(screen, "_hovered")

            rest = outdir / "hover-resting.svg"
            save_capture(app, str(rest))

            # A real MouseMove over the SECOND row, through the app's own
            # dispatch path (posted on the screen, so it bubbles like the real
            # driver's event does).
            row2_y = region.y + 2 + 1  # header rows 2, then row index 1
            screen.post_message(
                _mouse(body, events.MouseMove, screen_x=region.x + 4, screen_y=row2_y)
            )
            await pilot.pause()
            await pilot.pause()
            moved = outdir / "hover-over-row2.svg"
            save_capture(app, str(moved))
            report["pointer_after_move"] = str(screen.styles.pointer)
            report["hovered_attr_after"] = hasattr(screen, "_hovered")
            report["frame_changed_on_hover"] = hashlib.sha256(
                rest.read_bytes()
            ).hexdigest() != hashlib.sha256(moved.read_bytes()).hexdigest()

            # Now the SAME row position, clicked — the route the card does keep.
            await pilot.click(offset=(region.x + 4, row2_y))
            for _ in range(6):
                await pilot.pause()
            after = outdir / "click-row2-after.svg"
            save_capture(app, str(after))
            report["clicked_row2_opened"] = list(opener.urls)
            report["screen_after_click"] = type(app.screen).__name__
            report["clicked_row2_text"] = screen.render_lines_for_test()[2]
    print(json.dumps(report, indent=1))


asyncio.run(main())
