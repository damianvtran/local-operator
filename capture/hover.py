"""D3 for design round 2: does the `/links` card answer a pointer?

Captures the resting frame, the frame with the pointer over a row, the frame
with the pointer over the SELECTED row, and the frame after leaving — plus the
pointer shape and the hovered attribute at each step.

    LO_ROOT=<tree> .venv/bin/python hover.py OUTDIR
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


def _row_grounds(screen) -> list[str | None]:
    """The background each painted card row was given, top to bottom."""
    out = []
    for line in screen._card_text().split("\n"):
        bg = None
        for span in line.spans:
            color = getattr(span.style, "bgcolor", None)
            if color is not None:
                try:
                    bg = "#%02x%02x%02x" % color.get_truecolor()
                except Exception:  # noqa: BLE001
                    bg = str(color)
        out.append(bg)
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    opener = _Opener()
    app = OperatorApp(lambda: _factory(FakeSession()))
    report: dict = {"root": str(ROOT)}
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
            report["hovered_attr_before"] = getattr(screen, "_hovered", "ABSENT")
            report["grounds_resting"] = _row_grounds(screen)

            rest = outdir / "resting.svg"
            save_capture(app, str(rest))
            await pilot.pause()
            rest2 = outdir / "resting.f2.svg"
            save_capture(app, str(rest2))
            report["resting_settled"] = _sha(rest) == _sha(rest2)
            if report["resting_settled"]:
                rest2.unlink()

            # ---- pointer over row index 1 (the SECOND row), which is not the
            # selected one: the hover must show and must not speak for selection.
            row_y = region.y + 2 + 1
            screen.post_message(
                _mouse(body, events.MouseMove, screen_x=region.x + 4, screen_y=row_y)
            )
            await pilot.pause()
            await pilot.pause()
            moved = outdir / "hover-row1.svg"
            save_capture(app, str(moved))
            report["pointer_over_row1"] = str(screen.styles.pointer)
            report["hovered_over_row1"] = getattr(screen, "_hovered", "ABSENT")
            report["grounds_hover_row1"] = _row_grounds(screen)
            report["frame_changed_on_hover"] = _sha(rest) != _sha(moved)
            report["resting_sha"] = _sha(rest)
            report["moved_sha"] = _sha(moved)

            # ---- pointer over the SELECTED row (index 0): hover is additive,
            # it must not erase the selection.
            sel_y = region.y + 2
            screen.post_message(
                _mouse(body, events.MouseMove, screen_x=region.x + 4, screen_y=sel_y)
            )
            await pilot.pause()
            await pilot.pause()
            sel = outdir / "hover-selected.svg"
            save_capture(app, str(sel))
            report["grounds_hover_selected"] = _row_grounds(screen)
            report["hover_selected_sha"] = _sha(sel)
            report["hover_selected_differs_from_resting"] = _sha(sel) != _sha(rest)
            report["hover_selected_differs_from_row1_hover"] = _sha(sel) != _sha(moved)

            # ---- pointer off the card: hand and highlight both end.
            # Textual 8 spells `Leave(node)`; the real driver posts it when the
            # pointer leaves the widget's box.
            screen.post_message(events.Leave(body))
            await pilot.pause()
            await pilot.pause()
            left = outdir / "after-leave.svg"
            save_capture(app, str(left))
            report["pointer_after_leave"] = str(screen.styles.pointer)
            report["hovered_after_leave"] = getattr(screen, "_hovered", "ABSENT")
            report["grounds_after_leave"] = _row_grounds(screen)
            report["leave_restores_resting_frame"] = _sha(left) == _sha(rest)

            # ---- consecutive frames while the pointer moves across rows: a
            # highlight that settles late is motion the user sees.
            mid = outdir / "hover-row2.svg"
            screen.post_message(
                _mouse(body, events.MouseMove, screen_x=region.x + 4, screen_y=row_y + 1)
            )
            await pilot.pause()
            save_capture(app, str(mid))
            mid2 = outdir / "hover-row2.f2.svg"
            await pilot.pause()
            save_capture(app, str(mid2))
            report["hover_settled"] = _sha(mid) == _sha(mid2)
            if report["hover_settled"]:
                mid2.unlink()

            # ---- the click route is unchanged.
            await pilot.click(offset=(region.x + 4, row_y))
            for _ in range(6):
                await pilot.pause()
            report["clicked_row1_opened"] = list(opener.urls)
            report["screen_after_click"] = type(app.screen).__name__

    print(json.dumps(report, indent=1))


asyncio.run(main())
