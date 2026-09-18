"""Rendered frames for the three visual changes, on the production stylesheet.

Driven through the REAL app with the real `local_operator.tcss` (not the
CSS-less hosts in tests/, which would show no stylesheet at all), and through
`/links` typed into the real editor for the end-to-end cases. `--out DIR` is
required; every frame is written there as SVG.
"""
import asyncio, sys, pathlib
sys.path.insert(0, "/Users/damian/lo-wt/open-links-r3")
sys.path.insert(0, "/Users/damian/lo-wt/open-links-r3/tests")
from unittest.mock import patch
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import AssistantDelta, AssistantMessageEnd, AssistantMessageStart
from local_operator.tui.link_targets import LinkTarget
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.link_picker import LinkPickerScreen
from tests.unit.tui.test_app_pilot import FakeSession, _factory

OUT = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])


async def _settle(pilot, app):
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _type_links(pilot, app):
    editor = app.query_one(Editor)
    editor.text = "/links"
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


async def shoot(name, size, urls, hover_row=None):
    app = OperatorApp(lambda: _factory(FakeSession()))
    with patch("local_operator.mcp.auth.open_browser_quietly", lambda url: True):
        async with app.run_test(size=size) as pilot:
            await _settle(pilot, app)
            if True:
                text = "Links: " + " ".join(urls or URLS3)
                app.post_message(AssistantMessageStart())
                await pilot.pause()
                app.post_message(AssistantDelta(text))
                await pilot.pause()
                app.post_message(AssistantMessageEnd(text))
                await pilot.pause()
                await pilot.pause()
                await _type_links(pilot, app)
            screen = app.screen
            if hover_row is not None:
                body = screen._body
                region = body.region
                await pilot.hover(body, offset=(4, 2 + hover_row))
                await pilot.pause()
            path = OUT / f"{name}.svg"
            app.save_screenshot(path.name, str(OUT))
            strokes = ""
            if hover_row is not None:
                strokes = (f" pointer={app.screen._pointer_shape}"
                           f" hovered={getattr(screen, '_hovered', '<absent>')}")
            rows = []
            if isinstance(app.screen, LinkPickerScreen):
                rows = [r for r in app.screen.render_lines_for_test() if r.strip()]
                if not app.screen.is_drawable():
                    notice = app.screen.query_one("#link-picker-too-small")
                    rows = [f"<notice displayed={notice.display}> " + str(getattr(notice, '_content', notice))]
            print(f"{name}: {size} screen={app.screen.size}{strokes}")
            for row in rows:
                print(f"    {row}")


URLS3 = ["https://docs.example.test/rollout", "https://gitlab.com/minervaai/core-svc/-/merge_requests/412", "https://a.test/x"]
URLS24 = [f"https://example.test/report/{n}/index.html" for n in range(24)]


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    await shoot("hover-resting", (100, 30), URLS3)
    await shoot("hover-moved", (100, 30), URLS3, hover_row=1)
    await shoot("meta-many-100x30", (100, 30), URLS24)
    await shoot("notice-30x8", (30, 8), None)
    await shoot("notice-36x8", (36, 8), None)
    await shoot("notice-38x8", (38, 8), None)
    await shoot("card-floor-38x9", (38, 9), URLS3)


asyncio.run(main())
