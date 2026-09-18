"""End-to-end through the assembled OperatorApp: what the card PAINTS and what
the opener RECEIVES. No browser: the opener is a spy (no launch, no window)."""
import sys
sys.path.insert(0, "/Users/damian/lo-wt/open-links-r3")
sys.path.insert(0, "/Users/damian/lo-wt/open-links-r3/tests")
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import AssistantDelta, AssistantMessageEnd, AssistantMessageStart
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.link_picker import LinkPickerScreen
from tests.unit.tui.test_app_pilot import FakeSession, _factory
import asyncio


class Opener:
    def __init__(self):
        self.urls: list[str] = []

    async def __call__(self, url: str) -> bool:
        self.urls.append(url)
        return True


CASES = [
    "See [https://a.test/x] for docs.",
    "[[https://a.test/x]]",
    "[Source: https://a.test/x]",
    "[Source: https://a.test/wiki/Foo_(bar)]",
    "see https://[::1]] more",
    "https://[2001:db8::1]/path",
    "[see https://a.test/x](https://b.test/y)",
    "[https://a.test/x](https://a.test/x)",
]


async def run(text: str) -> tuple[list[str], list[str]]:
    app = OperatorApp(lambda: _factory(FakeSession()))
    opener = Opener()
    from unittest.mock import patch
    with patch("local_operator.mcp.auth.open_browser_quietly", opener):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app._session is not None:
                    break
            app.post_message(AssistantMessageStart())
            await pilot.pause()
            app.post_message(AssistantDelta(text))
            await pilot.pause()
            app.post_message(AssistantMessageEnd(text))
            await pilot.pause()
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.text = "/links"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            screen = app.screen
            rows = screen.render_lines_for_test() if isinstance(screen, LinkPickerScreen) else [f"!(no card) {type(screen).__name__}"]
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
    return rows, opener.urls


async def main():
    for text in CASES:
        rows, opened = await run(text)
        painted = [r for r in rows if r.strip()]
        print(f"text: {text}")
        for row in painted:
            print(f"  CARD    {row}")
        print(f"  OPENED  {opened}")
        print()


asyncio.run(main())
