"""Render the /mcp grant notices as a real frame.

These strings are rendered product copy: a user reads them in the transcript
after typing `/mcp reauth`. The PR changes their wording, so the standing rule
is to look at a painted frame rather than trust the assertions.

Uses the real OperatorApp so local_operator.tcss actually applies (the
lightweight test hosts declare no CSS_PATH and would not show the notice
styling at all).
"""

import asyncio
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

# The four notices a user can actually see from the grant path, in the order
# this PR's scenarios produce them. `variant` picks the wording under test.
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "after"

if VARIANT == "before":
    CANCELLED_CLEAN = "MCP reauth for 'notion' cancelled before the browser completed it."
else:
    CANCELLED_CLEAN = (
        "MCP login for 'notion' cancelled before the browser completed it; "
        "this attempt changed nothing — run /mcp login notion to authenticate "
        "the server."
    )

LINES = [
    (
        "authorizing MCP server 'notion' on this machine; the result appears "
        "here when it completes.",
        "info",
    ),
    (
        "/mcp reauth opens a browser and stores credentials on the machine "
        "running the session — run it from a terminal on that machine",
        "warning",
    ),
    (
        "MCP reauth for 'notion' was cancelled after its old credential was "
        "removed, so the server is now unauthenticated — run /mcp login notion "
        "to finish.",
        "warning",
    ),
    (CANCELLED_CLEAN, "warning"),
    ("authenticated MCP server 'notion'; 5 tools available.", "success"),
]


async def main() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        for text, kind in LINES:
            app._system_notice(text, kind)
            await pilot.pause()
        await pilot.pause()
        app.save_screenshot(sys.argv[1])


asyncio.run(main())
