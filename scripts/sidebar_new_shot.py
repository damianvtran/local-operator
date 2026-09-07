"""Capture `/new` and the band AFTER a sidebar switch, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/sidebar_new_shot.py OUT.svg [splash|band] [COLSxROWS]

Both frames are only reachable through the sidebar's prepare/commit pair, which
is why this drives that pair rather than `/new` alone: plain `/new` was never
broken. ``splash`` switches onto a conversation WITH history and then runs
`/new`; ``band`` parks on a conversation carrying cost and context, switches
back to the untouched `/new` conversation, and captures the status band.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import _factory  # noqa: E402
from tests.unit.tui.test_sidebar_swap_reset import SidebarRemote  # noqa: E402


def _message(role: str, text: str):
    return SimpleNamespace(role=role, text=text, tool_calls=None, content=text)


async def _switch(app: OperatorApp, pilot, remote: SidebarRemote) -> None:
    """The real prepare/commit pair, as `tests.unit.tui.test_sidebar_swap_reset`
    drives it — kept in step with the tests so the frame shows what they assert."""
    from local_operator.tui.session_interaction import SessionInteraction

    source = app._sidebar_sources.get(remote.session_id)
    if source is None:
        source = SessionInteraction(remote)
        app._sidebar_sources[remote.session_id] = source

    async def lease(_session_id, *, speculative=False):
        source.preparations += 1
        return source

    app._lease_sidebar_source = lease  # type: ignore[method-assign]
    prepare = asyncio.ensure_future(app._prepare_sidebar_session(remote.session_id))
    for _ in range(400):
        if prepare.done():
            break
        await pilot.pause()
    future = app._commit_sidebar_session(remote.session_id, prepare.result(), 0)
    for _ in range(20):
        await pilot.pause()
    if future is not None and not future.done():
        future.cancel()
    for _ in range(10):
        await pilot.pause()


async def main() -> None:
    out = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "splash"
    size = sys.argv[3] if len(sys.argv) > 3 else "100x30"
    columns, _, rows = size.partition("x")

    fresh = SidebarRemote("fresh-session")
    home = SidebarRemote("home-session")
    busy = SidebarRemote(
        "busy-session",
        history=[
            _message("user", "what does the sidebar switch preserve?"),
            _message("assistant", "the prepared presentation, not the band."),
        ],
        cost=12.3456,
        context=98_765,
    )

    async def resume_factory(_resume_id):
        return fresh

    app = OperatorApp(lambda: _factory(home), resume_factory=resume_factory)
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(int(columns), int(rows))) as pilot:
            for _ in range(20):
                await pilot.pause()

            await _switch(app, pilot, busy)
            if which == "splash":
                app._run_slash_command("/new")
                for _ in range(80):
                    await pilot.pause()
                await asyncio.sleep(0.4)
                for _ in range(40):
                    await pilot.pause()
            else:
                # Away and back: the frame that carried the other conversation's
                # money and context over a session that has never had a turn.
                await _switch(app, pilot, fresh)

            status = app._status
            assert status is not None
            view = app._transcript_view()
            print(f"cost={status._cost!r} context_tokens={status._context_tokens}")
            print(f"welcome={app._welcome!r} welcome_visible={app._welcome_visible}")
            print(f"blocks={len(view.blocks())} boot_class={app.screen.has_class('boot')}")
            print(f"screen size={tuple(app.screen.size)} virtual={tuple(app.screen.virtual_size)}")
            print(f"scrollbar={app.screen.show_vertical_scrollbar}")
            save_capture(app, out)


asyncio.run(main())
