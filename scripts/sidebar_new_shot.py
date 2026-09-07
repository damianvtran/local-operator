"""Capture `/new` and the band AFTER a sidebar switch, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/sidebar_new_shot.py OUT.svg [splash|band|return|notice] [COLSxROWS]

Every frame here is only reachable through the sidebar's prepare/commit pair,
which is why this drives that pair rather than `/new` alone: plain `/new` was
never broken. ``splash`` switches onto a conversation WITH history and then runs
`/new`; ``band`` parks on a conversation carrying cost and context, switches
back to the untouched `/new` conversation, and captures the status band.

``return`` and ``notice`` are the PARK-AND-RETURN frames — the leg where the
sidebar differs from `/new`, `/resume` and `/reload`, because it comes back to
the conversation it left instead of retiring it:

* ``return`` is the operator's reported flow with an infrastructure notice
  present (`/new`, switch to a busy session, switch back). The notice is posted
  through the production `_system_notice`, the same call `_check_build_skew`
  makes on every swap, so the frame shows the splash standing UNDER it rather
  than being retired by it.
* ``notice`` raises a real setup warning on the splash, parks the conversation,
  returns, and drives the `refresh_info()` repaint that used to take the row
  away — so the captured frame is the one AFTER the poll that exposed the loss,
  not the stale frame that survived the switch.
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

            if which == "notice":
                # The warning belongs to the conversation being PARKED, so it is
                # raised before the switch and read back after the return.
                app._announce_on_splash(
                    "/login openai to get started - no provider configured.", "warning"
                )
                for _ in range(10):
                    await pilot.pause()
                await _switch(app, pilot, busy)
                await _switch(app, pilot, home)
                if app._welcome is not None:
                    # The repaint that exposed the loss: the splash reads its
                    # facts through a closure, so a cleared notice left the drawn
                    # row standing and removed it at the next poll.
                    app._welcome.refresh_info()
                for _ in range(20):
                    await pilot.pause()
            elif which == "return":
                app._run_slash_command("/new")
                for _ in range(80):
                    await pilot.pause()
                await asyncio.sleep(0.4)
                for _ in range(40):
                    await pilot.pause()
                # The notice `_adopt_session` re-emits on every swap, posted the
                # way the product posts it. Counting it as conversation content
                # is what retired the splash on the return leg.
                app._system_notice(
                    "this session is running an older version than this window — it will "
                    "move to the new version when its current work finishes.",
                    "note",
                )
                for _ in range(10):
                    await pilot.pause()
                await _switch(app, pilot, busy)
                await _switch(app, pilot, fresh)
                for _ in range(20):
                    await pilot.pause()
            elif which == "splash":
                await _switch(app, pilot, busy)
                app._run_slash_command("/new")
                for _ in range(80):
                    await pilot.pause()
                await asyncio.sleep(0.4)
                for _ in range(40):
                    await pilot.pause()
            else:
                # Away and back: the frame that carried the other conversation's
                # money and context over a session that has never had a turn.
                await _switch(app, pilot, busy)
                await _switch(app, pilot, fresh)

            status = app._status
            assert status is not None
            view = app._transcript_view()
            print(f"cost={status._cost!r} context_tokens={status._context_tokens}")
            print(f"welcome={app._welcome!r} welcome_visible={app._welcome_visible}")
            print(f"blocks={len(view.blocks())} boot_class={app.screen.has_class('boot')}")
            # Defensive so this script can also be run from a pre-fix checkout to
            # capture a before-frame, where the predicate does not exist yet.
            started = getattr(view, "conversation_started", None)
            print(f"conversation_started={started() if started else 'n/a'}")
            print(f"splash_notice={app._splash_notice!r}")
            print(f"screen size={tuple(app.screen.size)} virtual={tuple(app.screen.virtual_size)}")
            print(f"scrollbar={app.screen.show_vertical_scrollbar}")
            save_capture(app, out)


asyncio.run(main())
