"""Capture `/new` and the band AFTER a sidebar switch, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/sidebar_new_shot.py OUT.svg [splash|band|return|notice|toast] [COLSxROWS]

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
* ``toast`` stops HALFWAY through ``notice``'s journey, on the session switched
  TO, which is where the notice's toast used to follow the user (design round 2,
  D2). The notice and the toast are raised together and have opposite lifetimes:
  the splash ROW is the parked conversation's own empty-state content and must
  come back, the TOAST is a transient overlay about the conversation being LEFT
  and must not travel. This frame is the second half of that pair — read it
  beside ``notice``, which shows the row surviving the return.

``fork_parked`` and ``fork_return`` are the two legs of the band's ``forking``
segment (design round 3, D3), and are only meaningful as a PAIR — each one alone
is satisfied by a fix that breaks the other:

* ``fork_parked`` parks a conversation whose ``/fork`` is still waiting for a
  turn boundary and captures the session switched TO, which has no fork. The
  segment must be ABSENT: it advertises ``esc``, and ``action_stop`` probes
  ``self._session``, so a segment left standing here offers a cancel that cannot
  reach the fork it names.
* ``fork_return`` completes the round trip and captures the conversation that
  owns the fork, where the segment must be PRESENT — the request is still live
  and still cancellable, and blanking it on the park leg is what review round 2
  raised as MAJOR-3.
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
    # The fork legs need the home conversation to REPORT a pending fork, because
    # the band reads `has_pending_fork()` off the session rather than tracking
    # it — which is the property the D3 frames are about: whose answer is being
    # painted, not whether a flag was set.
    home = SidebarRemote("home-session", pending_fork=which in ("fork_parked", "fork_return"))
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

            if which in ("fork_parked", "fork_return"):
                # The request, as `/fork` on a streaming session makes it: the
                # session defers to a turn boundary and the band is synced from
                # it, so the fork stays live and cancellable across the park.
                app._sync_fork_pending()
                for _ in range(10):
                    await pilot.pause()
                await _switch(app, pilot, busy)
                if which == "fork_return":
                    await _switch(app, pilot, home)
                for _ in range(20):
                    await pilot.pause()
            elif which == "toast":
                # Same setup as `notice`, captured one leg earlier: the toast is
                # still up when the conversation is parked, so the frame shows
                # what the session switched TO is wearing.
                app._announce_on_splash(
                    "/login openai to get started - no provider configured.", "warning"
                )
                for _ in range(10):
                    await pilot.pause()
                await _switch(app, pilot, busy)
                for _ in range(20):
                    await pilot.pause()
            elif which == "notice":
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
            # Counted from `display`, not from the owner tag: what the user sees
            # is a card on screen, and a hidden toast still holding a tag is
            # invisible. This is the D2 measurement.
            from local_operator.tui.widgets.toast import Toast

            print(f"live_toasts={len([t for t in app.query(Toast) if t.display])}")
            print(f"screen size={tuple(app.screen.size)} virtual={tuple(app.screen.virtual_size)}")
            print(f"scrollbar={app.screen.show_vertical_scrollbar}")
            # The D3 measurement, read off the RENDERED band rather than the
            # flag: a code check can confirm the leave-alone is correct and
            # still miss that the frame describes the wrong conversation.
            from local_operator.tui.widgets.status_line import FORK_PENDING_TEXT

            session = app._session
            probe = getattr(session, "has_pending_fork", None) if session is not None else None
            print(f"band_says_forking={FORK_PENDING_TEXT in status.render_text(100).plain}")
            print(f"session_has_pending_fork={probe() if callable(probe) else 'n/a'}")
            save_capture(app, out)


asyncio.run(main())
