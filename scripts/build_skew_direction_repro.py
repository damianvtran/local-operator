"""The reversed-direction build-skew notice, reproduced against the real app.

The defect this pins: a window sitting IDLE at the end of a session painted

    "<name>" is running 0.51.30@d7f12d3 -> 0.51.29@2412b1d - it will switch to
    the new version when it is next idle.

which is backwards. The RUNTIME is the newer side, and there was nothing to
say at all. The shape is routine on a host that runs ``lop-update`` several
times a day: a window keeps the build it imported at launch forever, and a
runtime spawned after the update resolves ``sys.executable`` fresh, so an old
window driving a new runtime is the steady state, not the edge.

Kept as a script rather than only as a unit cell because the evidence that
matters is the rendered SENTENCE - the unit test asserts on a fragment, while
this prints what the user actually read and can be diffed across builds.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/build_skew_direction_repro.py

Expected on a fixed tree: ``refresh requests sent to the NEWER runtime: 0``
and only the disk-drift notice, which names ``/reload``. On d7f12d3a7 it
prints 1 request and the reversed second line above.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.probe_isolation  # noqa: F401,E402  -- must precede app imports
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock  # noqa: E402
from local_operator.update import BuildStamp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_build_skew import _BoundViewer  # noqa: E402


async def main() -> None:
    import local_operator.update as update_mod

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 24)) as pilot:
        await pilot.pause()
        # This window imported 0.51.29 at launch and can never change.
        app._loaded_build = BuildStamp(version="0.51.29", source_ref="2412b1daf")
        app._skew_notice_shown.clear()
        # Disk now carries 0.51.30: `lop-update` ran after the window opened.
        on_disk = BuildStamp(version="0.51.30", source_ref="d7f12d3a7")
        update_mod.installed_build = lambda *_a, **_k: on_disk
        # The bound runtime was spawned AFTER that update, so it is 0.51.30 --
        # NEWER than this window -- and it is idle. Its own refresh check
        # compares itself against disk, finds a match, and answers "kept".
        viewer = _BoundViewer(
            owner_version="0.51.30",
            owner_source_ref="d7f12d3a7",
            session_id="s1",
            conversation_name="Investigating suspicious pwned notification source",
            idle=True,
            refresh_answer="kept: build on disk matches (or has not settled)",
        )
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = [block._text for block in app.query(NoticeBlock)]
        requests = viewer.refresh_requests

    print(f"refresh requests sent to the NEWER runtime: {requests}")
    for notice in notices:
        print("NOTICE:", notice)


asyncio.run(main())
