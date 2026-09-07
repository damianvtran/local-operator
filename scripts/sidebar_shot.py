"""Capture the session sidebar over a populated transcript, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/sidebar_shot.py OUT.svg [COLSxROWS]

Seeds ONE fixed catalog covering every state the list can draw — a live turn,
an idle runtime, an attached viewer, an armed wake, a dormant wake, a parked
gate, a wedged runtime and cold rows — so a single frame answers both questions
this change is about:

* **Which glyph each state draws**, across every mark the list can produce.
* **How much of a title survives.** The names here are real-length generated
  titles, so the frame shows the title budget rather than a curated short one.

**What this script can and cannot prove.** It renders from a FIXED catalog of
``SessionRow``s, so it exercises the display layer only. The rows named
"…(bg job)" and "…(subagent)" carry ``live_state="idle"`` because that is what
the fixed runtime now publishes for them — which means they render ``●`` in
this script against BOTH trees, and a before/after pair of these frames is not
evidence for the activity fix (round 1, D3). The change those rows stand for
happens upstream, where ``OwnedSessionHandle`` decides whether to publish
``busy`` at all; it is proven by
``tests/unit/session/runtime/test_activity_vs_residency.py`` and by the live
probe described on the PR, not here. To see the pre-fix rendering of those
rows, set ``LO_SIDEBAR_SHOT_STALE_BUSY=1``, which restores the state the OLD
record would have published for a session holding background work.

The frame is deterministic: the spinner is pinned to a known frame and the
catalog is fixed, so before/after captures differ only where the change does.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.resume import SessionRow  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_catalog import (  # noqa: E402
    CatalogEntry,
    SidebarSettings,
)
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

NOW = time.time()

#: ``(id, title, age_minutes, live_state, pending, wakes, dormant)``.
#: Titles are real generated-length names, several of them sharing a prefix,
#: because "Article-search-svc s…" vs "Article search servi…" is exactly the
#: discrimination the list failed to support at the base width.
#: Rows whose live_state models a session HOLDING BACKGROUND WORK with a
#: finished conversation. Under the old runtime these published ``busy``; under
#: the fixed one they publish idle. ``LO_SIDEBAR_SHOT_STALE_BUSY=1`` renders
#: them the old way so the two can be compared in one tree.
STALE_BUSY = os.environ.get("LO_SIDEBAR_SHOT_STALE_BUSY") == "1"
HOLDING_WORK = "busy" if STALE_BUSY else "idle"

ROWS = [
    ("aaaaaaaaaaa1", "Fix sidebar activity indicator accuracy", 1, "busy", None, 0, False),
    ("aaaaaaaaaaa2", "Update Provider Onboarding and OAuth UX", 4, "attached", None, 0, False),
    ("aaaaaaaaaaa3", "Article-search-svc schema review (bg job)", 11, HOLDING_WORK, None, 0, False),
    ("aaaaaaaaaaa4", "Article search service integration rollout", 6, "idle", None, 0, False),
    ("aaaaaaaaaaa5", "OSWorld benchmark evaluation (subagent)", 8, HOLDING_WORK, None, 0, False),
    ("aaaaaaaaaaa6", "Auto-update inactive session runtimes", 13, "idle", None, 2, False),
    ("aaaaaaaaaaa7", "Debugging session cost and naming drift", 43, "idle", None, 1, True),
    # The two states the round-1 design findings were about. Neither was in the
    # fixture when those findings were raised, so the frames could not show the
    # bugs OR their fixes and the reviewer had to build a scratch harness to
    # see them (round 2). A capture script that cannot frame the states it was
    # reviewed for is the same defect as D3 in a different place.
    #
    #   attached + armed wake -> must draw the ATTACHED mark, not the wake (D2)
    #   cold + dormant wake   -> draws the wake mark, so the words must follow
    #                            it: "Stopped (N wakes paused)" (D1)
    ("aaaaaaaaaab4", "Review provider onboarding copy", 7, "attached", None, 2, False),
    ("aaaaaaaaaab5", "Stopped nightly catalogue refresh", 120, "", None, 1, True),
    ("aaaaaaaaaaa8", "Add Flavia's Adverse Media Case", 3, "idle", "approval", 0, False),
    ("aaaaaaaaaaa9", "Review and merge open provider MRs", 52, "wedged", None, 0, False),
    ("aaaaaaaaaab1", "Toggleable Sidebar for Session Switching", 49, "", None, 3, False),
    ("aaaaaaaaaab2", "Mark Focused TUI Session as Read", 300, "", None, 0, False),
    ("aaaaaaaaaab3", "Address Local Operator packaging review", 35, "", None, 0, False),
]


def _entries() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            SessionRow(
                id=session_id,
                mtime=NOW - age * 60,
                name=name,
                live_state=state,
                pending=pending,
                wakes=wakes,
                wakes_dormant=dormant,
            )
        )
        for session_id, name, age, state, pending, wakes, dormant in ROWS
    ]


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._sidebar_settings = SidebarSettings(False, "left")
        # Seed a conversation so the frame shows the list BESIDE something,
        # which is the only way the width trade-off is visible at all.
        for turn in range(1, 7):
            app._append_block(UserBlock(f"Turn {turn}: what should we do about the stale rows?"))
            prose = AssistantBlock()
            prose.update_text(
                f"Answer {turn}: the audit log still has every row, so a backfill "
                "is possible. Nothing else reads that column today, which is why "
                "dropping it is on the table at all."
            )
            app._append_block(prose)
        await pilot.pause()

        await pilot.press("ctrl+b")
        await pilot.pause()
        sidebar = app._session_sidebar
        sidebar.set_entries(_entries())
        sidebar.current_id = "aaaaaaaaaaa1"
        sidebar.cursor_id = "aaaaaaaaaaa1"
        # Pin the animation: a capture is only comparable frame-to-frame if the
        # spinner is at a known phase in both.
        #
        # Pausing the timer is NOT enough, and a frame pinned that way was in
        # fact unstable across runs (design review round 2, D6: six runs, two
        # distinct glyphs). `_sync_animation()` runs from `set_entries` and from
        # several other paths, and it RE-RESUMES the timer whenever any visible
        # row is busy — so a tick can still land during the settling pauses
        # below and advance `_frame` out from under the pin.
        #
        # Stopping the timer outright and clearing the handle is what actually
        # holds: `_sync_animation` returns early when `_timer is None`, so
        # nothing can restart it, and `_frame` stays exactly where it is put.
        if sidebar._timer is not None:
            sidebar._timer.stop()
            sidebar._timer = None
        sidebar._frame = 2
        sidebar.refresh()
        await pilot.pause()
        await pilot.pause()
        # Assert the pin rather than trusting it: a capture script whose
        # determinism claim is false silently poisons every future comparison
        # captured from it.
        assert sidebar._frame == 2, f"spinner phase drifted to {sidebar._frame}"

        save_capture(app, out)
        conversation = app.query_one("#session-conversation")
        print(
            f"terminal={size[0]}x{size[1]} "
            f"sidebar_outer={sidebar.region.width} "
            f"sidebar_content={sidebar.content_region.width} "
            f"conversation={conversation.size.width} "
            f"virtual={app.screen.virtual_size} actual={app.screen.size} "
            f"scrollbar={app.screen.show_vertical_scrollbar}"
        )


asyncio.run(main())
