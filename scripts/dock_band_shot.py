"""Capture the composer's docked geometry with the sidebar open, for the band's column.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/dock_band_shot.py OUT.svg \
        [COLSxROWS] [failed|connecting|none]

The frame this sample exists for is 100x30 with the session drawer DOCKED on a
COLD session — the operator's report, and the one composition in which the boot
card's clamp and the drawer compete for the same columns. It seeds the drawer's
own catalog (so the list is not an empty shell) and deliberately leaves the
TRANSCRIPT EMPTY: the welcome splash is what puts the composer into its boot
layout, and a seeded conversation hides the whole defect, because the card is
withheld the moment a turn starts.

``failed`` drives the real connection row (``Saved · Reconnect failed · Select
again to retry``), which is the row the mis-measure clipped — the last row of the
band is the input panel's status line, and the band is a CHILD of ``#input-shell``,
so it inherits whatever width that panel resolves.

**What this script can and cannot prove.** It renders the real ``OperatorApp``
with its production stylesheet, so the pixels are the app's. It drives the
connection row through ``OperatorApp._show_sidebar_connection`` rather than a
network state, so it proves the LAYOUT, not the reconnect policy. The spinner is
pinned and the catalog fixed, so a before/after pair differs only where the
change does. The band's numbers are printed as well as drawn: the still shows the
symptom, and the numbers show the cause (the ``.geometry.json`` beside the SVG
carries the widget boxes).

Two more inputs are pinned for the same reason, and both belong to the SPLASH
rather than to the composer:

* The update line (``! latest is vX — /update``) comes from a LIVE PyPI probe on
  a background worker, so whether it lands before the shot is a race — and it
  adds a row to the splash, which moves the composition a pair is compared in.
  Pinned "not behind": the honest common case, and one fewer variable in the pair.
* The session's cwd, which the splash prints. Run both captures from the SAME
  working directory (any directory; the script reads its own tree) so the two
  frames differ in the layout and nothing else.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

import local_operator.update as _update  # noqa: E402


class _NoUpdateBehind:
    """Stand-in for :func:`local_operator.update.check_latest`.

    The splash's update line comes from a live PyPI probe on a background
    worker, so whether it lands before the shot is a race — and it adds a row to
    the splash, which moves the composition a before/after pair is compared in.
    Pinned to "not behind", the honest common case. Bound onto the module
    (``OperatorApp._check_for_update`` imports the name at call time) so the
    worker cannot race the capture; nothing else about the app is stubbed.
    """

    behind = False
    latest: str | None = None


def _no_update_behind(*args: object, **kwargs: object) -> _NoUpdateBehind:
    return _NoUpdateBehind()


_update.check_latest = _no_update_behind  # type: ignore[assignment]

from local_operator.resume import SessionRow  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_catalog import (  # noqa: E402
    CatalogEntry,
    SidebarSettings,
)
from local_operator.tui.session_interaction import SessionInteraction  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

NOW = time.time()

#: A real-length title so the band's right-hand name segment has something to
#: clip — the defect was invisible with a short one, which is why the operator's
#: own report names a session.
ROWS = [
    ("aaaaaaaaaaa1", "Fix sidebar reconnect on session switch", 0, ""),
    ("aaaaaaaaaaa2", "Update Provider Onboarding and OAuth UX", 4, "idle"),
    ("aaaaaaaaaaa3", "Article-search-svc schema review (bg job)", 8, "busy"),
]

#: Spinner frame pinned, so two captures of one state are byte-identical.
SPINNER_FRAME = 2


def region(app: OperatorApp, selector: str) -> tuple[int, int, int, int]:
    box = app.query_one(selector).region
    return (box.x, box.y, box.width, box.height)


async def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "dock-band.svg")
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"
    state = sys.argv[3] if len(sys.argv) > 3 else "failed"
    columns, rows = (int(part) for part in size.lower().split("x"))

    session = FakeSession()
    target = ROWS[0][0]
    session.set_conversation_name(ROWS[0][1])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(columns, rows)) as pilot:
        for _ in range(12):
            await pilot.pause()
        # The drawer, docked rather than floating: the overlay only appears below
        # the dock/overlay threshold, where nothing is displaced and this defect
        # cannot occur.
        app._sidebar_settings = SidebarSettings(False, "left")
        await pilot.press("ctrl+b")
        for _ in range(12):
            await pilot.pause()
        sidebar = app._session_sidebar
        sidebar.set_entries(
            [
                CatalogEntry(
                    SessionRow(id=s, mtime=NOW - age * 60, name=name, live_state=live_state)
                )
                for s, name, age, live_state in ROWS
            ]
        )
        sidebar.current_id = target
        sidebar.cursor_id = target
        if sidebar._timer is not None:
            sidebar._timer.stop()
            sidebar._timer = None
        for _ in range(6):
            await pilot.pause()

        if state != "none":
            source = SessionInteraction(session)
            source.display_only = True
            # `connection_error` is a plain `str`, and "" is the no-error state the
            # row's own `not source.connection_error` test reads.
            source.connection_error = "the runtime is not responding" if state == "failed" else ""
            app._interaction = source
            app._interactions[id(session)] = source
            status = app._status
            assert status is not None, "premise: compose built the band's status line"
            status.update(conversation_name=ROWS[0][1])
            app._show_sidebar_connection(source)
            status._spinner_index = SPINNER_FRAME
            for _ in range(4):
                await pilot.pause()

        save_capture(app, out)
        dock = region(app, "#input-dock")
        shell = region(app, "#input-shell")
        band = region(app, "#status-band")
        lane_right = dock[0] + dock[2]
        # The bound this frame has to satisfy, and it is the LANE rather than the
        # screen edge. A screen-edge bound is satisfied by a row that has already
        # overrun the composer's own lane whenever the drawer is docked on the
        # RIGHT: the compositor crops the band at the main lane, so the ink is cut
        # mid-word with every cell still inside the terminal (QA round 1, Q1 —
        # measured on the base's 72-cell row, which overran its lane by 8 cells
        # while a `content.x + held <= terminal_width` check passed).
        # Both terms come from the widget, never from arithmetic on its region:
        # `#status-band`'s padding is `1 1 0 0`, so its content box is one cell
        # narrower than its region and starts ON the region's left edge, where a
        # hand-derived `region.width - 3` / `region.x + 1` reported 60 cells held
        # at x=36 for the row that holds 62 at x=35 — a bound looser than the one
        # the compositor enforces, and looser in the direction that misses a crop.
        # `content_region` is the layout engine's own answer to "the box this row
        # is drawn in", and it is read on a settled frame (the pauses above), the
        # same discipline the tests use.
        content = app.query_one("#status-band").content_region
        held = content.width
        ink_right = content.x + held
        slack = lane_right - ink_right
        print(f"state={state} size={columns}x{rows} out={out}")
        print(f"  dock={dock} shell={shell} band={band}")
        print(
            f"  shell_right={shell[0] + shell[2]} band_right={band[0] + band[2]} "
            f"screen={columns} screen_content={columns - 2} lane_right={lane_right}"
        )
        print(
            f"  row_bound (band content box vs dock lane right edge): "
            f"content.x({content.x}) + held({held}) = {ink_right} "
            f"<= lane_right({lane_right}) -> slack={slack} "
            f"{'OK' if slack >= 0 else 'CROPPED'}"
        )
        print(
            f"  boot-card={app.screen.has_class('boot-card')} overlay="
            f"{app.query_one('#session-workspace').has_class('sidebar-overlay')}"
        )


asyncio.run(main())
