"""Rendered evidence for the status band's CONNECTION row at one terminal size.

The connection row is the band's one two-group row whose right group is a
conversation name the user did not choose to be short, so it is where the row's
own width arithmetic is visible: the band is a fixed 1-row box and a row that
asks for one cell more than it has does not paint short-and-honest, it WORD-WRAPS
its last word onto a row the box cannot show (D1 on #1040).

    LOP_REPO=<tree> env -u NO_COLOR TERM=xterm-256color \
        .venv/bin/python scripts/status_band_row_shot.py OUT.svg [SIZE] [STATE]

``SIZE`` defaults to ``100x30`` — the operator's worst case, because the sidebar
docks there and takes columns OFF the band, while at 80x24 it becomes an overlay
and the band keeps its width. ``STATE`` is ``failed`` (default) or ``connecting``.

``LOP_SHOT_GLYPH=0`` turns the connecting glyph off, which removes the two cells
that tip the row over; ``LOP_SHOT_SIDEBAR=0`` closes the sidebar; ``LOP_SHOT_T0=1``
also writes the first frame of the state beside the settled one, for the case
where the glyph moves the row.

Nothing about the band is stubbed: ``_show_sidebar_connection`` is the method the
app's connect path calls to publish this state, and the frame is a real
``OperatorApp`` — the app that loads ``local_operator.tcss`` — through
``scripts.visual_capture.save_capture``, so the SVG carries the native cell grid
and a ``.geometry.json``. Isolation comes from ``scripts.probe_isolation``,
imported before any application module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("LOP_REPO") or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(REPO))

import scripts.probe_isolation  # noqa: E402,F401  -- MUST be the first import

# ``probe_isolation`` forces shimmer off, which SUPPRESSES the connecting glyph
# (status_line.py gates it on ``shimmer_enabled()``). The glyph is half of what
# this evidence is about — it spends the two cells that tip the row over — so it
# is re-enabled unless the caller asked for the animation-off variant.
if os.environ.get("LOP_SHOT_GLYPH", "1") == "1":
    os.environ.pop("LOCAL_OPERATOR_NO_SHIMMER", None)

import asyncio  # noqa: E402

from rich.cells import cell_len  # noqa: E402

from local_operator.resume import SessionRow  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_catalog import (  # noqa: E402
    CatalogEntry,
    SidebarSettings,
)
from local_operator.tui.session_interaction import SessionInteraction  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.status_line import StatusLine  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The conversation being switched to. Its LENGTH is the point: this is the right
#: group the connection row has to share the row with, and it is the string the
#: other surfaces in the report call the same session by.
NAME = os.environ.get("LOP_SHOT_NAME", "Fix sidebar reconnect on session switch")

TARGET = "aaaaaaaaaaa1"

#: ``(id, title, live_state)``. The target row is the one whose viewer is cold: a
#: stopped runtime with a saved transcript publishes ``""`` (see
#: ``session_picker.row_state_mark`` — "empty glyph when cold").
ROWS = [
    (TARGET, NAME, ""),
    ("aaaaaaaaaaa2", "Update Provider Onboarding and OAuth UX", "idle"),
    ("aaaaaaaaaaa3", "Article-search-svc schema review (bg job)", "busy"),
    ("aaaaaaaaaaa4", "OSWorld benchmark evaluation (subagent)", "idle"),
    ("aaaaaaaaaaa5", "Debugging session cost and naming drift", "attached"),
    ("aaaaaaaaaaa6", "Review and merge open provider MRs", "idle"),
    ("aaaaaaaaaaa7", "Address Local Operator packaging review", "idle"),
]


def _entries() -> list[CatalogEntry]:
    return [
        CatalogEntry(SessionRow(id=sid, mtime=1_800_000_000 - i * 240, name=name, live_state=state))
        for i, (sid, name, state) in enumerate(ROWS)
    ]


def _seed_transcript(app: OperatorApp) -> None:
    """The saved conversation the preview is showing, in the repo's own blocks."""
    app._append_block(UserBlock("The sidebar keeps failing to reconnect when I switch to it."))
    prose = AssistantBlock()
    prose.update_text(
        "That is the prewarm lease handing the click a facade that cannot bind: "
        "the retry budget is spent on rounds that never dial, so the failure is "
        "permanent rather than transient. The lease needs to build the same "
        "viewer contract the click path builds."
    )
    app._append_block(prose)
    app._append_block(
        UserBlock("And the band sits on that verdict, so selecting again does nothing.")
    )
    prose2 = AssistantBlock()
    prose2.update_text(
        "Right — the source cache returns the same facade unchanged, so the "
        "affordance it offers reuses the thing that cannot dial."
    )
    app._append_block(prose2)


def _install_source(app: OperatorApp, session: FakeSession) -> SessionInteraction:
    source = SessionInteraction(session)
    source.display_only = True
    app._interaction = source
    app._interactions[id(session)] = source
    app._sidebar_sources[session.session_id] = source
    return source


async def main() -> None:
    out = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"
    state = sys.argv[3] if len(sys.argv) > 3 else "failed"
    cols, rows = (int(v) for v in size.split("x"))
    if state not in {"connecting", "failed"}:
        raise SystemExit(f"unknown state {state!r}: expected 'connecting' or 'failed'")

    session = FakeSession()
    session.set_conversation_name(NAME)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(cols, rows)) as pilot:
        for _ in range(20):
            await pilot.pause()
        _seed_transcript(app)
        await pilot.pause()

        # The sidebar is OPEN by default: the operator's report is a switch made
        # FROM the list, and an open sidebar takes columns off the band, which is
        # the width budget under review.
        app._sidebar_settings = SidebarSettings(False, "left")
        if os.environ.get("LOP_SHOT_SIDEBAR", "1") == "1":
            await pilot.press("ctrl+b")
            await pilot.pause()
            sidebar = app._session_sidebar
            sidebar.set_entries(_entries())
            sidebar.current_id = TARGET
            sidebar.cursor_id = TARGET
            if sidebar._timer is not None:
                sidebar._timer.stop()
                sidebar._timer = None
            sidebar._frame = 2
            sidebar.refresh()
            await pilot.pause()
            await pilot.pause()
            assert sidebar.region.width > 0, "the sidebar did not open"
        else:
            await pilot.pause()

        source = _install_source(app, session)
        assert app._status is not None
        # Pushed explicitly rather than left to the boot path: a capture whose
        # inputs vary between runs cannot back a before/after comparison.
        app._status.update(conversation_name=NAME)
        await pilot.pause()
        assert app._status._conversation_name == NAME, "the band did not take the name"

        if state == "connecting":
            source.connect_attempts = 2
            source.connection_error = ""
        else:
            source.connect_attempts = 0
            source.connection_error = "the runtime is not responding"
        app._show_sidebar_connection(source)
        await pilot.pause()

        # Bound once, so the narrowing the assert above bought survives into the
        # closure below (pyright does not carry it across a nested def).
        band_status = app._status

        # Pin the glyph: a capture is only comparable frame-to-frame if the
        # spinner sits at a known phase. Stop the timer outright (a paused
        # interval gets re-armed by ``_sync_spinner_timer``) and set the index.
        phase = int(os.environ.get("LOP_SHOT_PHASE", "2"))

        def pin_glyph() -> None:
            if band_status._spinner_timer is not None:
                band_status._spinner_timer.stop()
                band_status._spinner_timer = None
            band_status._spinner_index = phase
            band_status.refresh()

        if os.environ.get("LOP_SHOT_T0") == "1":
            # The FIRST frame of the new state, against the settled one below: a
            # first frame that differs from the settled frame is a reflow the
            # user sees as motion.
            pin_glyph()
            await pilot.pause()
            save_capture(app, str(Path(out).with_name(Path(out).stem + "-t0" + Path(out).suffix)))

        pin_glyph()
        await pilot.pause()
        assert band_status._spinner_index == phase, "the glyph phase drifted"
        save_capture(app, out)

        _report(app, band_status, state, cols, rows)


def _report(app: OperatorApp, band_status: StatusLine, state: str, cols: int, rows: int) -> None:
    """Print the numbers that back the still: asked vs painted, and the name.

    ``render_text`` is the REQUEST and ``render_line`` is the strip the
    compositor put on screen, so a row wider than its box shows up here as a
    strip that stops early. That difference is the whole "does the name fit"
    question, and it is the number a reader cannot get from the SVG alone.
    """
    band = app.query_one("#status-band")
    width = band.size.width
    row = band_status.render_text(width)
    print(
        f"state={state} terminal={cols}x{rows} "
        f"sidebar={'open' if os.environ.get('LOP_SHOT_SIDEBAR', '1') == '1' else 'closed'} "
        f"glyph={'on' if os.environ.get('LOP_SHOT_GLYPH', '1') == '1' else 'off'} "
        f"band_region={tuple(band.region)} band_content={tuple(band.content_region)} "
        f"budget={width}"
    )
    print(f"  asked={cell_len(row.plain)} text={row.plain!r}")
    for y in range(band.size.height or 1):
        try:
            strip = band.render_line(y)
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"  painted[{y}]=<{type(exc).__name__}: {exc}>")
            continue
        print(f"  painted[{y}]={cell_len(strip.text)} text={strip.text!r}")
    print(
        f"  band.size={tuple(band.size)} "
        f"renders_at="
        f"{[cell_len(band_status.render_text(x).plain) for x in (width - 1, width, width + 1)]}"
    )
    print(
        f"  conversation_name={band_status._conversation_name!r} "
        f"connecting={band_status._connecting}"
    )


asyncio.run(main())
