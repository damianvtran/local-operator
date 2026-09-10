"""Capture what the MOUSE does to the ``/analytics`` session table.

Sibling of ``analytics_cursor_shot.py`` (where is the KEYBOARD cursor) and
``analytics_collapse_shot.py`` (how much of the table is on screen). This one
is about the pointer: the hover highlight, the pointer shape, and what a click
does to a row — the three things a rendered frame can settle and an assertion
about ``_hovered`` cannot.

Four states, each named after what it shows and each ASSERTED before the
capture rather than reached by a fixed number of gestures:

1. ``hover-expandable`` — the pointer on a row with children. The highlight has
   to be visible AND distinguishable from the caret.
2. ``hover-childless`` — the pointer on a row with none. It still highlights
   (the row is under the pointer and saying so is honest), but the pointer
   SHAPE stays ``default`` and a click does nothing.
3. ``hover-vs-cursor`` — hover and caret on DIFFERENT rows, which is the frame
   that proves the two selection models are visually separable. If hover
   painted like the caret a reader would see two cursors and not know which one
   ``enter`` acts on.
4. ``after-click`` — the same row after a click expanded it.

**Captured at three geometries, not one.** The most expensive defect in #873
got through because every frame was taken at 120x40 — the one size where the
table is on screen at mount. 120x30 and 110x20 put the table below the fold and
force the scroll-then-hover path, which is where the coordinate arithmetic is
easiest to get wrong.

Usage: ``python -m scripts.analytics_mouse_shot <out-dir>``
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import: a headless
# pilot that inherits ``CMUX_WORKSPACE_ID`` renames the operator's real cmux
# workspaces (the incident ``analytics_collapse_shot`` records).
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.probe_isolation  # noqa: E402, F401  — isolates HOME and the config dir
from local_operator.analytics.store import AnalyticsStore  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from scripts.analytics_collapse_shot import (  # noqa: E402
    _factory_for,
    _make_session,
    seed,
)
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402

#: Geometries. 120x40 is the size the collapse PR's frames used; the other two
#: are the sizes it never looked at, where the table starts below the fold.
SIZES = ((120, 40), (120, 30), (110, 20))


async def _open(pilot, app) -> AnalyticsScreen:
    await pilot.pause()
    await _submit(pilot, app, "/analytics")
    await app.workers.wait_for_complete()
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, AnalyticsScreen), f"expected /analytics, got {type(screen)}"
    return screen


def _scroll_table_into_view(screen: AnalyticsScreen) -> None:
    """Put the first session row on screen, whatever the terminal height.

    The table sits below the totals and both charts, so on a short frame it is
    simply not painted until the viewport moves. Scrolled directly rather than
    by counting ``pagedown`` presses: how many pages reach the table depends on
    the height, which is exactly what varies across ``SIZES``.
    """
    scroll = screen._scroll
    first = screen._layout.session_first_line
    scroll.scroll_to(y=max(0, first - 1), animate=False)


def _row_y(screen: AnalyticsScreen, index: int) -> int:
    """Screen row of visible session row ``index``.

    Resolved from the VIEWPORT and the scroll offset, the same basis
    ``AnalyticsScreen._row_at`` uses — ``_body.region`` is arithmetically
    equivalent but lags a scroll by a frame, so a capture aiming by the body
    can shoot at a row the screen no longer thinks is there. Spelled here
    rather than calling the screen's own helper so the capture and the code
    under test agree only if BOTH are right.
    """
    scroll = screen._scroll
    viewport = scroll.scrollable_content_region
    line = screen._layout.session_first_line + index
    return viewport.y + line - int(scroll.scroll_offset.y)


def _find_row(screen: AnalyticsScreen, *, expandable: bool) -> int:
    """Index of the first VISIBLE row of the asked-for kind, or raise.

    Frames are named after the state they show, so the state is asserted rather
    than assumed — round 1 of #873 shipped a frame called ``cursor-expandable``
    holding the cursor on a childless root because press count was used as a
    proxy for row kind.
    """
    scroll = screen._scroll
    top = scroll.scroll_offset.y
    height = scroll.size.height
    first = screen._layout.session_first_line
    for index, row in enumerate(screen._layout.session_rows):
        line = first + index
        if row.expandable is expandable and top <= line <= top + height - 1:
            return index
    raise AssertionError(f"no visible row with expandable={expandable}")


def _hovered(screen: AnalyticsScreen) -> int | None:
    """Index of the hovered row, or ``None`` when nothing is hovered.

    The screen stores the hover as a SESSION ID (``_hover``) for the same
    reason it stores the cursor as one — an index slides when a row above it
    expands — so the index a frame is checked against is resolved here against
    the layout that was actually painted.

    Read through ``getattr`` so this SAME script runs unchanged against the
    pre-change code, where the attribute does not exist: the before-frames then
    come out of the identical harness as the after-frames, which is what makes
    the pair comparable (``analytics_collapse_shot`` does this for the expand
    keys for the same reason).
    """
    session_id = getattr(screen, "_hover", None)
    if session_id is None:
        return None
    return next(
        (i for i, row in enumerate(screen._layout.session_rows) if row.session_id == session_id),
        None,
    )


async def _capture(app, pilot, out: Path, name: str, state: dict[str, object]) -> dict[str, object]:
    path = out / f"{name}.svg"
    save_capture(app, str(path))
    await pilot.pause()
    return {"state": name, "svg": path.name, **state}


async def _states_for(width: int, height: int, out: Path) -> list[dict[str, object]]:
    tag = f"{width}x{height}"
    states: list[dict[str, object]] = []
    app = OperatorApp(lambda: _factory_for(_make_session()))
    async with app.run_test(size=(width, height)) as pilot:
        screen = await _open(pilot, app)
        _scroll_table_into_view(screen)
        await pilot.pause()

        body_x = screen._body.region.x + 6

        # 1. Hover an EXPANDABLE row.
        index = _find_row(screen, expandable=True)
        row = screen._layout.session_rows[index]
        await pilot.hover(screen, offset=(body_x, _row_y(screen, index)))
        await pilot.pause()
        states.append(
            await _capture(
                app,
                pilot,
                out,
                f"hover-expandable-{tag}",
                {
                    "hovered_index": _hovered(screen),
                    "aimed_at": index,
                    "session_id": row.session_id,
                    "expandable": row.expandable,
                    "pointer_shape": str(screen.styles.pointer),
                    "cursor_index": screen._cursor_index(),
                },
            )
        )

        # 2. Hover a CHILDLESS row — highlighted, but no hand and no action.
        childless = _find_row(screen, expandable=False)
        await pilot.hover(screen, offset=(body_x, _row_y(screen, childless)))
        await pilot.pause()
        before = "\n".join(screen.render_lines_for_test())
        states.append(
            await _capture(
                app,
                pilot,
                out,
                f"hover-childless-{tag}",
                {
                    "hovered_index": _hovered(screen),
                    "aimed_at": childless,
                    "session_id": screen._layout.session_rows[childless].session_id,
                    "expandable": False,
                    "pointer_shape": str(screen.styles.pointer),
                },
            )
        )

        # 3. Hover and CARET on different rows. The caret is moved with the
        #    keyboard to a row the pointer is not on, so the frame carries both
        #    marks at once and their difference is what the reader checks.
        target = index if index != screen._cursor_index() else childless
        hover_on = childless if target == index else index
        screen._cursor = screen._layout.session_rows[target].session_id
        screen._repaint()
        await pilot.hover(screen, offset=(body_x, _row_y(screen, hover_on)))
        await pilot.pause()
        states.append(
            await _capture(
                app,
                pilot,
                out,
                f"hover-vs-cursor-{tag}",
                {
                    "hovered_index": _hovered(screen),
                    "cursor_index": screen._cursor_index(),
                    "distinct_rows": _hovered(screen) != screen._cursor_index(),
                },
            )
        )

        # 4. CLICK the expandable row and capture what it opened.
        rows_before = len(screen._layout.session_rows)
        await pilot.click(screen, offset=(body_x, _row_y(screen, index)))
        await pilot.pause()
        await pilot.pause()
        rows_after = len(screen._layout.session_rows)
        states.append(
            await _capture(
                app,
                pilot,
                out,
                f"after-click-{tag}",
                {
                    "rows_before": rows_before,
                    "rows_after": rows_after,
                    "expanded_by_click": rows_after > rows_before,
                    "cursor_followed_click": screen._cursor == row.session_id,
                    "hovered_index": _hovered(screen),
                },
            )
        )

        # 5. A click on a CHILDLESS row must be inert: same rows, same viewport.
        offset_before = screen._scroll.scroll_offset.y
        text_before = "\n".join(screen.render_lines_for_test())
        childless = _find_row(screen, expandable=False)
        await pilot.click(screen, offset=(body_x, _row_y(screen, childless)))
        await pilot.pause()
        states.append(
            {
                "state": f"childless-click-inert-{tag}",
                "viewport_unmoved": screen._scroll.scroll_offset.y == offset_before,
                "row_count_unchanged": len(screen._layout.session_rows) == rows_after,
                "frame_changed": "\n".join(screen.render_lines_for_test()) != text_before,
            }
        )
        assert before  # the childless hover frame was captured above

    for state in states:
        state["size"] = tag
    return states


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    store = AnalyticsStore()
    seed(store)
    store.close()

    states: list[dict[str, object]] = []
    for width, height in SIZES:
        states.extend(await _states_for(width, height, out))

    (out / "mouse-states.json").write_text(json.dumps(states, indent=2) + "\n")
    for state in states:
        print(json.dumps(state))


if __name__ == "__main__":
    asyncio.run(main())
