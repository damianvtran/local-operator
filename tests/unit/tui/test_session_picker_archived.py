"""The `/resume` picker's Archived reveal: hidden by default, one gesture to show.

Three properties are load-bearing, and each is a defect this feature was
pictured shipping:

* **The toggle is ABSENT for a user who has never archived.** A control that
  answers with an empty list teaches a feature that does not exist, and it costs
  every such user a row of their picker. ``_archived_total > 0`` is the whole
  condition.
* **Revealing is a MOUSE gesture as well as a key.** The picker rows are
  clickable (``test_clicking_a_row_resumes_it``), so the one other thing on this
  pane a click can do has to be clickable too — and the row it lives on shifts
  the hit-test for every row beneath it, which is the second test below.
* **The binding is not a printable character.** Inside a filter every printable
  character belongs to the query (``on_key`` feeds them to ``set_query``), so a
  letter binding would make that letter untypable — a rule this codebase has
  already recorded for the picker.

The rows are built as ``SessionRow``s rather than through a store, because what
is under test is what the WIDGET does with an archived row; the scan that stamps
the flag is pinned in ``tests/unit/session/test_archive_listings.py``.
"""

from __future__ import annotations

import pytest
from rich.color import Color
from rich.style import Style
from rich.text import Text

from local_operator.resume import SessionRow
from local_operator.tui.widgets.session_picker import (
    ARCHIVE_MARKER,
    SessionPickerScreen,
)
from tests.unit.tui.test_session_picker import _PickerHost, _row

RESULTS = "#session-picker-results"


def _rows() -> list[SessionRow]:
    return [
        _row("first1", "one"),
        # ``_replace``, because ``SessionRow`` is a NamedTuple and
        # ``dataclasses.replace`` refuses it.
        _row("second", "two")._replace(archived=True),
        _row("third3", "three"),
    ]


def _lines(screen: SessionPickerScreen) -> list[str]:
    return screen.render_lines_for_test()


@pytest.mark.asyncio
async def test_an_archived_row_is_hidden_and_the_toggle_says_it_exists() -> None:
    app = _PickerHost(_rows())
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        assert [row.id for row in screen.visible_rows] == ["first1", "third3"]
        first = _lines(screen)[0]
        assert first.startswith("Archived (1) hidden"), first


@pytest.mark.asyncio
async def test_ctrl_a_reveals_the_rows_with_their_mark() -> None:
    app = _PickerHost(_rows())
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("ctrl+a")
        await pilot.pause()

        assert [row.id for row in screen.visible_rows] == ["first1", "second", "third3"]
        lines = _lines(screen)
        assert lines[0].startswith("Archived (1) shown"), lines[0]
        marked = [line for line in lines if ARCHIVE_MARKER in line]
        assert len(marked) == 1, "exactly the revealed row carries the mark"
        assert "two" in marked[0], marked

        # And again to put them away: the same key is both verbs.
        await pilot.press("ctrl+a")
        await pilot.pause()
        assert [row.id for row in screen.visible_rows] == ["first1", "third3"]
        assert not [line for line in _lines(screen) if ARCHIVE_MARKER in line]


@pytest.mark.asyncio
async def test_the_toggle_is_absent_when_nothing_is_archived() -> None:
    """A user who has never archived gets the picker that shipped before this.

    Asserted on the painted column too: the mark's cells are reserved only when
    a row in the result set carries it, so an un-archived list is byte-identical
    to the one without this feature.
    """
    rows = [_row("first1", "one"), _row("third3", "three")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        assert not [line for line in _lines(screen) if "Archived (" in line]
        assert not [line for line in _lines(screen) if ARCHIVE_MARKER in line]


@pytest.mark.asyncio
async def test_the_key_is_not_a_printable_character() -> None:
    """``ctrl+a``, and a typed ``a`` still belongs to the filter.

    Both halves are asserted: the binding's key names a control chord, and
    typing the letter filters rather than toggling. A binding on a printable
    character would make that character untypable inside the query.
    """
    app = _PickerHost(_rows())
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()

        # ``getattr`` on both halves because ``BINDINGS`` is a heterogeneous
        # list — a tuple shorthand is legal beside a ``Binding`` — and the
        # assertion is about the pair this feature added, not about the others'
        # spelling.
        keys = {
            getattr(binding, "key", "")
            for binding in SessionPickerScreen.BINDINGS
            if getattr(binding, "action", "").startswith("toggle_archived")
        }
        assert keys == {"ctrl+a"}, keys

        await pilot.press("a")
        await pilot.pause()
        assert screen.filter_query == "a", "the letter went to the query"
        assert all(not row.archived for row in screen.visible_rows), "and did not reveal anything"


@pytest.mark.asyncio
async def test_clicking_the_toggle_row_reveals_them() -> None:
    app = _PickerHost(_rows())
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        body = screen.query_one(RESULTS)

        await pilot.click(body, offset=(4, 0))
        await pilot.pause()

        assert [row.id for row in screen.visible_rows] == ["first1", "second", "third3"]

        # And off again, from the same line.
        await pilot.click(body, offset=(4, 0))
        await pilot.pause()
        assert [row.id for row in screen.visible_rows] == ["first1", "third3"]


@pytest.mark.asyncio
async def test_the_toggle_row_shifts_the_hit_test_for_the_rows_below_it() -> None:
    """One gesture still means what it says with the toggle drawn.

    The pane is one ``Static`` and the rows are lines in it, so a chrome line
    added at the top moves every row down by one. Without the header accounting
    in ``_index_at``, clicking the FIRST session would resume the one below it —
    and the row under the pointer would not be the row that answers.
    """
    app = _PickerHost(_rows())
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("ctrl+a")
        await pilot.pause()
        body = screen.query_one(RESULTS)

        # Line 0 is the toggle, line 1 is the first session.
        await pilot.click(body, offset=(4, 1))
        await pilot.pause()
        assert app.chosen == ["first1"], app.chosen


@pytest.mark.asyncio
async def test_a_filter_does_not_find_an_archived_row_while_the_toggle_is_off() -> None:
    """The default SEARCH is narrowed the same way the default list is.

    The rows all arrive in one list, so a filter applied downstream of the pool
    would surface the archived conversation by name in a picker that is not
    offering it — the row would answer a query about a conversation the user
    cannot then see in the unfiltered list.
    """
    app = _PickerHost(_rows())
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for char in "two":
            await pilot.press(char)
        await pilot.pause()
        assert screen.visible_rows == [], "'two' names the archived row and only that row"

        await pilot.press("ctrl+a")
        await pilot.pause()
        assert [row.id for row in screen.visible_rows] == ["second"]


@pytest.mark.asyncio
async def test_the_revealed_rows_are_searchable_by_name_like_any_other() -> None:
    """Revealing is not a second, weaker list: the filter sees what is shown."""
    app = _PickerHost(_rows())
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("ctrl+a")
        await pilot.pause()
        for char in "two":
            await pilot.press(char)
        await pilot.pause()
        assert [row.id for row in screen.visible_rows] == ["second"]


# ---------------------------------------------------------------------------
# The two treatments design round 1 asked for (D3 ink, D5 hover)
# ---------------------------------------------------------------------------


def _as_style(value: str | Style | None) -> Style:
    """Normalise a ``Text.style`` / span style, which may be a str or a Style."""
    return Style.parse(value) if isinstance(value, str) else (value or Style())


def _span_style(text: Text, needle: str) -> Style:
    """The style actually painted on ``needle`` inside a composed ``Text``."""
    index = text.plain.index(needle)
    span = next(s for s in text.spans if s.start <= index < s.end)
    return _as_style(text.style) + _as_style(span.style)


def _painted_hex(colour: Color | None) -> str | None:
    """A rich ``Color`` as the hex string this suite compares against."""
    return None if colour is None or colour.triplet is None else colour.triplet.hex


@pytest.mark.asyncio
async def test_the_marker_is_painted_in_the_ink_that_explains_a_row() -> None:
    """D3: `dim` measures 3.43:1 dark / 2.72:1 light on the row ground — under AA.

    This file's own body-match note rejects those numbers for a mark that explains
    a row, and the archive mark is exactly that: while the toggle is on it is the
    only thing saying a row is not offered by default. So it takes `muted` (6.51:1
    / 5.18:1) while the age and the id keep `dim` as lookup keys.
    """
    from local_operator.tui import theme as theme_mod

    app = _PickerHost(_rows())
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("ctrl+a")
        await pilot.pause()

        # Compared as the painted hex, the way this suite's other colour checks
        # are (`test_analytics_mouse`): a `Color` never equals a hex string.
        style = _span_style(screen._results_text(), ARCHIVE_MARKER.strip())
        assert _painted_hex(style.color) == theme_mod.semantic_color("muted")
        assert _painted_hex(style.color) != theme_mod.semantic_color("dim")


@pytest.mark.asyncio
async def test_hovering_the_toggle_line_paints_it_like_a_hovered_row() -> None:
    """D5: the whole line is the hit box, so the whole line has to say so.

    Rows get a `tint-select` ground under the pointer; the toggle is a control with
    the same hit box and had no hover state at all, so the mouse path was invisible
    until it was used. The ground is padded to the pane for the reason the rows are
    — a highlight that stops at the last word reads as emphasis on the text rather
    than as "this line is the target".
    """
    from local_operator.tui import theme as theme_mod

    app = _PickerHost(_rows())
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        body = screen.query_one(RESULTS)

        await pilot.hover(body, offset=(4, 0))
        await pilot.pause()
        assert screen._toggle_hovered is True
        hovered = screen._toggle_line_text()
        assert _painted_hex(_as_style(hovered.style).bgcolor) == theme_mod.semantic_color(
            "tint-select"
        )
        assert len(hovered.plain) == screen._usable(), "the ground covers the line"

        # A row below it is not the toggle, and the ground goes with the pointer.
        await pilot.hover(body, offset=(4, 1))
        await pilot.pause()
        assert screen._toggle_hovered is False
        assert _as_style(screen._toggle_line_text().style).bgcolor is None
