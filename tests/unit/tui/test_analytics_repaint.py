"""The ``/analytics`` body repaints a hover change, and how MUCH it repaints.

Separate from ``test_analytics_mouse.py`` (which asks whether the surface
behaves when driven) because these are not questions about the pointer: they are
about the WORK a pointer event causes, and they are the guard for the defect the
whole ``ReportView`` exists to remove.

The defect, measured on the operator's real 156 MB ledger at 120x45: the body was
one ``Static`` whose height was the whole report, and ``Widget._render_content``
converts the widget's OWN height to strips on every dirty repaint, so a pointer
crossing one row re-stripped all 846 body lines / 85,446 cells to change two of
them — 205 ms mean / 261 ms max of loop-thread CPU per row crossed (and 134-147
ms per wheel line with the pointer resting). 3.4% of the report was visible.

**So these tests assert the work, not the clock.** A CPU ceiling calibrated on
this laptop would be a bet on machine load rather than a statement about the
screen (AGENTS.md "Calibrate ceilings from CI"; the ``test_launch_subagent.py``
account is the cautionary tale). The invariants below are the ones that hold at
any speed, and each one is false on the unfixed tree:

* a hover change does not re-enter the renderer (stock: ``build_report`` runs
  once per crossing);
* a hover change converts at most a handful of lines to strips (stock: 846 per
  crossing);
* the row that IS recomposed is byte-equal to the same row composed by a full
  rebuild, styling included — the invariant that catches a patch taking its
  column widths from the row it was handed instead of from the table (see
  ``_session_row_line``: 24 of 26 patched rows were misaligned that way);
* the hover adds no CELL to the row it lands on — a highlight that moves text
  would reflow the table under a moving pointer.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.text import Text
from textual.visual import Visual

from local_operator.analytics.model import UsageAggregate
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import analytics_panel as ap
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen, _flatten_blocks
from tests.unit.tui.test_analytics_mouse import _row_x, _row_y, _show_table
from tests.unit.tui.test_analytics_panel import _push, _tall_report_agg
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


def _ragged_columns_agg() -> UsageAggregate:
    """A session table whose rows DISAGREE about their column widths.

    The trap ``test_a_recomposed_row_equals_the_full_rebuild`` exists for cannot
    be seen in a table whose rows are all the same width: on the operator-shaped
    fixture the first twelve rows all carry a six-cell cost and a calls count at
    the four-cell floor, so a patch that measured the single row it was handed
    would still agree with the full paint and the test would pass while the bug
    it is named after went unnoticed (verified by mutation: the naive patch
    survived on that fixture and dies on this one).

    Here the rows differ in every column that is a maximum over the table — a
    five-cell ``$1.00`` beside a seven-cell ``$123.46``, a one-digit call count
    beside a six-digit one — so a patch reading its own row's figures produces a
    row that is not the painted one, and the comparison catches it.
    """

    def scope(cost_micro: int, calls: int, tokens: int) -> UsageAggregate:
        return UsageAggregate(
            calls=calls,
            ok_calls=calls,
            input_tokens=tokens,
            context_tokens=tokens,
            cost_micro=cost_micro,
            cost_known_calls=calls,
        )

    agg = scope(123_456_789 + 1_000_000 + 9_000 + 4, 317_977 + 96 + 7 + 1, 4_200_000_000)
    agg.by_session = {
        "aaaa11112222": scope(123_456_789, 317_977, 4_200_000_000),
        "bbbb22223333": scope(1_000_000, 96, 1_500),
        "cccc33334444": scope(9_000, 7, 1_500_000),
        "dddd44445555": scope(4, 1, 900),
    }
    return agg


def _styled(line: Text) -> tuple[str, tuple[tuple[int, int, str], ...]]:
    """A row's text AND its styling, as a comparable pair.

    ``rich.Text.__eq__`` compares the plain text and the span list, so a plain
    ``Text == Text`` would in fact catch a lost hover ground — what this adds is
    the SPANS IN READABLE FORM, so a failure prints ``(plain, spans)`` instead of
    a bare ``False``, and the assertion below can compare one row at a time
    without building a second composition to diff against. (It also makes the
    comparison explicit about what "the same row" means, which the width-trap
    test leans on.)
    """
    return (line.plain, tuple((span.start, span.end, str(span.style)) for span in line.spans))


def _full_rows(screen: AnalyticsScreen) -> list[Text]:
    """What a FULL repaint would paint right now, as one ``Text`` per body line.

    The screen's own renderer, fed the screen's own state — so this is the
    reference a patch has to agree with, not a second implementation of it.
    """
    return _flatten_blocks(screen._report_lines())


def _body_lines(screen: AnalyticsScreen) -> list[Text]:
    """The body widget's own lines: what it will hand the compositor as strips."""
    return screen._body.lines_for_test()


def test_a_hover_crossing_does_not_recompose_the_report(monkeypatch):
    """A pointer crossing a row must not re-run the report renderer.

    The stock tree re-enters ``build_report`` for every crossing, which is the
    whole defect: 846 lines recomposed to move a tint two rows. This asserts the
    renderer is NOT reached, and that the frame still shows the new highlight —
    a body that skipped the work by not highlighting anything would satisfy the
    first half alone.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            assert isinstance(screen._body, ap.ReportView), "the body is not virtualized"

            calls: list[int] = []
            real = ap.build_report

            def counted(*args: Any, **kwargs: Any) -> list[Text]:
                calls.append(1)
                return real(*args, **kwargs)

            monkeypatch.setattr(ap, "build_report", counted)
            # Land the pointer on row 0 first: the crossing being measured is
            # the row-to-row one, which is the operator's gesture.
            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, 0)))
            await pilot.pause()
            calls.clear()

            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, 1)))
            await pilot.pause()

            assert calls == [], "a hover crossing recomposed the whole report"
            assert screen._hover_index() == 1, "the fixture did not cross a row"
            # The new row IS tinted, out of the model the strips are built from.
            from local_operator.tui import theme as theme_mod

            tint = str(theme_mod.semantic_color("tint-select"))
            line = _body_lines(screen)[screen._layout.session_first_line + 1]
            assert tint in [str(span.style) for span in line.spans] or any(
                tint in str(span.style) for span in line.spans
            ), f"the crossed-into row painted no hover ground: {line.spans}"

    asyncio.run(run())


def test_a_hover_crossing_converts_only_the_lines_that_changed(monkeypatch):
    """One crossing converts a handful of lines, and goes through the PATCH path.

    Counted at ``Visual.to_strips`` — the call that IS the cost (measured at
    1,850 ms of the 3,006 ms a 12-step crossing run spent on the stock tree).
    The budget is deliberately loose: two rows change, so two conversions are
    expected and four allows for a line being built twice (a cache miss after a
    scroll, say). The stock tree spends 846 per crossing, so this cannot pass
    without the fix.

    **The strip budget alone does not prove the patch path ran**, and saying so
    here matters more than the assertion: ``set_lines`` reuses the strips of
    lines that came back byte-identical, so a full ``_repaint()`` also converts
    about two lines and this budget passes under an always-``_repaint`` mutant.
    What the budget catches is a body that renders every line again (the
    virtualize-only shape: 31 conversions per crossing). The patch path itself
    is asserted below by the call that distinguishes them — a crossing must call
    ``set_line`` and must NOT call ``set_lines`` — and by
    ``test_a_hover_crossing_does_not_recompose_the_report``, which counts the
    renderer.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()

            seen: list[tuple[Any, int]] = []
            real = Visual.to_strips
            body = screen._body
            sets: list[tuple[str, int]] = []
            real_set_lines = type(body).set_lines
            real_set_line = type(body).set_line

            def counting(  # type: ignore[no-untyped-def]
                cls, widget, visual, width, height, style, **kwargs
            ):
                seen.append((widget, height or 0))
                return real(widget, visual, width, height, style, **kwargs)

            def counting_set_lines(self, *args: Any, **kwargs: Any) -> None:
                sets.append(("set_lines", len(args[0]) if args else -1))
                return real_set_lines(self, *args, **kwargs)

            def counting_set_line(self, *args: Any, **kwargs: Any) -> None:
                sets.append(("set_line", args[0] if args else -1))
                return real_set_line(self, *args, **kwargs)

            monkeypatch.setattr(Visual, "to_strips", classmethod(counting))
            monkeypatch.setattr(type(body), "set_lines", counting_set_lines)
            monkeypatch.setattr(type(body), "set_line", counting_set_line)
            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, 0)))
            await pilot.pause()
            seen.clear()
            sets.clear()

            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, 1)))
            await pilot.pause()

            assert [kind for kind, _ in sets].count(
                "set_lines"
            ) == 0, f"the crossing rebuilt the whole body instead of patching rows: {sets}"
            assert [kind for kind, _ in sets].count(
                "set_line"
            ) == 2, f"the crossing did not repaint exactly the two rows involved: {sets}"

            body_lines = sum(height for widget, height in seen if widget is screen._body)
            assert body_lines <= 4, (
                f"a hover crossing converted {body_lines} body lines to strips; the "
                f"report has {len(_body_lines(screen))} lines and only two rows changed"
            )
            assert body_lines > 0, "the frame was not repainted at all, so this proves nothing"

    asyncio.run(run())


def test_a_recomposed_row_equals_the_full_rebuild(monkeypatch):
    """A patched row must be byte-equal to the same row from a full recompose.

    THE test for the width trap. The table's ``tokens``/``cost``/``calls``
    columns are maxima over every row in it, so a patch that measures the row it
    is composing — rather than reading the columns the table was painted with —
    gets a narrower column and shifts the row sideways. Measured on the real
    ledger: 24 of 26 patched rows differed from a full recompose that way, and
    all 26 matched once the widths came from the layout.

    Every row, compared after each crossing, styling included, because a row
    that matches in text but not in style is still a wrong row. The fixture is
    deliberately ragged (see ``_ragged_columns_agg``): on an evenly sized
    table this test cannot fail even when the trap is planted.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            agg = _ragged_columns_agg()
            screen = await _push(pilot, app, agg)
            _show_table(screen)
            await pilot.pause()
            first = screen._layout.session_first_line
            rows = len(screen._layout.session_rows)
            assert rows >= 4, f"the ragged fixture collapsed to {rows} rows"
            visited: list[int] = []

            for index in range(rows):
                await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, index)))
                await pilot.pause()
                assert screen._hover_index() == index, f"the pointer did not land on row {index}"
                visited.append(index)
                patched = _body_lines(screen)
                full = _full_rows(screen)
                for seen in visited:
                    assert _styled(patched[first + seen]) == _styled(full[first + seen]), (
                        f"row {seen} as painted differs from a full recompose — a patch "
                        f"composed it against the wrong columns"
                    )

    asyncio.run(run())


def test_the_hover_adds_no_cell_to_the_row_it_lands_on():
    """No reflow on hover, asserted on the PATCHED row rather than a rebuild.

    The highlight is a background span: the row's text and its cell count must be
    identical hovered and not, or the table shifts under a pointer that is moving
    across it — the one place a layout shift is least forgivable (three review
    rounds of #873 pinned this for the full paint; this pins it for the patch,
    which is now the path that paints every hovered row).
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            first = screen._layout.session_first_line

            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, 2)))
            await pilot.pause()
            assert screen._hover_index() == 2, "the fixture did not hover a row"

            hovered = _styled(_body_lines(screen)[first + 2])

            # The same row, composed with no hover at all. Cleared on the screen
            # and in the widget: the renderer reads the screen's state, and the
            # widget's own line is the thing being compared, so it has to be
            # recomposed rather than read back.
            saved = screen._hover
            screen._hover = None
            try:
                plain = _styled(_full_rows(screen)[first + 2])
            finally:
                screen._hover = saved

            assert hovered[0] == plain[0], "the hover changed the row's TEXT"
            assert len(hovered[0]) == len(plain[0]), "the hover changed the row's cell count"
            assert hovered[1] != plain[1], (
                "the hover changed nothing about the row's styling, so it painted no "
                "highlight at all"
            )

    asyncio.run(run())


def test_entering_and_leaving_the_table_touch_one_row_each(monkeypatch):
    """The pointer arriving on, or leaving, the table is a one-row change too.

    Both are ordinary gestures (moving up off the table onto the totals, or back
    down onto it) and both were worth a full 846-line recompose under the
    obvious "if either row index is missing, rebuild everything" rule: measured
    at 71 ms to enter the table and 40 ms to leave it, against 2-4 ms to patch
    the single row whose tint actually changed. This pins that they patch.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()

            calls: list[int] = []
            real = ap.build_report

            def counted(*args: Any, **kwargs: Any) -> list[Text]:
                calls.append(1)
                return real(*args, **kwargs)

            monkeypatch.setattr(ap, "build_report", counted)
            first = screen._layout.session_first_line

            # ENTER: no row was hovered, now row 1 is.
            await pilot.hover(screen, offset=(_row_x(screen), _row_y(screen, 1)))
            await pilot.pause()
            assert screen._hover_index() == 1
            tinted = _styled(_body_lines(screen)[first + 1])
            assert calls == [], "entering the table recomposed the whole report"

            # LEAVE: onto the hint line below the card, which is no row at all.
            await pilot.hover(screen, offset=(_row_x(screen), screen._hint.region.y))
            await pilot.pause()
            assert screen._hover is None, "the pointer left the rows and the highlight stayed"
            cleared = _styled(_body_lines(screen)[first + 1])
            assert calls == [], "leaving the table recomposed the whole report"
            assert tinted[1] != cleared[1], "the highlight was not cleared from the row"

    asyncio.run(run())


def test_a_style_update_drops_the_cached_strips():
    """A cached strip pins the style it was rendered with, so styles must clear it.

    Two halves, both of which the widget owns: the cache is DROPPED (or the next
    frame pins the old ramp — a mixed-ramp body the moment anything restyles the
    app while this screen is up) and the body actually repaints from the new
    styles (a cache nobody reads from would be no bug at all). ``RichLog`` and
    ``OptionList`` clear their line caches on the same hook for the same reason.
    """

    async def run():
        app = _app()
        async with app.run_test(size=(110, 40)) as pilot:
            screen = await _push(pilot, app, _tall_report_agg())
            _show_table(screen)
            await pilot.pause()
            body = screen._body
            # Render a frame first, so there is a cache to drop.
            strips = [body.render_line(y) for y in range(4)]
            assert body._strips, "nothing was cached, so this test would prove nothing"
            stale = [id(strip) for strip in strips]

            body.notify_style_update()
            assert not body._strips, "a style update left the stale strips in the cache"
            await pilot.pause()
            assert [
                id(strip) for strip in (body.render_line(y) for y in range(4))
            ] != stale, "the body repainted the same strip objects after a style update"

    asyncio.run(run())


def test_a_live_resize_republishes_the_width_it_composed_for():
    """What the body claims to be as wide as must survive a live resize.

    ``set_lines`` sizes the content from the lines, the width they were composed
    for and the content box — and the box is read BEFORE layout settles on a
    resize, so the pre-resize width would stay published (measured: 101 kept
    after 120x45 -> 90x40, where a fresh open at 90x40 reports 86). No frame
    difference was found for it, and it is asserted anyway: a stale published
    width is a wrong claim about the widget that anything reading
    ``virtual_size`` — a scroll extent, a later feature — would act on.
    """

    async def run():
        async def width_after(size: tuple[int, int], resize: tuple[int, int] | None) -> int:
            app = _app()
            async with app.run_test(size=size) as pilot:
                screen = await _push(pilot, app, _tall_report_agg())
                _show_table(screen)
                await pilot.pause()
                if resize is not None:
                    await pilot.resize_terminal(*resize)
                    await pilot.pause()
                    await pilot.pause()
                return int(screen._body.virtual_size.width)

        fresh = await width_after((90, 40), None)
        resized = await width_after((120, 45), (90, 40))
        assert resized == fresh, (
            f"a live resize to 90x40 published content width {resized}, "
            f"where a fresh open at the same size publishes {fresh}"
        )

    asyncio.run(run())
