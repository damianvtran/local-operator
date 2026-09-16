"""Pure geometry for the ``/resume`` picker: rendered output as a function of size.

No Textual app and no pilot. Everything here is a function of
``(rows, query, size)``, which is the seam the two-pane redesign hangs on: the
breakpoint, the name cap and the row budget are arithmetic, and arguing them
on paper is exactly how design round 2 shipped its BLOCKER.
"""

from __future__ import annotations

from local_operator.resume import SessionRow
from local_operator.tui.widgets.session_picker import (
    FILTER_ROWS,
    LIST_FR_STACKED,
    LIST_MIN,
    NAME_MAX,
    NAME_P75,
    OUTER_INSET_ROWS,
    PICKER_MIN_WIDTH,
    PREVIEW_FR_STACKED,
    PREVIEW_MIN,
    STACK_BELOW_COLS,
    fit_rows,
    plan_columns,
    plan_layout,
    render_rows,
    scroll_into_window,
)


def _row(session_id: str, name: str, mtime: float = 0.0) -> SessionRow:
    return SessionRow(
        id=session_id,
        name=name,
        mtime=mtime,
        created_at=0.0,
        forked=False,
        live_state="",
        pending=None,
        wakes=0,
        wakes_dormant=False,
        kind="tui",
    )


def test_the_name_field_never_narrows_as_the_terminal_grows() -> None:
    """D16, the round-2 BLOCKER, and the single most important test in the slice.

    Uncapped, the stacked field gains a full cell per terminal column while the
    side-by-side field gains only ``split``, so their gap DIVERGES and no
    breakpoint value can satisfy the invariant. ``NAME_MAX`` makes both layouts
    saturate at the same value, so past the breakpoint the two are equal. This
    fails against any breakpoint-only fix.

    THE SWEEP STARTS AT :data:`PICKER_MIN_WIDTH`, not at 80, and that floor is
    the difference between a test and a decoration. A review re-introduced the
    conditional-gutter form of the D16 bug and swept 80-240: **zero shrinks**,
    green, the invariant apparently intact. The bug's whole footprint lives
    below 80, where the ``show_id`` flip moves — so a sweep that starts at 80
    passes the exact defect it exists to prevent.

    THE ONE PERMITTED EXCEPTION is the ``show_id`` flip, and it is pinned here
    rather than excused by a floor. When the id column first fits, the name
    gives up its 14 cells: a real narrowing, and a deliberate trade — the id is
    what a user copies into ``/resume <id>``, so it appears as soon as there is
    room. Pinning it means asserting THREE things a regression would break: it
    happens at exactly one width, that width is identical on both query paths,
    and nothing else in 30-240 narrows at all. The path-identity clause is what
    catches the conditional gutter specifically — with that bug the flip sits
    at 70 unfiltered and 72 while querying, because the reservation differs by
    query state, so a test that merely allowed "one shrink somewhere" would
    still wave it through.
    """
    flips: dict[bool, list[int]] = {}
    for querying in (False, True):
        widths = {
            width: plan_layout(width, 40, querying=querying)
            for width in range(PICKER_MIN_WIDTH, 241)
        }
        shrinks = [
            (width, widths[width - 1].name_width, widths[width].name_width)
            for width in range(PICKER_MIN_WIDTH + 1, 241)
            if widths[width].name_width < widths[width - 1].name_width
        ]
        # Every shrink must BE the id-column flip, and there must be exactly one.
        for width, before, after in shrinks:
            assert not widths[width - 1].show_id and widths[width].show_id, (
                f"name narrowed at {width} without the id appearing "
                f"(querying={querying}): {before} -> {after}"
            )
        assert len(shrinks) == 1, (
            f"expected exactly one narrowing — the id flip — over "
            f"{PICKER_MIN_WIDTH}-240 (querying={querying}), got {shrinks}"
        )
        flips[querying] = [
            width
            for width in range(PICKER_MIN_WIDTH + 1, 241)
            if widths[width].show_id and not widths[width - 1].show_id
        ]
        assert flips[querying] == [shrinks[0][0]], (
            f"the id column flips at {flips[querying]} but the name narrows at "
            f"{shrinks[0][0]} (querying={querying}) — those must be the same width"
        )

    assert flips[False] == flips[True], (
        f"the id column appears at a different width depending on whether a "
        f"query is active ({flips[False]} unfiltered vs {flips[True]} querying) — "
        f"the reservation is leaking the query state into the layout, which is "
        f"the conditional-gutter form of D16"
    )


def test_the_breakpoint_is_where_side_by_side_reaches_the_cap() -> None:
    """The breakpoint is not a taste value: it is where the cap starts binding.

    Below it the side-by-side name field would be narrower than the stacked one
    at the same width, which is the comparison D16 is about.
    """
    assert plan_layout(STACK_BELOW_COLS - 1, 40).mode == "stacked"
    assert plan_layout(STACK_BELOW_COLS, 40).mode == "side-by-side"
    for width in (STACK_BELOW_COLS, STACK_BELOW_COLS + 1, 180, 240):
        assert plan_layout(width, 40).name_width == NAME_MAX, width


def test_the_row_count_follows_the_terminal_height() -> None:
    """``PAGE_ROWS_MAX`` clamped this to 10 at every height; it is deleted.

    Today's picker draws 10 rows out of 140 at a 60-row terminal with room for
    41. Fails on ``origin/main`` by construction.
    """
    counts = [plan_layout(160, height).list_rows for height in (24, 30, 40, 50, 60, 80)]
    assert counts == sorted(counts) and len(set(counts)) == len(counts), counts
    assert plan_layout(160, 40).list_rows > 10


def test_the_id_column_appears_only_when_the_name_still_clears_p75() -> None:
    """The old ``width >= 52`` turned the id on at 142 cols and cost the name 53 points.

    One rule for both layouts: the id is shown only when the name field still
    clears the p75 of the real store's names after reserving it.
    """
    for width in range(80, 241):
        plan = plan_layout(width, 40)
        if plan.show_id:
            assert plan.name_width >= NAME_P75, (width, plan.name_width)
    # The rule is a boundary, so assert it AT the boundary rather than at a
    # guessed width: the last width that hides the id is one where showing it
    # would have dropped the name under p75, and the first that shows it lands
    # the name exactly on p75.
    hides = [width for width in range(60, 241) if not plan_layout(width, 40).show_id]
    last_hidden = max(hides)
    assert plan_layout(last_hidden, 40).name_width >= NAME_P75
    assert plan_layout(last_hidden + 1, 40).show_id is True
    assert plan_layout(last_hidden + 1, 40).name_width == NAME_P75


def test_the_age_column_is_one_fixed_right_aligned_column() -> None:
    """D3: the age was rendering at 13 distinct start columns across 37 rows.

    Ages span the full range ``format_age`` can produce, so the column is sized
    by the widest one drawn and every age ENDS at the same column.
    """
    ages = ["just now", "1m ago", "4h ago", "9d ago", "99d ago", "1000d ago"]
    rows = [_row(f"{index:012x}", f"session {index}") for index in range(len(ages))]
    name_col, age_col, id_col = plan_columns(rows, 100, ages)
    assert age_col == max(len(age) for age in ages)
    lines = [line.plain for line in render_rows(rows, 0, 100, now=0.0)]
    # The age is right-aligned into a fixed column, so its END is what must
    # agree across rows — a left-aligned column is what D3 measured.
    ends = {line.index(id_text) - 2 for line, id_text in zip(lines, [row.id for row in rows])}
    assert len(ends) == 1, f"age column ends at {len(ends)} distinct columns: {ends}"
    assert name_col > 0 and id_col == 12


def test_a_row_with_a_context_line_costs_two_lines_and_the_cursor_stays_drawn() -> None:
    """The two-pass fit: the window depends on the rows, the clamp on the window.

    One pass can leave the cursor on an undrawn row, which is the exact defect
    ``_page_rows`` exists to prevent.
    """
    plan = plan_layout(160, 30)
    budget = plan.list_rows
    # Every row draws a context line, so the window holds half as many rows.
    costs = [2] * 40
    window = fit_rows(costs, top=0, budget=budget)
    assert window == budget // 2

    # A cursor below the window scrolls it, and the row it lands on is drawn.
    top = scroll_into_window(costs, top=0, cursor=30, budget=budget)
    drawn = fit_rows(costs, top=top, budget=budget)
    assert top <= 30 < top + drawn


def test_a_stacked_preview_takes_the_majority_share_proportionally() -> None:
    """T3: stacked, the preview takes 7/10 of the content rows, clamped.

    The old third-plus-14-cap handed the LIST 41 of 55 content rows on a
    60-row terminal — "kind of unusable". The pins are the measured sanity
    table; the sweep is the cap being gone: the preview first exceeds the
    old 14 at height 27 and keeps growing to 52 rows at height 80.
    """
    for height in (24, 30, 40, 50, 60, 80):
        cols_h = height - FILTER_ROWS - 2 * OUTER_INSET_ROWS
        plan = plan_layout(120, height)
        assert plan.mode == "stacked"
        share = cols_h * PREVIEW_FR_STACKED // (LIST_FR_STACKED + PREVIEW_FR_STACKED)
        expected = min(max(share, PREVIEW_MIN), cols_h - LIST_MIN)
        assert plan.preview_rows == expected, (height, plan.preview_rows)
        assert abs(plan.preview_rows - 0.7 * cols_h) <= 1, (height, plan.preview_rows)
        assert plan.list_rows == cols_h - plan.preview_rows >= LIST_MIN, (height, plan.list_rows)
        assert plan.preview_rows >= PREVIEW_MIN, (height, plan.preview_rows)
    # Pins as literals off the sanity table — the formula above mirrors the
    # production arithmetic, these do not.
    assert (plan_layout(120, 60).preview_rows, plan_layout(120, 60).list_rows) == (38, 17)
    assert (plan_layout(120, 30).preview_rows, plan_layout(120, 30).list_rows) == (17, 8)
    assert (plan_layout(120, 24).preview_rows, plan_layout(120, 24).list_rows) == (13, 6)
    # The cap is gone: monotone growth, no plateau, 52 rows at height 80.
    vals = [plan_layout(120, h).preview_rows for h in range(27, 81)]
    assert vals == sorted(vals)
    assert max(vals) == 52
    assert len(set(vals)) == 38


def test_the_stacked_split_is_monotonic_and_never_starves_either_pane() -> None:
    """Every stacked height keeps the row budget whole and both floors held.

    The clamps bind at the short end only: the list sits at ``LIST_MIN``
    for heights 19-25, the preview at ``PREVIEW_MIN`` only at 19, and the
    preview never shrinks as the terminal grows.
    """
    previous = 0
    for height in range(19, 121):
        plan = plan_layout(120, height)
        cols_h = height - FILTER_ROWS - 2 * OUTER_INSET_ROWS
        assert plan.preview_rows + plan.list_rows == cols_h, (height, plan)
        assert plan.preview_rows >= PREVIEW_MIN, (height, plan.preview_rows)
        assert plan.list_rows >= LIST_MIN, (height, plan.list_rows)
        assert plan.preview_rows >= previous, (height, plan.preview_rows)
        previous = plan.preview_rows
    for height in range(19, 26):
        assert plan_layout(120, height).list_rows == LIST_MIN, height
    for height in range(26, 121):
        assert plan_layout(120, height).list_rows > LIST_MIN, height
    at_floor = [h for h in range(19, 121) if plan_layout(120, h).preview_rows == PREVIEW_MIN]
    assert at_floor == [19], at_floor


def test_the_side_by_side_geometry_is_untouched_by_the_stacked_flip() -> None:
    """T3 is stacked-only; this is the byte-identical guard for wide terminals.

    Table captured at e368e69a, before the flip: GREEN before and after is
    the whole point, alongside the untouched width sweeps in this file.
    """
    for width, list_width, preview_width in (
        (165, 93, 61),
        (166, 94, 61),
        (180, 102, 67),
        (200, 114, 75),
        (240, 138, 91),
    ):
        for querying in (False, True):
            plan = plan_layout(width, 40, querying=querying)
            assert plan.mode == "side-by-side", (width, querying)
            geometry = (
                plan.list_width,
                plan.preview_width,
                plan.list_rows,
                plan.preview_rows,
                plan.name_width,
                plan.show_id,
            )
            assert geometry == (list_width, preview_width, 35, 35, 64, True), (width, querying)


def test_the_panes_never_ask_for_more_cells_than_the_terminal_has() -> None:
    """Textual clips SILENTLY, so the budget is asserted rather than trusted."""
    for width in range(60, 241):
        plan = plan_layout(width, 40)
        if plan.mode == "side-by-side":
            assert plan.list_width + plan.preview_width < width
        else:
            assert plan.list_width < width
            assert plan.preview_width < width
        assert plan.context_width <= plan.list_width
