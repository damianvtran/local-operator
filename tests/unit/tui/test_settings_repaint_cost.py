"""Structural guards on what ONE arrow press costs on ``/settings``.

WHY THESE ARE COUNTS AND NOT TIMINGS
====================================

The operator reported perceptible lag arrow-keying through ``/settings``, and
``docs/settings-keypress-profile.md`` measured why: a press re-rasterised the
ENTIRE list (96 lines at 96 rows, 218 at 218) to repaint a 14-line viewport in
which two lines had changed, plus one ``_row_text`` call and five
``semantic_color`` lookups per row per paint. The cost was cleanly linear at
~30 µs per row per press, which is what made it urgent: a ~60-row Hotkeys
section is queued behind this work and would have inherited the scaling.

Nothing here measures a duration. AGENTS.md is emphatic on the point
("Prefer a structural invariant to a numeric one"; "Calibrate ceilings from CI,
never from your laptop"), and the profiler explicitly declined to propose a
wall-clock bound because every number it had came from one M3 Max. A count of
operations per press is a fact about what the code DID, not about how fast the
machine was, so these cannot flake under load — which is the property that lets
them sit in the default suite alongside 2,700 other tests.

WHY THE BOUNDS ARE VIEWPORT-RELATIVE
------------------------------------

Every bound below is expressed against the BODY VIEWPORT (how many rows are on
screen), never against ``len(view._rows)``. That is the whole point: adding the
Hotkeys section must not move any of these numbers, and if it does, the guard
has done its job. A bound of "≤ 3 × rows" would pass vacuously on any list
length and would have passed on the code this test was written against.

PROVEN TO FAIL BEFORE THE FIX (AGENTS.md, "Prove the test can still fail").
Measured on the pre-fix tree at 100x30, 96 rows, viewport 14, 10 bounced
presses — each guard's failing value, against the bound it now holds:

    G1  semantic_color   552.0/press   bound  80   FAIL
    G2  list strip lines  96.0/press   bound  42   FAIL
    G3  _row_text calls   96.0/press   bound  28   FAIL
    G4  _build_rows        1.0/press   bound   0   FAIL

EVERY SAMPLED PRESS MUST MOVE THE CURSOR. ``/settings`` is AGENTS.md's
documented wrap-vs-clamp exception, so a press at the end of the list is a
silent no-op that repaints nothing; a naive ``down`` loop dilutes the sample
with those and under-reports the waste (the profile records this as one of two
measurement bugs it had to correct). ``scripts/settings_repaint_probe`` drives
the presses through ``scripts.settings_press_driver.Bouncer``, which reverses
direction before either end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.tui.app import OperatorApp
from scripts.settings_repaint_probe import PressCounts, count_presses, open_settings
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: Terminal the guards are stated at. 100x30 resolves to a 14-row body
#: viewport and 96 list rows on the shipped registry — the exact shape the
#: profile measured, so its numbers and these bounds describe one page.
_SIZE = (100, 30)

#: Enough presses that a one-off (a lazily-built cache filling on the first
#: press, a scroll that happened to fire) is averaged down rather than
#: dominating, and few enough that the test stays inside a second or two.
_PRESSES = 10


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Read and write an empty scratch config, never the developer's own.

    The page writes on Enter and reads on every paint; a test on the real
    config dir would both edit the developer's settings and count a row list
    that differs per machine.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.tui.settings import settings_reload

    settings_reload()
    return tmp_path


async def _measure(monkeypatch: pytest.MonkeyPatch) -> PressCounts:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        view = await open_settings(app, pilot)
        counts = await count_presses(pilot, view, _PRESSES, monkeypatch)
    return counts


@pytest.mark.asyncio
async def test_a_cursor_move_does_not_rasterise_the_whole_settings_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G2 — the load-bearing one: lines rasterised scale with the VIEWPORT.

    This is the direct assertion that a press repaints what is on screen rather
    than the whole widget. Pre-fix it measured 96.0 lines/press at 96 rows and
    218.0 at 218 — i.e. exactly ``len(rows)``, the signature of
    ``Static.update`` invalidating the render cache and
    ``Widget._render_content`` rasterising ``self.size``.

    The bound is ``3 × viewport`` rather than ``1 × viewport`` because a press
    legitimately dirties more than one band: the cursor's old row and its new
    row are in different places, a scroll can shift the window, and Textual
    renders a line on demand per compositor pass. Three screenfuls is
    comfortably above that and still an order of magnitude below the 96 the
    pre-fix code produced, so the guard has real headroom without being
    vacuous.
    """
    counts = await _measure(monkeypatch)

    assert counts.rows > 3 * counts.viewport, (
        "the guard is only meaningful while the list is much longer than the "
        f"viewport; got {counts.rows} rows against {counts.viewport} visible"
    )
    assert counts.per_press("strip_lines") <= 3 * counts.viewport, counts.summary()


@pytest.mark.asyncio
async def test_a_cursor_move_composes_only_the_rows_it_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G3 — ``_row_text`` is called for a screenful, not for every row.

    Pre-fix: 96.0 calls/press, one per row, so 82 of 96 rows were composed to
    repaint 14. Bounded at ``2 × viewport`` to leave room for the row cache
    filling around the edges of a scroll.
    """
    counts = await _measure(monkeypatch)

    assert counts.per_press("row_text") <= 2 * counts.viewport, counts.summary()


@pytest.mark.asyncio
async def test_a_cursor_move_does_not_resolve_a_theme_colour_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G1 — theme lookups are per PAINT, not per row per paint.

    Pre-fix: 554.1 lookups/press, of which 480 were the five-``Style`` prelude
    ``_row_text`` built before it examined the row's kind — so a header row paid
    for the accent and faint styles it never used.

    The bound is an ABSOLUTE number rather than a multiple of the viewport,
    because after the fix the count is a property of the PAINTERS and no longer
    of the list at all. Measured post-fix at ``_SIZE``, and flat under a
    Hotkeys-sized registry, which is the property that matters here:

        rows  96 -> 216 (registry inflated) : 61.0/press both
        size  100x30, 120x32, 140x40, 200x50: 61.0/press
        size  80x24, 70x20, 60x20 (footer sheds): 44.0/press

    THE BOUND IS 65, AND IT IS DELIBERATELY TIGHT ENOUGH TO FAIL ALONE.
    It was 80, and at 80 this guard did not do its job: reverting the
    ``_RowStyles`` hoist — the exact optimisation it names — took the count to
    only 71.0 and the test stayed GREEN, so it was really re-detecting the lazy
    compose that G3 already covers. A guard that cannot go red for the thing it
    guards is a decoration.

    65 is chosen against the measured pair, not by taste: 61.0 for the correct
    code and 71.0 for the hoist reverted, both stable to the count across every
    geometry above and under an inflated registry. It sits inside that 10-count
    gap with 4 counts of headroom below and 6 above, and this suite pins the
    terminal size, so the 44.0 of the shed footer ladder is not in play. The
    mutation is re-run in the PR's remediation round rather than asserted here.

    WHY NOT THE PROFILE'S PROPOSED 40. That figure assumed the residual would
    be the settings painters alone. It is not: 44 of the 61 come from
    ``HintButton._build``, the FOOTER, which resolves two-to-four colours per
    hint on every paint and is shared with ``OrgChartView`` and
    ``SubagentView`` — independently confirmed by a caller-attributing spy at
    exactly 44.0/press. Reaching 40 means changing a widget three views depend
    on, which is outside this change; a bound the code cannot meet is a red
    suite rather than a guard. What this change owns is the per-ROW cost, and
    that is gone (552 -> 61, flat in row count).
    """
    counts = await _measure(monkeypatch)

    assert counts.per_press("semantic_color") <= 65, counts.summary()


@pytest.mark.asyncio
async def test_a_cursor_move_does_not_rebuild_the_row_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G4 — moving the cursor re-derives no rows at all.

    ``_build_rows`` was structurally identical on 59/59 consecutive transitions
    in the profile: a cursor move cannot change the STRUCTURE of the list, only
    which row is inked as selected. Exactly zero, not "few": a rebuild on a pure
    move means the cache's invalidation has stopped tracking what actually
    changes structure, and any nonzero count is that bug.
    """
    counts = await _measure(monkeypatch)

    assert counts.build_rows == 0, counts.summary()


@pytest.mark.asyncio
async def test_a_row_rendered_alone_is_identical_to_the_whole_block() -> None:
    """The correctness premise under the per-line render, asserted not assumed.

    ``_ListStatic.render_line`` rasterises ONE row at a time, where Textual's
    default path rasterises the widget's full height in a single pass. That is
    only sound if a row rendered alone is byte-identical to the same row
    rendered as line ``y`` of the whole block — which holds because the rows are
    composed ``no_wrap`` and each is truncated to the list width before it
    reaches the renderer, so no row's rendering depends on its neighbours.

    The one thing that does NOT survive the split by itself is the segment
    metadata: Textual stamps each segment with ``meta={"offset": (x, y)}``
    naming its position in the content, and the compositor reads it back in
    ``get_widget_at``. A row rendered alone is at y=0, so ``render_line``
    re-stamps it with ``apply_offsets(0, y)``. Written without that call, this
    test fails on 95 of 96 rows — which is what makes it a real assertion about
    the mechanism rather than a restatement of it.
    """
    from textual.visual import Visual, visualize

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=_SIZE) as pilot:
        await pilot.pause()
        view = await open_settings(app, pilot)
        widget = view._list

        # The whole list in ONE pass — the pre-fix rendering, as the control.
        whole = Visual.to_strips(
            widget,
            visualize(widget, view._list_text),
            widget.size.width,
            widget.size.height,
            widget.visual_style,
        )
        assert len(whole) == len(view._rows) > 3 * view._body.size.height

        mismatched = [y for y in range(len(whole)) if list(widget.render_line(y)) != list(whole[y])]

    assert not mismatched, (
        "a row rendered on its own differs from the same row rendered as part "
        f"of the whole block, at line(s) {mismatched[:5]}"
    )
