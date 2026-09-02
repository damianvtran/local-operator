"""Every row of a ``/btw`` aside is reachable by some gesture. The guard.

This file replaces ``test_aside_reachability_investigation.py``, the QA
artifact that reproduced the defect. That file asserted the BROKEN behaviour
by design and went red with the fix, as its own docstring instructed. What
survives it — and the reason this file exists rather than a deletion — is the
ORACLE: a property-style sweep that drives the real scroll entry points to
exhaustion and unions everything painted, so a row absent from the result is
reachable by no sequence of gestures at all.

The property, stated once:

    For every terminal size, every dock/band state, and every shape of
    exchange, the union of what the card paints across all reachable scroll
    offsets contains EVERY row of EVERY answer and EVERY question.

It is asserted against BOTH gestures independently. The wheel
(``_scroll_by``, what ``on_mouse_scroll_*`` calls) and the keyboard page
(``scroll_page``, what the app's ``ctrl+pageup``/``ctrl+pagedown`` chords
call) are separate code paths over one offset, and a fix that satisfied only
the mouse would leave the card's own ``↑ … · scroll`` marker naming content
that is unreachable under ``tmux set -g mouse off``.

**The geometry numbers here were measured, not assumed.** A standalone panel
gets ``overlay.FALLBACK_SCREEN`` and a budget of 16, which is not a budget any
real terminal produces; the ones in :data:`MEASURED_GEOMETRY` were read off
``AsidePanel._fit()`` in the assembled app at each size, with and without a
populated todo band. ``test_the_pinned_geometry_matches_the_assembled_app``
re-derives them through the real app so the matrix cannot drift into testing a
fiction.

One case is NOT fixed and is recorded as a strict xfail rather than omitted:
see ``test_a_one_row_budget_hides_every_answer_row``.
"""

from __future__ import annotations

import base64
import re

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import overlay
from local_operator.tui.widgets.aside_panel import AsidePanel, AsideTurn
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.todo_panel import TodoPanel
from local_operator.tui.widgets.transcript import TranscriptView
from tests.unit.tui.test_app_pilot import _factory
from tests.unit.tui.test_aside import AsideSession

#: ``rows_above_dock`` at each size and band state, MEASURED through the
#: assembled app (``_fit()`` after ``/btw``, then again after ``_refresh_band``
#: with a populated store). The body budget each one yields is in the comment,
#: and ``test_the_pinned_geometry_matches_the_assembled_app`` proves the pair
#: still holds. Pinned rather than driven through the pilot in every case
#: because the matrix below sweeps hundreds of scroll offsets per entry, and a
#: full app boot per offset would trade the coverage for the runtime.
MEASURED_GEOMETRY = {
    "80x24, no band": 17,  # budget 9
    "80x24, todo band": 11,  # budget 3 — the worst case the band produces
    "120x40, no band": 33,  # budget 25
    "120x40, todo band": 27,  # budget 19
}

#: The todo rows the band test seeds. Three items is what ``_refresh_band``
#: needs to take the six body rows the band costs.
BAND_TODOS = [
    {"text": "wire the band", "status": "done"},
    {"text": "capture frames", "status": "pending"},
    {"text": "measure the squeeze", "status": "pending"},
]

_OSC52_RE = re.compile(r"\x1b]52;c;([A-Za-z0-9+/=]*)\x07")


@pytest.fixture(autouse=True)
def _clean_todo_store():
    """The todo store is module-global in the tool registry; one test's seed
    leaks into the next otherwise (same fixture as ``test_band_panels``)."""
    from local_operator.tools import builtin

    builtin.TODO_STORE.clear()
    yield
    builtin.TODO_STORE.clear()


def _panel(
    turns: list[AsideTurn], rows_above_dock: int, monkeypatch: pytest.MonkeyPatch
) -> AsidePanel:
    """A standalone card whose body budget is a REAL one.

    ``overlay.rows_above_dock`` is patched rather than ``AsidePanel._fit``:
    the ceiling above the dock is the only input the card takes from its host,
    so pinning it exercises the whole of ``_fit`` — the gutter threshold and
    the notice row included — instead of substituting an answer for it.
    """
    monkeypatch.setattr(overlay, "rows_above_dock", lambda widget: rows_above_dock)
    panel = AsidePanel()
    panel.display = True
    panel._turns = list(turns)
    return panel


def _long_answer(rows: int, tag: str = "ANSWER") -> str:
    """``rows`` uniquely identifiable, already-short lines.

    Short enough never to wrap at any width the card takes, so "row" in the
    assertions means one source line and the counts are not a function of the
    terminal's column count. ``test_wrapped_prose_is_reachable_row_by_row``
    covers the case where they differ.
    """
    return "\n".join(f"{tag}-ROW-{index:03d}" for index in range(rows))


def _exhaust_the_wheel(panel: AsidePanel) -> set[str]:
    """Every distinct row the card can EVER render, wheeling both ways.

    The oracle this file exists to keep. It drives the same entry point the
    wheel handlers do (``_scroll_by``), walks back until the clamp stops
    moving it, then walks forward again, unioning what is painted at each
    stop. A row absent from this set cannot be reached by any sequence of
    wheel gestures.
    """
    seen: set[str] = set()
    panel._scroll_back_rows = 0
    for _ in range(5000):
        seen.update(panel.render_lines_for_test())
        before = panel._scroll_back_rows
        panel._scroll_by(1)
        if panel._scroll_back_rows == before:
            break
    for _ in range(5000):
        seen.update(panel.render_lines_for_test())
        before = panel._scroll_back_rows
        panel._scroll_by(-1)
        if panel._scroll_back_rows == before:
            break
    return seen


def _exhaust_the_keyboard(panel: AsidePanel) -> set[str]:
    """The same oracle over ``scroll_page`` — the chords' entry point.

    Separate from the wheel sweep on purpose. The two gestures share an offset
    but not a step: the wheel moves one row and the key moves a page, and the
    page is the rows SHOWN rather than the budget precisely so the key does
    not step over the rows the overlay covers. Only running both proves the
    marker's promise holds for a reader with no mouse.
    """
    seen: set[str] = set()
    panel._scroll_back_rows = 0
    for _ in range(5000):
        seen.update(panel.render_lines_for_test())
        if not panel.scroll_page(down=False):
            break
    seen.update(panel.render_lines_for_test())
    for _ in range(5000):
        seen.update(panel.render_lines_for_test())
        if not panel.scroll_page(down=True):
            break
    seen.update(panel.render_lines_for_test())
    return seen


def _rows_present(seen: set[str], count: int, tag: str = "ANSWER") -> set[int]:
    blob = "\n".join(seen)
    return {index for index in range(count) if f"{tag}-ROW-{index:03d}" in blob}


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(80):
        await pilot.pause()
        if app._session is not None:
            return
    raise AssertionError("the session never booted")


async def _open_aside(pilot, app: OperatorApp, question: str) -> AsidePanel:
    app.query_one(Editor).focus()
    await pilot.pause()
    app.query_one(Editor).load_text(f"/btw {question}")
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()
    panel = app.query_one(AsidePanel)
    assert panel.is_open
    return panel


# -- the property, across the matrix ---------------------------------------
@pytest.mark.parametrize("geometry", sorted(MEASURED_GEOMETRY))
@pytest.mark.parametrize("shape", ["one row", "exactly the budget", "one over", "200", "500"])
def test_every_row_of_one_long_answer_is_reachable(
    geometry: str, shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the fix, swept over sizes, band states and lengths.

    The defect was that a turn was the smallest addressable unit, so an answer
    taller than the card had a middle no gesture could reach — 191 of 200 rows
    at 80x24, and worse with the band up because the band costs six body rows.
    A row is the unit now, so the answer is all of them at every size.

    ``exactly the budget`` and ``one over`` bracket the boundary the old model
    could not express: at the budget the card holds the whole answer and no
    gesture should be needed, one row over is the smallest exchange that must
    scroll, and a fix that windowed with an off-by-one would lose exactly one
    of them.
    """
    rows_above_dock = MEASURED_GEOMETRY[geometry]
    probe = _panel([], rows_above_dock, monkeypatch)
    budget = probe._fit()[2]
    total = {
        "one row": 1,
        "exactly the budget": budget,
        "one over": budget + 1,
        "200": 200,
        "500": 500,
    }[shape]

    panel = _panel(
        [AsideTurn(question="QUESTION-ZERO?", answer=_long_answer(total), state="done")],
        rows_above_dock,
        monkeypatch,
    )

    for gesture, seen in (
        ("wheel", _exhaust_the_wheel(panel)),
        ("keyboard", _exhaust_the_keyboard(panel)),
    ):
        present = _rows_present(seen, total)
        assert present == set(range(total)), (
            f"{gesture} at {geometry} (budget {budget}) cannot reach "
            f"{sorted(set(range(total)) - present)[:8]}… of {total} rows"
        )
        # The question that produced the rows is reachable too. A window that
        # opens mid-answer pins it, so it is never a fragment with no owner.
        assert "QUESTION-ZERO?" in "\n".join(seen), f"{gesture} at {geometry} lost the question"


@pytest.mark.parametrize("geometry", sorted(MEASURED_GEOMETRY))
def test_every_row_of_many_turns_of_mixed_sizes_is_reachable(
    geometry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turns that fit, turns that do not, and one far taller than the card.

    The turn-grouped cut handled short turns correctly and only failed on a
    turn bigger than the budget, so a suite of uniform turns can pass while
    the defect is live. The sizes here straddle every budget in the matrix on
    purpose: 1 and 2 fit anywhere, 40 and 200 fit nowhere, and 17 fits at
    120x40 but not at 80x24.
    """
    rows_above_dock = MEASURED_GEOMETRY[geometry]
    sizes = [1, 3, 40, 2, 17, 200, 1, 9]
    turns = [
        AsideTurn(
            question=f"QUESTION-{index}?", answer=_long_answer(size, f"T{index}"), state="done"
        )
        for index, size in enumerate(sizes)
    ]
    panel = _panel(turns, rows_above_dock, monkeypatch)

    for gesture, seen in (
        ("wheel", _exhaust_the_wheel(panel)),
        ("keyboard", _exhaust_the_keyboard(panel)),
    ):
        blob = "\n".join(seen)
        for index, size in enumerate(sizes):
            present = _rows_present(seen, size, f"T{index}")
            assert present == set(range(size)), (
                f"{gesture} at {geometry}: turn {index} ({size} rows) cannot reach "
                f"{sorted(set(range(size)) - present)[:8]}…"
            )
            assert f"QUESTION-{index}?" in blob, f"{gesture} at {geometry} lost question {index}"


@pytest.mark.parametrize("geometry", sorted(MEASURED_GEOMETRY))
def test_an_empty_aside_scrolls_nowhere_and_says_so(
    geometry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero turns: the invitation row, no marker, and no gesture that moves.

    ``_window`` indexes ``flat.owners[first]`` and ``flat.heads[owner]``, both
    of which are empty here — the guard that keeps this off an IndexError is
    the early return in ``_body``, and an exchange with nothing in it is the
    state a card spends most of its life in.
    """
    panel = _panel([], MEASURED_GEOMETRY[geometry], monkeypatch)

    rendered = panel.render_lines_for_test()
    assert any("Ask anything about this session." in line for line in rendered)
    assert not any("earlier" in line for line in rendered), "nothing is above an empty exchange"
    assert panel._max_scroll_back() == 0
    assert panel.scroll_page(down=False) is False
    assert panel.scroll_page(down=True) is False
    # And the oracle still terminates rather than spinning on a clamped offset.
    assert _exhaust_the_wheel(panel) == _exhaust_the_keyboard(panel)


@pytest.mark.parametrize("geometry", ["80x24, no band", "80x24, todo band"])
def test_wrapped_prose_is_reachable_row_by_row(
    geometry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One paragraph, no newlines: painted rows are a function of the width.

    Everywhere else a source line is a row, which keeps the counts honest but
    also means the whole matrix could pass while the wrap path was windowed
    wrongly. Here the answer has ONE source line and sixty-odd painted ones.
    """
    words = [f"word{index:03d}" for index in range(600)]
    panel = _panel(
        [AsideTurn(question="Q?", answer=" ".join(words), state="done")],
        MEASURED_GEOMETRY[geometry],
        monkeypatch,
    )
    assert len(panel._flat_body().lines) > panel._fit()[2], "this must overflow to prove anything"

    for gesture, seen in (
        ("wheel", _exhaust_the_wheel(panel)),
        ("keyboard", _exhaust_the_keyboard(panel)),
    ):
        blob = "\n".join(seen)
        missing = [word for word in words if word not in blob]
        assert not missing, f"{gesture} at {geometry} cannot reach {missing[:8]}…"


def test_a_notice_costs_a_body_row_without_hiding_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal is chrome, so it comes out of the budget — not out of reach.

    ``_fit`` charges the notice row to the body rather than growing the card,
    which makes the budget one smaller than any other test here sees. The
    reachability property has to survive that, because the notice is raised by
    ``^f`` — pressed on exactly the long exchange this card is for.
    """
    panel = _panel(
        [AsideTurn(question="Q?", answer=_long_answer(80), state="done")],
        MEASURED_GEOMETRY["80x24, no band"],
        monkeypatch,
    )
    before = panel._fit()[2]
    panel.set_notice("nothing to fork yet")
    assert panel._fit()[2] == before - 1, "the notice must be charged to the body"

    for gesture, seen in (
        ("wheel", _exhaust_the_wheel(panel)),
        ("keyboard", _exhaust_the_keyboard(panel)),
    ):
        present = _rows_present(seen, 80)
        assert present == set(range(80)), f"{gesture} lost {sorted(set(range(80)) - present)[:8]}…"


# -- the marker tells the truth --------------------------------------------
@pytest.mark.parametrize("geometry", sorted(MEASURED_GEOMETRY))
def test_a_single_long_answer_admits_what_it_cut(
    geometry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silence was the sharpest half of the defect, and it is gone.

    With one turn ``hidden == 0``, so the old ``_body`` returned a bare
    ``lines[-budget:]`` slice and emitted NO marker: the reader was not told
    anything had been cut and read a truncated answer as a complete one. The
    identical answer announced itself as soon as a second turn existed.

    The count is in LINES here and questions elsewhere, which is the point —
    the question count is zero inside one answer, and "how much of this am I
    missing" is a row question.
    """
    panel = _panel(
        [AsideTurn(question="QUESTION-ZERO?", answer=_long_answer(200), state="done")],
        MEASURED_GEOMETRY[geometry],
        monkeypatch,
    )

    at_home = panel.render_lines_for_test()
    marker = [line for line in at_home if "earlier" in line]
    assert len(marker) == 1, at_home
    assert re.search(r"↑ \d+ earlier lines · scroll", marker[0]), marker
    assert "question" not in marker[0], "there is no question above; the unit is lines"

    # At the far end the exchange starts on screen, so a marker there would
    # claim rows were withheld that the reader is looking at.
    panel._scroll_back_rows = panel._max_scroll_back()
    at_top = panel.render_lines_for_test()
    assert not any("earlier" in line for line in at_top), at_top
    assert any("QUESTION-ZERO?" in line for line in at_top), at_top
    assert any("ANSWER-ROW-000" in line for line in at_top), at_top


def test_an_exchange_that_fits_claims_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: no overflow, no marker, and no gesture that moves.

    The marker states a cost, so a card that shows one when nothing was cut is
    the same class of lie as the card that cut silently.
    """
    panel = _panel(
        [AsideTurn(question="Q?", answer=_long_answer(3), state="done")],
        MEASURED_GEOMETRY["120x40, no band"],
        monkeypatch,
    )
    rendered = panel.render_lines_for_test()
    assert not any("earlier" in line for line in rendered), rendered
    assert panel._max_scroll_back() == 0
    assert panel.scroll_page(down=False) is False


def test_the_marker_counts_questions_once_whole_turns_are_above(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two markers, two units, and the boundary between them.

    Whole turns above the window are counted as QUESTIONS — a user remembers
    asking three things and never counted the rows an answer wrapped to. The
    line count is the fallback for the case where that number is zero, not a
    replacement for it.
    """
    # The third answer must itself be taller than the budget. With a SHORT
    # third turn the window never opens far enough into it for two whole
    # questions to sit above, so the plural marker is unreachable and this
    # test would turn on the turn sizes rather than on the wording.
    panel = _panel(
        [
            AsideTurn(question="FIRSTQ?", answer=_long_answer(30, "A"), state="done"),
            AsideTurn(question="SECONDQ?", answer=_long_answer(30, "B"), state="done"),
            AsideTurn(question="THIRDQ?", answer=_long_answer(12, "C"), state="done"),
        ],
        MEASURED_GEOMETRY["80x24, no band"],
        monkeypatch,
    )

    seen = "\n".join(_exhaust_the_wheel(panel))
    assert "2 earlier questions · scroll" in seen
    assert "1 earlier question · scroll" in seen, "singular, not '1 earlier questions'"
    # And the line-count marker is still what a window inside ONE answer says.
    assert re.search(r"↑ \d+ earlier lines · scroll", seen)


# -- the two gestures are one scroll ---------------------------------------
def test_the_wheel_and_the_keyboard_drive_the_same_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They cannot diverge, because a page is N notches of the same wheel.

    ``scroll_page`` is not an independent scroll implementation — it calls
    ``_scroll_by``, which is what the wheel handlers call. This pins that:
    one page back is exactly as many rows as it takes the wheel to undo, and
    the two interleave without either losing the other's position.
    """
    panel = _panel(
        [AsideTurn(question="Q?", answer=_long_answer(200), state="done")],
        MEASURED_GEOMETRY["80x24, no band"],
        monkeypatch,
    )

    assert panel.scroll_page(down=False) is True
    page = panel._scroll_back_rows
    assert page > 1, "a page must be worth pressing"

    notches = 0
    while panel._scroll_back_rows > 0 and notches < 500:
        panel._scroll_by(-1)
        notches += 1
    assert notches == page, "the key moved rows the wheel does not count in the same unit"

    # Interleaved, the later gesture reads the earlier one's offset.
    panel._scroll_by(5)
    assert panel._scroll_back_rows == 5
    panel.scroll_page(down=False)
    assert panel._scroll_back_rows == 5 + page
    panel.scroll_page(down=True)
    assert panel._scroll_back_rows == 5


def test_the_two_gestures_reach_the_same_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same set, not merely "both complete".

    The wheel steps a row and the key steps a page, so the two visit different
    OFFSETS; the guarantee is about the union of what is painted. A page sized
    to the budget rather than the rows shown would step over the rows the
    overlay covers and this is where that shows up.
    """
    turns = [
        AsideTurn(question=f"Q{index}?", answer=_long_answer(size, f"T{index}"), state="done")
        for index, size in enumerate([60, 4, 120])
    ]
    for geometry in sorted(MEASURED_GEOMETRY):
        panel = _panel(turns, MEASURED_GEOMETRY[geometry], monkeypatch)
        wheel = _exhaust_the_wheel(panel)
        keyboard = _exhaust_the_keyboard(panel)
        for index, size in enumerate([60, 4, 120]):
            assert _rows_present(wheel, size, f"T{index}") == _rows_present(
                keyboard, size, f"T{index}"
            ), f"{geometry}: turn {index} differs between the wheel and the keyboard"


def test_the_card_keeps_one_height_at_every_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrolling must not move the text being read.

    The marker and the pinned question OVERLAY the window's top rows rather
    than reserving budget below them, because reserving cannot fill the budget
    exactly at every offset — stepping past a turn boundary changes the pin and
    the content count together. The card is sized to its content, so a height
    that changes with the scroll position drags the prose under the reader.
    """
    panel = _panel(
        [
            AsideTurn(question=f"Q{index}?", answer=_long_answer(9, f"T{index}"), state="done")
            for index in range(8)
        ],
        MEASURED_GEOMETRY["80x24, no band"],
        monkeypatch,
    )
    heights = set()
    for offset in range(panel._max_scroll_back() + 1):
        panel._scroll_back_rows = offset
        heights.add(len(panel._compose_rows()) + panel._fit()[1])
    assert len(heights) == 1, f"the card changed height while scrolling: {sorted(heights)}"
    assert max(heights) <= panel._fit()[0] - 2, "the card must not reach over the dock"


# -- the streaming anchor, both halves -------------------------------------
def test_a_parked_reader_holds_still_across_many_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holding the offset NUMBER still is not the rule; holding the ROWS is.

    The offset counts back from the tail, so rows arriving at the tail slide
    the window forward under a reader who has not touched anything. Measured
    against the unanchored form, a reader parked 120 rows back watched their
    top row walk from 067 to 127 across 60 deltas.
    """
    panel = _panel([], MEASURED_GEOMETRY["80x24, no band"], monkeypatch)
    generation = panel.ask("why?")
    for index in range(200):
        panel.append_answer(generation, f"ANSWER-ROW-{index:03d}\n")

    panel._scroll_by(120)
    parked = [line for line in panel.render_lines_for_test() if "ANSWER-ROW-" in line]
    assert parked, "the reader must be looking at content for this to mean anything"

    for index in range(200, 320):
        panel.append_answer(generation, f"ANSWER-ROW-{index:03d}\n")

    after = [line for line in panel.render_lines_for_test() if "ANSWER-ROW-" in line]
    assert after == parked, "120 deltas dragged the reader off the rows they were reading"


def test_a_reader_at_the_tail_follows_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of the rule. A fix that pinned everyone would break it."""
    panel = _panel([], MEASURED_GEOMETRY["80x24, no band"], monkeypatch)
    generation = panel.ask("why?")
    for index in range(50):
        panel.append_answer(generation, f"ANSWER-ROW-{index:03d}\n")
    assert "ANSWER-ROW-049" in "\n".join(panel.render_lines_for_test())

    for index in range(50, 300):
        panel.append_answer(generation, f"ANSWER-ROW-{index:03d}\n")
    assert "ANSWER-ROW-299" in "\n".join(panel.render_lines_for_test())
    assert panel._scroll_back_rows == 0, "an untouched reader must stay home"


def test_asking_again_re_acquires_the_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming does not move a parked reader; the user's own question does.

    The three-state rule the transcript follows, in this card's units: a new
    question is a re-acquire, so the answer to it must not arrive off screen
    above the window the reader left behind.
    """
    panel = _panel([], MEASURED_GEOMETRY["80x24, no band"], monkeypatch)
    generation = panel.ask("first?")
    panel.settle_answer(generation, _long_answer(200))
    panel._scroll_by(100)
    assert panel._scroll_back_rows == 100

    panel.ask("second?")
    assert panel._scroll_back_rows == 0


def test_a_settle_that_shrinks_the_answer_does_not_strand_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``settle_answer`` replaces streamed text with the authoritative text,
    which is routinely shorter. The offset is clamped in ``_window`` and not
    only in ``_scroll_by`` for exactly this: the exchange can shrink under a
    parked reader without any gesture having been made."""
    panel = _panel([], MEASURED_GEOMETRY["80x24, no band"], monkeypatch)
    generation = panel.ask("why?")
    panel.append_answer(generation, _long_answer(300))
    panel._scroll_back_rows = panel._max_scroll_back()
    stranded = panel._scroll_back_rows

    panel.settle_answer(generation, _long_answer(10))

    # The STORED offset is deliberately left past the new end — no gesture ran,
    # so nothing clamped it — and `_window` clamps on the way to the paint
    # instead. The frame is therefore the TOP of the shorter answer rather than
    # a blank card or an IndexError.
    assert panel._scroll_back_rows == stranded > panel._max_scroll_back()
    rendered = panel.render_lines_for_test()
    assert any("▌ why?" in line for line in rendered), rendered
    assert any("ANSWER-ROW-000" in line for line in rendered), rendered
    assert not any("earlier" in line for line in rendered), "the top is on screen"

    # And the reader is not stranded there: every row of the settled answer is
    # still reachable, which is the property this whole file is about.
    assert _rows_present(_exhaust_the_wheel(panel), 10) == set(range(10))


# -- the smallest budgets, and each direction on its own -------------------
@pytest.mark.parametrize("rows_above_dock", [4, 5, 6, 7, 8, 9, 10])
def test_the_smallest_budgets_still_show_the_answer(
    rows_above_dock: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA-1. At budget 1 the overlay must yield the row to the content.

    The marker and the pin OVERLAY the window's top rows, which costs nothing
    in reachability only while the overlay is strictly SMALLER than the
    budget. At budget 1 it is not: the marker covered the only row at every
    offset, so the card painted ``↑ N earlier lines · scroll`` and nothing
    else, forever — 0 of 40 rows reachable by any gesture, measured. That is a
    sharper form of the defect this file exists to guard, and a REGRESSION:
    the same geometry before the row model painted the single tail row.

    These budgets are reachable, not theoretical. ``_fit()`` yields 1 for 4 to
    7 rows above the dock, which a populated todo band produces at 80x20 and
    60x16, and at 80x24 with a draft wrapped to five rows.
    """
    panel = _panel(
        [AsideTurn(question="Q?", answer=_long_answer(40), state="done")],
        rows_above_dock,
        monkeypatch,
    )
    budget = panel._fit()[2]

    for gesture, seen in (
        ("wheel", _exhaust_the_wheel(panel)),
        ("keyboard", _exhaust_the_keyboard(panel)),
    ):
        present = _rows_present(seen, 40)
        assert present == set(range(40)), (
            f"{gesture} at budget {budget} cannot reach " f"{sorted(set(range(40)) - present)[:8]}…"
        )


@pytest.mark.parametrize("geometry", sorted(MEASURED_GEOMETRY))
def test_paging_forward_alone_reaches_every_row(
    geometry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA-2. Each DIRECTION must reach every row on its own.

    A sweep that unions both directions hides this, and mine did: paging back
    to the top and then reading DOWN is the natural way to read a long answer
    from the start, and it skipped rows. The overlay covers rows at the
    window's TOP, so a backward page leaves the covered rows below the new
    top and the next step re-reads them — backward self-corrects. Forward
    moves the top the other way, so a step sized by the CURRENT window lands
    past rows the destination will cover, and they are painted at no offset at
    all. Measured on the pre-fix tree: rows 8-9 of 60 at budget 9, 2-3 at
    budget 3, 24-25 at budget 25 — every size, always the pair straddling the
    budget.

    Paging has to be REVERSIBLE, or the two directions disagree about which
    rows exist.
    """
    panel = _panel(
        [AsideTurn(question="Q?", answer=_long_answer(60), state="done")],
        MEASURED_GEOMETRY[geometry],
        monkeypatch,
    )

    # Start at the oldest row and read DOWN, the way someone reads an answer
    # from its beginning. Nothing walks back afterwards to paper over a gap.
    panel._scroll_back_rows = panel._max_scroll_back()
    seen: set[str] = set()
    for _ in range(5000):
        seen.update(panel.render_lines_for_test())
        if not panel.scroll_page(down=True):
            break
    seen.update(panel.render_lines_for_test())

    present = _rows_present(seen, 60)
    assert present == set(
        range(60)
    ), f"paging forward at {geometry} skipped {sorted(set(range(60)) - present)[:8]}…"
    assert panel._scroll_back_rows == 0, "forward paging must land home"


@pytest.mark.parametrize("geometry", sorted(MEASURED_GEOMETRY))
def test_paging_back_alone_reaches_every_row(
    geometry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction on its own, for the same reason: no union."""
    panel = _panel(
        [AsideTurn(question="Q?", answer=_long_answer(60), state="done")],
        MEASURED_GEOMETRY[geometry],
        monkeypatch,
    )

    panel._scroll_back_rows = 0
    seen: set[str] = set()
    for _ in range(5000):
        seen.update(panel.render_lines_for_test())
        if not panel.scroll_page(down=False):
            break
    seen.update(panel.render_lines_for_test())

    present = _rows_present(seen, 60)
    assert present == set(
        range(60)
    ), f"paging back at {geometry} skipped {sorted(set(range(60)) - present)[:8]}…"


# -- through the assembled app ---------------------------------------------
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
@pytest.mark.asyncio
async def test_the_pinned_geometry_matches_the_assembled_app(size) -> None:
    """The matrix above is only worth its runtime if its numbers are real.

    Re-derives ``rows_above_dock`` at both sizes, clean and with the band up,
    through the same ``_refresh_band()`` path the 1 Hz poll uses, and checks
    them against :data:`MEASURED_GEOMETRY`. A layout change that moves the
    ceiling makes this fail rather than quietly leaving the sweep testing a
    geometry the app no longer produces.
    """
    from local_operator.tools import builtin

    label = f"{size[0]}x{size[1]}"
    session = AsideSession(answer=_long_answer(200))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        panel = await _open_aside(pilot, app, "why is this long?")

        clean_rows, _, clean_budget = panel._fit()
        assert (
            clean_rows == MEASURED_GEOMETRY[f"{label}, no band"]
        ), f"{label} clean ceiling moved: {clean_rows}"

        builtin.TODO_STORE["sess"] = list(BAND_TODOS)
        app._refresh_band()
        await pilot.pause()
        assert app.query_one(TodoPanel).display is True
        panel.sync_layout(force=True)
        await pilot.pause()

        band_rows, _, band_budget = panel._fit()
        assert (
            band_rows == MEASURED_GEOMETRY[f"{label}, todo band"]
        ), f"{label} band ceiling moved: {band_rows}"
        # The band still costs the body rows — that has not changed, and it is
        # why the band columns of the matrix are the ones worth sweeping.
        assert band_budget < clean_budget
        # But the rows it costs are no longer rows the reader loses: the card
        # says what it cut, at both sizes, where it used to say nothing.
        assert any("earlier" in line for line in panel.render_lines_for_test())


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
@pytest.mark.asyncio
async def test_the_chords_reach_every_row_without_touching_the_draft(size) -> None:
    """The keyboard path as a user has it: Editor focused, draft half-typed.

    The card is ``can_focus = False``, so the chords are bound at APP level and
    have to reach past a focused ``TextArea`` that claims most keys. Two things
    are asserted together because either alone would be a false pass: that the
    chords reach all 200 rows, and that the composer's text and caret come
    through untouched — a chord the editor also handles would scroll the card
    AND edit the draft the aside promised to hand back.
    """
    draft = "a half-typed thought I do not want to lose"
    session = AsideSession(answer=_long_answer(200))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await _boot(pilot, app)
        panel = await _open_aside(pilot, app, "explain the whole loop")

        editor = app.query_one(Editor)
        editor.load_text(draft)
        await pilot.pause()
        editor.cursor_location = (0, 7)
        await pilot.pause()
        assert app.screen.focused is editor, "the composer must hold focus inside an aside"

        seen: set[str] = set()
        for _ in range(500):
            seen.update(panel.render_lines_for_test())
            await pilot.press("ctrl+pageup")
            await pilot.pause()
            if panel._scroll_back_rows >= panel._max_scroll_back():
                break
        seen.update(panel.render_lines_for_test())
        for _ in range(500):
            await pilot.press("ctrl+pagedown")
            await pilot.pause()
            seen.update(panel.render_lines_for_test())
            if panel._scroll_back_rows == 0:
                break

        present = _rows_present(seen, 200)
        assert present == set(range(200)), f"unreachable: {sorted(set(range(200)) - present)[:8]}…"
        assert "explain the whole loop" in "\n".join(seen)

        assert editor.text == draft, "a chord edited the draft"
        assert editor.cursor_location == (0, 7), "a chord moved the caret"
        assert app.screen.focused is editor, "a chord stole focus from the composer"


@pytest.mark.asyncio
async def test_copy_lifts_the_whole_exchange_and_leaves_no_trace() -> None:
    """``ctrl+r`` rescues what ``esc`` discards, without writing to the record.

    Read off the DRIVER rather than ``app._clipboard``, which is assigned
    before the driver check and so cannot prove a write went out
    (``test_copy_picker_qa.py:372``). All three claims in one test because
    they are one contract: the payload is the whole answer (not the painted
    window, which is a fraction of it), it carries no card chrome, and nothing
    reaches the transcript or the session — the aside's off-the-record promise
    is what makes copy the second door rather than a second fork.
    """
    session = AsideSession(answer=_long_answer(200))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        panel = await _open_aside(pilot, app, "explain the whole loop")

        painted = _rows_present(set(panel.render_lines_for_test()), 200)
        assert len(painted) < 200, "the card must be windowing for this to prove anything"

        blocks_before = len(app.query_one(TranscriptView).blocks())
        history_before = len(session._history)

        driver = app._driver
        assert driver is not None, "no driver: the pilot would prove nothing about the write"
        sink: list[str] = []
        original = driver.write

        def write(data: str) -> None:
            sink.append(data)
            original(data)

        driver.write = write  # type: ignore[method-assign]
        await pilot.press("ctrl+r")
        await pilot.pause()

        payloads = [
            base64.b64decode(match.group(1)).decode("utf-8")
            for chunk in sink
            for match in _OSC52_RE.finditer(chunk)
        ]
        assert len(payloads) == 1, f"expected one OSC 52 write, got {len(payloads)}"
        copied = payloads[0]

        assert _rows_present({copied}, 200) == set(range(200)), "the copy is a window, not the text"
        assert "explain the whole loop" in copied, "the question belongs with its answer"
        for chrome in ("off the record", "esc discard", "─", "↑ ", "▌"):
            assert chrome not in copied, f"card chrome {chrome!r} reached the clipboard"

        assert len(app.query_one(TranscriptView).blocks()) == blocks_before
        assert len(session._history) == history_before
        assert session.forked == [], "copy is not a fork"
        assert panel.is_open, "copying does not dismiss the card"


@pytest.mark.asyncio
async def test_copy_is_the_way_out_while_fork_is_refused() -> None:
    """The case ``ctrl+r`` exists for: a running turn, with ``^f`` unavailable.

    ``^f`` refuses mid-stream because splicing a message into a live batch
    produces a request no provider accepts — a constraint on WRITING, which
    the clipboard does not do. So the partial answer must still be copyable,
    and it must include the running turn ``fork_messages`` drops.
    """
    session = AsideSession(answer=_long_answer(200))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        panel = await _open_aside(pilot, app, "explain the whole loop")

        session.streaming = True
        await pilot.pause()
        app._clipboard = ""
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert session.forked == [], "the fork was supposed to be refused here"

        await pilot.press("ctrl+r")
        await pilot.pause()
        assert _rows_present({app._clipboard}, 200) == set(range(200))
        assert panel.is_open

    # A RUNNING turn is in the payload where `fork_messages` drops it, and a
    # FAILED one is in neither: its prose is an error string this app wrote,
    # never something the model said.
    running = AsidePanel()
    running.display = True
    running._turns = [AsideTurn(question="RUNQ?", answer=_long_answer(30, "R"), state="running")]
    assert running.fork_messages() == []
    assert _rows_present({running.copy_text()}, 30, "R") == set(range(30))
    assert "RUNQ?" in running.copy_text()

    failed = AsidePanel()
    failed.display = True
    failed._turns = [AsideTurn(question="FAILQ?", answer="", state="error", error="boom")]
    assert failed.copy_text() == ""


@pytest.mark.asyncio
async def test_copy_excludes_turns_the_model_never_stood_behind() -> None:
    """A turn that streamed and THEN failed still holds its partial text.

    Filtering on ``answer.strip()`` alone let that text through, formatted
    exactly like an answer the model completed — the same reason
    ``fork_messages`` drops these turns and the reason ``AsideTurn`` keeps a
    failure's cause in its own field. Cancelled goes with it: the user moved
    on, and the card marks it on screen where the clipboard cannot.

    Asserted through the APP's ``ctrl+r`` as well as the panel, because the
    two had separate implementations of the payload and the leak was in both.
    """
    turns = [
        AsideTurn(question="DONEQ?", answer=_long_answer(4, "OK"), state="done"),
        AsideTurn(question="FAILQ?", answer=_long_answer(4, "LEAKED"), state="error", error="boom"),
        AsideTurn(question="GONEQ?", answer=_long_answer(4, "DROPPED"), state="cancelled"),
        AsideTurn(question="RUNQ?", answer=_long_answer(4, "LIVE"), state="running"),
    ]

    panel = AsidePanel()
    panel.display = True
    panel._turns = list(turns)
    payload = panel.copy_text()
    assert "OK-ROW-000" in payload, "a settled answer must be copied"
    assert "LIVE-ROW-000" in payload, "a running turn is the case ^f cannot cover"
    assert "LEAKED" not in payload, "a failed turn's partial text reached the clipboard"
    assert "DROPPED" not in payload, "a cancelled turn's text reached the clipboard"
    assert "boom" not in payload, "the failure's cause is not an answer"

    session = AsideSession(answer=_long_answer(4, "OK"))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        live = await _open_aside(pilot, app, "explain")
        live._turns = list(turns)
        app._clipboard = ""
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert "OK-ROW-000" in app._clipboard
        assert "LIVE-ROW-000" in app._clipboard
        assert "LEAKED" not in app._clipboard, "the app path leaked a failed turn"
        assert "DROPPED" not in app._clipboard, "the app path leaked a cancelled turn"
