"""The blank separator row: the only clipboard evidence the rail was stripped.

Written by QA against this slice and adopted into it. Both defects it names were
reproduced before it was taken, and the mutation matrix in
``test_assistant_rail.py``'s module docstring records what each mutant scores.

A CLIPBOARD-level pin for the de-rail before ``_copy_markdown.align``, and for
the blankness predicate in ``get_selection`` -- the two halves of "the rail must
not reach the aligner" that the geometry tests cannot see.

Why a whole file for one row class. The rail is stripped from the rows before
they reach the aligner; revert that and the aligner reads ``▌`` as a
blockquote bar. Measured on ``MIXED_CONSTRUCTS`` at 100x30:

    rows    ['▌ Here is prose.', '▌', '▌  • alpha item',
             '▌  • beta item', '▌', '▌ ▌ a quoted line']
    correct [0, 1, 2, 3, 4, 5]     <- align over BARE rows
    railed  [0, 0, 2, 3, 3, 5]     <- align over PAINTED rows (the bug)

The two mappings disagree on rows 1 and 4 ONLY — the blank separator rows.
Every row that carries a glyph maps to the same source line either way, so a
clipboard assertion driven from a content row CANNOT fail when the strip is
reverted. That is not a hypothetical: the first version of this slice's own
alignment pin asserted on a bullet row and slept through a coherent revert.

The blank row is the discriminator, and it is also the honest one: under the
railed mapping a blank separator is attributed to the PRECEDING source line, so
a take that includes it inherits that line's text.

These tests use the smallest input that exhibits it — two paragraphs, three
rendered rows — and assert on what ``get_selection`` puts on the CLIPBOARD, not
on a mapping the test recomputed for itself. Verified to go red with
``bare = [row[RAIL_COLS:] for row in rows]`` reverted to ``bare = rows``, at
widths 20, 40, 60, 100 and 150.
"""

from __future__ import annotations

import pytest
from textual.geometry import Offset
from textual.selection import Selection

from local_operator.tui.widgets.assistant import RAIL_COLS
from tests.unit.tui.test_assistant_rail import _block

#: Two paragraphs: the minimum shape with a blank separator row between two
#: source lines. Renders as exactly three rows at every width under test, so
#: the coordinates below need no width-dependent row finding.
TWO_PARAGRAPHS = "Alpha.\n\nBravo.\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [20, 40, 60, 100, 150])
async def test_a_blank_separator_row_copies_as_nothing(width: int) -> None:
    """A take of ONLY the blank row between two paragraphs is empty.

    With the rail left on for the aligner, the blank row aligns to the
    PRECEDING source line and this take pastes ``Alpha.`` — text the reader
    never highlighted, from a row that has no glyphs on it at all.
    """
    block, rows = await _block(TWO_PARAGRAPHS, size=(width, 20))
    bare = [row[RAIL_COLS:] for row in rows]
    blank = next(index for index, row in enumerate(bare) if not row.strip())

    got = block.get_selection(Selection(Offset(0, blank), Offset(len(rows[blank]), blank)))

    text = "" if got is None else got[0].strip()
    assert text == "", (
        f"a blank separator row pasted {text!r}: the aligner is reading the "
        "rail as a blockquote bar and attributing the blank row to the "
        "preceding source line (D5 stripping reverted)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [20, 40, 60, 100, 150])
async def test_the_last_paragraph_copies_as_itself(width: int) -> None:
    """A whole-row take of the second paragraph pastes that paragraph ALONE.

    The positive half of the same defect: with the rail on, row 2 and the blank
    row 1 both align to source line 0, so the take is read as spanning the
    document and pastes ``Alpha.\\n\\nBravo.`` — the whole message for a
    one-row gesture.
    """
    block, rows = await _block(TWO_PARAGRAPHS, size=(width, 20))
    bare = [row[RAIL_COLS:] for row in rows]
    last = max(index for index, row in enumerate(bare) if row.strip())

    got = block.get_selection(Selection(Offset(0, last), Offset(len(rows[last]), last)))

    assert got is not None
    assert got[0].strip() == "Bravo.", (
        f"a one-row take of the last paragraph pasted {got[0]!r} instead of "
        "'Bravo.': the rail is being aligned as a quote bar (D5 reverted)"
    )


@pytest.mark.asyncio
async def test_the_two_mappings_disagree_only_on_blank_rows() -> None:
    """WHY the existing pin sleeps, asserted so it cannot regress unnoticed.

    This is the meta-assertion: on the fixture the existing test uses, the
    railed and bare mappings differ ONLY on blank rows. Any clipboard assertion
    driven from a row with glyphs on it — ``first_bullet`` among them — is
    incapable of failing when D5 is reverted. If this ever stops holding, the
    D5 pins can be simplified; while it holds, a blank-row take is the only
    honest clipboard-level discriminator.
    """
    from local_operator.tui.widgets import _copy_markdown  # noqa: PLC0415
    from tests.unit.tui.test_assistant_rail import MIXED_CONSTRUCTS  # noqa: PLC0415

    block, rows = await _block(MIXED_CONSTRUCTS)
    bare = [row[RAIL_COLS:] for row in rows]

    correct = _copy_markdown.align(block._full_text, bare)
    railed = _copy_markdown.align(block._full_text, rows)

    moved = [index for index, (a, b) in enumerate(zip(correct, railed)) if a != b]
    assert moved, "the rail no longer confuses the aligner; the de-rail is unpinned"
    # The BLANK rows must be among the casualties, because they are what the
    # clipboard pins in this file key on. They were once the ONLY casualties —
    # true while the rail was ``U+258C`` and the aligner read it as a quote bar,
    # which is exactly why an assertion on a content row could sleep through a
    # revert. With the narrower ``U+258E`` the aligner cannot place the railed
    # rows at all and most of them degrade to ``None``, so the hazard is wider
    # now, not narrower. Asserted as "blanks move", not "only blanks move": the
    # second form was a property of the old glyph and would fail here for a
    # reason that is not a regression.
    blanks = [index for index, row in enumerate(bare) if not row.strip()]
    assert blanks, bare
    assert set(blanks) <= set(moved), (
        f"blank rows {blanks} no longer move under the railed mapping (moved: "
        f"{moved}); the clipboard pins in this file may have stopped "
        "discriminating and must be re-derived before they are trusted"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [20, 40, 60, 100])
async def test_a_drag_ending_on_a_blank_row_keeps_its_partial_take(width: int) -> None:
    """The L932 blankness filter, which no test in the slice currently pins.

    ``content`` filters on ``bare[i].strip()``. Reverting that one expression to
    ``rows[i].strip()`` — the railed row, which is NEVER blank — leaves all 28
    tests in ``test_assistant_rail.py`` green AND all 11 D5 pins above green,
    but it is not inert: it changes the CLIPBOARD on 76 of 368 measured
    blank-row takes.

    The shape: a drag that starts mid-word and stops ON the blank separator
    row. Correctly, the blank contributes nothing and the take is the partial
    glyphs the reader lit — ``pha.``. With the filter reading the painted row,
    the blank is admitted to ``content`` as a second element, the take stops
    reading as a single-source-line sub-line take, and the whole source line
    ``Alpha.`` is pasted instead.
    """
    block, rows = await _block("Alpha.\n\nBravo.\n", size=(width, 24))
    bare = [row[RAIL_COLS:] for row in rows]
    blank = next(index for index, row in enumerate(bare) if not row.strip())

    got = block.get_selection(
        Selection(Offset(RAIL_COLS + 2, blank - 1), Offset(len(rows[blank]), blank))
    )

    assert got is not None
    assert got[0] == "pha.", (
        f"a drag ending on the blank separator pasted {got[0]!r} instead of "
        "'pha.': the blankness filter is reading the painted row, so the blank "
        "entered `content` and escalated a sub-line take to a whole source line"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [20, 40, 60, 100])
async def test_a_drag_starting_on_a_blank_row_keeps_its_partial_take(width: int) -> None:
    """The mirror of the above: the drag BEGINS on the blank separator row.

    Correct behaviour pastes only the glyphs lit on the following row
    (``Bra``); with the filter inverted the whole of ``Bravo.`` arrives.
    """
    block, rows = await _block("Alpha.\n\nBravo.\n", size=(width, 24))
    bare = [row[RAIL_COLS:] for row in rows]
    blank = next(index for index, row in enumerate(bare) if not row.strip())

    got = block.get_selection(Selection(Offset(0, blank), Offset(RAIL_COLS + 3, blank + 1)))

    assert got is not None
    assert got[0] == "Bra", (
        f"a drag starting on the blank separator pasted {got[0]!r} instead of "
        "'Bra': the blankness filter is reading the painted row"
    )
