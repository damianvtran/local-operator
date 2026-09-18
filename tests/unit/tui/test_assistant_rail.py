"""The assistant message's gutter rail — the ANSWER's delineation, on every row.

A user prompt has carried a rule down its left edge since ``UserBlock`` grew one
(``test_user_block.py``, which asks exactly these questions of the other rule).
The model's answer carried nothing, so the transcript drew a line beside the
shorter half of the conversation and left the half a reader actually scrolls
back to read unmarked.

**The rail marks the ANSWER, not every assistant block.** It was painted on every
``AssistantBlock`` in PR #1229, which railed the mid-turn progress sentence
exactly as it railed the outcome — the one distinction the mark exists to draw,
and the shape a maintainer reported ("Progress messages shouldn't have the left
bar"). A block the surfaces mark as narration paints no gutter and folds to the
lane's full width, which is the rail-OFF geometry; ``test_rail_toggle.py`` owns
that equivalence. The tests below are written for a RAILED block, which is what
an answer is.

This file holds two invariants, and the second is the expensive one.

**Geometry.** Every row a RAILED block paints carries the rail — wrapped
continuations and the blank rows between paragraphs alike, for the reason
``test_user_block`` states: a marker on one row marks a LINE, a marker on every
row marks a BLOCK. The width is ``SPINE_INDENT``, the same two cells the prompt
rule spends, so the transcript has ONE text origin rather than two.

**The copy path is unchanged by it.** ``▌`` is already the blockquote bar in
``_copy_markdown``, so a rail on every row makes every row look like a quote row
to the aligner, and the rail makes no row ``.strip()``-blank any more. Either
failure is SILENT: the reader lights one thing and pastes another. The tests
under "copy / selection" are the ones that matter, and each is written to go red
against a plausible half-implementation rather than to describe the finished one.

**The mutation matrix, measured — and why SINGLE-SITE mutants are not enough.**
A developer who did not believe the rail needed handling would not delete one
expression; they would write the whole method without it. Scored as
``test_assistant_rail.py`` / ``test_copy_blank_row.py``:

===========================================  =========  ==========
mutant                                       this file  blank_row
===========================================  =========  ==========
A  ``bare = rows`` (one line)                3 red      8 red
B  A + the three ``RAIL_COLS`` compensations 3 red      8 red
C  ``content`` filters ``rows[i].strip()``   green      8 red
D  ``copy_gutter`` returns 0                 5 red      5 red
E  ``RAIL`` back to ``U+258C``               3 red      green
F  the quote-row reservation, restored       4 red      green
===========================================  =========  ==========

Row F USED to read "``_quote_owns_row`` returns ``False``", scoring 2 red, and
it pinned the opposite ruling: that a quote row reserves the rail's cells and
paints nothing into them. PR #1229 reversed that (see
``test_a_quoted_row_carries_both_the_rail_and_the_quote_bar``), the function is
gone, and F is now the FAITHFUL REVERT of the deletion — re-add
``_quote_owns_row`` and its ``blank_gutter`` branch in ``rail_rows`` by hand.
Measured at 4 red in this file, and the honest revert shape rather than a
one-line mutation, because a developer who disagreed with the reversal would
restore the mechanism whole rather than flip one expression.

B is the honest revert and it is the one that caught this file out: the
alignment pin originally asserted on a BULLET row, went red under A, and passed
under B. On ``MIXED_CONSTRUCTS`` under the OLD ``U+258C`` rail the two mappings
differed on rows 1 and 4 only — both blank — so no assertion on a content row
could ever fail. C is why the adopted file exists: it moves 76 of 368 measured
blank-row takes on the clipboard while every geometry test here stays green.

E and F pin the design round's two rulings, and both are about how the rail
RELATES to marks it sits beside rather than about the rail alone — which is the
class of defect the colour and geometry tests above are structurally unable to
see, since each measures the rail against the background or against itself.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.content import Content
from textual.geometry import Offset
from textual.selection import Selection

from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import _copy_markdown
from local_operator.tui.widgets.assistant import (
    MIN_BODY,
    QUOTE_BAR,
    RAIL,
    RAIL_COLS,
    RAIL_TOKEN,
    AssistantBlock,
)
from local_operator.tui.widgets.subagent_view import (
    SubagentEntry,
    SubagentView,
    entry_block,
)
from local_operator.tui.widgets.transcript import (
    SPINE_INDENT,
    TranscriptView,
    UserBlock,
)
from tests.unit.tui.conftest import StyledTranscriptApp
from tests.unit.tui.test_band_panels import FakeSession
from tests.unit.tui.test_subagent_view import (
    TRAJECTORY,
    _async_factory,
    _fake_jobs,
    _job_with,
)

THREE_PARAGRAPHS = (
    "The ingest path reads each source once and writes a manifest.\n"
    "\n"
    "Failures are retried with backoff, and a source that fails three times is "
    "quarantined rather than dropped, so nothing is lost silently.\n"
    "\n"
    "The manifest is what the reconciler reads on the next pass."
)

#: Paragraph, then a bullet list, then a blockquote — the exact shape that broke
#: the aligner. The blockquote is the point: its ``▌`` is CONTENT, painted by
#: Rich from the model's ``>``, and it sits one row after a blank separator so a
#: mismeasured ``opens_line`` shows up here first.
MIXED_CONSTRUCTS = "Here is prose.\n\n- alpha item\n- beta item\n\n> a quoted line\n"


def _rendered(block: AssistantBlock) -> list[str]:
    """The rows the block PAINTED, from the same content the selection sees.

    Read through ``_render()`` rather than ``renderable`` because that is the
    surface ``get_selection`` indexes into: a row list taken from anywhere else
    could agree with the assertions here while disagreeing with the clipboard,
    which is the only disagreement that matters in this file.
    """
    visual = block._render()
    # Narrowed with an assert rather than a cast, the way ``_rendered_rows`` in
    # ``test_transcript_selection`` does it: the type is the assumption every
    # row assertion in this file rests on, so a block that started returning
    # something else should fail here and say so.
    assert isinstance(visual, Content)
    return visual.plain.split("\n")


async def _block(
    text: str, size: tuple[int, int] = (100, 30), *, narration: bool = False
) -> tuple[AssistantBlock, list[str]]:
    """A finalized assistant block at ``size``, and its painted rows.

    Two pauses after the text, as ``test_user_block._rows`` does it: the first
    mounts and lays the block out, the second lets the resize re-fold it against
    its REAL width. Without the second, every assertion here would be made
    against the 80-column fallback rather than the frame under test.

    ``narration`` marks the block BEFORE its text, which is the order both
    production paths use and the only one that produces a settled frame: the
    mark is read at paint rate, so a mark set after ``finalize_text`` would leave
    the committed rows carrying a rail nothing would repaint away.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=size) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        if narration:
            block.mark_narration()
        block.update_text(text)
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()
        return block, _rendered(block)


def _luminance(value: str) -> float:
    """WCAG relative luminance, the same helper ``test_user_block`` L338 uses."""
    raw = value.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(fg: str, bg: str) -> float:
    first, second = _luminance(fg), _luminance(bg)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def _furniture_by_row(block: AssistantBlock, rows: list[str]) -> list[int]:
    """Per-row furniture in BARE columns, measured the way the block measures it.

    Goes through the block's own ``_furniture_width`` against rail-stripped rows,
    so it reproduces the production call exactly rather than re-deriving an
    answer that could be right while the block's is wrong.
    """
    bare = [row[RAIL_COLS:] for row in rows]
    mapping = _copy_markdown.align(block._full_text, bare)
    return [block._furniture_width(bare, mapping, index) for index in range(len(bare))]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (60, 40)])
async def test_every_row_of_a_multi_paragraph_message_carries_the_rail(
    size: tuple[int, int],
) -> None:
    """The invariant, at both widths the treatment was designed against.

    Asserted over EVERY row rather than as ``RAIL in text``: the latter passes
    when a single row carries the gutter, which is the line-marking failure this
    rail exists to avoid, wearing the right glyph.
    """
    _, rows = await _block(THREE_PARAGRAPHS, size)
    assert len(rows) >= 5
    assert all(row.startswith(RAIL) for row in rows), rows


@pytest.mark.asyncio
async def test_the_blank_row_between_paragraphs_carries_the_rail_and_nothing_else() -> None:
    """The case that decides block-marking versus line-marking.

    A blank line between paragraphs must still paint a row and that row must be
    the rail alone. Skipping it breaks the rail into one segment per paragraph —
    the same "three separate things" reading the treatment was built to fix.
    """
    _, rows = await _block("first\n\nsecond\n\nthird", (120, 40))
    # Spelled from the constant: a test that hard-codes the glyph asserts about
    # whichever glyph was current when it was written, which is how a treatment
    # change turns into a test edit instead of a design decision.
    assert [row.rstrip() for row in rows] == [
        f"{RAIL} first",
        RAIL,
        f"{RAIL} second",
        RAIL,
        f"{RAIL} third",
    ]


@pytest.mark.asyncio
async def test_the_prose_starts_in_the_same_column_on_every_row() -> None:
    """One text origin, wrapped continuations included.

    The rail is worth nothing if the prose beside it starts at a different
    column depending on whether the row opened a paragraph or continued one.
    """
    _, rows = await _block(THREE_PARAGRAPHS, (60, 40))
    for row in rows:
        assert row.startswith(RAIL)
        # The cells between the glyph and the prose are pad, not content: the
        # gutter is RAIL_COLS wide whatever the row carries.
        assert len(row[:RAIL_COLS]) == RAIL_COLS
        body = row[RAIL_COLS:]
        assert body == body.lstrip() or not body.strip(), row


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [40, 60, 80, 120])
async def test_a_narrow_frame_keeps_every_word_and_never_overhangs(width: int) -> None:
    """The test that fails without the width subtraction (D4).

    Without ``_flat_width`` taking the rail's cells off the fold, the markdown
    is folded for the full lane and then has two cells pushed onto it, so every
    wrapped row overhangs the block by two. Asserted as "no painted row is wider
    than the frame", which is the reader-visible form of that bug.
    """
    # The frame is read INSIDE the running app: ``region`` collapses to zero
    # once the pilot's context exits, which would make a width assertion taken
    # afterwards vacuously true.
    app = StyledTranscriptApp()
    async with app.run_test(size=(width, 40)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text(THREE_PARAGRAPHS)
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()
        rows = _rendered(block)
        frame = block.region.width

        assert frame > 0, "the block never got a real width"
        for row in rows:
            assert len(row.rstrip()) <= frame, (width, frame, row)
        # And the words survive the narrower fold: a subtraction that "fixed"
        # the overhang by truncating would pass the assertion above.
        painted = " ".join(row[RAIL_COLS:] for row in rows).split()
        assert "quarantined" in painted


@pytest.mark.asyncio
async def test_resizing_rewraps_without_losing_the_rail() -> None:
    """120 -> 60 -> 120: the rail survives a re-fold in both directions."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text(THREE_PARAGRAPHS)
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()
        wide = _rendered(block)
        wide_built = block._built_width

        app.console.width = 60
        await pilot.resize_terminal(60, 40)
        await pilot.pause()
        await pilot.pause()
        narrow = _rendered(block)

        assert len(narrow) > len(wide), "the text never re-folded"
        assert all(row.startswith(RAIL) for row in narrow), narrow
        assert block._built_width < wide_built, "the build width did not track the resize"

        await pilot.resize_terminal(120, 40)
        await pilot.pause()
        await pilot.pause()
        again = _rendered(block)
        assert all(row.startswith(RAIL) for row in again), again
        assert len(again) == len(wide), "the re-widened fold did not return"


@pytest.mark.asyncio
async def test_the_lane_walk_folds_the_prose_inside_the_rail() -> None:
    """``authored_width``'s half of D4, driven through the CONTAINER's walk.

    The block has two rebuild triggers and they must name one width. Its own
    ``on_resize`` goes through ``_flat_width``; the container's lane walk
    (``TranscriptView._refit_authored_blocks``) instead asks each block what
    width to rebuild at for a published lane, precisely so a block whose box is
    not the lane can say so. Without the override the walk hands over the whole
    lane, the prose is folded two cells wider than the box the rail leaves it,
    and every wrapped row overhangs.

    Driven by calling the walk with a lane the block has NOT just been resized
    to, which is the only way to exercise the override rather than
    ``_flat_width``: a plain resize repairs the width through the other path and
    hides the missing override entirely.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text(THREE_PARAGRAPHS)
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()

        # A narrower lane than the block's box, as a sidebar opening publishes.
        lane = 60
        assert block.authored_width(lane) == lane - RAIL_COLS, (
            "the block answered the lane unchanged, so the walk will fold its "
            "prose wider than the box the rail leaves"
        )
        view._refit_authored_blocks(lane)
        await pilot.pause()

        rows = _rendered(block)
        assert all(row.startswith(RAIL) for row in rows), rows
        for row in rows:
            assert len(row.rstrip()) <= lane, (lane, row)


def test_the_rail_matches_the_user_rule_width() -> None:
    """Geometry mirrors the prompt rule; the colour deliberately does not.

    Both read from ``SPINE_INDENT`` rather than from the literal 2, so this
    pins that they share an ORIGIN — two gutters that agree by coincidence
    drift the first time one is tuned.
    """
    assert RAIL_COLS == SPINE_INDENT
    assert RAIL_COLS == UserBlock.RULE_COLS
    assert MIN_BODY == UserBlock.MIN_BODY


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------


def test_the_rail_is_not_the_user_rule_colour() -> None:
    """The verdict's whole point, in both ramps.

    ``signal`` — the prompt's rule — was rendered for this and REJECTED in
    review of the frames precisely because it makes a user block and an
    assistant block look alike, which removes the distinction the rail is for.
    """
    for ramp in ("dark", "light"):
        rail = theme_mod.semantic_color(RAIL_TOKEN, ramp)
        rule = theme_mod.semantic_color(UserBlock.RULE_TOKEN, ramp)
        assert rail != rule, (ramp, rail, rule)


def test_the_rail_never_spends_the_accent() -> None:
    """``accent`` is a closed five-site budget and this is not one of them."""
    assert RAIL_TOKEN != "accent"
    for ramp in ("dark", "light"):
        assert theme_mod.semantic_color(RAIL_TOKEN, ramp) != theme_mod.semantic_color(
            "accent", ramp
        )


def test_the_rail_is_told_from_the_user_rule_without_colour() -> None:
    """The property the whole treatment exists for, on a NON-COLOUR channel.

    This is the assertion the design round turned on, and its absence is why a
    FAIL was possible with every colour test green. The 54-theme sweep measured
    the rail against the BACKGROUND — legibility — which is a different question
    from the one the user asked: can I tell the model's line from my own?

    Measured, colour answers that with nothing. ``label`` and ``signal`` are
    effectively ISOLUMINANT: 1.08:1 against each other on ``dark``, 1.00:1 on
    ``tokyo-night-day`` — identical luminance — and below 1.5:1 in 46 of the 54
    themes. Rendered in greyscale, or to a reader with any of the common colour
    vision deficiencies (simulation never exceeded 1.24:1), the two bars were
    the same mark. The one-line case was the proof: a short user message above a
    short assistant reply, same glyph, same column, same weight, alternating.

    So the distinction is carried by GLYPH WIDTH, which no rendering can
    collapse: ``U+258E`` quarter block against ``U+258C`` half block. This test
    asserts the channel exists and is theme-INDEPENDENT — a glyph is not a
    palette entry, so one assertion covers all 54 — and deliberately does NOT
    assert any luminance relationship, because relying on one is the defect.
    """
    assert RAIL != UserBlock.RULE, (
        "the rail and the user rule are the same glyph again; with the two "
        "tokens isoluminant, colour cannot carry this distinction on its own"
    )
    # Same reserved width, different ink coverage: the lane geometry is shared
    # (one text origin) while the mark is not.
    assert RAIL_COLS == UserBlock.RULE_COLS
    assert cell_len(RAIL) == cell_len(UserBlock.RULE) == 1

    # And the channel survives every theme, because it is not a colour at all.
    for name in theme_mod.available_themes():
        rail = theme_mod.semantic_color(RAIL_TOKEN, name)
        rule = theme_mod.semantic_color(UserBlock.RULE_TOKEN, name)
        # Stated as a FACT about this palette rather than as a requirement: the
        # inks may be near-identical in luminance and that is tolerable now,
        # precisely because nothing depends on their difference.
        assert isinstance(rail, str) and isinstance(rule, str), name
    assert RAIL != UserBlock.RULE, "glyph distinction must not be theme-dependent"


@pytest.mark.asyncio
async def test_a_quoted_row_carries_both_the_rail_and_the_quote_bar() -> None:
    """A quote row carries TWO marks, and that is the nesting — not a collision.

    **This test asserted the opposite until PR #1229's review, and the reversal
    is deliberate. Read this before you change it back.**

    Design round 1 ruled that a quote row must RESERVE the rail's two cells and
    paint nothing into them. That ruling was correct for the tree it was made
    in: the rail and Rich's blockquote bar were the SAME glyph, ``U+258C``, in
    the same ``label`` token, landing one space apart. Two identical glyphs,
    identically filled and identically inked, read as a double-paint glitch
    rather than as nesting — nesting is legible only when something differs, and
    there nothing did.

    Design finding D2 (isoluminance) then moved the rail to ``U+258E``, a
    quarter block, so the distinction would survive greyscale and CVD. That
    killed the collision rationale: a quarter block at painted column 0 beside a
    half block at painted column 2 cannot read as one glyph painted twice. What
    the reservation left instead was a HOLE — the gutter column breaking and
    restarting around every quote — which is what the maintainer reported on
    PR #1229 ("the left bar should be uniform").

    So the rail now runs through quote rows and the quote's bar sits beside it,
    OUTSIDE-IN: block gutter first, then the quote's own mark. The rows above
    and below are still asserted, but they now prove CONTINUITY rather than
    bracketing a hole.
    """
    block, rows = await _block(
        "Intro line here.\n\n> quoted line one\n> quoted line two\n\nAfter the quote.\n",
        (60, 24),
    )
    quoted = [index for index, row in enumerate(rows) if QUOTE_BAR in row]
    assert quoted, rows

    for index in quoted:
        row = rows[index]
        assert RAIL in row, f"row {index} lost the rail beside the quote bar: {row!r}"
        assert row.count(QUOTE_BAR) == 1, row
        # The rail opens the row, in the same column every other row opens in:
        # the gutter is PAINTED now, not reserved blank.
        assert row.startswith(RAIL), row
        # Outside-in. The rail belongs to the block and the bar to the quote
        # nested inside it, so their order is the nesting the maintainer asked
        # for — a bar OUTSIDE the rail would be a different, wrong picture.
        assert row.index(RAIL) < row.index(QUOTE_BAR), row

    # The column is unbroken around it: the rail runs above and below.
    above = max((i for i in range(min(quoted)) if rows[i].strip()), default=None)
    below = next((i for i in range(max(quoted) + 1, len(rows)) if rows[i].strip()), None)
    assert above is not None and below is not None, (quoted, rows)
    assert rows[above].startswith(RAIL), rows[above]
    assert rows[below].startswith(RAIL), rows[below]


@pytest.mark.asyncio
async def test_every_row_including_a_quote_row_carries_the_rail() -> None:
    """The uniform rail, on the shape that has one of everything (T1-a).

    The necessity test for PR #1229's deletion. ``MIXED_CONSTRUCTS`` is prose, a
    bullet list and a blockquote separated by blank rows, so one render exercises
    every row kind the block paints: content rows, wrapped continuations, the
    blank separators, a list row, and the quote row that used to be the
    exception. The claim is that NO row kind is exempt any more.

    Asserted PER ROW with the offenders in the message rather than as
    ``RAIL in text``: a membership test on the whole block passes while any
    single row is missing its rail, which is the only failure this can actually
    have. The faithful revert it is written against is the restoration of
    ``_quote_owns_row`` and its ``blank_gutter`` branch (matrix row F), under
    which the quote row fails ``startswith(RAIL)``.
    """
    _, rows = await _block(MIXED_CONSTRUCTS, (60, 24))
    # The quote row must actually be present, or this proves nothing about the
    # case it exists for.
    assert any(QUOTE_BAR in row for row in rows), rows

    bare = [row for row in rows if row.strip()]
    assert bare, rows
    missing = [
        (index, row) for index, row in enumerate(rows) if row.strip() and not row.startswith(RAIL)
    ]
    assert not missing, f"rows without the rail: {missing!r}\nall rows: {rows!r}"


@pytest.mark.asyncio
async def test_the_de_rail_slice_is_unchanged_by_a_railed_quote_row() -> None:
    """The copy path still lands after the quote row started painting (T1-c).

    This is the re-proof the deletion needs and the one that would catch a real
    break. ``get_selection`` slices a FIXED ``RAIL_COLS`` off every row to get
    back to bare columns, and quote rows used to have two SPACES there while now
    they have a glyph plus a pad. Slicing a fixed two cells is blind to what is
    in them — so ``bare`` is unmoved and the alignment, the furniture
    measurement and the blankness predicates all read exactly what they read
    before. That argument is sound, and it is still an argument; this pins it.

    Two blockquotes so a quote row is not also the last row, which is the case
    where an off-by-one in the walk hides.
    """
    source = (
        "Here is prose.\n\n- alpha item\n- beta item\n\n> first quoted line\n\n"
        "Between the quotes.\n\n> second quoted line\n\nAfter both.\n"
    )
    block, rows = await _block(source, (60, 30))

    bare = [row[RAIL_COLS:] for row in rows]
    mapping = _copy_markdown.align(block._full_text, bare)

    # Every row carrying text is PLACED. A ``None`` on a content row is the
    # aligner having lost the source line, which is what reading the rail as a
    # quote bar would cause.
    unplaced = [
        (index, bare[index])
        for index in range(len(bare))
        if bare[index].strip() and mapping[index] is None
    ]
    assert not unplaced, f"content rows the aligner could not place: {unplaced!r}"

    # The quote rows map to the source's own ``>`` lines rather than drifting
    # onto a neighbour.
    src_lines = source.split("\n")
    quote_rows = [index for index, row in enumerate(rows) if QUOTE_BAR in row]
    assert len(quote_rows) >= 2, rows
    for index in quote_rows:
        placed = mapping[index]
        assert placed is not None, (index, bare[index])
        assert src_lines[placed].lstrip().startswith(">"), (
            f"quote row {index} ({bare[index]!r}) mapped to {src_lines[placed]!r}, "
            "which is not a quote line"
        )
        # The bar is furniture the copy must strip, measured in BARE columns.
        # Zero here would mean the aligner stopped seeing the quote at all.
        assert (
            block._furniture_width(bare, mapping, index) > 0
        ), f"quote row {index} measured no furniture; the bar would reach the clipboard"

    # ...and the real path agrees: a drag across the whole block puts markdown
    # on the clipboard, not furniture.
    copied = _whole_message(block, rows)
    assert copied is not None
    assert RAIL not in copied, copied
    assert QUOTE_BAR not in copied, copied
    assert "> first quoted line" in copied, copied
    assert "> second quoted line" in copied, copied


def test_the_rail_stays_legible_in_every_theme() -> None:
    """3:1 against the ground — the WCAG floor for a non-text graphical object.

    Across every REGISTERED theme, not just the two default ramps: ``label``
    moves per palette, and a rail the reader cannot see costs a column and
    delivers no delineation. The floor is asserted rather than the measured
    worst case (``tokyo-night-day`` at 4.32) because the ramp may legitimately
    move; a failure here after neither this file nor the token changed means the
    palette moved, and the fix is in ``theme.py``.
    """
    worst: tuple[float, str] | None = None
    for name in theme_mod.available_themes():
        ratio = _contrast(
            theme_mod.semantic_color(RAIL_TOKEN, name),
            theme_mod.semantic_color("bg", name),
        )
        if worst is None or ratio < worst[0]:
            worst = (ratio, name)
        assert ratio >= 3.0, (name, ratio)
    assert worst is not None


def test_the_rail_is_never_the_prompt_rule_in_any_theme() -> None:
    """Not just the default ramps: no palette may collapse the two inks."""
    for name in theme_mod.available_themes():
        rail = theme_mod.semantic_color(RAIL_TOKEN, name)
        rule = theme_mod.semantic_color(UserBlock.RULE_TOKEN, name)
        assert rail != rule, (name, rail)


@pytest.mark.asyncio
async def test_a_theme_change_reinks_the_rail() -> None:
    """``retheme`` re-enters ``_apply_rows``, so the colour re-resolves (D6).

    Pins that the style is resolved at PAINT time rather than cached on the
    instance — a ``Style`` held on the block is exactly what would make a theme
    change leave a stale rail behind.
    """

    def rail_colour(block: AssistantBlock) -> str | None:
        renderable = block.renderable
        assert isinstance(renderable, Text)
        for span in renderable.spans:
            if span.start == 0:
                # A span's style is a ``Style`` or the name of one; only the
                # former carries the resolved colour this test compares.
                style = span.style
                if isinstance(style, Style) and style.color is not None:
                    return str(style.color)
        return None

    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 30)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text("Some prose for the rail to run beside.")
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()
        before = rail_colour(block)

        # A theme whose `label` genuinely differs from the starting ramp's.
        start = theme_mod.semantic_color(RAIL_TOKEN)
        target = next(
            name
            for name in theme_mod.available_themes()
            if theme_mod.semantic_color(RAIL_TOKEN, name) != start
        )
        theme_mod.set_theme(target)
        block.retheme()
        await pilot.pause()
        after = rail_colour(block)

        assert before is not None and after is not None
        assert before != after, (before, after, target)
        assert all(row.startswith(RAIL) for row in _rendered(block))


# --------------------------------------------------------------------------
# Copy / selection — the silent-failure surface
# --------------------------------------------------------------------------


def _whole_message(block: AssistantBlock, rows: list[str]) -> str | None:
    selection = Selection(Offset(0, 0), Offset(len(rows[-1]), len(rows) - 1))
    got = block.get_selection(selection)
    return None if got is None else got[0]


@pytest.mark.asyncio
async def test_a_copy_excludes_the_rail() -> None:
    """A copied message pastes the model's markdown, gutter-free.

    The rail is chrome the app painted; the blockquote's bar on the last line is
    CONTENT the model wrote. Asserted as "one ``▌`` survives, and it is the
    quote's" rather than "no ``▌`` at all", which would pass an implementation
    that over-strips and silently eats the quote.
    """
    block, rows = await _block(MIXED_CONSTRUCTS)
    copied = _whole_message(block, rows)
    assert copied is not None
    assert RAIL not in copied, copied
    assert "- alpha item" in copied
    assert "> a quoted line" in copied


@pytest.mark.asyncio
async def test_the_alignment_is_unchanged_by_the_rail() -> None:
    """The direct pin for D5: the aligner must see the rows Rich folded.

    ``▌`` is the blockquote bar in ``_copy_markdown``, so aligning the RAILED
    rows makes every row look like a quote row. Measured on this exact source,
    railed rows map to ``[0, 0, 2, 3, 3, 5]`` where the correct mapping is
    ``[0, 1, 2, 3, 4, 5]`` — rows collapse onto the wrong source lines and a
    copy pastes the wrong text with no visible symptom.

    **Asserted on the BLANK separator rows, and that is the whole point.** The
    two mappings above differ on rows 1 and 4 only — both blank. Every row
    carrying a glyph maps to the same source line either way, so a clipboard
    assertion driven from a content row cannot fail when the strip is reverted.
    An earlier version of this test took a bullet row and passed against a
    coherent revert of all four de-rail sites; only a single-site mutant caught
    it, which is exactly the false clearance single-site mutation testing gives.

    Under the railed mapping a blank row is attributed to the PRECEDING source
    line, so a take of it inherits that line's text: row 1 pastes
    ``Here is prose.`` — a row with no glyphs on it producing a sentence — and
    row 4 pastes ``- beta item``. Fifteen characters of gesture, and it is the
    only clipboard-level evidence the aligner ever saw the rail.

    ``tests/unit/tui/test_copy_blank_row.py`` holds the parametrised form of
    this across widths; this stays here because the file that claims to pin the
    alignment must be the file that actually does.
    """
    block, rows = await _block(MIXED_CONSTRUCTS)
    bare = [row[RAIL_COLS:] for row in rows]

    correct = _copy_markdown.align(block._full_text, bare)
    railed = _copy_markdown.align(block._full_text, rows)

    # The hazard is real at this width, or the rest of this test proves nothing:
    # the two mappings must genuinely disagree.
    assert railed != correct, (
        "the rail no longer confuses the aligner; this test has stopped "
        "discriminating and the D5 stripping is now unpinned"
    )
    assert correct == list(range(len(bare))), correct

    # Asserted against what PRODUCTION puts on the clipboard, not against a
    # mapping this test computed for itself: recomputing both mappings here and
    # comparing them passes against any implementation, including one that
    # aligns the railed rows, because neither number ever came from the block.
    blanks = [index for index, row in enumerate(bare) if not row.strip()]
    assert blanks, bare
    for blank in blanks:
        got = block.get_selection(Selection(Offset(0, blank), Offset(len(rows[blank]), blank)))
        pasted = "" if got is None else got[0].strip()
        assert pasted == "", (
            f"row {blank} has no glyphs on it but pasted {pasted!r}: the block "
            "aligned its rows with the rail on, so a blank separator was "
            "attributed to the preceding source line"
        )


@pytest.mark.asyncio
async def test_the_furniture_survives_a_blank_separator_row() -> None:
    """The failure ``align`` alone does NOT catch: blankness predicates invert.

    A railed separator row is ``'▌'`` plus pad, which is TRUTHY under
    ``.strip()``. Every blankness test in the copy path therefore inverts once
    the rail is painted, and the load-bearing one is ``_furniture_width``'s scan
    for the previous PAINTED row: it skips blank rows precisely so a separator
    does not make the row after it read as opening its source line. With no row
    ever blank, ``previous`` becomes the immediately preceding row, ``opens_line``
    flips after every paragraph break, and a list row's furniture is measured as
    indent instead of a marker — the issue #395 class.

    Routing only ``align`` and ``_furniture_width`` through de-railed rows
    passes ``test_the_alignment_is_unchanged_by_the_rail`` while leaving this
    broken, which is why this test exists separately. The rows immediately AFTER
    each blank separator are the assertion.
    """
    block, rows = await _block(MIXED_CONSTRUCTS)
    bare = [row[RAIL_COLS:] for row in rows]
    mapping = _copy_markdown.align(block._full_text, bare)

    measured = _furniture_by_row(block, rows)

    # The pre-rail answer, computed with no block involved: furniture_width over
    # the bare rows with `opens_line` derived from the frame the way the block
    # derives it.
    lines = block._full_text.split("\n")
    covered, _ = _copy_markdown.classify(lines)
    expected: list[int] = []
    for index, row in enumerate(bare):
        source = mapping[index] if index < len(mapping) else None
        previous: int | None = None
        for back in range(index - 1, -1, -1):
            if bare[back].strip():
                previous = mapping[back] if back < len(mapping) else None
                break
        expected.append(
            _copy_markdown.furniture_width(
                row,
                lines[source] if source is not None and source < len(lines) else None,
                opens_line=source != previous,
                fenced=source is not None and source in covered,
            )
        )

    assert measured == expected, (measured, expected)
    # The list rows are the ones that carry a painted marker, and row 2 is the
    # one immediately after a blank separator — the row the inverted predicate
    # mismeasures. Stated explicitly so a future reader sees the case.
    bullets = [index for index, row in enumerate(bare) if "•" in row]
    assert bullets, bare
    assert all(measured[index] > 0 for index in bullets), (measured, bare)


@pytest.mark.asyncio
async def test_a_bullet_after_a_blank_row_copies_its_marker_not_a_mismeasure() -> None:
    """The blankness inversion, observed on the CLIPBOARD rather than asserted.

    This is the test that fails on the half-fix, and the half-fix is the
    plausible one: route ``align()`` through de-railed rows — which is what D5
    literally asks for — and leave ``_furniture_width``, its previous-painted-row
    scan, and the ``content`` blankness filter reading the PAINTED rows. The
    alignment is then correct and every geometry assertion still passes, so
    nothing above catches it.

    What breaks is the scan at the heart of ``_furniture_width``: it walks back
    for the previous row that PAINTS something, deliberately skipping blanks so a
    paragraph separator cannot make the row after it read as opening its source
    line. With the rail on every row nothing is ever blank, the separator becomes
    the previous painted row, ``opens_line`` flips, and the bullet row is
    measured as a wrapped continuation — furniture of pure indent — instead of a
    marker row.

    Measured: a drag from inside the bullet's furniture pastes ``' alpha item'``,
    the item's text with the marker replaced by a stray leading space, where the
    correct answer is the markdown line ``'- alpha item'``. A reader lit a bullet
    and pasted something that is no longer a list item, which is the issue #395
    harm class in new clothes.
    """
    block, rows = await _block("Here is prose.\n\n- alpha item\n- beta item\n")
    bare = [row[RAIL_COLS:] for row in rows]

    separator = next(index for index, row in enumerate(bare) if not row.strip())
    bullet = next(index for index, row in enumerate(bare) if "alpha" in row and index > separator)
    assert bullet == separator + 1, (
        "the bullet must sit immediately after the blank separator, or the "
        "previous-painted-row scan is not exercised"
    )

    # Started INSIDE the painted furniture — past the rail, on the marker's
    # cells. That is the column whose answer the mismeasure changes.
    selection = Selection(Offset(RAIL_COLS + 2, bullet), Offset(len(rows[bullet].rstrip()), bullet))
    got = block.get_selection(selection)
    assert got is not None
    assert got[0].strip() == "- alpha item", (
        f"the bullet after a blank separator was measured as a continuation: " f"pasted {got[0]!r}"
    )


@pytest.mark.asyncio
async def test_a_blank_separator_row_is_still_blank_to_the_copy_path() -> None:
    """The other half of the inversion, asserted where a reader can see it.

    A whole-message drag spans the separator rows. If those rows stop reading as
    blank they enter ``content``, and the first or last element of the selection
    can then be a row with no glyphs — which decides ``starts_full`` /
    ``ends_full`` and so decides whether the copy is the markdown source or a
    run of rendered glyphs.
    """
    block, rows = await _block(MIXED_CONSTRUCTS)
    bare = [row[RAIL_COLS:] for row in rows]
    assert not all(row.strip() for row in bare), "the fixture has no blank separator row"
    assert all(row.strip() for row in rows), "the rail should make every PAINTED row non-blank"

    copied = _whole_message(block, rows)
    assert copied is not None
    # The markdown path, not the glyph path: blank rows treated as content would
    # push this take down the rendered-glyph branch and paste bullets as `•`.
    assert "- alpha item" in copied and "•" not in copied, copied


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   \n  \n", "\n\n"])
async def test_an_empty_message_copies_as_nothing_not_as_a_rail(text: str) -> None:
    """The one path where ``copy_gutter`` is the ONLY thing stripping the rail.

    A message with no printable content short-circuits to
    ``TranscriptBlock.get_selection`` (the ``not self._full_text.strip()``
    guard), which never reaches this block's own bare-column accounting and
    clamps past :meth:`copy_gutter` instead. The block still PAINTS a row — the
    rail goes on every row, including the only row an empty message has — so
    without the override that row copies as ``▌``: a reader drags over what
    looks like empty space and pastes a glyph that is nowhere in the document.

    Reviewer finding: ``copy_gutter`` returning ``RAIL_COLS`` had no test that
    failed without it, because every other take goes through ``get_selection``'s
    own de-rail and never consults it.
    """
    block, rows = await _block(text)
    assert rows == [RAIL] or rows == [RAIL + " "], rows

    got = block.get_selection(Selection(Offset(0, 0), Offset(len(rows[0]), 0)))

    pasted = "" if got is None else got[0]
    assert pasted.strip() == "", (
        f"an empty message pasted {pasted!r}: the rail reached the clipboard "
        "through the empty-text fallback, which clamps past `copy_gutter` "
        "rather than through this block's own stripping"
    )


@pytest.mark.asyncio
async def test_a_quoted_line_still_copies_its_own_bar() -> None:
    """The model's quote bar is content; only the block's gutter is chrome.

    Guards over-stripping: an implementation that removes every bar from the row
    would pass ``test_a_copy_excludes_the_rail`` and lose the quote.

    The quote's bar is ``QUOTE_BAR`` and the rail is ``RAIL`` — different
    glyphs, in different columns. The row carries both: the block's rail in the
    gutter and exactly ONE quote bar, which is the model's. Counting the bar is
    what guards this test's real subject — an implementation that double-painted
    the quote's own mark, or dropped it, would fail here.

    (This paragraph used to assert the rail was ABSENT on a quote row. That was
    design round 1's reservation ruling, reversed in PR #1229 — see
    ``test_a_quoted_row_carries_both_the_rail_and_the_quote_bar`` for why. The
    copy assertion below is what this test exists for and is unchanged by it.)
    """
    block, rows = await _block("> a quoted line\n")
    quote_rows = [row for row in rows if QUOTE_BAR in row]
    assert quote_rows, rows
    assert all(row.count(QUOTE_BAR) == 1 for row in quote_rows), (
        "a quote row must carry exactly one quote bar, the model's",
        quote_rows,
    )
    assert all(RAIL in row for row in quote_rows), (
        "a quote row carries the block's rail beside the quote's own bar",
        quote_rows,
    )
    copied = _whole_message(block, rows)
    assert copied is not None
    assert copied.strip() == "> a quoted line"


@pytest.mark.asyncio
async def test_a_sub_line_selection_starts_past_the_rail() -> None:
    """Column 0 and column ``RAIL_COLS`` are the same take (design round 1, D1).

    A whole-row gesture must read as whole whether or not the reader's drag
    began on the gutter cell, or the same gesture one cell left would fall to a
    different branch and paste a different document.
    """
    block, rows = await _block("Here is a single prose line.\n")
    end = len(rows[0].rstrip())
    from_zero = block.get_selection(Selection(Offset(0, 0), Offset(end, 0)))
    from_rail = block.get_selection(Selection(Offset(RAIL_COLS, 0), Offset(end, 0)))
    assert from_zero is not None and from_rail is not None
    assert from_zero[0] == from_rail[0], (from_zero[0], from_rail[0])
    assert RAIL not in from_zero[0]


@pytest.mark.asyncio
async def test_a_fenced_code_line_copies_byte_for_byte() -> None:
    """Code is returned verbatim, ``▌`` and leading digits included.

    ``furniture_width`` refuses to strip inside a fence because a code line that
    starts with a digit is indistinguishable from a rendered ordered marker by
    glyph alone. The rail must not become a second way to eat a code byte.
    """
    source = "```\n\u258c literal bar\n1 / 0\n```\n"
    block, rows = await _block(source)
    copied = _whole_message(block, rows)
    assert copied is not None
    assert "\u258c literal bar" in copied, copied
    assert "1 / 0" in copied, copied


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_frozen_prefix_never_holds_the_rail() -> None:
    """D8, proved rather than asserted: the cache can only hold bare prose.

    ``_flat_rows`` caches ``_frozen_flat`` and CONCATENATES it with a freshly
    flattened tail on every delta. A rail baked into that prefix would be railed
    again on the next flush, growing by two cells per delta — ``▌ ▌ text``. The
    rail is applied in ``_apply_rows``, to the assembled output, which the cache
    never passes through.

    Streamed past a blank-line boundary so the prefix genuinely populates, then
    flushed three more times: the assertion is that no row ever carries two
    gutters, which is what a reader would see.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 30)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()

        block.update_text("First settled paragraph.\n\nSecond ")
        await pilot.pause()
        assert block._frozen_flat is not None, "the frozen prefix never populated"
        assert RAIL not in block._frozen_flat.plain, block._frozen_flat.plain

        for delta in ("paragraph ", "growing ", "steadily."):
            block.update_text(block.text() + delta)
            await pilot.pause()
            assert block._frozen_flat is not None
            assert RAIL not in block._frozen_flat.plain, block._frozen_flat.plain
            for row in _rendered(block):
                assert not row.startswith(RAIL + " " + RAIL), row
                assert row.count(RAIL) <= 1, row


# --------------------------------------------------------------------------
# Narration — the rail marks the ANSWER, and the mark is the whole difference
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_narration_block_paints_no_rail_and_keeps_its_prose() -> None:
    """The report this change answers, at the block.

    A settled block marked as mid-turn narration — the state
    ``app.py::on_assistant_message_end`` commits when a message finalizes into
    tool calls — paints no gutter on any row. The PROSE is untouched: dropping
    it is ``display.narration``'s separate job, and this flag must not do it by
    accident, so the text and its wrap are asserted alongside the absent mark.
    """
    block, rows = await _block(MIXED_CONSTRUCTS, (60, 24), narration=True)
    assert block.is_narration() is True
    assert block.text().strip() == MIXED_CONSTRUCTS.strip()
    assert len(rows) >= 5, f"this fixture must paint several rows; got {rows!r}"
    assert all(not row.startswith(RAIL) for row in rows), rows
    # Not "railed geometry minus a glyph": the gutter is not reserved, so the
    # prose starts at column 0 and the copy path strips nothing.
    assert rows[0].startswith(MIXED_CONSTRUCTS.splitlines()[0]), rows[0]
    assert block.copy_gutter(0) == 0


@pytest.mark.asyncio
async def test_the_mark_is_the_only_difference_between_progress_and_the_answer() -> None:
    """Same text, same width, one mark — and the two frames are opposite.

    This is the discrimination the rail exists to draw, asserted as a PAIR
    rather than as two independent claims about one block: a rule that railed
    nothing, or one that railed everything, would satisfy either half alone.
    """
    answer, answer_rows = await _block(MIXED_CONSTRUCTS, (60, 24))
    progress, progress_rows = await _block(MIXED_CONSTRUCTS, (60, 24), narration=True)

    assert answer.is_narration() is False and progress.is_narration() is True
    assert all(row.startswith(RAIL) for row in answer_rows), answer_rows
    assert all(not row.startswith(RAIL) for row in progress_rows), progress_rows
    assert answer.text() == progress.text(), "the mark must not move the prose"


# --------------------------------------------------------------------------
# The settle — the rail is a settled-state property
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width", [16, 24, 40, 60, 120])
@pytest.mark.asyncio
async def test_a_streaming_message_paints_no_rail_and_folds_at_the_lane(
    width: int,
) -> None:
    """The report, at the block: no gutter until the message stops arriving.

    The block is driven through ``update_text`` alone — every delta, no
    finalize — which is the state a reader watches. Asserted as three things at
    once because each alone passes a different half-implementation: NO row
    carries the glyph (a rail that is merely not painted would still be here);
    the fold is taken at the LANE (a mark that skipped the paint but kept the
    two cells would leave the prose indented with nothing in the space, which
    the next assertion's arithmetic catches); and the copy gutter is 0, so the
    clipboard does not strip two cells of real content from a frame that never
    had them. Then the SAME block settles, and the pair is the whole claim.

    The widths run from the ``MIN_BODY`` clamp region upward: below the floor
    the arithmetic and the paint have to agree about a body narrower than the
    floor, which is where a gate that special-cased either state would show.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(width, 40)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text(THREE_PARAGRAPHS)
        await pilot.pause()
        await pilot.pause()

        lane = block.region.width
        assert lane > 0, "the block never got a real width"
        streaming_rows = _rendered(block)
        assert sum(1 for row in streaming_rows if row.strip()) >= 3, streaming_rows
        assert all(not row.startswith(RAIL) for row in streaming_rows), streaming_rows
        assert block._painted_rail_cols == 0
        assert block.copy_gutter(0) == 0
        # The lane itself is the fold: rail-OFF geometry, not railed geometry
        # with the glyph left off.
        assert block._built_width == lane, (block._built_width, lane)

        block.finalize_text()
        await pilot.pause()

        settled_rows = _rendered(block)
        assert all(row.startswith(RAIL) for row in settled_rows), settled_rows
        assert block._painted_rail_cols == RAIL_COLS
        assert block.copy_gutter(0) == RAIL_COLS
        # The two cells came off the fold, and the box did not move: the box is
        # the whole lane and the rail lives inside it.
        assert block._built_width == max(lane - RAIL_COLS, 0), (block._built_width, lane)
        for row in settled_rows:
            assert len(row.rstrip()) <= block.region.width, (row, block.region.width)


@pytest.mark.asyncio
async def test_the_paint_that_commits_the_message_is_the_one_that_adds_the_rail() -> None:
    """D3, with NO pause between the commit and the read.

    The frame the reader gets at the settle is the frame that was painted BY
    the settle. A settle flag raised after ``finalize_text``'s paint leaves the
    committed rows un-railed until something else repaints the block, and every
    paused assertion in this file would still pass — the rail arrives one
    repaint late, which on a screen that then goes quiet is never. So the rows
    are read twice with no await between them: bare before the call, railed
    immediately after it.

    ``_render`` returns what the block is painting NOW, so this reads the frame
    rather than a re-derivation of it.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(60, 24)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text(THREE_PARAGRAPHS)
        await pilot.pause()

        before = _rendered(block)
        assert all(not row.startswith(RAIL) for row in before), before

        block.finalize_text()  # deliberately NOT followed by a pause

        after = _rendered(block)
        assert after != before, "the settle did not repaint the block"
        assert all(row.startswith(RAIL) for row in after), after
        assert block._painted_rail_cols == RAIL_COLS


@pytest.mark.asyncio
async def test_narration_takes_no_rail_while_streaming_or_after_it_settles() -> None:
    """Both routes to zero, on one block — and the prose untouched by both.

    Narration is never railed: not while it streams (the settle gate answers 0)
    and not after it settles (the mark answers 0). Two independent "no"s on the
    same block, which is the state the live path commits at
    ``app.py::on_assistant_message_end`` — the mark is applied and the rows
    committed in that order, one event apart from the streaming frame above.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(60, 24)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.mark_narration()
        block.update_text(MIXED_CONSTRUCTS)
        await pilot.pause()

        streaming_rows = _rendered(block)
        assert all(not row.startswith(RAIL) for row in streaming_rows), streaming_rows
        assert block.copy_gutter(0) == 0

        block.finalize_text()
        await pilot.pause()

        settled_rows = _rendered(block)
        assert all(not row.startswith(RAIL) for row in settled_rows), settled_rows
        assert block.copy_gutter(0) == 0
        # The mark must not move or drop the prose: the message comes back whole
        # and still folded at the LANE, which is what makes this the rail-OFF
        # geometry rather than a rail whose glyph went missing.
        assert block.text().strip() == MIXED_CONSTRUCTS.strip()
        assert block._built_width == block.region.width, (
            block._built_width,
            block.region.width,
        )


@pytest.mark.asyncio
async def test_a_theme_switch_keeps_the_rail_on_a_settled_block() -> None:
    """D4, the trap: ``retheme`` clears ``_finalized`` to force a repaint.

    A rail gated on ``_finalized`` is therefore painted OFF on a theme switch —
    the bar vanishing when the user changes theme — and the flag is restored
    afterwards without a second repaint, so nothing puts it back. Gated on the
    message's own settled state, the rebuild paints exactly what the frame had.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(60, 24)) as pilot:
        view = app.query_one(TranscriptView)
        settled = AssistantBlock()
        streaming = AssistantBlock()
        view.append_block(settled)
        view.append_block(streaming)
        await pilot.pause()
        settled.update_text(THREE_PARAGRAPHS)
        settled.finalize_text()
        streaming.update_text(THREE_PARAGRAPHS)
        await pilot.pause()
        assert all(row.startswith(RAIL) for row in _rendered(settled))

        settled.retheme()
        streaming.retheme()
        await pilot.pause()

        rebuilt = _rendered(settled)
        assert all(row.startswith(RAIL) for row in rebuilt), rebuilt
        assert settled._painted_rail_cols == RAIL_COLS
        # ...and the half that matters as much: a theme switch must not hand one
        # to a message that has not settled.
        assert all(not row.startswith(RAIL) for row in _rendered(streaming))
        assert streaming._painted_rail_cols == 0


@pytest.mark.asyncio
async def test_the_copy_strips_the_rail_only_once_it_has_been_painted() -> None:
    """Selection parity across the settle: what is painted is what is stripped.

    A whole-message take, taken at both states of ONE block. While streaming
    there is no gutter on any row, so the take is the message byte for byte; at
    the settle the same take must come back with the rail removed and nothing
    else — the two cells are chrome, and a copy that reported a gutter it never
    painted would eat real prose instead (review round 2, M1).
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(60, 24)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text(MIXED_CONSTRUCTS)
        await pilot.pause()

        streaming_rows = _rendered(block)
        streaming_copy = _whole_message(block, streaming_rows)
        assert streaming_copy is not None
        assert RAIL not in streaming_copy
        assert "- alpha item" in streaming_copy and "> a quoted line" in streaming_copy

        block.finalize_text()
        await pilot.pause()

        settled_rows = _rendered(block)
        settled_copy = _whole_message(block, settled_rows)
        assert settled_copy is not None
        assert settled_copy == streaming_copy, (settled_copy, streaming_copy)


@pytest.mark.asyncio
async def test_a_subagent_row_gains_its_rail_when_its_own_message_settles() -> None:
    """The page settles per ROW, so its rail arrives per row.

    ``entry_block`` is the page's own factory, and a row whose message may still
    receive deltas is built UNFINALIZED on purpose (a finalized block ignores
    ``update_text``). So the page's live row is exactly the streaming state, and
    its rail must arrive when the row is committed — the same seam
    ``_refresh_row`` drives, one row at a time, rather than at job end.
    """
    entry = SubagentEntry("m1", "text", text=MIXED_CONSTRUCTS)
    live = entry_block(entry, fold_width=60, settled=False)
    done = entry_block(entry, fold_width=60, settled=True)
    assert isinstance(live, AssistantBlock) and isinstance(done, AssistantBlock)
    assert not live.is_finalized(), "a live row must not be frozen at building time"
    assert done.is_finalized()

    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 30)) as pilot:
        view = app.query_one(TranscriptView)
        view.append_block(live)
        view.append_block(done)
        await pilot.pause()
        await pilot.pause()

        live_rows = _rendered(live)
        assert all(not row.startswith(RAIL) for row in live_rows), live_rows
        assert live.copy_gutter(0) == 0
        done_rows = _rendered(done)
        assert all(row.startswith(RAIL) for row in done_rows), done_rows

        # ...and the row that commits IN PLACE (deltas arrived, then the row's own
        # message ended) gains the rail in the paint that commits it.
        live.finalize_text()
        assert all(row.startswith(RAIL) for row in _rendered(live))


# --------------------------------------------------------------------------
# Reuse surfaces and focus
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_subagent_page_marks_the_answer_and_not_the_progress_sentence() -> None:
    """D7's deliberate part, under the rail's new meaning.

    The subagent page renders the same conversation shape, and the delegated
    PROMPT there already carries its own rule
    (``test_user_block::test_a_prompt_in_the_nested_subagent_body_paints_every_row_it_reserves``).
    Its prose rows are all model responses, so the page is where a mark that
    discriminates has to be shown to survive the second fold: the fixture's
    opening sentence is prose streamed before a tool batch (progress), and its
    closing sentence is the last thing the child said (the answer). The two
    blocks are told apart BY TEXT and both outcomes are asserted, because a
    page that had stopped marking anything would satisfy either one alone.
    """
    job = _job_with(TRAJECTORY)
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._open_subagent_view(str(job.id))
        for _ in range(8):
            await pilot.pause()
        prose = {
            block.text().strip(): block
            for block in app.query_one(SubagentView)._body.blocks()
            if isinstance(block, AssistantBlock)
        }
        assert set(prose) == {
            "Reading the ingest path.",
            "Two tests fail on the retry budget.",
        }, f"the page stopped mounting agent prose: {sorted(prose)}"

        progress = prose["Reading the ingest path."]
        assert progress.is_narration() is True
        assert all(not row.startswith(RAIL) for row in _rendered(progress)), _rendered(progress)

        answer = prose["Two tests fail on the retry budget."]
        assert answer.is_narration() is False
        assert all(row.startswith(RAIL) for row in _rendered(answer)), _rendered(answer)


@pytest.mark.asyncio
async def test_a_subagent_prompt_and_answer_use_different_inks() -> None:
    """The prompt rule and the answer rail in ONE frame, resolved side by side.

    The colour tests above compare tokens; this compares what the page actually
    paints, which is the form the reader meets. Same geometry, different ink —
    that is the whole verdict.
    """
    job = _job_with(TRAJECTORY)
    # The delegated prompt is the job's, not a trajectory entry — the same way
    # ``test_user_block`` seeds this page.
    job.prompt = "summarise the ingest path"
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._open_subagent_view(str(job.id))
        for _ in range(8):
            await pilot.pause()
        blocks = list(app.query_one(SubagentView)._body.blocks())
        prompts = [block for block in blocks if isinstance(block, UserBlock)]
        answers = [block for block in blocks if isinstance(block, AssistantBlock)]
        assert prompts and answers, (len(prompts), len(answers))
        railed = [block for block in answers if not block.is_narration()]
        assert len(railed) == 1, "the page rails the answer, and only the answer"

        rail = theme_mod.semantic_color(RAIL_TOKEN)
        rule = theme_mod.semantic_color(UserBlock.RULE_TOKEN)
        assert rail != rule, (rail, rule)
        # Both are painting a gutter of the same width, which is what makes the
        # colour the only thing telling them apart. It is a claim about RAILED
        # blocks: the progress sentence beside them paints no gutter at all, so
        # it has no bar to be told apart from.
        assert prompts[0].copy_gutter(0) == railed[0].copy_gutter(0) == RAIL_COLS
        progress = [block for block in answers if block.is_narration()]
        assert progress, "the fixture's progress sentence stopped being marked"
        assert all(block.copy_gutter(0) == 0 for block in progress)


def test_an_assistant_block_is_not_a_focus_stop() -> None:
    """A rail is ink, not an affordance (D9).

    The prototype made every assistant block focusable because one class there
    carried mutable treatments the reader could cycle. Nothing here is
    interactive, and a transcript where every message is a tab stop is a
    different change from this one.
    """
    assert AssistantBlock.can_focus is False
    assert "BINDINGS" not in AssistantBlock.__dict__
