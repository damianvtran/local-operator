"""The user prompt's gutter rule — the delineation, on every row.

Reported from the field: "have user messages have a more obvious delineation
[…] consider that they will often be multi-line with paragraphs". The old
block put a ``❯`` on its FIRST row only, so at 60 columns even a single
paragraph lost its marker the moment it wrapped, and a three-paragraph prompt
read as three assistant paragraphs.

The fix turns on one invariant, and it is the one this file exists to hold:
**every row the block paints carries the rule** — wrapped continuations and
the blank rows between paragraphs alike. A marker on one row marks a LINE; a
marker on every row marks a BLOCK, and the block is what a reader scrolling
back is looking for. The blank paragraph rows are the case that decides it:
skip them and the bar breaks into segments, which is the same "three separate
things" failure wearing the new treatment.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from rich.text import Text

from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.subagent_view import SubagentView
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
    "I want the transcript to make my own messages obvious at a glance.\n"
    "\n"
    "Right now a prompt is a chevron and a run of prose, so scrolling back "
    "through a long session I cannot find where I asked for something.\n"
    "\n"
    "These are often long and have paragraphs in them, like this one."
)


def _painted(block: UserBlock) -> list[str]:
    """The rows the block authored; a ``UserBlock`` always renders a ``Text``.

    Narrowed with an assert rather than a cast, the way ``_card_text`` in
    ``test_tool_card`` does it: the type is the assumption every row assertion
    in this file rests on, so a block that started returning something else
    should fail here and say so, not fail four tests down on a missing
    attribute.
    """
    renderable = block.renderable
    assert isinstance(renderable, Text)
    return renderable.plain.split("\n")


async def _rows(text: str, size: tuple[int, int]) -> list[str]:
    """The prompt's painted rows at ``size``, through the REAL stylesheet.

    Via ``renderable.plain`` rather than the strips: the strips are padded out
    to the widget width, which would make "does this row carry the rule"
    trivially true of a row of spaces. The plain text is what the block
    actually authored.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=size) as pilot:
        view = app.query_one(TranscriptView)
        block = UserBlock(text)
        view.append_block(block)
        # Two pauses: the first mounts and lays the block out, the second lets
        # the `on_resize` re-wrap it against its REAL width. Without the second
        # every assertion here would be made against the 80-column fallback the
        # constructor uses before the widget has a size.
        await pilot.pause()
        await pilot.pause()
        return _painted(block)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (60, 40)])
async def test_every_row_of_a_multi_paragraph_prompt_carries_the_rule(
    size: tuple[int, int],
) -> None:
    """The invariant, at both widths the treatment was designed against.

    At 120 the three paragraphs are five rows (two of them blank); at 60 they
    are eleven. Neither count is asserted — the point is that the rule holds
    however the text happens to fall.
    """
    rows = await _rows(THREE_PARAGRAPHS, size)
    assert len(rows) >= 5
    assert all(row.startswith(UserBlock.RULE) for row in rows), rows


@pytest.mark.asyncio
async def test_the_blank_row_between_paragraphs_carries_the_rule_and_nothing_else() -> None:
    """The case the whole design turns on.

    A blank line in the prompt must still paint a row, and that row must be
    the rule alone. Emitting nothing would break the bar into one segment per
    paragraph; emitting the rule plus text would lose the paragraph break.
    """
    rows = await _rows("first\n\nsecond\n\nthird", (120, 40))
    assert [row.rstrip() for row in rows] == [
        "▌ first",
        "▌",
        "▌ second",
        "▌",
        "▌ third",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello\n", ["▌ hello"]),
        ("\n\nhello", ["▌ hello"]),
        ("a\n\nb\n\n", ["▌ a", "▌", "▌ b"]),
        ("", ["▌"]),
    ],
)
async def test_a_blank_row_at_the_EDGE_is_trimmed_rather_than_painted(
    text: str, expected: list[str]
) -> None:
    """The other half of the blank-row rule: meaningful between, not at the end.

    A blank row between paragraphs keeps the bar continuous. A blank row at the
    edge separates content from nothing and paints a stub of rule beside no
    text. The empty prompt still yields one row, so a block always has a height
    for the spacing machinery to reason about.
    """
    rows = await _rows(text, (120, 40))
    assert [row.rstrip() for row in rows] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [120, 60, 40, 24, 12])
async def test_a_narrow_frame_keeps_every_word_and_never_overhangs(width: int) -> None:
    """Content survives the squeeze; the rule is never what gets dropped.

    ``12`` is below any terminal anyone runs and above :attr:`UserBlock.MIN_BODY`
    plus the gutter, which is where the two constraints — fit the frame, keep
    the delineation — can still both be met. Below it the block CLIPS the row
    with an ellipsis rather than dropping the rule (see :attr:`UserBlock.MIN_BODY`),
    so the fits-the-frame assertion is only made where fitting is possible.
    """
    rows = await _rows(THREE_PARAGRAPHS, (width, 40))
    assert all(row.startswith(UserBlock.RULE) for row in rows), rows
    for row in rows:
        assert cell_len(row) <= width, (width, row)
    # Nothing was CLIPPED. Compared with whitespace removed on both sides,
    # because at 12 columns a word longer than the body is broken across rows
    # ("something." → "somethin" + "g."): that is the wrap doing its job, and a
    # word-for-word comparison would fail on correct output. This assertion is
    # deliberately whitespace-BLIND, so it is not the one that defends layout —
    # see the indentation test below for that.
    painted = "".join(row[UserBlock.RULE_COLS :] for row in rows)
    assert "".join(painted.split()) == "".join(THREE_PARAGRAPHS.split())


@pytest.mark.asyncio
async def test_a_pasted_snippet_keeps_the_indentation_it_was_pasted_with() -> None:
    """Leading whitespace is layout, and a prompt is where code gets pasted.

    ``wrap_cells`` splits on ``" "`` and rebuilds with a single separator, which
    keeps an INTERIOR run of spaces and silently drops a LEADING one. Rich kept
    it before this block started wrapping itself, so once it did, a pasted
    snippet came out flush left one line at a time — a regression landed by the
    very change that exists to make multi-line prompts legible.
    """
    snippet = "here is the failure:\n\ndef f(x):\n    if x:\n        return 1\n    return 0"
    rows = await _rows(snippet, (120, 40))
    assert [row[UserBlock.RULE_COLS :] for row in rows] == [
        "here is the failure:",
        "",
        "def f(x):",
        "    if x:",
        "        return 1",
        "    return 0",
    ]


@pytest.mark.asyncio
async def test_a_wrapped_indented_line_keeps_its_indent_on_every_row() -> None:
    """The continuation rows of an indented line stay under it, not at the rule.

    Hanging the wrap back at column 0 would put a continuation of the deepest
    nesting level flush against the outermost — which reads as a different
    statement, and is the specific way "keeps the indentation" decays once the
    frame narrows.
    """
    rows = await _rows("        deeply indented and long enough to wrap twice over", (40, 40))
    bodies = [row[UserBlock.RULE_COLS :] for row in rows]
    assert len(bodies) > 1
    assert all(body.startswith("        ") for body in bodies), bodies
    assert all(cell_len(row) <= 40 for row in rows), rows


@pytest.mark.asyncio
async def test_the_prose_starts_in_the_same_column_on_every_row() -> None:
    """A uniform hanging indent — the rule is a gutter, not a first-row prefix.

    ``RULE_COLS`` is pinned to :data:`SPINE_INDENT` so the prompt's text lands
    in the column the old ``❯ `` prefix put it in — two cells, which is also
    where a notice puts its GLYPH (a notice's own text is two further cells in).
    A rule that pushed the prose one cell further would move every user message
    relative to the rest of the transcript.

    Asserted on unindented text: the block adds no padding of its OWN beyond the
    gutter. Indentation the author typed is a different thing and is kept — see
    the pasted-snippet test above.
    """
    assert UserBlock.RULE_COLS == SPINE_INDENT
    rows = await _rows(THREE_PARAGRAPHS, (60, 40))
    for row in rows:
        assert row[: UserBlock.RULE_COLS].rstrip() == UserBlock.RULE
        assert not row[UserBlock.RULE_COLS :].startswith(" "), row


@pytest.mark.asyncio
async def test_resizing_rewraps_the_prompt_without_losing_the_rule() -> None:
    """The block is FINALIZED, and still has to survive a width change.

    Finalization stops the container re-rendering a block; it must not stop the
    block re-wrapping itself, because this one wraps its OWN text (Rich's fold
    would return continuation rows to column 0 and eat the rule). The bug this
    guards is a prompt wrapped for 120 columns still painting 120-column rows
    after the terminal is dragged to 60.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        block = UserBlock(THREE_PARAGRAPHS)
        view.append_block(block)
        await pilot.pause()
        await pilot.pause()
        wide = _painted(block)

        await pilot.resize_terminal(60, 40)
        await pilot.pause()
        await pilot.pause()
        narrow = _painted(block)

    assert block.is_finalized()
    assert len(narrow) > len(wide)
    assert all(row.startswith(UserBlock.RULE) for row in narrow), narrow
    assert all(cell_len(row) <= 60 for row in narrow), narrow


@pytest.mark.asyncio
async def test_a_prompt_in_the_nested_subagent_body_paints_every_row_it_reserves() -> None:
    """Reserved rows equal painted rows, in the configuration that broke.

    Reported from the subagent page with numbers: a three-paragraph delegated
    prompt at 60 columns reserved 10 rows and painted 8, leaving a two-row hole
    that pushed the child's reply down the frame. The cause is that ``auto``
    height makes the layout engine MEASURE this widget: the measurement is
    cached on ``Widget._content_height_cache`` keyed on the WIDTH ALONE, the
    first one is taken of the pre-layout 80-column build folded to fit, and
    ``Static.update`` never clears it. :meth:`UserBlock._build` therefore pins
    the height instead of being measured.

    Driven through the REAL page rather than a synthetic host, and that is
    load-bearing rather than thoroughness: the trigger is the mount ordering
    ``SubagentView`` produces — the block is built during the pending replay,
    before the body has its final width — and a plain ``TranscriptView`` does
    not reproduce it in either direction. An earlier version of this test primed
    the cache by hand in a flat host; it passed with the pin removed, i.e. it
    was guarding nothing. This one fails with the pin removed (reserved 10,
    painted 8), which is the only reason to keep it.
    """
    job = _job_with(TRAJECTORY)
    job.prompt = THREE_PARAGRAPHS
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(60, 30)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._open_subagent_view(str(job.id))
        for _ in range(8):
            await pilot.pause()
        prompts = [
            block
            for block in app.query_one(SubagentView)._body.blocks()
            if isinstance(block, UserBlock)
        ]
        assert len(prompts) == 1, "the page stopped mounting the delegated prompt"
        block = prompts[0]
        rows = _painted(block)
        assert len(rows) > 1, "a one-row prompt cannot exhibit the hole"
        assert block.region.height == len(rows)
        assert all(row.startswith(UserBlock.RULE) for row in rows), rows


def test_the_rule_never_spends_the_accent() -> None:
    """The accent green is spent on four sites and means "a turn is live".

    A green column beside every prompt would be the largest accent surface in
    the app and would mean nothing — see the exhaustive list in
    ``local_operator.tcss``. Pinned in both ramps, because a token that resolves
    to the accent in only one of them is the same violation half the time.
    """
    assert UserBlock.RULE_TOKEN != "accent"
    for ramp in ("dark", "light"):
        rule = theme_mod.semantic_color(UserBlock.RULE_TOKEN, ramp)
        assert rule != theme_mod.semantic_color("accent", ramp)


def test_the_rule_stays_legible_as_a_graphical_element_in_both_ramps() -> None:
    """3:1 against the ground, the WCAG floor for a non-text graphical object.

    The rule is structure, not prose, so it is held to the graphical-object bar
    rather than the 4.5:1 text one — but it has to clear that bar in the light
    ramp too, where the warm-paper ground is far closer to the neutral inks.
    Measured: 4.55 on ``dark``, 3.77 on ``light``. A quieter token (``faint``,
    at 1.97) draws a bar the reader cannot see, which costs a column and
    delivers no delineation at all.

    This reads the palette, so it can fail for two different reasons. Failing
    after :attr:`UserBlock.RULE_TOKEN` changed is this block's bug — the
    tempting edit is lightening the rule so messages "stand out", and the
    brighter neutral moves toward the ink on dark and toward the GROUND on
    paper. Failing after neither the token nor this file changed means the ramp
    itself moved; the fix is in ``local_operator/tui/theme.py``, not here.
    """

    def luminance(value: str) -> float:
        raw = value.lstrip("#")
        channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    for ramp in ("dark", "light"):
        rule = luminance(theme_mod.semantic_color(UserBlock.RULE_TOKEN, ramp))
        ground = luminance(theme_mod.semantic_color("bg", ramp))
        ratio = (max(rule, ground) + 0.05) / (min(rule, ground) + 0.05)
        assert ratio >= 3.0, (ramp, ratio)


@pytest.mark.asyncio
async def test_a_one_line_prompt_is_one_row_of_rule_and_no_slab() -> None:
    """The short-message case that rejected the background-tint variant.

    A full-width fill behind a 26-character prompt paints a mostly-empty slab
    the width of the terminal, which reads as a card — the bordered box the
    minimalism contract forbids, drawn in fill instead of line. The rule is
    one cell wide whatever the message length, so a short prompt costs one
    glyph. Asserted as "the row is exactly the prompt", which a tint would
    break by padding the row out to the frame.
    """
    rows = await _rows("summarise the ingest path", (120, 40))
    assert rows == ["▌ summarise the ingest path"]
