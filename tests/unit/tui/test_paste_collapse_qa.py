"""Independent QA probes of large-paste consolidation: the edges the suite leaves.

Written against the RUNNING app rather than against the helpers, for the reason
``test_copy_picker_qa.py`` states about the same split: these questions are ones
only the assembled widget answers. ``test_paste_collapse.py`` (the implementing
slices' own file) covers the mechanism — the predicate, the splice, the marker
grammar, the submit chain — and this file deliberately does not repeat it.

What is here is the ground that file leaves uncovered:

* **The DISPATCH boundary, not the predicate.** ``_collapse_paste`` is only
  reached for a paste that ``_on_paste`` routes to the text branch, and the
  whitespace guard above it returns first. A test that calls the predicate
  directly cannot see that, which is how a payload the predicate would collapse
  reaches the buffer raw.
* **The caret the gesture actually leaves.** ``ctrl+o`` is asserted elsewhere
  from a caret placed inside the marker. The caret a PASTE leaves is one column
  further right, past the trailing space the collapse itself inserted, and that
  is the position the user's finger is at when they decide they wanted the text
  raw.

Two of these were written as ``xfail(strict=True)`` against the defects they
found, following the precedent at ``test_copy_picker_qa.py:355`` and
``test_transcript_selection.py:4966``: the assertion IS the acceptance check, so
it starts passing the moment the defect is fixed. Both defects are now fixed
(QA round 1, DEFECT-1 and DEFECT-2), so both are ordinary passing tests and the
marks are gone — an ``xfail`` left on a passing test is a lie about the state of
the code. The assertions themselves are unchanged.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.app import App, ComposeResult

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.editor import COLLAPSE_ROWS, Editor, PastedText
from tests.unit.tui.conftest import TCSS_PATH

#: Wide enough that a pasted line never soft-wraps, so "lines pasted" and "rows
#: consumed" are the same number — the same reason and the same value
#: ``test_paste_collapse.py`` uses.
WIDE = (100, 30)


class Host(App[None]):
    """The editor under the SHIPPED stylesheet.

    ``max-height: 8`` is what the row budget is derived from, so a host with a
    convenient inline style would assert this file's opinion rather than the
    shipped one.
    """

    CSS_PATH = TCSS_PATH

    def compose(self) -> ComposeResult:
        yield Editor()

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(theme_mod.tcss_variable_map())
        return variables


async def _paste(app: App[None], pilot, text: str) -> None:
    """Post to the APP, never the widget: the widget route delivers twice."""
    app.post_message(events.Paste(text))
    await pilot.pause()
    await pilot.pause()


def _lines(count: int, body: str = "ERROR line {}") -> str:
    return "\n".join(body.format(index) for index in range(count))


# -- the dispatch boundary above the predicate --------------------------------
@pytest.mark.asyncio
async def test_a_whitespace_only_paste_that_overruns_the_field_still_collapses() -> None:
    """The guard that protects the indent gesture must not swallow a 40-row one.

    ``_on_paste`` returns early for any payload whose ``strip()`` is empty —
    correct for the D9 case (an indent copied inside a multi-line prompt must
    stay an indent), but the test for it is emptiness, not SIZE. Forty blank
    lines strip to nothing too, so the collapse is never asked about a paste
    that occupies forty rows of an eight-row field: exactly the legibility
    defect the feature exists to answer, and the one shape of it that the
    predicate itself already answers correctly.

    Asserted through the real paste EVENT rather than by calling
    ``_collapse_paste``, because the predicate is not where this goes wrong —
    called directly it returns a marker for this very input.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        blank_block = "\n" * 40
        await _paste(app, pilot, blank_block)

        assert editor.wrapped_document.height <= COLLAPSE_ROWS, (
            "a 41-row payload is sitting in an 8-row field: "
            f"{editor.wrapped_document.height} rows"
        )
        (pasted,) = editor.attachments().values()
        assert isinstance(pasted, PastedText)
        assert pasted.text == blank_block


@pytest.mark.asyncio
async def test_an_indent_paste_is_still_left_alone() -> None:
    """The other half of the pair, and the reason the guard exists (round 2, D9).

    Pinned beside the xfail above so that a fix for it cannot quietly re-break
    this: an indent copied and pasted inside a multi-line prompt is an ordinary
    editing gesture, it worked before the feature existed, and it must not turn
    into a marker the user did not ask for. Whatever distinguishes the two cases
    has to be SIZE, not emptiness.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        await _paste(app, pilot, "    ")

        assert editor.text == "    "
        assert editor.attachments() == {}


# -- the escape hatch at the caret the gesture leaves -------------------------
@pytest.mark.asyncio
async def test_ctrl_o_expands_at_the_caret_the_PASTE_leaves() -> None:
    """The escape hatch, pressed the moment the user wants it.

    ``ctrl+o`` is the feature's ONLY inverse — design §1 rejects a setting on
    the strength of it — so the gesture it has to serve is "that collapsed and I
    did not want it to", pressed immediately after the paste. At that instant
    the caret is one column past the marker, because ``_collapse_paste`` returns
    ``marker + " "``; ``_marker_span`` asks about the character the caret is
    touching, finds a space, and the action raises ``SkipAction``. The key does
    nothing, with no notice, at the only moment it is reached by reflex.

    ``ctrl+w`` already crosses that exact space, deliberately
    (``_delete_marker_past_spaces``, round 22 D13), so the composer has both the
    precedent and the helper for treating the trailing space as part of the
    gesture's reach.
    """
    payload = _lines(30, "L{}")
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        assert editor.text == "[Paste #1, 30 lines] ", "the fixture stopped collapsing"
        # NOT moved: this is where the paste left it.
        assert editor.selection.end == (0, len("[Paste #1, 30 lines] "))

        await pilot.press("ctrl+o")
        await pilot.pause()

        assert editor.text.startswith("L0"), "the escape hatch did nothing"
        assert editor.attachments() == {}


@pytest.mark.asyncio
async def test_ctrl_w_DOES_cross_the_same_trailing_space() -> None:
    """The precedent the xfail above points at, pinned as a fact.

    Kept because it is the evidence that crossing the space is an established
    behaviour of this composer rather than a new rule invented for the expand
    action — and because if this ever stops holding, the argument for fixing
    ``ctrl+o`` the same way needs to be re-made rather than assumed.
    """
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, _lines(30, "L{}"))
        assert editor.attachments() != {}

        await pilot.press("ctrl+w")
        await pilot.pause()

        assert editor.text == ""
        assert editor.attachments() == {}


@pytest.mark.asyncio
async def test_ctrl_o_still_expands_from_INSIDE_the_marker() -> None:
    """The control for the xfail: the action itself works, the CARET is the bug.

    Without this a reader cannot tell whether the failure above is "the expand
    action is broken" or "the expand action is unreachable from one position",
    and the fix for those two is not the same.
    """
    payload = _lines(30, "L{}")
    app = Host()
    async with app.run_test(size=WIDE) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, payload)
        editor.move_cursor((0, len("[Paste #1, 30 lines]")))
        await pilot.pause()

        await pilot.press("ctrl+o")
        await pilot.pause()

        assert editor.text.startswith("L0")
        assert editor.attachments() == {}
