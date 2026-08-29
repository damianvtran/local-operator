"""Inline ghost text in the composer — the Tab invariant and its gates.

The whole feature rests on ONE promise: **the dimmed cells the user sees are
exactly the characters Tab commits**, i.e. ``buffer + ghost == buffer after
Tab``. Everything here exists to hold that promise against the shape of the
completion code, which does not naturally have it — every completion site
replaces a SPAN (``text[:start] + name + text[end:]``), and a span replacement
only looks like an append when the row happens to extend what was typed. For a
FUZZY match it rewrites characters already on screen (``/lg`` + Tab yields
``/login ``), so no ghost can honestly describe it and the correct ghost is
none at all.

The cases are driven through ``run_test`` with REAL key presses rather than by
calling the completion methods, because that is the only way the span
arithmetic is exercised end to end: a unit-level assertion about
``completion_for`` would agree with itself while the widget did something else
with the caret. The pilot runs the real ``OperatorApp``, so the picker rows
come from the shipped ``SLASH_COMMANDS`` registry rather than a fixture that
could drift from it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from local_operator.tui.app import OperatorApp
from textual.widgets.text_area import Selection

from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.test_app_pilot import (
    FakeMcpManager,
    FakeSession,
    McpSession,
    _factory,
)


def _oauth_configs() -> dict[str, Any]:
    """Two OAuth servers, which is what makes the ``/mcp`` compound rows exist.

    ``_mcp_argument_choices`` offers server rows as compound ``login notion``
    names because the matcher compares against the WHOLE argument. That shape
    is the hardest completion in the app to predict from the list alone, and is
    the case the feature was asked for.
    """
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    return {
        "linear": MCPHttpServerConfig(
            url="https://mcp.linear.app/mcp", auth=MCPAuthConfig(type="oauth")
        ),
        "notion": MCPHttpServerConfig(
            url="https://mcp.notion.com/mcp", auth=MCPAuthConfig(type="oauth")
        ),
    }


def _mcp_app() -> OperatorApp:
    from local_operator.session.mcp_status import McpStartupOutcome

    configs = _oauth_configs()
    manager = FakeMcpManager(["linear", "notion"], ["linear"])
    manager._configs = configs
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    return OperatorApp(lambda: _factory(session))


async def _settle(pilot: Any, times: int = 8) -> None:
    for _ in range(times):
        await pilot.pause()


async def _type(pilot: Any, keys: str) -> None:
    """Type ``keys`` one press at a time, settling between each.

    Per-key settling matters: the argument rows are filled by the app answering
    a posted message, so a burst of presses can outrun the list the ghost is
    derived from.
    """
    for char in keys:
        await pilot.press("space" if char == " " else char)
        await _settle(pilot, 5)


#: ``(seed, typed, expected_ghost)``. ``seed`` is set on the buffer first (the
#: faithful shortcut for "the user already got here"), then ``typed`` goes in
#: as real presses. An expected ghost of ``""`` is the honest-refusal case, not
#: an absent assertion — it is what makes a fuzzy or case-mismatched row show
#: nothing rather than a lie.
GHOST_CASES = [
    # Command WORD, prefix match: the ghost carries the trailing space, because
    # `/mc` + Tab yields `/mcp ` WITH it. Invisible on screen and deliberately
    # asserted, since dropping it would break the invariant silently.
    ("", "/mc", "p "),
    ("", "/te", "am "),
    # Enum-tail ARGUMENT: no trailing space, matching `_complete_argument`'s
    # rule that the space would terminate the argument and close the list.
    ("/mcp ", "lo", "gin"),
    ("/mcp ", "l", "ogin"),
    # The COMPOUND `/mcp` row the issue is about. Reached through `/mcp login `
    # because that is the route on which the app fills the server rows.
    ("/mcp login ", "n", "otion"),
    ("/mcp login ", "l", "inear"),
    # FUZZY: `/lg` + Tab produces `/login `, which is not `/lg` plus anything.
    # A span rewrite cannot be described by characters appended at the caret,
    # so the only honest ghost is none.
    ("", "/lg", ""),
    ("/mcp ", "lgn", ""),
    # CASE: `/MCP lo` + Tab inserts the registry's own casing, so a
    # case-INSENSITIVE startswith would pass here and then paint characters Tab
    # does not produce. Pinned so the check is never "helpfully" relaxed.
    ("", "/MC", ""),
    ("/mcp ", "LO", ""),
    ("", "/Te", ""),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("seed,typed,expected", GHOST_CASES)
async def test_tab_commits_exactly_the_ghost(seed: str, typed: str, expected: str) -> None:
    """THE invariant: ``buffer + ghost == buffer after Tab``, or no ghost.

    Asserted in both directions on purpose. The expected-ghost check pins WHICH
    characters are dimmed (so a case-insensitive or fuzzy ghost fails here),
    and the concatenation check pins that those exact characters are what Tab
    writes (so a completion that rewrote the span differently fails there).
    Either alone would pass a version of the bug.
    """
    app = _mcp_app()
    configs = _oauth_configs()
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        with patch("local_operator.mcp.config.load_all_mcp_configs", return_value=(configs, {})):
            if seed:
                editor.text = seed
                editor.move_cursor(editor._end_of_buffer())
                editor._sync_picker()
                await _settle(pilot, 10)
            await _type(pilot, typed)
            await _settle(pilot, 10)

            before = editor.text
            ghost = editor.suggestion
            assert ghost == expected, f"{before!r}: ghost {ghost!r} != {expected!r}"

            await pilot.press("tab")
            await _settle(pilot, 10)
            after = editor.text

    if ghost:
        assert before + ghost == after, (
            f"invariant broken: {before!r} + {ghost!r} != {after!r} — the dimmed "
            "cells are not the characters Tab committed"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "typed",
    ["review this /team ", "fix the bug /agent cod"],
)
async def test_an_inline_name_command_shows_no_ghost(typed: str) -> None:
    """The reassembly path previews nothing, because it is not an append.

    ``/team`` and ``/agent`` are NAME+message commands, and when a draft
    survives outside the command token, accepting a row does not just fill the
    span — ``_complete_name_argument`` moves the whole ``/<cmd> <name>``
    construct to the FRONT of the buffer with the draft as its message. The
    ghost previewed only the span replacement, so ``review this /team `` showed
    a dimmed ``chart `` and Tab produced ``/team chart review this`` (review
    round 1, B1).

    ``completion_for`` now models both edits, so the predicted buffer is a
    reordering rather than an append and ``ghost_for``'s ``startswith`` rule
    withholds the preview on its own. Asserted through real keys, and the
    invariant is re-checked here rather than assumed: whatever the ghost says,
    it must still describe Tab.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, typed)
        await _settle(pilot, 10)
        before = editor.text
        ghost = editor.suggestion
        await pilot.press("tab")
        await _settle(pilot, 10)
        after = editor.text

    assert ghost == "", f"the reassembly path previewed {ghost!r}, which Tab does not append"
    # Tab still reassembles — the fix is to the prediction, not to the edit.
    assert after != before and after.startswith(("/team ", "/agent ")), after


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "typed,width,expected",
    [
        # `/analytic` (9 cells) + ghost `s ` (2) needs 11 free cells.
        ("/analytic", 20, "s "),  # 12 available: room to spare
        ("/analytic", 19, ""),  # 11 available: EXACTLY the boundary
        ("/mc", 14, "p "),  # 6 available
        ("/mc", 13, ""),  # 5 available: EXACTLY the boundary
    ],
)
async def test_the_width_gate_rejects_the_exact_boundary(
    typed: str, width: int, expected: str
) -> None:
    """At ``col + len(ghost) == width`` the ghost must be refused, not admitted.

    Textual reserves the cell AT the caret for the caret itself, so a ghost
    ending exactly at the content edge still pushed the rendered strip one cell
    past the box — measured as a strip one wider than the same row with no
    ghost, at w=19 and w=13 (review round 1, B2). The original ``>`` admitted
    precisely that case, and the existing width test sampled widths comfortably
    either side of it.

    Both a fitting and a boundary width per ghost, so the test discriminates
    rather than just asserting emptiness at narrow widths.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(width, 20)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, typed)
        await _settle(pilot, 8)
        ghost = editor.suggestion
        with_ghost = editor.render_line(0).cell_length
        editor.suggestion = ""
        await _settle(pilot, 3)
        without_ghost = editor.render_line(0).cell_length

    assert ghost == expected, f"w={width}: ghost {ghost!r} != {expected!r}"
    # The point of the gate, asserted directly: a ghost never widens the row.
    assert with_ghost == without_ghost, (
        f"w={width}: the ghost pushed the strip from {without_ghost} to "
        f"{with_ghost} cells — it overran the content box"
    )


@pytest.mark.asyncio
async def test_the_ghost_is_dropped_when_it_would_not_fit() -> None:
    """Gate 2: a ghost wider than the row is withheld, not cropped.

    Textual injects the suggestion AFTER the wrap sections are divided, so a
    long ghost neither wraps nor crops — it simply overruns the composer. The
    gate withholds it ENTIRELY rather than truncating, because a cropped ghost
    would show fewer characters than Tab inserts and break the invariant from
    the other direction. Same buffer at two widths, so the width is provably
    the only variable.
    """
    ghosts = {}
    for width in (100, 18):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(width, 24)) as pilot:
            await _settle(pilot, 6)
            editor = app.query_one(Editor)
            editor.focus()
            await _type(pilot, "/analytic")
            await _settle(pilot, 8)
            ghosts[width] = (editor.text, editor.suggestion)

    assert ghosts[100] == ("/analytic", "s "), ghosts[100]
    # 18 columns leaves 10 text cells, so `/analytic` (9) plus `s ` cannot fit.
    assert ghosts[18] == ("/analytic", ""), ghosts[18]


@pytest.mark.asyncio
async def test_right_arrow_does_not_accept_the_ghost() -> None:
    """``→`` stays a pure caret key.

    ``TextArea.action_cursor_right`` inserts ``suggestion`` natively (the
    fish/zsh-autosuggest convention). This widget overrides that: Tab is the
    single accept key the invariant is stated over, and issue #370 wants
    ``alt+←/→``/``cmd+←/→`` as caret motion in this same composer — a ``→``
    that sometimes types five characters would make that family of chords mean
    two different things depending on whether a list is open.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, "/mc")
        await _settle(pilot, 8)
        assert editor.suggestion == "p ", editor.suggestion

        await pilot.press("right")
        await _settle(pilot, 8)
        assert editor.text == "/mc", f"→ accepted the ghost: {editor.text!r}"


@pytest.mark.asyncio
async def test_moving_the_caret_clears_the_ghost() -> None:
    """Gate 1: a ghost is only ever rendered with the caret at the line end.

    The ghost adds cells the DOCUMENT does not have, while ``_slash_cells`` and
    ``_marker_cells`` compute their x-ranges from document columns — so a ghost
    left standing after the caret moves back into the word renders mid-word.
    Verified by rendering the actual strip, not just reading the reactive:
    driving the widget with the gate bypassed paints row 0 as ``/p mc``.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, "/mc")
        await _settle(pilot, 8)
        assert editor.suggestion == "p "
        assert editor.render_line(0).text.startswith("/mcp"), editor.render_line(0).text

        await pilot.press("left")
        await _settle(pilot, 8)
        assert editor.suggestion == "", "a ghost survived a caret move"
        assert editor.render_line(0).text.startswith("/mc "), editor.render_line(0).text


@pytest.mark.asyncio
async def test_gate_one_alone_withholds_a_mid_caret_ghost() -> None:
    """Gate 1's OWN contribution, isolated from ``watch_selection``.

    ``test_moving_the_caret_clears_the_ghost`` presses ``left``, which fires
    ``watch_selection`` — and that clears the ghost independently, so the
    assertion passes whether or not gate 1 exists. Deleting the gate left the
    whole suite green (review round 1, M1), which is the worst kind of gap:
    the docstring warns a future editor not to relax the gate, and nothing
    held them to it.

    This asks :meth:`_ghost_completion` directly with the caret parked
    mid-buffer, which is the one question the gate answers by itself. The state
    is reachable rather than synthetic: brute-forcing the pure functions finds
    34 caret-mid combinations that yield a real ghost, of which ``/mcp `` with
    the caret inside the word is one.

    Verified to FAIL with the gate removed (it returns ``' '``, painting a
    one-space ghost mid-word) and pass with it present.
    """
    app = _mcp_app()
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, "/mcp")
        await _settle(pilot, 8)
        # The list is open on `mcp`, so a ghost is genuinely on offer here.
        assert editor.picker.highlighted_name() == "mcp"
        assert editor._ghost_completion() == " ", "precondition: caret at end still ghosts"

        # Park the caret INSIDE the word without going through a key press, so
        # `watch_selection`'s own clearing cannot be what produces the result.
        editor.selection = Selection((0, 2), (0, 2))
        assert editor._ghost_completion() == "", (
            "gate 1 admitted a ghost with the caret mid-word — it would render "
            "between the typed characters (`/p mc`)"
        )


@pytest.mark.asyncio
async def test_enter_completes_to_the_same_text_as_tab() -> None:
    """Enter's COMPLETING path is byte-identical to Tab's.

    Both keys deliberately insert the same completion so a row can never mean
    two different commands depending on the key, and the ghost is stated over
    Tab — so if Enter's completion diverged, the ghost would be a lie for
    whichever key the user actually pressed. Only the completing cases are
    compared: an UNAMBIGUOUS Enter runs the command instead, and an ambiguous
    Enter on the command WORD takes the common-prefix path
    (``_extend_to_common_prefix``), which is not a row completion and is
    deliberately never ghosted.
    """
    configs = _oauth_configs()
    results = {}
    for key in ("tab", "enter"):
        app = _mcp_app()
        async with app.run_test(size=(100, 24)) as pilot:
            await _settle(pilot, 6)
            editor = app.query_one(Editor)
            editor.focus()
            with patch(
                "local_operator.mcp.config.load_all_mcp_configs", return_value=(configs, {})
            ):
                editor.text = "/mcp login "
                editor.move_cursor(editor._end_of_buffer())
                editor._sync_picker()
                await _settle(pilot, 10)
                await _type(pilot, "n")
                await _settle(pilot, 10)
                ghost = editor.suggestion
                await pilot.press(key)
                await _settle(pilot, 10)
                results[key] = (ghost, editor.text)

    assert results["tab"] == results["enter"] == ("otion", "/mcp login notion"), results


@pytest.mark.asyncio
async def test_a_multiline_draft_shows_no_ghost() -> None:
    """Gate 3: single-line buffers only.

    The wrap machinery the suggestion is injected around is per-line, and a
    multi-line draft is not a state the command lists are live in anyway — so
    the gate costs nothing and removes a whole class of misplacement.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "draft line\n/mc"
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await _settle(pilot, 8)
        assert editor.suggestion == "", editor.suggestion


@pytest.mark.asyncio
@pytest.mark.parametrize("typed,downs", [("/", 1), ("/", 2), ("/lo", 1), ("/lo", 2)])
async def test_arrowing_the_command_list_moves_the_ghost_with_it(typed: str, downs: int) -> None:
    """The ghost follows the ACCEPT TARGET, including in COMMAND mode.

    The ghost used to ride ``on_highlight``, which reports only ARGUMENT rows —
    so in COMMAND mode nothing reported during an arrow press and the only sync
    was the one at the top of ``_on_key``, which runs BEFORE ``picker.move()``.
    The preview sat one row behind: `/` then `down` showed `help ` and Tab
    inserted `/exit ` (review round 1, U1).

    Parametrised over both lists and two depths because the defect persisted for
    every subsequent arrow, so a single-press test could have passed on an
    off-by-one that merely shifted.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, typed)
        for _ in range(downs):
            await pilot.press("down")
            await _settle(pilot, 6)
        highlighted = editor.picker.highlighted_name()
        before = editor.text
        ghost = editor.suggestion
        await pilot.press("tab")
        await _settle(pilot, 10)
        after = editor.text

    assert ghost, "arrowing a list left no preview at all"
    assert before + ghost == after, (
        f"after {downs} down press(es) the ghost promised {(before + ghost)!r} "
        f"but Tab inserted {after!r} (highlighted row was {highlighted!r})"
    )


@pytest.mark.asyncio
async def test_hovering_the_list_does_not_repaint_the_ghost() -> None:
    """The ghost follows the KEYBOARD selection, never the pointer.

    ``_report_highlight`` deliberately prefers the hover — correct for the row
    grounds, wrong for a prediction about a key. Resting the pointer over the
    third row previewed `/mcp reauth` while Tab still inserted `/mcp login`,
    and no keystroke existed to correct it (review round 1, U2).
    """
    app = _mcp_app()
    configs = _oauth_configs()
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        with patch("local_operator.mcp.config.load_all_mcp_configs", return_value=(configs, {})):
            await _type(pilot, "/mcp ")
            await _settle(pilot, 10)
            keyboard_ghost = editor.suggestion
            for row in range(3):
                await pilot.hover(editor.picker, offset=(2, row))
                await _settle(pilot, 6)
                assert editor.suggestion == keyboard_ghost, (
                    f"hovering row {row} repainted the ghost to {editor.suggestion!r}; "
                    "Tab acts on the keyboard selection, so the preview must too"
                )
            before = editor.text
            ghost = editor.suggestion
            await pilot.press("tab")
            await _settle(pilot, 10)
            after = editor.text

    assert before + ghost == after, f"{(before + ghost)!r} != {after!r}"


@pytest.mark.asyncio
async def test_escape_clears_the_ghost_with_the_list() -> None:
    """Dismissing the list retires the preview it was explaining.

    Escape hid the rows but left the dimmed cells painted, and Tab with no open
    picker is a literal tab: the screen promised `/mcp ` and the buffer became
    `/mc ` (review round 1, U3). The ghost must not outlive its own legend.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, "/mc")
        await _settle(pilot, 8)
        assert editor.suggestion == "p ", "precondition: a ghost is showing"

        await pilot.press("escape")
        await _settle(pilot, 8)

        assert editor.picker.is_open() is False
        assert editor.suggestion == "", "the ghost outlived the list Escape dismissed"


@pytest.mark.asyncio
async def test_resizing_re_checks_the_width_gate() -> None:
    """Gate 2's answer depends on the width, so a resize has to re-ask it.

    Nothing re-derived the ghost on resize: one admitted at 100 columns stayed
    painted when the terminal was narrowed to 13, where it overran and cropped
    the user's own text — the exact failure the gate exists to prevent. The
    inverse was equally wrong: a ghost correctly withheld at a narrow width did
    not return on widening until another character was typed (review round 1,
    U4).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, "/us")
        await _settle(pilot, 8)
        assert editor.suggestion == "age ", editor.suggestion

        await pilot.resize_terminal(13, 20)
        await _settle(pilot, 10)
        assert editor.suggestion == "", "a ghost survived a narrowing that made it overrun"

        await pilot.resize_terminal(100, 24)
        await _settle(pilot, 10)
        assert editor.suggestion == "age ", "the ghost did not come back when the room did"
