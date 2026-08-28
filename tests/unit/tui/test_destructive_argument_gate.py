"""The destructive-argument gate: which Enter may FIRE and which must FILL.

The gate exists because the registry's blast radius is uneven and the matcher
is a fuzzy SUBSEQUENCE matcher, so a query that spells nothing can still leave
a single survivor — ``/logout oer`` reached openrouter. On a destructive list
that makes one mis-keystroke unrecoverable, so those lists demand the name in
full or an explicit arrow before Enter acts.

``Editor.DESTRUCTIVE_COMMANDS`` matches the COMMAND WORD, which is the wrong
granularity for a two-level command. Under ``/mcp`` the word is ``mcp`` for
every verb, so ``/mcp remove fsy`` — three characters that spell nothing —
narrowed to one survivor and DELETED A SERVER CONFIG FROM DISK on a single
Enter. Reproduced on the live key path; the same hole existed for
``/mcp logout``. The fix reads the highlighted ROW's ``alert`` flag, which the
app already sets on exactly the destructive rows, and keeps the command-word
tuple as a floor.

Every case drives REAL keys through ``run_test``. The defect was invisible to
reasoning about the source and only appeared under an actual Enter, so a test
that called the gate directly would have agreed with the bug.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: ``/mcp remove`` rows, shaped exactly as the app builds them: compound names
#: (the matcher compares against the WHOLE argument) carrying ``alert=True``
#: because choosing one deletes a server from the config file.
#:
#: Constructed HERE rather than driven through the app's own
#: ``_mcp_argument_choices`` because the ``/mcp remove`` verb ships in a
#: concurrent PR. The gate under test reads the flag off the row, so rows with
#: the right shape exercise it faithfully and this file does not depend on that
#: branch landing first.
REMOVE_ROWS = [
    ArgumentChoice("remove filesystem", "Remove the filesystem server", alert=True),
    ArgumentChoice("remove grafana", "Remove the grafana server", alert=True),
]

#: ``/mcp logout`` — destructive for the same reason (an OAuth credential costs
#: another browser round trip to get back). The app already sets ``alert`` here.
LOGOUT_ROWS = [
    ArgumentChoice("logout filesystem", "Forget the stored credential", alert=True),
    ArgumentChoice("logout grafana", "Forget the stored credential", alert=True),
]

#: ``/mcp login`` — the NON-destructive sibling under the same command word.
#: This is why the fix is not ``DESTRUCTIVE_COMMANDS = ("logout", "mcp")``:
#: that would tax this flow to protect the ones above.
LOGIN_ROWS = [
    ArgumentChoice("login filesystem", "Authorize the filesystem server"),
    ArgumentChoice("login grafana", "Authorize the grafana server"),
]


async def _settle(pilot: Any, times: int = 8) -> None:
    for _ in range(times):
        await pilot.pause()


async def _fuzzy_enter(rows: list[ArgumentChoice], typed: str, key: str = "enter") -> Any:
    """Type ``typed`` into ``/mcp `` against ``rows``, then press ``key``.

    Returns ``(buffer_before, buffer_after, destructive, matched_rows)``. An
    empty buffer afterwards means the command FIRED (the submit path clears
    it); a filled one means the gate held and completed instead.

    The rows are re-pushed after typing because the app answers the argument
    query one message-loop tick behind the keystroke, and this harness has no
    app-side handler to refill them.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/mcp "
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await _settle(pilot, 8)
        editor.picker.set_choices(list(rows))
        await _settle(pilot, 6)
        for char in typed:
            await pilot.press("space" if char == " " else char)
            await _settle(pilot, 5)
        editor.picker.set_choices(list(rows))
        await _settle(pilot, 6)

        matched = [name for name, _ in editor.picker.suggestions()]
        destructive = editor._argument_is_destructive()
        before = editor.text
        await pilot.press(key)
        await _settle(pilot, 10)
        return before, editor.text, destructive, matched


@pytest.mark.asyncio
async def test_a_fuzzy_mcp_remove_fills_instead_of_deleting() -> None:
    """THE data-loss case: ``/mcp remove fsy`` + Enter must not delete anything.

    ``fsy`` is a subsequence of ``filesystem`` that spells nothing — the user
    never named the server. Before the fix the matcher narrowed to one survivor,
    the gate answered False because the command word is ``mcp``, and Enter fired:
    buffer cleared, config gone from disk.

    The gate now fills the exact name instead, so the buffer holds what the user
    would have had to type — one match, so a SECOND Enter runs it deliberately.
    """
    before, after, destructive, matched = await _fuzzy_enter(REMOVE_ROWS, "remove fsy")

    assert matched == ["remove filesystem"], matched
    assert destructive is True, "an alert row must make the slot destructive"
    assert before == "/mcp remove fsy"
    assert after == "/mcp remove filesystem", f"Enter fired on a fuzzy delete: {after!r}"


@pytest.mark.asyncio
async def test_a_fuzzy_mcp_logout_fills_the_preexisting_hole() -> None:
    """``/mcp logout <fuzzy>`` + Enter fills — the hole that predates the fix.

    Same shape as ``/mcp remove`` and destructive for the same reason. It was
    already reachable before either PR: ``/mcp logout`` has always been gated
    only by the command word ``mcp``, which is not in the tuple.
    """
    _before, after, destructive, matched = await _fuzzy_enter(LOGOUT_ROWS, "logout fsy")

    assert matched == ["logout filesystem"], matched
    assert destructive is True
    assert after == "/mcp logout filesystem", f"Enter fired on a fuzzy logout: {after!r}"


@pytest.mark.asyncio
async def test_a_fuzzy_mcp_login_still_runs() -> None:
    """The non-destructive sibling is NOT taxed by its destructive relatives.

    This is the whole reason the fix reads the row rather than adding ``mcp`` to
    ``DESTRUCTIVE_COMMANDS``. ``/mcp login`` opens a browser and re-uses an
    existing grant if there is one; the worst case of a wrong pick is an
    authorization the user cancels, not a deletion. It keeps the single-match
    Enter that every harmless list has.
    """
    _before, after, destructive, matched = await _fuzzy_enter(LOGIN_ROWS, "login fsy")

    assert matched == ["login filesystem"], matched
    assert destructive is False, "a login row must not be gated as destructive"
    assert after == "", f"the harmless sibling stopped running on Enter: {after!r}"


@pytest.mark.asyncio
async def test_a_gated_enter_fills_the_same_text_as_tab() -> None:
    """A fill-instead-of-fire Enter is byte-identical to Tab on the same row.

    Both keys reach ``_complete_argument``, which derives its text from the
    shared ``completion_for``. Pinned because the gate turns Enter into a
    completion key for these rows, and a completion that differed from Tab's
    would mean the row named two different commands depending on the key — the
    exact property the picker's Tab/Enter design exists to prevent.
    """
    _b1, tab_text, _d1, _m1 = await _fuzzy_enter(REMOVE_ROWS, "remove fsy", key="tab")
    _b2, enter_text, _d2, _m2 = await _fuzzy_enter(REMOVE_ROWS, "remove fsy", key="enter")

    assert tab_text == enter_text == "/mcp remove filesystem", (tab_text, enter_text)


@pytest.mark.asyncio
async def test_the_command_word_floor_survives_a_row_without_flags() -> None:
    """``DESTRUCTIVE_COMMANDS`` still gates on its own, with no flag in sight.

    The regression guard for the constraint that the row check ADDS to the
    command-word check rather than replacing it. Every ``/logout`` row the app
    builds carries ``alert=True`` today, so a replacement would pass the suite
    while making credential safety depend on data the app happened to set — and
    any future row that shipped without the flag would silently lose the
    protection. Rows here deliberately carry ``alert=False`` to prove the floor
    holds without them.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle(pilot, 6)
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/logout "
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await _settle(pilot, 8)
        unflagged = [
            ArgumentChoice("openrouter", "Forget the key"),
            ArgumentChoice("deepseek", ""),
        ]
        editor.picker.set_choices(unflagged)
        await _settle(pilot, 6)

        assert (
            editor._argument_is_destructive() is True
        ), "the command-word floor must gate /logout even when no row carries alert"


@pytest.mark.asyncio
async def test_logout_rows_all_carry_the_alert_flag() -> None:
    """The premise the OR relies on, asserted rather than assumed.

    Not a test of the gate but of the data it now reads: if a ``/logout`` row
    ever ships without ``alert``, the row condition contributes nothing there
    and the floor is doing all the work — which is fine, and is exactly why the
    floor was kept. This states that expectation where a future change to the
    row builder will see it.
    """
    from tests.unit.tui.test_command_picker import _logout_choices

    rows = _logout_choices()
    assert rows, "the fixture must offer rows for this to mean anything"
    assert all(choice.alert for choice in rows), [c.name for c in rows if not c.alert]
