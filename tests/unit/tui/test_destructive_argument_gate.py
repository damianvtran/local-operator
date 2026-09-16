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

import asyncio
from typing import Any

import pytest
from textual.message import Message

from local_operator.tui.app import OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.waiting import MessageWaiter

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


async def _fuzzy_enter(rows: list[ArgumentChoice], typed: str, key: str = "enter") -> Any:
    """Type ``typed`` into ``/mcp `` against ``rows``, then press ``key``.

    Returns ``(buffer_before, buffer_after, destructive, matched_rows)``. An
    empty buffer afterwards means the command FIRED (the submit path clears
    it); a filled one means the gate held and completed instead.

    The rows are re-pushed after typing because the app answers the argument
    query one message-loop tick behind the keystroke, and this harness has no
    app-side handler to refill them.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    messages = MessageWaiter()
    async with app.run_test(size=(100, 24), message_hook=messages.on_message) as pilot:
        await messages.wait_for(
            lambda: app._session is session, description="boot worker adopted the session"
        )
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/mcp "
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await pilot.pause()
        editor.picker.set_choices(list(rows))
        await pilot.pause()
        # press() already waits for the screen's message queues. These are
        # parser/gate assertions, not animation geometry: five more idle
        # frames per character prove nothing beyond that same barrier.
        for char in typed:
            await pilot.press("space" if char == " " else char)
        editor.picker.set_choices(list(rows))
        await pilot.pause()

        matched = [name for name, _ in editor.picker.suggestions()]
        destructive = editor._argument_is_destructive()
        before = editor.text
        await pilot.press(key)
        # The real Enter/Tab handler and its screen barrier ran in press().
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
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    messages = MessageWaiter()
    async with app.run_test(size=(100, 24), message_hook=messages.on_message) as pilot:
        await messages.wait_for(
            lambda: app._session is session, description="boot worker adopted the session"
        )
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/logout "
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await pilot.pause()
        unflagged = [
            ArgumentChoice("openrouter", "Forget the key"),
            ArgumentChoice("deepseek", ""),
        ]
        editor.picker.set_choices(unflagged)
        await pilot.pause()

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verb,alert,expected",
    [
        ("remove", True, True),
        ("logout", True, True),
        ("reauth", True, True),
        ("login", False, False),
    ],
)
async def test_the_gate_is_armed_on_the_typed_path(verb: str, alert: bool, expected: bool) -> None:
    """The gate must arm through REAL keystrokes, not just a seeded buffer.

    ``_argument_is_destructive`` reads the highlighted row's flag, and a CLOSED
    list has no highlighted row — so the gate silently answered False for every
    ``/mcp`` row while the server slot was unreachable by typing (the refresh
    key missed the verb→space transition). The alert flags and this gate were
    both inert on the one path a user actually takes.

    This test does NOT bind the refresh-key fix, despite typing. It calls
    ``set_choices`` itself, so the rows are present whether or not the widget
    posted ``RefreshArgumentChoices`` — reverting that fix leaves this file
    10/10 green (UX review round 3, U11). What it binds is the GATE: that a
    row's ``alert`` flag arms ``_argument_is_destructive`` on a buffer built by
    real key presses, which is the half that was silently answering False.

    The refresh fix is bound by
    ``test_ghost_text.py::test_typing_into_the_mcp_server_slot_opens_its_rows``,
    which asserts the rows arrive without staging them and fails 3/3 on revert.
    Look there, not here, when deciding whether a change to the refresh key is
    safe.

    The rows are constructed here because ``/mcp remove`` and the ``reauth``
    alert flag ship in the concurrent PR; the gate reads the flag off the row,
    so rows of the right shape exercise it faithfully.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    messages = MessageWaiter()
    async with app.run_test(size=(100, 24), message_hook=messages.on_message) as pilot:
        await messages.wait_for(
            lambda: app._session is session, description="boot worker adopted the session"
        )
        editor = app.query_one(Editor)
        editor.focus()
        # Keep every real key event; press() includes its message barrier.
        for char in f"/mcp {verb} ":
            await pilot.press("space" if char == " " else char)
        editor.picker.set_choices(
            [
                ArgumentChoice(f"{verb} linear", "", alert=alert),
                ArgumentChoice(f"{verb} notion", "", alert=alert),
            ]
        )
        await pilot.pause()

        assert editor.picker.is_open(), f"/mcp {verb} left the picker closed by typing"
        assert editor._argument_is_destructive() is expected


@pytest.mark.asyncio
async def test_message_waiter_handles_early_and_reentrant_publication() -> None:
    messages = MessageWaiter()
    messages.on_message(Message())
    await messages.wait_for(lambda: True, description="already ready")
    assert not messages._pending
    checks = 0

    def predicate() -> bool:
        nonlocal checks
        checks += 1
        if checks == 1:
            # A clear-after-check implementation loses this synchronous edge.
            messages.on_message(Message())
            return False
        return True

    await messages.wait_for(predicate, description="reentrant edge")
    assert checks == 2
    assert not messages._pending


@pytest.mark.asyncio
async def test_message_waiter_cancellation_does_not_steal_another_consumers_edge() -> None:
    messages = MessageWaiter()
    entered = asyncio.Event()
    ready = False

    # ``_pending`` is read directly, deliberately: the property pinned here is
    # that cancellation removes the CANCELLED consumer's registration, and a
    # leaked registration has no observable consequence to assert on -- the
    # hook setting an event nobody awaits is inert, and ``MessageWaiter``
    # exposes no public registry to assert against. Inventing a public accessor
    # would rename the private read rather than replace it. The behavioural half
    # of the claim (the surviving consumer still receives its wake) is the final
    # ``await second``, which expires its own guard with
    # ``never became ready: surviving consumer`` if cancellation stole the edge.
    def predicate() -> bool:
        entered.set()
        return ready

    first = asyncio.create_task(messages.wait_for(predicate, description="cancelled consumer"))
    await entered.wait()
    entered.clear()
    second = asyncio.create_task(messages.wait_for(predicate, description="surviving consumer"))
    await entered.wait()
    assert len(messages._pending) == 2
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert len(messages._pending) == 1
    ready = True
    messages.on_message(Message())
    await second
    assert not messages._pending


@pytest.mark.asyncio
async def test_message_waiter_cleans_up_timeout_and_predicate_failure() -> None:
    messages = MessageWaiter()
    # A zero guard tests readiness vs a missing edge without another time bet.
    await messages.wait_for(lambda: True, description="ready", timeout=0)
    with pytest.raises(AssertionError, match="never became ready: absent edge"):
        await messages.wait_for(lambda: False, description="absent edge", timeout=0)
    assert not messages._pending

    def broken() -> bool:
        raise ValueError("predicate failed")

    with pytest.raises(ValueError, match="predicate failed"):
        await messages.wait_for(broken, description="broken predicate")
    assert not messages._pending

    def timed_out() -> bool:
        raise TimeoutError("predicate timeout")

    with pytest.raises(TimeoutError, match="predicate timeout"):
        await messages.wait_for(timed_out, description="predicate, not guard, timed out")
    assert not messages._pending


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.asyncio
async def test_message_waiter_observes_real_delayed_boot_and_failure(fail: bool) -> None:
    release = asyncio.Event()
    entered = asyncio.Event()
    session = FakeSession()

    async def factory() -> Any:
        entered.set()
        await release.wait()
        if fail:
            raise RuntimeError("deliberate boot failure")
        return session

    messages = MessageWaiter()
    app = OperatorApp(factory)
    async with app.run_test(size=(100, 24), message_hook=messages.on_message) as pilot:
        await entered.wait()
        assert app._session is None, "premise: real boot is still loading"
        release.set()
        if fail:
            await messages.wait_for(
                lambda: any(
                    "deliberate boot failure" in row.text() for row in app.query(NoticeBlock)
                ),
                description="boot failure notice mounted",
            )
            assert app._session is None
            assert app._boot_failed
        else:
            await messages.wait_for(
                lambda: app._session is session, description="delayed session adoption"
            )
            assert app.query_one(Editor).is_mounted
        await pilot.pause()
        assert not messages._pending
