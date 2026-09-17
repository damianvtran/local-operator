"""The flag forms of ``/goal`` and ``/loop``, and the picker row that teaches them.

Two things are being held in place here at once, and they pull in opposite
directions:

* ``--clear``/``--stop`` are FLAGS. ``/goal --clear`` must unset the goal and
  never become the standing objective ``--clear``, on every host that implements
  ``/goal``: the TUI's local handler, its routed one (the owner backend a
  follower calls), and the detached runtime's (``session/runtime/serving.py``,
  exercised in ``tests/unit/session/runtime/test_goal_submission.py`` and
  ``test_desktop_loop.py``).
* ``/goal`` and ``/loop`` are still FREE-TEXT commands. They now open a value
  list at the space so the flag row is discoverable, and the two behaviours that
  a value list would otherwise take away — the ``$skill`` claim inside the
  argument, and the inline reassembly of a bare ``/goal`` — are asserted
  unchanged, because the code that decided them used to read "has a value list"
  as "has a NAME slot" and those two facts were only ever the same by accident.

The picker half is driven through the real ``OperatorApp``: the row is a
rendered surface, so the assertion is on the rows the widget derived, not on a
call the test made.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len

from local_operator.session.goal import (
    CLEARED_GOAL_ECHO_CHARS,
    GOAL_CLEAR_ARGS,
    cleared_goal_receipt,
)
from local_operator.session.goal_loop import LOOP_CLEAR_ARGS, LOOP_STOP_ARGS
from local_operator.slash_commands import SLASH_COMMANDS, slash_command_for
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.command_picker import (
    CommandPicker,
    PickerMode,
    skill_token,
)
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

from .test_app_pilot import FakeSession, _factory

#: Every host's vocabulary, read from the one definition each host imports. A
#: test that spelled the words out would pass while a host drifted.
CLEAR_WORDS = ("clear", "none", "reset")


async def _boot(pilot, app: OperatorApp) -> None:
    """Settle until the session exists — `_cmd_goal` rejects a set before then."""
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


def _notice_texts(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


async def _settle(pilot, app: OperatorApp) -> None:
    """Let the worker a submit started finish, as the reported path does.

    The clear/submit path runs as a worker, so the receipt is not painted by the
    time Enter returns.
    """
    await app.workers.wait_for_complete()
    await pilot.pause()
    await pilot.pause()


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    """Type a line into the real editor and press Enter — the reported path.

    The picker is dismissed first where it is showing: Enter on an open list
    COMPLETES the highlighted row and runs THAT, so a test aiming at the typed
    form would otherwise be testing the picker. That row path is covered on its
    own by ``test_accepting_the_goal_row_fills_before_it_runs``; this helper drives
    the other one, the words the user typed with the list dismissed.
    """
    editor = app.query_one(Editor)
    editor.focus()
    editor.text = text
    editor.move_cursor(editor._end_of_buffer())
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    # The submit path runs as a worker, so the prompt the handler started is not
    # recorded by the time Enter returns; settle it before asserting on it.
    await app.workers.wait_for_complete()
    await pilot.pause()
    await pilot.pause()


async def _draft(app: OperatorApp, pilot, text: str) -> Editor:
    """Put ``text`` in the composer with the caret at its END.

    The caret position is the whole subject for the picker assertions:
    `slash_argument` is caret-anchored, so a caret left at offset 0 is not
    "inside the argument region" and the contract would go untested.
    """
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.load_text(text)
    editor.move_cursor(editor._end_of_buffer())
    for _ in range(50):
        await pilot.pause()
        if editor.picker.is_open():
            break
    return editor


def _row_names(editor: Editor) -> list[str]:
    return [name for name, _ in editor.picker._matches]


# ---------------------------------------------------------------------------
# `/goal --clear` — local TUI, on the typed path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_goal_flag_clears_the_goal_and_runs_no_turn() -> None:
    """`/goal --clear` takes the goal away and sends nothing.

    The pairing matters: a clear that also submitted would hand the model the
    literal argument, which is the one outcome the flag exists to prevent.
    """
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal --clear")
        assert session.goal == ""
        assert session.prompts == []
        # The receipt NAMES the goal it took away: it is the only place the
        # removed text is still visible (design D4 / UX U3).
        assert "goal cleared: land the OAuth refresh fix" in _notice_texts(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("word", CLEAR_WORDS)
async def test_the_bare_clear_words_still_work(word: str) -> None:
    """Backwards compatibility: the undocumented forms keep working."""
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/goal {word}")
        assert session.goal == ""
        assert session.prompts == []


@pytest.mark.asyncio
async def test_the_goal_flag_is_never_stored_as_a_goal_body() -> None:
    """The flag belongs to ``GOAL_CLEAR_ARGS``, and the goal is empty after it.

    Asserting on ``session.goal`` rather than on the notice is the point: the
    failure this guards would show "goal set" over a standing objective that
    reads ``--clear``, and only the state tells the two apart.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal --clear")
        assert session.goal == ""
        assert "--clear" not in session.goal
        assert session.prompts == []


@pytest.mark.asyncio
async def test_a_goal_body_that_opens_with_the_flag_is_still_a_goal() -> None:
    """The flag is the WHOLE argument, never a prefix.

    ``/goal --clear the flaky job`` is free text a user meant as an objective;
    treating it as a flag would silently eat the tail of a real goal — the one
    command whose argument the model is told.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal --clear the flaky job")
        assert session.goal == "--clear the flaky job"
        assert session.prompts == ["--clear the flaky job"]


# ---------------------------------------------------------------------------
# `/goal --clear` — the routed (owner-backend) path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("arg", ["--clear", *CLEAR_WORDS])
async def test_the_routed_goal_handler_clears_without_submitting(arg: str) -> None:
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        result = await app.run_slash_authoritative("goal", arg)
        assert result["kind"] == "notice"
        # The receipt NAMES what went (design D4 / UX U3): a standing goal is
        # invisible in the UI and there is no undo, so this line is the user's
        # only chance to see and retype what a mistaken clear took away.
        assert result["text"] == "goal cleared: land the OAuth refresh fix"
        assert session.goal == ""
        assert session.prompts == []


@pytest.mark.asyncio
async def test_the_routed_goal_flag_is_not_a_goal_body() -> None:
    """A follower's ``/goal --clear`` must not store the flag on the owner."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        result = await app.run_slash_authoritative("goal", "--clear")
        assert session.goal == ""
        # No `goal_set` receipt means no viewer is told to submit anything.
        assert "data" not in result or not result.get("data")


# ---------------------------------------------------------------------------
# `/loop --stop` and `/loop --clear` — local TUI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("arg", ["--stop", "stop", "cancel", "abort"])
async def test_the_loop_stop_forms_ask_a_running_loop_to_stop(arg: str) -> None:
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._loop_running = True
        await _submit(pilot, app, f"/loop {arg}")
        assert app._loop_cancelled is True
        assert "loop will stop after the current turn" in _notice_texts(app)


@pytest.mark.asyncio
async def test_the_loop_stop_forms_still_refuse_when_nothing_runs() -> None:
    """The honest sentence stays exactly as it was: what THIS terminal knows."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/loop --stop")
        assert app._loop_cancelled is False
        assert "no loop is running in THIS terminal" in _notice_texts(app)


@pytest.mark.asyncio
async def test_loop_clear_while_running_clears_this_terminals_loop() -> None:
    """In the TUI `--clear` means the only thing this host can mean by it.

    This surface publishes no loop state, so a running loop IS the state a
    clear would remove — refusing and naming `/loop --stop` taught a flag that
    does nothing here (round 1: UX U4, reviewer NIT-6). The receipt says what
    happened rather than borrowing the stop sentence.
    """
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._loop_running = True
        await _submit(pilot, app, "/loop --clear")
        assert app._loop_cancelled is True
        notices = _notice_texts(app)
        assert any("loop cleared" in text for text in notices), notices


@pytest.mark.asyncio
async def test_loop_clear_when_idle_answers_a_clear_request() -> None:
    """In the TUI there is no published loop state to clear.

    The app-local loop is published nowhere, so the honest answer is about
    CLEARING — the `--stop` sentence names running, which is a different
    question and read as though `--clear` were a typo (NIT-6).
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/loop --clear")
        assert "nothing to clear in THIS terminal — no loop is running here" in _notice_texts(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("arg", ["--stop", "--clear"])
async def test_the_routed_loop_handler_answers_the_flags(arg: str) -> None:
    """The owner-backend path carries the same two flags, the same two ways."""
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        idle = await app.run_slash_authoritative("loop", arg)
        assert idle["kind"] == "notice"
        assert "no loop is running" in idle["text"]
        # The idle SENTENCE differs by what the host can see (QA Q5: a
        # pre-existing divergence this change does not normalise), so a clear
        # asks about clearing here too.
        assert ("nothing to clear" in idle["text"]) is (arg == "--clear")

        app._loop_running = True
        running = await app.run_slash_authoritative("loop", arg)
        assert running["kind"] == "notice"
        assert running["text"] == (
            "loop cleared — stopping after the current turn"
            if arg == "--clear"
            else "loop will stop after the current turn"
        )
        assert app._loop_cancelled is True


@pytest.mark.asyncio
@pytest.mark.parametrize("arg", ["--stop ", "--stop\t", "cancel "])
async def test_a_trailing_space_does_not_turn_a_flag_into_a_loop_start(arg: str) -> None:
    """`Command.args` is a plain `str` on the wire: the space arrives verbatim.

    `serving.py`'s loop branch matched the flag UNSTRIPPED, so `/loop --stop `
    silently no-opped with `loop_busy` while the loop kept running and
    `/loop --clear ` on an idle driver started a paid goal-mode loop toward the
    literal goal `--clear` (round 1, reviewer MAJOR-2). Both TUI handlers strip
    the same argument, so the hosts disagreed about one word.
    """
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._loop_running = True
        running = await app.run_slash_authoritative("loop", arg)
        assert running["kind"] == "notice"
        assert "already running" not in running["text"], running["text"]
        assert app._loop_cancelled is True
        # And the bare word is not stored as a goal either.
        assert session.goal == "land the OAuth refresh fix"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "arg", "supported"),
    [
        ("goal", "--stop", "/goal --clear"),
        ("goal", "--cli", "/goal <text> sets a goal"),
        ("loop", "--clearx", "/loop <n> runs n turns"),
        ("loop", "--stopx", "/loop --stop cancels"),
    ],
)
async def test_a_bare_unknown_flag_is_refused_for_both_commands(
    command: str, arg: str, supported: str
) -> None:
    """Two flag vocabularies are taught, so mixing them is the expected mistake.

    Under the whole-argument flag rule the mismatch became the VALUE —
    `/goal --stop` stored `--stop` as the standing objective and submitted a
    turn carrying it (round 1: UX U6, reviewer NIT-5).
    """
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        result = await app.run_slash_authoritative(command, arg)
        assert result["kind"] == "notice"
        assert result.get("style") == "warning"
        assert f"unknown flag {arg}" in result["text"]
        assert supported in result["text"]
        assert session.goal == "land the OAuth refresh fix"
        assert session.prompts == []


@pytest.mark.asyncio
async def test_an_unknown_flag_typed_locally_is_refused_not_stored() -> None:
    """The local handlers refuse it too, and start no turn."""
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal --stop")
        notices = _notice_texts(app)
        assert any("unknown flag --stop" in text for text in notices), notices
        assert session.goal == "land the OAuth refresh fix"
        assert session.prompts == []
        await _submit(pilot, app, "/loop --clearx")
        notices = _notice_texts(app)
        assert any("unknown flag --clearx" in text for text in notices), notices
        assert app._loop_running is False


@pytest.mark.asyncio
async def test_a_whole_argument_flag_with_a_tail_still_becomes_a_goal() -> None:
    """The refusal is narrow BY CONTRACT: only a bare token is a flag attempt.

    `/goal --clear the flaky job` must stay an objective — eating the tail of a
    real goal would be silent data loss in the one command whose argument the
    MODEL is told.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        result = await app.run_slash_authoritative("goal", "--clear the flaky job")
        assert result["text"] == "goal set"
        assert session.goal == "--clear the flaky job"


# ---------------------------------------------------------------------------
# The picker row — rendered through the real app
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_goal_argument_offers_the_clear_flag() -> None:
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/goal ")
        assert editor.picker.mode is PickerMode.ARGUMENT
        assert _row_names(editor) == ["--clear"]
        row = editor.picker.render_rows(100)[0].plain
        assert "Clear the standing goal" in row


@pytest.mark.asyncio
async def test_the_goal_row_goes_once_anything_is_typed() -> None:
    """Free text is the argument's ordinary content, so the offer withdraws."""
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/goal ship it")
        assert _row_names(editor) == []
        assert not editor.picker.is_open()


@pytest.mark.asyncio
async def test_no_goal_means_no_goal_row() -> None:
    """A palette advertising `--clear` with nothing to clear is a dead end."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/goal ")
        assert _row_names(editor) == []
        assert not editor.picker.is_open()


@pytest.mark.asyncio
async def test_a_running_loop_offers_the_stop_flag() -> None:
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._loop_running = True
        editor = await _draft(app, pilot, "/loop ")
        assert editor.picker.mode is PickerMode.ARGUMENT
        assert _row_names(editor) == ["--stop"]
        assert "Stop the running loop" in editor.picker.render_rows(100)[0].plain


@pytest.mark.asyncio
async def test_an_idle_loop_offers_nothing() -> None:
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/loop ")
        assert _row_names(editor) == []
        assert not editor.picker.is_open()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "bare"),
    [("goal", "clear"), ("loop", "stop")],
)
async def test_the_bare_word_still_acts_on_one_enter(command: str, bare: str) -> None:
    """The row is `alert`, and the BARE word is still spelled-in-full.

    The row is named with its dashes (`--clear`, `--stop`) while the same action
    has a bare spelling both commands have always honoured and this PR keeps
    (`clear`/`none`/`reset`, `stop`/`cancel`/`abort`) — the words the pre-existing
    `test_app_pilot` pins type. So `Editor._picker_choice_is_unambiguous` counts
    either spelling as "typed in full": the first cut of the `alert` gate treated
    only `--clear` as spelled, which turned `/goal clear` + Enter (one keystroke
    before the row existed) into a completion needing a second one.
    """
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._loop_running = command == "loop"
        expected_row = "--clear" if command == "goal" else "--stop"
        editor = await _draft(app, pilot, f"/{command} {bare}")
        assert _row_names(editor) == [expected_row], editor.picker.render_rows(100)
        await pilot.press("enter")
        await _settle(pilot, app)
        if command == "goal":
            assert session.goal == ""
            assert session.prompts == []
        else:
            assert app._loop_cancelled is True
        # The row was RUN, not completed into the buffer: one keystroke, as before.
        assert editor.text == ""


@pytest.mark.asyncio
async def test_accepting_the_goal_row_fills_before_it_runs() -> None:
    """Two Enters, because one Enter used to destroy what the keystroke READS.

    The row is pre-selected and is the only match, so without `alert` the
    editor's `_picker_choice_is_unambiguous` RUNS it — and `/goal ` + Enter, the
    keystroke that reports the standing goal and the one `/goal`'s own
    description advertises, cleared it instead (round 1: design D1, UX U1,
    reviewer MAJOR-1). This is the app's established shape for a row that
    removes something: first Enter fills, second Enter runs.
    """
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/goal ")
        assert _row_names(editor) == ["--clear"]
        assert editor._argument_is_destructive()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        # The FIRST Enter only fills: the goal is untouched and nothing ran.
        assert editor.text == "/goal --clear"
        assert session.goal == "land the OAuth refresh fix"
        assert "goal cleared" not in " ".join(_notice_texts(app))
        # The second one accepts the completed row.
        await pilot.press("enter")
        await _settle(pilot, app)
        assert session.goal == ""
        assert session.prompts == []
        assert any(text.startswith("goal cleared") for text in _notice_texts(app))


@pytest.mark.asyncio
async def test_accepting_the_loop_row_fills_before_it_runs() -> None:
    """The same gate on `/loop `: one Enter must not stop a running loop.

    On the base tree the identical keys hit the "a loop is already running"
    REFUSAL, so an unguarded row turned a report into a cancel (UX U2).
    """
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._loop_running = True
        editor = await _draft(app, pilot, "/loop ")
        assert _row_names(editor) == ["--stop"]
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert editor.text == "/loop --stop"
        assert app._loop_cancelled is False
        await pilot.press("enter")
        await _settle(pilot, app)
        assert app._loop_cancelled is True


@pytest.mark.asyncio
async def test_a_deliberate_move_onto_the_row_still_runs_it_in_one_press() -> None:
    """The gate costs the IMPLICIT path only.

    An explicit down-arrow onto the row is the editor's own definition of
    unambiguous, so a user who deliberately reaches for the flag keeps the single
    keystroke.
    """
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/goal ")
        assert _row_names(editor) == ["--clear"]
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await _settle(pilot, app)
        assert session.goal == ""
        assert editor.text == ""


@pytest.mark.asyncio
async def test_clicking_the_goal_row_fills_instead_of_clearing() -> None:
    """The MOUSE path takes the same gate, not a second one.

    A click used to run a non-destructive row outright (`_apply_command` →
    `_run_argument`), so one click on `--clear` cleared — the reviewer's note on
    MAJOR-1, and the reason the fix is `alert=True` rather than a keyboard-only
    guard. A click names one exact row, which the editor already treats as an
    explicit choice; `alert` is what makes a DESTRUCTIVE row still stop to be
    confirmed, exactly as it does on `/logout`.
    """
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/goal ")
        assert _row_names(editor) == ["--clear"]
        # Row 0 is at the picker's first content row: the one-row offer has no
        # notice row above it, so `content_region` == `region` here.
        await pilot.click(CommandPicker, offset=(4, 0))
        for _ in range(60):
            await pilot.pause()
            if editor.text != "/goal ":
                break
        assert editor.text == "/goal --clear"
        assert session.goal == "land the OAuth refresh fix"
        assert not [text for text in _notice_texts(app) if text.startswith("goal cleared")]


# ---------------------------------------------------------------------------
# What the value list must NOT have taken away
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_skill_token_still_opens_inside_the_goal_argument() -> None:
    """`/goal` has a value list now, and still no NAME slot.

    The floor that lets `$skill` open inside the argument asks whether the
    command's first token is a roster NAME. Reading "has a value list" there
    swallowed the token for `/goal` and `/loop` when the flag row landed, which
    is why the fact lives on the registry as `name_argument`.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/goal $cte")
        token = skill_token(
            editor.text,
            editor._caret_offset(),
            editor._command_names,
            editor._prompt_command_names,
            editor._name_prompt_commands,
        )
        assert token is not None and token.query == "cte"
        assert editor._picker_phase() == "skill"


@pytest.mark.asyncio
async def test_a_typed_goal_still_submits_as_free_text() -> None:
    """The list is an OFFER: nothing here filters what may be submitted."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal land the OAuth refresh fix")
        assert session.goal == "land the OAuth refresh fix"
        assert session.prompts == ["land the OAuth refresh fix"]


@pytest.mark.asyncio
async def test_the_picker_row_does_not_start_a_turn_when_the_draft_is_typed_over() -> None:
    """Typing over the offer leaves an ordinary goal: no row, no interference."""
    session = FakeSession()
    session.set_goal("land the OAuth refresh fix")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/goal ")
        assert _row_names(editor) == ["--clear"]
        editor.load_text("/goal ship the flag work")
        editor.move_cursor(editor._end_of_buffer())
        for _ in range(10):
            await pilot.pause()
        assert not editor.picker.is_open()
        await _submit(pilot, app, "/goal ship the flag work")
        assert session.goal == "ship the flag work"


# ---------------------------------------------------------------------------
# The registry declarations the mechanism rests on
# ---------------------------------------------------------------------------


def test_the_palette_descriptions_name_the_flag_forms() -> None:
    """Discoverability is the requirement, so the palette must carry the flag.

    ``--clear`` is deliberately absent from the `/loop` row: the TUI cannot
    dismiss a published loop state it does not have, and the runtime refuses it
    while a loop runs, so the 53-cell form sizes against the `/help` budget at 80
    columns instead and names only the flag that acts on every host.
    """
    goal = slash_command_for("/goal")
    loop = slash_command_for("/loop")
    assert goal is not None and "--clear" in goal.description
    assert loop is not None and "--stop" in loop.description
    # The 80-column `/help` description budget is `W - 26`, i.e. 54 cells.
    assert cell_len(loop.description) <= 54


def test_the_clear_vocabularies_hold_the_flag_and_the_legacy_words() -> None:
    """One definition per command, read by every host that implements it."""
    assert "--clear" in GOAL_CLEAR_ARGS
    assert {"clear", "none", "reset"} <= GOAL_CLEAR_ARGS
    assert {"stop", "cancel", "abort"} <= LOOP_STOP_ARGS and "--stop" in LOOP_STOP_ARGS
    assert LOOP_CLEAR_ARGS == frozenset({"--clear"})
    # The two loop sets are disjoint on purpose: collapsing them would make
    # `--clear` cancel live work by accident.
    assert not (LOOP_STOP_ARGS & LOOP_CLEAR_ARGS)


def test_the_clear_receipt_names_what_it_removed() -> None:
    """The receipt is the only place a cleared goal is still readable.

    A standing goal is invisible in the UI and there is no undo, so the echo is
    the user's whole chance to retype what a mistaken clear took away (design
    D4 / UX U3). One line, because a receipt is a single terminal row.
    """
    assert cleared_goal_receipt("land the OAuth refresh fix") == (
        "goal cleared: land the OAuth refresh fix"
    )
    assert cleared_goal_receipt("") == "goal cleared"
    multi = cleared_goal_receipt("first line\n\nsecond line")
    assert multi == "goal cleared: first line second line"
    long = cleared_goal_receipt("x" * 400)
    assert long == "goal cleared: " + "x" * CLEARED_GOAL_ECHO_CHARS + "…"


def test_the_name_slot_flag_is_only_on_the_roster_commands() -> None:
    """`name_argument` states the fact the picker floor and the composer read.

    Pinned because it is a defaulted field: without this a new command silently
    inherits "no name slot", and — the reason the field exists — a command with
    a value list would inherit a name slot it does not have.
    """
    flagged = {command.name for command in SLASH_COMMANDS if command.name_argument}
    assert flagged == {"team", "agent"}
    spellings = {
        name for command in SLASH_COMMANDS if command.name_argument for name in command.names
    }
    assert spellings == set(Editor.NAME_ARGUMENT_COMMANDS)


@pytest.mark.asyncio
async def test_the_editor_and_the_picker_derive_the_same_name_slot_set() -> None:
    """Both copies come from the registry flag, so they cannot disagree."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        assert editor._name_prompt_commands == editor.picker._name_prompt_commands
        assert editor._name_prompt_commands == frozenset(Editor.NAME_ARGUMENT_COMMANDS)
