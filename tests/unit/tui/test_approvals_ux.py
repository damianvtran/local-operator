"""Choosing an approval mode: the scope, the list, and what the band claims.

Three complaints answered here, and each maps to a group below.

* The mode applied to the running session only, so "auto-approve everything
  forever" had no spelling at all — while ``/model default`` had one, on the
  same word, one command away.
* The modes were named in the command's DESCRIPTION and nowhere else, so a user
  who could not remember them typed blind into an argument the app knew the
  answers to. ``/effort`` had the identical shape.
* An owner's frame showed ``! auto-approve`` beside two tool calls reporting
  ``User denied approval``. The denial turned out to be a swallowed exception
  elsewhere, but the hour spent establishing whether the band was lying is the
  cost of a band nothing ties to the gate. So the last group asserts the tie.

Everything here drives the REAL editor and the REAL command dispatch. The
argument list is a keystroke-level feature: a test that called the handler
directly would pass with the list wired to nothing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import SLASH_COMMANDS, OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice, ArgumentMode
from local_operator.tui.widgets.approval import ApprovalBlock, ApprovalPrompt
from local_operator.tui.widgets.command_picker import PickerMode
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _band, _factory
from tests.unit.tui.test_effort import EffortSession


class GatedSession(FakeSession):
    """A fake that keeps the approval handler the app installs on it."""

    def __init__(self) -> None:
        super().__init__()
        self.approval_handler: Any | None = None

    def set_approval_handler(self, handler: object | None) -> None:
        self.approval_handler = handler


def _gate(session: GatedSession):
    """The gate the app installed, narrowed — this is what a tool call reaches."""
    handler = session.approval_handler
    assert handler is not None, "the app installed no approval handler"
    return cast("Any", handler)


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _type(pilot, app: OperatorApp, text: str) -> None:
    """Put ``text`` in the composer the way typing does, and settle the lists.

    Assignment rather than per-character presses: the editor funnels every
    mutation through ``edit()``/``load_text()`` into the same ``_sync_picker``,
    so the list state is identical and the test does not spend a second on
    keystrokes. The tests that are ABOUT keystrokes press real keys.
    """
    app.query_one(Editor).text = text
    await pilot.pause()
    await pilot.pause()


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    """Type a line and send it, dismissing the list first.

    Enter on an open list completes the highlighted row instead of submitting
    what was typed, so a test that skipped the Esc would exercise the
    completion rather than the command it meant to run.
    """
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor.picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _rows(app: OperatorApp) -> list[tuple[str, str, str]]:
    """``(name, description, detail)`` for every row the argument list offers."""
    picker = app.query_one(Editor).picker
    assert picker.mode is PickerMode.ARGUMENT, "the picker is not in argument mode"
    rows: list[tuple[str, str, str]] = []
    for name, choice in picker.suggestions():
        assert isinstance(choice, ArgumentChoice)
        rows.append((name, choice.description, choice.detail))
    return rows


def _saved_mode(config_dir: Path) -> str | None:
    """The mode as it is on DISK — read back through YAML, not through the app.

    Asserting on the app's own attribute would prove only that it remembers
    what it was told; the claim a receipt makes is about a file.
    """
    config_file = config_dir / "config.yml"
    if not config_file.is_file():
        return None
    return yaml.safe_load(config_file.read_text(encoding="utf-8"))["values"].get(
        "tool_approval_mode"
    )


@pytest.fixture()
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the config at a temp dir — these tests WRITE one."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path


# -- scope: this session versus every session ---------------------------------


@pytest.mark.asyncio
async def test_a_session_mode_changes_nothing_on_disk(config_dir: Path) -> None:
    """The default of the two scopes is the reversible one.

    `/approvals auto` is the same command it always was, and it still ends when
    the window does. The receipt has to SAY so, because nothing else on the
    frame distinguishes it from the durable form — that indistinguishability is
    the complaint.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals auto")

        assert app._approve_all is True
        assert app._approvals_default_auto is False
        assert _saved_mode(config_dir) is None, "a session switch wrote to the config file"
        receipt = _notices(app)[-1]
        assert "(this session)" in receipt
        assert "/approvals default auto" in receipt, "the durable form is unreachable"


@pytest.mark.asyncio
async def test_the_default_form_writes_the_config_and_names_the_file(config_dir: Path) -> None:
    """`/approvals default auto` — the promotion, spelled `/model`'s way.

    The receipt names the file AND the key for the reason `/model default`'s
    does: "saved" alone is a claim the user cannot check without quitting.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals default auto")

        # Both halves: the durable one AND this session, so nobody has to run
        # two commands to end up in the state they asked for.
        assert _saved_mode(config_dir) == "auto"
        assert app._approve_all is True
        assert app._approvals_default_auto is True
        receipt = _notices(app)[-1]
        assert "config.yml" in receipt and "tool_approval_mode auto" in receipt
        assert "every new one" in receipt


@pytest.mark.asyncio
async def test_bare_default_keeps_the_mode_the_session_is_already_in(config_dir: Path) -> None:
    """ "Make THIS the default" is the sentence a user has right after switching.

    Same affordance as bare `/model default`, and for the same reason: making
    them retype the word they just typed is a transcription exercise.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals auto")
        assert _saved_mode(config_dir) is None
        await _submit(pilot, app, "/approvals default")

        assert _saved_mode(config_dir) == "auto"


@pytest.mark.asyncio
async def test_the_saved_default_is_in_force_in_the_next_session(config_dir: Path) -> None:
    """A config value nothing reads back is a file the app writes to itself.

    So this boots a SECOND app against the same config dir and asks the gate
    itself — not the flag, the callable a tool call actually awaits.
    """
    first = OperatorApp(lambda: _factory(GatedSession()))
    async with first.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, first)
        await _submit(pilot, first, "/approvals default auto")
    assert _saved_mode(config_dir) == "auto"

    session = GatedSession()
    relaunched = OperatorApp(lambda: _factory(session))
    async with relaunched.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, relaunched)

        assert relaunched._approve_all is True
        assert relaunched._approvals_default_auto is True
        # The gate, not the flag: a write-only default would pass every
        # assertion above and still stop the first tool of the session.
        assert await _gate(session)("bash", "run: ls") is True
        assert not relaunched.query(ApprovalPrompt), "auto-approve still mounted a prompt"
        assert "auto-approve always" in _band(relaunched)


@pytest.mark.asyncio
async def test_a_session_switched_away_from_its_default_says_both(config_dir: Path) -> None:
    """The two-valued state, reported without leaving the screen.

    A user who booted on a saved `auto` and turned it off for this session is
    one relaunch away from a mode they last chose days ago. Nothing else on the
    frame would ever mention it: the band goes quiet when the gate is armed,
    which is correct, and correct is not the same as complete.
    """
    first = OperatorApp(lambda: _factory(GatedSession()))
    async with first.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, first)
        await _submit(pilot, first, "/approvals default auto")

    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals ask")
        await _submit(pilot, app, "/approvals")

        report = _notices(app)[-1]
        assert "ask (this session)" in report
        assert "new sessions open in auto" in report
        assert "/approvals default ask" in report


@pytest.mark.asyncio
async def test_a_matched_pair_reports_one_state(config_dir: Path) -> None:
    """When the session and the default agree there is one fact, said once.

    The split sentence is for the split state; using it unconditionally would
    make every bare `/approvals` read like a discrepancy report.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals")

        report = _notices(app)[-1]
        assert "tool approvals: ask" in report
        assert "new sessions open the same way" in report
        assert "this session" not in report


@pytest.mark.asyncio
async def test_an_unwritable_config_still_switches_the_session(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    """A read-only config dir is a reason not to promise the next launch
    anything, not a reason to refuse this session the mode it asked for.

    And the band must not claim `always` off a write that failed — that is the
    same class of lie as the band the last group is about.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        monkeypatch.setattr(
            "local_operator.config.ConfigManager.set_config_value",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only file system")),
        )
        await _submit(pilot, app, "/approvals default auto")

        assert app._approve_all is True, "the session was denied the mode it asked for"
        assert app._approvals_default_auto is False
        assert "could not save default" in _notices(app)[-1]
        band = _band(app)
        assert "auto-approve" in band and "always" not in band


# -- the list: offered, not remembered ----------------------------------------


@pytest.mark.asyncio
async def test_approvals_offers_both_modes_and_both_scopes(config_dir: Path) -> None:
    """The reported UX bug: the modes were named in prose and typed from memory.

    Four rows, because scope is the axis that was invisible. A `default` row
    leading to a second list would put the mode back behind a keystroke the
    list cannot show, which is the same failure one level down.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, app, "/approvals ")

        assert [row[0] for row in _rows(app)] == ["ask", "auto", "default ask", "default auto"]
        scopes = {name: detail for name, _description, detail in _rows(app)}
        assert scopes["ask"].startswith("this session")
        assert scopes["auto"].startswith("this session")
        assert scopes["default ask"].startswith("every session")
        assert scopes["default auto"].startswith("every session")
        # Every row says what the mode DOES, not just what it is called.
        assert all(description for _name, description, _detail in _rows(app))


@pytest.mark.asyncio
async def test_the_list_marks_the_live_mode_and_the_saved_one_separately(
    config_dir: Path,
) -> None:
    """Two marks, because the user can be in two places at once.

    `· current` is the mode running now and `· saved` is what the next launch
    opens in; a list carrying only one of them cannot show a session that has
    been switched away from its default, which is the state most worth seeing
    at the moment of choosing.
    """
    first = OperatorApp(lambda: _factory(GatedSession()))
    async with first.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, first)
        await _submit(pilot, first, "/approvals default auto")

    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/approvals ask")
        await _type(pilot, app, "/approvals ")

        marks = {name: detail for name, _description, detail in _rows(app)}
        assert marks["ask"] == "this session · current"
        assert marks["auto"] == "this session"
        assert marks["default auto"] == "every session · saved"
        assert marks["default ask"] == "every session"


@pytest.mark.asyncio
async def test_choosing_a_row_runs_the_command_it_spells(config_dir: Path) -> None:
    """The list completes into the ARGUMENT and submits the same line a typist
    would have typed — one implementation of what `/approvals default auto`
    means, not a second path that can drift from the first."""
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, app, "/approvals ")
        # Arrow onto the row rather than trusting the matcher's pick: an
        # explicit move is what the editor's ambiguity gate accepts as "the
        # user chose this", and it is the gesture the feature is for.
        for _ in range(3):
            await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert app.query_one(Editor).text == ""
        assert app._approve_all is True
        assert _saved_mode(config_dir) == "auto"


@pytest.mark.asyncio
async def test_typing_the_mode_still_works_without_the_list(config_dir: Path) -> None:
    """The list is an addition for people who do not remember the options, not
    a gate in front of people who do.

    Both the canonical word and an accepted alias, pressed as real keys, with
    the list dismissed — the muscle memory that existed before this change must
    survive it.
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        for character in "/approvals yolo":
            await pilot.press(
                "slash" if character == "/" else ("space" if character == " " else character)
            )
        await pilot.pause()
        await pilot.press("escape")  # dismiss the list; the typist does not need it
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert app._approve_all is True

        await _submit(pilot, app, "/approvals ask")
        assert app._approve_all is False


@pytest.mark.asyncio
async def test_a_bare_approvals_still_reports_instead_of_opening_a_list(
    config_dir: Path,
) -> None:
    """`/approvals` answers "what am I on", so Enter on its row SENDS it.

    That is the difference between an OPTIONAL argument and `/login`'s REQUIRED
    one, and it is why the two are separate values rather than one boolean: a
    command with a useful bare form must not have Enter silently repurposed
    into "open a list".
    """
    app = OperatorApp(lambda: _factory(GatedSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).focus()
        for character in "/approvals":
            await pilot.press("slash" if character == "/" else character)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert any("tool approvals: ask" in notice for notice in _notices(app))


@pytest.mark.asyncio
async def test_effort_offers_this_models_rungs_with_the_current_one_marked(
    config_dir: Path,
) -> None:
    """The second customer of the same mechanism, converted in the same change.

    `/effort` printed a ladder the user then had to transcribe. The rungs are
    the model's, read off the spec the request is built from, so a list that
    offered a fixed set would be wrong on the next model.
    """
    session = EffortSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, app, "/effort ")

        names = [name for name, _description, _detail in _rows(app)]
        assert names[0] == "auto", "the way back has to be offered, not remembered"
        assert names[1:] == list(session.model.reasoning_efforts)
        marked = [name for name, _description, detail in _rows(app) if detail == "current"]
        assert marked == [session.model.reasoning_effort or "auto"]


@pytest.mark.asyncio
async def test_choosing_an_effort_row_puts_it_on_the_spec(config_dir: Path) -> None:
    """The rung has to reach the REQUEST, not just the transcript."""
    session = EffortSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _type(pilot, app, "/effort low")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert session.model.reasoning_effort == "low"


def test_the_registry_states_which_commands_offer_values() -> None:
    """One declaration per command, and the two kinds are not interchangeable.

    Pinned as a table for the reason ``ECHO_POLICY`` is: the field has a safe
    default, so without this a new command silently inherits "no list" and the
    absence never shows up as a failure.
    """
    modes = {
        command.name: command.arguments
        for command in SLASH_COMMANDS
        if command.arguments is not ArgumentMode.NONE
    }
    assert modes == {
        "effort": ArgumentMode.OPTIONAL,
        "approvals": ArgumentMode.OPTIONAL,
        "login": ArgumentMode.REQUIRED,
        "logout": ArgumentMode.REQUIRED,
    }
    # `/provider` was the third candidate and is deliberately not here: it takes
    # no argument at all — `_cmd_providers` ignores what follows it — so a list
    # would offer values the handler discards.
    assert next(c for c in SLASH_COMMANDS if c.name == "provider").arguments is ArgumentMode.NONE


# -- the band cannot outrun the gate ------------------------------------------


@pytest.mark.asyncio
async def test_the_band_and_the_gate_agree_through_every_route(config_dir: Path) -> None:
    """The invariant the owner's confusing frame is the argument for.

    Each route that can change the mode — the prompt's `A`, the command, both
    directions — is followed by asking the GATE what it does and the BAND what
    it says. They are set in one place precisely so this cannot drift; the test
    is what stops the next author reintroducing a second writer.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        gate = _gate(session)

        async def gate_runs_without_asking() -> bool:
            pending = asyncio.ensure_future(gate("bash", "run: echo hi"))
            await pilot.pause(0.2)
            if pending.done():
                return await pending
            # A prompt is on screen: the gate is armed. Answer it so the test
            # leaves nothing parked on a future.
            await pilot.press("n")
            await asyncio.wait_for(pending, 2)
            return False

        assert await gate_runs_without_asking() is False
        assert "auto-approve" not in _band(app)

        await _submit(pilot, app, "/approvals auto")
        assert await gate_runs_without_asking() is True
        assert "auto-approve" in _band(app) and "always" not in _band(app)

        await _submit(pilot, app, "/approvals default auto")
        assert await gate_runs_without_asking() is True
        assert "auto-approve always" in _band(app)

        await _submit(pilot, app, "/approvals ask")
        assert await gate_runs_without_asking() is False
        assert "auto-approve" not in _band(app)


@pytest.mark.asyncio
async def test_the_live_prompt_is_untouched_by_the_default_machinery(
    config_dir: Path,
) -> None:
    """The in-flight gate is a SEPARATE mechanism and stays exactly as it was.

    Three behaviours pinned here because this change touched the mode they read:
    a question still mounts and still waits; `n` refuses that one tool and
    leaves the turn's next ask free to ask again; and the turn-scoped deny latch
    still drains a stopped turn's queued asks without a card. None of them is
    about the default — that is the point.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        gate = _gate(session)

        first = asyncio.ensure_future(gate("bash", "run: one"))
        await pilot.pause(0.3)
        # The live question is the docked card; the transcript keeps the
        # RECEIPT once it is answered. Two widgets, two jobs.
        assert app.query(ApprovalPrompt), "the prompt no longer mounts"
        assert not first.done(), "the gate stopped waiting for an answer"
        await pilot.press("n")
        assert await asyncio.wait_for(first, 2) is False

        # …and one refusal is not a mode: the next ask still asks.
        second = asyncio.ensure_future(gate("write", "write: two"))
        await pilot.pause(0.3)
        assert app.query(ApprovalPrompt)
        await pilot.press("y")
        assert await asyncio.wait_for(second, 2) is True
        # The turn-scoped latch: a stop drains what the stopped turn queued,
        # with no card, and that is still true with a saved default in play.
        await _submit(pilot, app, "/approvals default ask")
        receipts = len(app.query(ApprovalBlock))
        app._deny_queued_approvals()
        third = asyncio.ensure_future(gate("bash", "run: three"))
        assert await asyncio.wait_for(third, 2) is False
        await pilot.pause(0.2)
        # No question was raised for the stopped turn's ask...
        assert not app.query(ApprovalPrompt), "a stopped turn's ask mounted a question"
        # ...and no receipt was written for a decision the user never made.
        # Counted rather than queried for emptiness: the two answered prompts
        # above left receipts and stay on screen, which is what a ledger is for.
        assert len(app.query(ApprovalBlock)) == receipts


@pytest.mark.asyncio
async def test_the_allow_all_key_reports_and_paints_like_the_command(
    config_dir: Path,
) -> None:
    """`A` on the prompt is the other route into auto, and it is session-scoped.

    It must not write a default — a keystroke answering one question cannot
    reasonably be a standing preference — but it must paint the same band the
    command does, through the same writer.
    """
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        pending = asyncio.ensure_future(_gate(session)("bash", "run: one"))
        await pilot.pause(0.3)
        await pilot.press("A")
        assert await asyncio.wait_for(pending, 2) is True

        assert app._approve_all is True
        assert "auto-approve" in _band(app)
        assert "always" not in _band(app), "a keystroke must not claim a saved default"
        assert _saved_mode(config_dir) is None
