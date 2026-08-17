"""Which slash commands write a user row into the visible ledger, and which do not.

The echo used to be unconditional: every ``/`` line the user pressed Enter on
landed in the transcript as a :class:`UserBlock` before its handler ran. The
owner's report was that this filled the reading record with rows saying nothing
— ``/usage`` opens a panel that already IS the answer, so the row above it was
a keystroke log pretending to be conversation.

The policy now lives on the registry entry (``SlashCommand.echo``), and the
table below pins it entry by entry. That pin is the forcing function: a command
added to ``SLASH_COMMANDS`` without an opinion about its own echo fails here,
which is the only way "state your choice" can be enforced on a field that has
to keep a default for the picker fixtures that have no opinion at all.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.session.goal import MAX_GOAL_CHARS, GoalState
from local_operator.tui.app import SLASH_COMMANDS, OperatorApp, slash_command_for
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock
from local_operator.tui.widgets.usage_panel import UsagePanel
from local_operator.tui.widgets.welcome import WelcomeView
from tests.unit.tui.test_app_pilot import FakeProviderController, FakeSession, _factory

#: The settled policy, one row per registry entry.
#:
#: ``/goal`` is the only ``True``: it is the one command whose ARGUMENT reaches
#: the model — the goal rides the system prompt's volatile tail on every later
#: turn — so the transcript's job of showing what the model was told applies to
#: it and to nothing else here. Every other handler already prints, lists, or
#: opens its own receipt, and the reasoning per command is a comment beside the
#: entry in ``SLASH_COMMANDS``.
ECHO_POLICY = {
    "help": False,
    "exit": False,
    "clear": False,
    "new": False,
    "reload": False,
    "resume": False,
    "model": False,
    # A setting, not words the model is told: the receipt names the level that
    # is now in force and the band carries it from then on.
    "effort": False,
    "provider": False,
    "search": False,
    "accounts": False,
    "usage": False,
    "goal": True,
    "loop": False,
    # The aside's whole promise is that the exchange leaves no trace in the
    # ledger; a user row would be the one it left. See `SLASH_COMMANDS`.
    "btw": False,
    "compact": False,
    "context": False,
    "approvals": False,
    "skills": False,
    "mcp": False,
    "login": False,
    "logout": False,
}


def _user_rows(app: OperatorApp) -> list[str]:
    return [
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock)
    ]


def _notice_texts(app: OperatorApp) -> list[str]:
    """Notice bodies, unwrapped — a 2000-character row pushes them off a frame."""
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _painted(app: OperatorApp) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def _boot(pilot, app: OperatorApp) -> None:
    """Settle until the session exists.

    Load-bearing for anything touching `/goal`: `_cmd_goal` rejects a set while
    the session is still starting, and the row is written by the handler on the
    branch that stored something — so a test that submitted too early would
    assert against the REJECTION path while reading like it tested the set.
    """
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    """Type a line into the real editor and press Enter — the reported path.

    Calling ``_run_slash_command`` directly would skip the editor and the
    submit handler, which is the pair the reported bug lived in.

    The picker is dismissed first when it is showing. Enter on an open command
    picker COMPLETES the highlighted row and then submits THAT — so a typo like
    ``/USGE`` fuzzy-matches its way to ``/usage`` and the unknown-command path
    is never reached. Esc is the key a user actually presses to keep what they
    typed, so this is the real path to the real branch, not a way around it.
    """
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def test_every_registered_command_states_an_echo_policy() -> None:
    """A new command must land in the table above, with a reason beside its
    registry entry. Without this, the field's default silently decides."""
    assert {command.name: command.echo for command in SLASH_COMMANDS} == ECHO_POLICY


def test_no_two_registry_entries_claim_the_same_word() -> None:
    """`slash_command_for` returns the FIRST entry whose `names` contains the
    token, and it is now the single authority for BOTH the echo permission and
    the dispatch. A duplicate — an alias colliding with a later entry's primary
    name — would silently re-point that command at the earlier handler, and
    neither of the tests around this one would notice: the policy pin keys on
    primary names only, and the alias walk is satisfied by ANY handler running.
    """
    spellings = [spelling for entry in SLASH_COMMANDS for spelling in entry.names]
    duplicates = sorted({s for s in spellings if spellings.count(s) > 1})
    assert duplicates == [], duplicates


def test_an_alias_inherits_its_command_policy() -> None:
    """``/quit`` and ``/exit`` are one entry, so they cannot disagree about
    whether they echo — a per-spelling policy is a policy that drifts."""
    assert slash_command_for("/quit") is slash_command_for("/exit")
    assert slash_command_for("/recall") is slash_command_for("/resume")
    # Case-insensitive, matching the dispatcher: one resolver decides what a
    # word means, so a spelling cannot echo as one command and run as another.
    assert slash_command_for("/USAGE now") is slash_command_for("/usage")
    assert slash_command_for("/usge") is None
    assert slash_command_for("hello") is None


@pytest.mark.asyncio
async def test_every_registered_name_and_alias_actually_runs() -> None:
    """Making the resolver shared caught a live bug: the dispatcher matched raw
    literals, so the two aliases no branch repeated by hand — `/models` and
    `/recall` — answered "unknown command" even though `/help` printed them and
    the picker completed them. `/quit` only worked because its branch listed it
    twice, which is the same bug wearing a patch.

    Driven off ``SLASH_COMMANDS`` rather than those two literals, because the
    invariant is "a registry name has a branch" and the chain still matches
    hardcoded strings: a 19th command spelled differently in its branch reopens
    exactly this.

    Read from the TRANSCRIPT, not the painted frame, and per spelling: a
    handler that opens a panel or pushes a screen covers the rows a later
    spelling would write, so a frame-scraping assertion would go quietly blind
    for the rest of the walk.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        for entry in SLASH_COMMANDS:
            if entry.name == "exit":
                continue  # `self.exit()` would end the pilot mid-loop
            for spelling in entry.names:
                app._transcript_view().clear_blocks()
                app._run_slash_command(f"/{spelling}")
                await pilot.pause()
                unknown = [n for n in _notice_texts(app) if "unknown command" in n]
                assert unknown == [], (spelling, unknown)


@pytest.mark.asyncio
async def test_a_panel_opening_command_leaves_no_user_row() -> None:
    """The reported bug, verbatim: `/usage` opens the panel that IS the answer,
    so a row above it restates the keystroke and nothing else."""
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/usage")
        rows = _user_rows(app)
        panel_open = app.query_one(UsagePanel).display
    assert rows == [], rows
    assert panel_open is True


@pytest.mark.asyncio
async def test_a_listing_names_what_it_lists() -> None:
    """The policy's load-bearing claim is "the listing IS the receipt", and a
    receipt has to say what it is a receipt for. Rendered at 120x40, three
    consecutive listings with the echo removed stacked into one anonymous run
    of tree glyphs — and the provider list and the credential list are the pair
    a reader is most likely to confuse, since both are one row per provider id.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/provider")
        await _submit(pilot, app, "/accounts")
        await _submit(pilot, app, "/login")
        painted = _painted(app)
    assert "providers" in painted, painted
    assert "stored credentials" in painted, painted
    # `/login` bare lists the SAME set as `/provider`, so its caption is the
    # one carrying the whole distinction between two adjacent identical trees.
    assert "providers with interactive login" in painted, painted


@pytest.mark.asyncio
async def test_repeated_panel_commands_do_not_accumulate_a_ledger() -> None:
    """The complaint was cumulative — the rows pile up across a session. Three
    panel-class commands in a row must leave the user-attributed ledger empty."""
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for command in ("/usage", "/provider", "/accounts", "/usage openrouter"):
            await _submit(pilot, app, command)
        rows = _user_rows(app)
    assert rows == [], rows


@pytest.mark.asyncio
async def test_setting_the_goal_writes_the_users_words_to_the_ledger() -> None:
    """`/goal <text>` is the one argument that reaches the model, on every later
    turn. The ledger shows what the model was told, attributed to whoever said
    it, so this row is content rather than a keystroke.

    The notice beside the row reports status ONLY — repeating the goal there
    would be the duplicate row this change removed everywhere else. That the
    row carries the STORED text rather than the typed one is not visible here
    (the editor strips too, so only the length cap separates them): it is
    ``test_a_goal_cut_by_the_cap_says_so`` that pins it.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal   land the OAuth refresh fix  ")
        rows = _user_rows(app)
        painted = _painted(app)
    assert rows == ["/goal land the OAuth refresh fix"], rows
    assert session.goal == "land the OAuth refresh fix"
    assert "goal set — applies from the next turn" in painted, painted


@pytest.mark.asyncio
async def test_a_goal_cut_by_the_cap_says_so() -> None:
    """`GoalState.set` caps at 2000 characters silently. That was survivable
    while the receipt was a system notice; now the text sits in a row
    ATTRIBUTED TO THE USER, so a silent cut leaves the ledger claiming they
    typed something ending mid-word."""

    class _CappingSession(FakeSession):
        """`FakeSession.set_goal` only strips — this applies the real cap."""

        def __init__(self) -> None:
            super().__init__()
            self._state = GoalState()

        @property
        def goal(self) -> str:
            return self._state.text

        def set_goal(self, text: str) -> str:
            return self._state.set(text)

    session = _CappingSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal " + "x" * (MAX_GOAL_CHARS + 100))
        rows = _user_rows(app)
        notices = _notice_texts(app)
    assert rows == ["/goal " + "x" * MAX_GOAL_CHARS], len(rows[0]) if rows else rows
    assert any(f"shortened to the {MAX_GOAL_CHARS}-character cap" in n for n in notices), notices


@pytest.mark.asyncio
async def test_the_read_only_and_clearing_forms_of_goal_write_no_row() -> None:
    """The permission is per COMMAND but the row is per INVOCATION, which is why
    the handler writes it. Neither of these hands the model any words: a bare
    `/goal` reports the current one, and `/goal clear` takes it away. A row
    written before dispatch claimed a goal for both."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal")
        await _submit(pilot, app, "/goal clear")
        rows = _user_rows(app)
    assert rows == [], rows


@pytest.mark.asyncio
async def test_a_goal_rejected_before_the_session_is_ready_writes_no_row() -> None:
    """The row's whole claim is that the model was given these words. Written
    before dispatch it appeared even when `_cmd_goal` rejected the set outright,
    and — being an `_append_block` — it also retired the boot splash for a
    command that changed nothing, defeating the `_system_notice` that handler
    uses for exactly this case."""

    async def _never_finishes() -> FakeSession:
        # A factory that never resolves holds the app in the exact window
        # `_cmd_goal` guards against; the ordinary fake boots in one tick.
        await asyncio.sleep(3600)
        return FakeSession()

    app = OperatorApp(_never_finishes)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app._session is None
        await _submit(pilot, app, "/goal ship it")
        rows = _user_rows(app)
        welcome = app.query_one(WelcomeView).display
        painted = _painted(app)
    assert rows == [], rows
    assert welcome is True
    assert "session is still starting" in painted, painted


@pytest.mark.asyncio
async def test_the_splash_survives_a_first_action_that_draws_no_block() -> None:
    """What the echo was incidentally protecting: the submit handler retired the
    splash unconditionally, which was only ever safe because a row always
    followed. Without one, retiring it here would leave a frame holding neither
    the splash nor any ledger — so the edge belongs to ``_append_block``, which
    is what actually knows whether something was drawn."""
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/usage")
        welcome = app.query_one(WelcomeView).display
        boot = app.screen.has_class("boot")
        panel_open = app.query_one(UsagePanel).display
    assert welcome is True
    assert boot is True
    assert panel_open is True


@pytest.mark.asyncio
async def test_an_echoing_first_action_still_retires_the_splash() -> None:
    """The other half of the same edge: a command that DOES write a row must
    still end the empty state, or the row lands behind a splash still claiming
    the session has not started."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal ship it")
        welcome = app.query_one(WelcomeView).display
        boot = app.screen.has_class("boot")
    assert welcome is False
    assert boot is False


@pytest.mark.asyncio
async def test_an_unknown_command_names_what_was_typed() -> None:
    """With the echo gone the warning is the ONLY place the mistyped word
    appears; "unknown command" without the command is a dead end. Cased as
    typed, too — reporting `/USGE` as `/usge` sends the user hunting for a
    second typo they did not make."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/USGE")
        painted = _painted(app)
        rows = _user_rows(app)
        # A typo changed nothing, so the conversation has not started and the
        # splash stays — the same rule a rejected `/model` selector follows.
        welcome = app.query_one(WelcomeView).display
    assert "/USGE" in painted, painted
    assert "unknown command" in painted, painted
    assert rows == [], rows
    assert welcome is True
