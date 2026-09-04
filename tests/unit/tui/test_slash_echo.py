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
import tempfile
from pathlib import Path

import pytest
from rich.style import Style
from textual.containers import Container

from local_operator.session.goal import MAX_GOAL_CHARS, GoalState
from local_operator.tui import theme as theme_mod
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
    # Opens a picker, and nothing here reaches the model. The receipt is
    # whatever that interaction produces — a clipboard toast naming how much
    # landed there, or nothing at all when the user cancels — and both are
    # strictly more informative than echoing the typed word. `/approvals`' rule.
    "copy": False,
    "new": False,
    "reload": False,
    "update": False,
    "resume": False,
    # TRUE, and the second command to earn it for the same reason `/goal` does:
    # the argument becomes a user turn a model is given — in the FORK. The
    # receipt in THIS window names both session ids but not the instruction, so
    # without the echo the only record of what the branch was asked to do is in
    # another window entirely.
    "fork": True,
    # The label of the conversation, not words the model is told: the receipt
    # quotes the title that ended up in force, which is more than what was typed.
    "rename": False,
    "model": False,
    # A setting, not words the model is told: the receipt names the level that
    # is now in force and the band carries it from then on.
    "effort": False,
    # A setting, not words the model is told: the receipt names the theme now
    # in force, and the screen itself is wearing the change.
    "theme": False,
    "provider": False,
    # The PAGE is the receipt (same rule as `/usage` and `/analytics`): it
    # replaces the transcript region, so a user row printed behind it would
    # only be readable after leaving. It takes no argument at all.
    "settings": False,
    "search": False,
    "accounts": False,
    # The cascade tree IS the receipt, and there is no argument to restate.
    "failovers": False,
    "usage": False,
    # The screen it opens IS the receipt (same rule as `/usage`); the argument
    # names a view, never words the model is told.
    "analytics": False,
    "goal": True,
    "loop": False,
    # The aside's whole promise is that the exchange leaves no trace in the
    # ledger; a user row would be the one it left. See `SLASH_COMMANDS`.
    "btw": False,
    "compact": False,
    # The stop receipt names what was stopped; nothing reaches the model.
    "stop": False,
    "context": False,
    "approvals": False,
    "skills": False,
    "mcp": False,
    "login": False,
    "logout": False,
    # The listing or the masked paste is the receipt; the argument is a key
    # name, never the secret, so a user row would only restate the notice.
    "credential": False,
    # The listing is the receipt; a named request is sent as a real user
    # turn by `_submit_prompt`, which already writes the row.
    "team": False,
    # Same reasoning as `/team`, which `/agent` mirrors: the listing or the
    # attach notice is the receipt, and a message is sent as a real user
    # turn by `_submit_prompt`, which already writes the row.
    "agent": False,
}


#: Which commands CONSUME the rest of the composer as a prompt when engaged
#: inline (``SlashCommand.consumes_prompt``). True for the free-text commands
#: whose argument becomes a message the model is given — engaging one mid-draft
#: reassembles it to the FRONT with the draft as its argument rather than
#: splicing-and-running, so a plausible mid-sentence gesture cannot silently eat
#: the user's text. Pinned entry-by-entry for the same reason ``ECHO_POLICY`` is:
#: a new command must state whether its argument is a prompt.
PROMPT_POLICY = {
    "help": False,
    "exit": False,
    "clear": False,
    # Takes no argument at all: WHICH message is chosen in the picker the
    # command opens, so there is nothing typed for an inline engage to consume.
    # (`/copy me` and `/copy <n>` are deliberately not built — the picker is the
    # answer to "which message", and a typed selector would be a second one.)
    "copy": False,
    "new": False,
    "reload": False,
    "update": False,
    "resume": False,
    # Free text destined for a model, so an inline `/fork` reassembles to the
    # front of the composer rather than splicing into the middle of a sentence.
    "fork": True,
    "rename": False,
    "model": False,
    "effort": False,
    "theme": False,
    "provider": False,
    # The PAGE is the receipt (same rule as `/usage` and `/analytics`): it
    # replaces the transcript region, so a user row printed behind it would
    # only be readable after leaving. It takes no argument at all.
    "settings": False,
    "search": False,
    "accounts": False,
    "failovers": False,
    "usage": False,
    # A view selector, not a prompt: `/analytics [view]` names which screen to
    # open, so it splices-and-runs inline like `/usage` rather than reassembling.
    "analytics": False,
    # Free text the model is given (the objective / a loop instruction / a side
    # question), so an inline engage reassembles to the front.
    "goal": True,
    "loop": True,
    "btw": True,
    "compact": False,
    "stop": False,
    "context": False,
    "approvals": False,
    "skills": False,
    "mcp": False,
    "login": False,
    "logout": False,
    "credential": False,
    # The request after the name is a prompt the manager / persona is given.
    "team": True,
    "agent": True,
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


def _band_text(app: OperatorApp) -> str:
    """The status band's current rendered text (U2 segment assertions).

    Rendered wide so the drop ladder keeps the static-identity segments — the
    band sheds them at narrow widths by design, which the status_line suite
    covers; here we only assert they are wired to the command.
    """
    status = app._status
    assert status is not None
    return status.render_text(200).plain


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
    # The same forcing function for the inline-prompt choice: a new command must
    # state whether its argument is a prompt consumed on an inline engage.
    assert {c.name: c.consumes_prompt for c in SLASH_COMMANDS} == PROMPT_POLICY


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
            if entry.name in {"exit", "reload", "update"}:
                # ``exit`` ends the pilot; ``reload``/``update`` now exit 75
                # to re-exec, which would do the same mid-loop.
                continue
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
async def test_bare_team_lists_without_a_user_row() -> None:
    """`/team` is a listing; the listing is the receipt."""
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    session = FakeSession()
    # In-memory: point the registry at a throwaway dir via the session.
    tmp = tempfile.mkdtemp()
    registry = TeamRegistry(Path(tmp))
    registry.create_team(
        TeamEditFields(
            name="feature-release",
            manager="manager",
            members=[TeamMember(role="coder")],
            description="Ship a change",
        )
    )
    session.team_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/team")
        rows = _user_rows(app)
        painted = _painted(app)
    assert rows == [], rows
    assert "feature-release" in painted, painted
    assert "Led by manager · 2 roles" in painted, painted
    assert "Ship a change" in painted, painted
    assert "Send: /team <name> <message>" in painted, painted
    assert "manager=" not in painted, painted


@pytest.mark.asyncio
async def test_team_request_attaches_and_sends() -> None:
    """`/team <name> <request>` stamps the team and sends the request as a turn."""
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    session = FakeSession()
    registry = TeamRegistry(Path(tempfile.mkdtemp()))
    registry.create_team(
        TeamEditFields(
            name="feature-release",
            manager="manager",
            members=[TeamMember(role="coder")],
        )
    )
    session.team_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/team feature-release ship the dashboard")
        rows = _user_rows(app)
        band = _band_text(app)
    assert session.attached_teams, "the team must be attached before the turn"
    assert session.attached_teams[0].name == "feature-release"
    assert session.prompts == ["ship the dashboard"]
    assert rows == ["ship the dashboard"], rows
    # U2: the band names the active roster after the attach.
    assert "feature-release" in band, band


@pytest.mark.asyncio
async def test_inline_team_reassembles_to_the_front_and_then_sends_under_the_manager() -> None:
    """The end-to-end mid-trajectory gesture the user asked to confirm.

    A user types a message, remembers to route it to a team, appends ``/team``
    and picks the team from the autofill. ``/team`` is a PROMPT command (its
    request is free text the manager is given), so it does NOT auto-run and eat
    the draft as a name — it REASSEMBLES to the front as ``/team <name> <the
    draft>``, STAGED. The user reads the assembled line and presses Enter, which
    attaches the team and sends the message as the request — under the manager,
    because the roster rides the session's volatile prompt tail (see
    ``session_factory``'s per-turn provider).
    """
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    session = FakeSession()
    registry = TeamRegistry(Path(tempfile.mkdtemp()))
    registry.create_team(
        TeamEditFields(
            name="feature-release",
            manager="manager",
            members=[TeamMember(role="coder")],
        )
    )
    session.team_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        # Type the message, then the inline command at the caret.
        for char in "ship the dashboard ":
            await pilot.press("space" if char == " " else char)
        for char in "/team":
            await pilot.press("slash" if char == "/" else char)
        await pilot.pause()
        # Word complete: Enter opens the team list rather than running (the name
        # is picked from the autofill, not guessed from the draft).
        await pilot.press("enter")
        await pilot.pause()
        assert editor.picker.is_open(), "the team argument list must open"
        # Pick the team by name.
        for char in "feature-release":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        # Reassembled to the front, STAGED — nothing attached or sent yet, and
        # the draft is preserved verbatim as the request.
        assert editor.text == "/team feature-release ship the dashboard", editor.text
        assert session.attached_teams == [], "reassembly must not attach yet"
        assert session.prompts == [], "reassembly must not send yet"
        # The user reviews the assembled line and submits it.
        if editor.picker.is_open():
            await pilot.press("escape")
            await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert session.attached_teams, "the team attaches on the staged submit"
        assert session.attached_teams[0].name == "feature-release"
        assert session.prompts == ["ship the dashboard"], session.prompts


@pytest.mark.asyncio
async def test_completing_a_team_name_keeps_the_parked_hint_in_the_real_app() -> None:
    """D5 regression (design review round 3), asserted against the REAL app.

    Picking a team name at the START of the buffer fills ``/team <name> `` and
    parks the caret, and the picker shows the switch/send hint. The hint is set
    by the editor and cleared by the app's ``on_argument_query_opened`` — so a
    stray extra picker resync (which nulls ``_argument_command`` and re-fires the
    query) wipes it. The bespoke harness's ``on_argument_query_opened`` does not
    clear the notice, so this is only observable against ``OperatorApp``: exactly
    the "green tests aren't the rendered frame" gap the round-3 review named.
    """
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    session = FakeSession()
    registry = TeamRegistry(Path(tempfile.mkdtemp()))
    registry.create_team(
        TeamEditFields(name="security", manager="manager", members=[TeamMember(role="coder")])
    )
    session.team_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        # Start-of-line /team, open the list, then fill the single team with Tab.
        for char in "/team ":
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()
        assert editor.picker.is_open(), "the team list must open"
        await pilot.press("tab")
        await pilot.pause()
        await pilot.pause()
        # Name filled, caret parked, and the switch/send hint survives the app's
        # own argument-query handling — the D5 fix.
        assert editor.text == "/team security ", editor.text
        assert (
            "switch" in editor.picker._notice and "send" in editor.picker._notice
        ), editor.picker._notice


@pytest.mark.asyncio
async def test_session_adoption_catches_up_team_picker_and_name_highlight() -> None:
    """Typing `/team lop` before boot must recover when the registry appears."""
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    session = FakeSession()
    registry = TeamRegistry(Path(tempfile.mkdtemp()))
    registry.create_team(
        TeamEditFields(
            name="lopdev",
            manager="manager",
            members=[TeamMember(role="coder")],
        )
    )
    session.team_registry = registry
    release = asyncio.Event()

    async def delayed_factory() -> FakeSession:
        await release.wait()
        return session

    app = OperatorApp(delayed_factory)
    # The committed real-app evidence uses the product's standard 100x30 frame;
    # pin the regression to that same viewport so these cell coordinates map
    # directly to the SVG y positions cited in the PR body.
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/team lop":
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()
        assert app._session is None
        assert editor.picker.is_pending()
        assert editor.picker._query == "lop"

        # D1 (design review round 1): while the registry is genuinely still
        # arriving, the picker reserves exactly one NON-SELECTABLE notice row.
        # The real row must replace it in place — not expand the dock and jump
        # the composer/welcome/status geometry under a mid-keystroke user.
        empty_geometry = {
            "composer": editor.region.y,
            "welcome": app.query_one(WelcomeView).region.y,
            "status": app.query_one("#status-band").region.y,
            "picker": editor.picker.region.y,
            "picker_height": editor.picker.region.height,
            "screen": app.screen.size,
            "virtual": app.screen.virtual_size,
        }
        assert editor.picker.display is True
        assert editor.picker.is_open() is False  # the reserve is not selectable
        assert editor.picker._notice == "loading teams…"
        assert empty_geometry["picker_height"] == 1

        release.set()
        await _boot(pilot, app)
        # Consecutive post-adoption frames must be settled at the same geometry.
        await pilot.pause()
        filled_geometry = {
            "composer": editor.region.y,
            "welcome": app.query_one(WelcomeView).region.y,
            "status": app.query_one("#status-band").region.y,
            "picker": editor.picker.region.y,
            "picker_height": editor.picker.region.height,
            "screen": app.screen.size,
            "virtual": app.screen.virtual_size,
        }
        await pilot.pause()
        settled_geometry = {
            "composer": editor.region.y,
            "welcome": app.query_one(WelcomeView).region.y,
            "status": app.query_one("#status-band").region.y,
            "picker": editor.picker.region.y,
            "picker_height": editor.picker.region.height,
            "screen": app.screen.size,
            "virtual": app.screen.virtual_size,
        }
        # The picker itself may not ADD geometry: its one pending row is
        # replaced by the one real row. FakeSession adoption also changes
        # unrelated welcome/status content (model label + status segments), so
        # compare the picker's invariant as relative geometry and pin the two
        # consecutive post-adoption frames exactly. The committed real-Session
        # evidence carries the absolute pre/post coordinates with stable facts.
        assert empty_geometry["picker_height"] == filled_geometry["picker_height"] == 1
        assert empty_geometry["status"] - empty_geometry["composer"] == 2
        assert filled_geometry["status"] - filled_geometry["composer"] == 2
        assert empty_geometry["picker"] - empty_geometry["composer"] == 1
        assert filled_geometry["picker"] - filled_geometry["composer"] == 1
        # FakeSession's welcome/status facts can settle one tick after the
        # registry row (the same unrelated recentering excluded above), so
        # compare the dock invariant again rather than claiming absolute
        # frame equality from this synthetic host. The committed real-Session
        # triplet pins absolute first/settled equality at 24/25/26/1.
        assert settled_geometry["picker_height"] == 1
        assert settled_geometry["status"] - settled_geometry["composer"] == 2
        assert settled_geometry["picker"] - settled_geometry["composer"] == 1
        assert filled_geometry["screen"] == settled_geometry["screen"]
        assert filled_geometry["virtual"] == settled_geometry["virtual"]
        assert editor.text == "/team lop"
        assert editor.picker.is_open()
        assert editor.picker.highlighted_name() == "lopdev"
        assert editor._name_choices == frozenset({"lopdev"})

        # Complete the exact name by hand and type a message; the snapshot
        # delivered during adoption must paint it without another list-opening.
        editor.text = "/team lopdev fix it"
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await pilot.pause()
        green = Style.parse(theme_mod.semantic_color("string")).color
        assert green is not None
        line = editor.render_line(0)
        painted = [
            segment
            for segment in line._segments
            if segment.text == "lopdev" and segment.style and segment.style.color == green
        ]
        assert painted


def _picker_geometry(
    app: OperatorApp, editor: Editor
) -> tuple[int, int, int, int, int, object, object]:
    """Exact dock/screen coordinates for pending-name geometry assertions."""
    return (
        editor.region.y,
        editor.picker.region.y,
        editor.picker.region.height,
        app.query_one("#status-band").region.y,
        app.query_one(WelcomeView).region.y,
        app.screen.size,
        app.screen.virtual_size,
    )


@pytest.mark.asyncio
async def test_session_adoption_catches_up_agent_picker_without_geometry_change() -> None:
    """U2-1: `/agent aud` reserves one row until the registry arrives."""
    session = FakeSession()
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    release = asyncio.Event()

    async def delayed_factory() -> FakeSession:
        await release.wait()
        return session

    app = OperatorApp(delayed_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/agent aud":
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()

        pending = _picker_geometry(app, editor)
        assert editor.picker.is_loading()
        assert editor.picker.is_open() is False
        assert editor.picker._notice == "loading agent roster…"
        # FakeSession carries different welcome/status facts from the real
        # Session used by committed evidence, so its absolute y coordinates
        # are intentionally not hard-coded here. Pin the exact dock geometry
        # and dimensions it must preserve across the three frames; the real
        # Session capture below supplies the absolute 24/25/26/1 coordinates.
        composer, picker_y, picker_h, status_y, _welcome_y, screen, virtual = pending
        assert picker_h == 1
        assert picker_y - composer == 1
        assert status_y - composer == 2
        assert screen == virtual

        release.set()
        await _boot(pilot, app)
        await pilot.pause()
        first = _picker_geometry(app, editor)
        await pilot.pause()
        settled = _picker_geometry(app, editor)
        # Session adoption replaces FakeSession's welcome/status facts and may
        # recenter that content one tick after the row. Pin the DOCK invariant
        # in all three synthetic frames rather than claiming absolute frame
        # equality from this host; the committed real-Session triplet has stable
        # facts and asserts absolute first/settled equality at 24/25/26/1.
        for geometry in (pending, first, settled):
            frame_composer, frame_picker, frame_h, frame_status, *_ = geometry
            assert frame_h == 1
            assert frame_picker - frame_composer == 1
            assert frame_status - frame_composer == 2
        assert first[-2:] == settled[-2:]
        assert editor.text == "/agent aud"
        assert editor.picker.is_loading() is False
        assert editor.picker.is_open()
        assert editor.picker.highlighted_name() == "auditor"
        assert editor._name_choices == frozenset(
            {
                "architect",
                "auditor",
                "coder",
                "dashboard-sme",
                "designer",
                "hollow-role",
                "manager",
                "reviewer",
                "scout",
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "query", "expected"),
    [("team", "lop", "lopdev"), ("agent", "aud", "auditor")],
)
async def test_pending_name_tab_is_noop_then_completes_after_adoption(
    command: str, query: str, expected: str
) -> None:
    """U2-2: pending Tab preserves exact text/caret; real rows restore Tab."""
    from local_operator.teams import TeamEditFields, TeamRegistry

    session = FakeSession()
    teams = TeamRegistry(Path(tempfile.mkdtemp()))
    teams.create_team(TeamEditFields(name="lopdev", manager="manager"))
    session.team_registry = teams
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    release = asyncio.Event()

    async def delayed_factory() -> FakeSession:
        await release.wait()
        return session

    app = OperatorApp(delayed_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        original = f"/{command} {query}"
        for char in original:
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()
        caret = editor._caret_offset()
        assert editor.picker.is_loading()
        assert editor.picker.suggestions() == []

        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == original
        assert editor._caret_offset() == caret
        assert editor.picker.is_loading()

        release.set()
        await _boot(pilot, app)
        await pilot.pause()
        assert editor.picker.highlighted_name() == expected
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == f"/{command} {expected} "


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "query"), [("team", "lop"), ("agent", "aud")])
async def test_pending_name_enter_is_noop_until_rows_arrive(command: str, query: str) -> None:
    """U2-2: pending Enter never submits or clears the delayed query."""
    from local_operator.teams import TeamEditFields, TeamRegistry

    session = FakeSession()
    teams = TeamRegistry(Path(tempfile.mkdtemp()))
    teams.create_team(TeamEditFields(name="lopdev", manager="manager"))
    session.team_registry = teams
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    release = asyncio.Event()

    async def delayed_factory() -> FakeSession:
        await release.wait()
        return session

    app = OperatorApp(delayed_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        original = f"/{command} {query}"
        for char in original:
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()
        caret = editor._caret_offset()
        assert editor.picker.is_loading()

        await pilot.press("enter")
        await pilot.pause()
        assert editor.text == original
        assert editor._caret_offset() == caret
        assert getattr(session, "attached_teams", []) == []
        assert getattr(session, "active_agent", "") == ""

        release.set()
        await _boot(pilot, app)
        await pilot.pause()
        assert editor.text == original
        assert editor.picker.is_open()


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "query"), [("team", "lop"), ("agent", "aud")])
async def test_pending_name_placeholder_is_not_mouse_selectable(command: str, query: str) -> None:
    """U2-2: the loading reserve has no keyboard or mouse selection target."""

    async def never_adopts() -> FakeSession:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    app = OperatorApp(never_adopts)
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        original = f"/{command} {query}"
        for char in original:
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()
        assert editor.picker.is_loading()
        assert editor.picker.is_open() is False
        assert editor.picker.highlighted_name() is None
        assert editor.picker.suggestions() == []

        await pilot.click(type(editor.picker), offset=(4, 0))
        await pilot.pause()
        assert editor.text == original
        assert editor.picker.is_loading()
        assert editor.picker.highlighted_name() is None


@pytest.mark.asyncio
async def test_pending_name_row_collapses_on_escape_and_stays_dismissed_after_adoption() -> None:
    """Esc releases D1's reserve; the late registry must not resurrect it."""
    from local_operator.teams import TeamEditFields, TeamRegistry

    session = FakeSession()
    registry = TeamRegistry(Path(tempfile.mkdtemp()))
    registry.create_team(TeamEditFields(name="lopdev", manager="manager"))
    session.team_registry = registry
    release = asyncio.Event()

    async def delayed_factory() -> FakeSession:
        await release.wait()
        return session

    app = OperatorApp(delayed_factory)
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/team lop":
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()
        assert editor.picker.is_pending()
        assert editor.picker.display is True
        assert editor.picker.region.height == 1
        reserved_y = editor.region.y

        await pilot.press("escape")
        await pilot.pause()
        assert editor.text == "/team lop"
        assert editor.picker.display is False
        assert editor.picker._dismissed_query == "lop"
        assert editor.region.y == reserved_y + 1  # the reserved row collapsed

        release.set()
        await _boot(pilot, app)
        await pilot.pause()
        assert editor.text == "/team lop"
        assert editor.picker.display is False
        assert editor.picker.is_open() is False
        assert editor.picker._dismissed_query == "lop"
        # Adoption may change unrelated status/welcome rows in FakeSession;
        # dismissal's invariant is that no picker row is reintroduced.
        assert editor.picker.region.height == 0
        assert app.query_one("#status-band").region.y - editor.region.y == 1


@pytest.mark.asyncio
async def test_adopted_empty_roster_does_not_leave_a_pending_name_row() -> None:
    """A genuinely empty adopted roster is final, not an eternal blank hole."""
    from local_operator.teams import TeamRegistry

    session = FakeSession()
    session.team_registry = TeamRegistry(Path(tempfile.mkdtemp()))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/team lop":
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()
        assert app._session is session
        assert editor.picker.is_pending()
        # Pending is the input state (an argument list with no matches), not a
        # promise that more rows are coming; adopted-empty must take no space.
        assert editor.picker.display is False
        assert editor.picker._notice == ""
        assert editor.picker.region.height == 0


@pytest.mark.asyncio
async def test_a_second_inline_command_of_the_same_kind_is_plain_argument_text() -> None:
    """Once a prompt command is engaged at the front, a second occurrence of the
    same command inside its argument is plain text — no picker, no re-engagement.

    Reported: composing ``/team a improve the /team command`` must treat the
    second ``/team`` as part of the request, not a nested command to highlight or
    run.
    """
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    session = FakeSession()
    registry = TeamRegistry(Path(tempfile.mkdtemp()))
    registry.create_team(
        TeamEditFields(name="alpha", manager="manager", members=[TeamMember(role="coder")])
    )
    session.team_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        for char in "/team alpha improve the ":
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        # Now type a nested "/team" inside the request.
        for char in "/team":
            await pilot.press("slash" if char == "/" else char)
        for char in " command":
            await pilot.press("space" if char == " " else char)
        await pilot.pause()
        # The picker must be closed: the caret is inside the first /team's
        # argument, and the nested /team is plain text.
        assert not editor.picker.is_open(), "a nested /team must not open the picker"
        assert editor.text == "/team alpha improve the /team command"
        # Submitting sends the whole request verbatim to the manager.
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert session.attached_teams and session.attached_teams[0].name == "alpha"
        assert session.prompts == ["improve the /team command"], session.prompts


def _agent_registry(tmp: str):
    """A real registry holding one role, one specialist, one PRIVATE chat row.

    The private row is the load-bearing fixture: `/agent` shares the registry
    with ordinary conversational agents, and the scope rule under test is
    that only rows tagged as roles or marked as specialists are listed or
    accepted.
    """
    from local_operator.agents import AgentEditFields, AgentRegistry

    registry = AgentRegistry(Path(tmp))

    def _fields(**kwargs):
        base = dict(
            name=None,
            security_prompt=None,
            hosting=None,
            model=None,
            description=None,
            tags=None,
            categories=None,
            last_message=None,
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=None,
        )
        base.update(kwargs)
        return AgentEditFields(**base)

    role = registry.create_agent(
        _fields(name="auditor", description="Audit changes for risk", tags=["role"])
    )
    registry.set_agent_system_prompt(role.id, "You audit changes.")
    specialist = registry.create_agent(
        _fields(
            name="dashboard-sme",
            description="Knows the dashboard release practices",
            categories=["specialist"],
        )
    )
    registry.set_agent_system_prompt(specialist.id, "Follow the dashboard checklist.")
    private = registry.create_agent(_fields(name="private-chat", description="My diary"))
    registry.set_agent_system_prompt(private.id, "PRIVATE USER CONTEXT")
    # A role that RESOLVES by name but carries no instructions — the A2 case:
    # the attach layers nothing, and the notice must say so rather than
    # claiming the persona is active.
    hollow = registry.create_agent(
        _fields(name="hollow-role", description="Resolves but says nothing", tags=["role"])
    )
    registry.set_agent_system_prompt(hollow.id, "")
    return registry


@pytest.mark.asyncio
async def test_bare_agent_lists_without_a_user_row() -> None:
    """`/agent` is a listing; the listing is the receipt. Only roles and
    specialists appear — a plain conversational agent stays private."""

    session = FakeSession()
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    app = OperatorApp(lambda: _factory(session))
    # Tall on purpose: the listing carries the six packaged starters too, and
    # the transcript keeps scrolled to the bottom, so a short frame would have
    # scrolled the registered rows out of the paint this test reads.
    async with app.run_test(size=(100, 74)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/agent")
        rows = _user_rows(app)
        painted = _painted(app)
    assert rows == [], rows
    assert "auditor" in painted, painted
    assert "dashboard-sme" in painted, painted
    assert "specialist" in painted, painted
    assert "private-chat" not in painted, painted
    assert "Send: /agent <name> <message>" in painted, painted


@pytest.mark.asyncio
async def test_agent_name_alone_attaches_without_a_turn() -> None:
    """`/agent <name>` adopts the profile and prints the notice — no prompt."""

    session = FakeSession()
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/agent auditor")
        rows = _user_rows(app)
        notices = _notice_texts(app)
        band = _band_text(app)
    assert session.attached_agents == ["auditor"]
    assert session.prompts == []
    assert rows == [], rows
    # U2: the band names the active profile after the attach.
    assert "auditor" in band, band
    # U3/U4: the notice states the profile now governs the session and points
    # at the detach verb, rather than the thinner "is active".
    assert any("auditor is ready and now governs" in n for n in notices), notices
    assert any("/agent clear" in n for n in notices), notices


@pytest.mark.asyncio
async def test_agent_message_attaches_and_sends() -> None:
    """`/agent <name> <message>` stamps the profile then sends the message as
    a turn — the `/team <name> <request>` shape, including a specialist."""

    session = FakeSession()
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/agent dashboard-sme review the release")
        rows = _user_rows(app)
    assert session.attached_agents == ["dashboard-sme"]
    assert session.prompts == ["review the release"]
    assert rows == ["review the release"], rows


@pytest.mark.asyncio
async def test_agent_rejects_unknown_and_private_names() -> None:
    """An unknown name warns; so does a PRIVATE conversational agent — its
    exact name must not be enough to pull its prompt into this session."""

    session = FakeSession()
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/agent no-such-profile do things")
        await _submit(pilot, app, "/agent private-chat do things")
        notices = _notice_texts(app)
    assert session.attached_agents == []
    assert session.prompts == []
    assert any("no agent named 'no-such-profile'" in n for n in notices), notices
    assert any("no agent named 'private-chat'" in n for n in notices), notices


@pytest.mark.asyncio
async def test_agent_clear_detaches_the_active_profile() -> None:
    """U1: `/agent clear` detaches the active profile and reports the session is
    back on its base instructions. `clear` is the detach verb, not a name to
    look up, so nothing is "attached" by it."""
    session = FakeSession()
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/agent auditor")
        assert session.agent_brief, "profile should be attached first"
        band_attached = _band_text(app)
        await _submit(pilot, app, "/agent clear")
        cleared_brief = session.agent_brief
        cleared_count = session.cleared_agents
        band_cleared = _band_text(app)
        notices = _notice_texts(app)
    # U2: the band named the profile while attached, and drops it on clear.
    assert "auditor" in band_attached, band_attached
    assert "auditor" not in band_cleared, band_cleared
    assert cleared_brief == "", "clear must blank the agent brief"
    assert cleared_count == 1, "clear must reach the session detach"
    # `clear` was the verb, not an attach — attached_agents stays as it was.
    assert session.attached_agents == ["auditor"]
    assert any("base instructions" in n for n in notices), notices
    # D4: the noun is standardized on "agent" — the detach notice no longer says
    # "agent profile". A drift back to "profile" here re-opens the finding.
    assert not any("profile" in n for n in notices), notices


@pytest.mark.asyncio
async def test_agent_none_is_an_alias_for_clear() -> None:
    """`/agent none` detaches too, so a user reaching for either word lands on
    base instructions rather than an 'unknown agent' warning."""
    session = FakeSession()
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/agent auditor")
        await _submit(pilot, app, "/agent none")
        notices = _notice_texts(app)
    assert session.agent_brief == ""
    assert session.cleared_agents == 1
    assert not any("no agent named" in n for n in notices), notices
    assert any("base instructions" in n for n in notices), notices


@pytest.mark.asyncio
async def test_agent_with_no_instructions_says_nothing_was_applied() -> None:
    """A2: a role/specialist that resolves but carries no instructions must
    NOT claim to be active — the notice states nothing was applied, and with a
    message the message is still sent (labelled as carrying no persona)."""
    session = FakeSession()
    session.agent_registry = _agent_registry(tempfile.mkdtemp())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/agent hollow-role")
        bare_notices = list(_notice_texts(app))
        await _submit(pilot, app, "/agent hollow-role do the thing")
        notices = _notice_texts(app)
    # The name resolved (so it was "attached"), but no persona layered.
    assert session.attached_agents == ["hollow-role", "hollow-role"]
    assert any(
        "no instructions" in n or "nothing was applied" in n for n in bare_notices
    ), bare_notices
    # With a message: still sent, but the notice does not claim a persona.
    assert session.prompts == ["do the thing"]
    assert not any("hollow-role now governs" in n for n in notices), notices
    assert not any("hollow-role is ready" in n for n in notices), notices


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


def _first_text_styles(block) -> list[tuple[str, Style]]:
    """Flatten a listing block to (text, style) pairs, header first."""
    from rich.console import Group
    from rich.padding import Padding
    from rich.text import Text

    out: list[tuple[str, Style]] = []

    def walk(node) -> None:
        if isinstance(node, Group):
            for child in node.renderables:
                walk(child)
        elif isinstance(node, Padding):
            walk(node.renderable)
        elif isinstance(node, Text):
            # Only the styled entries matter; the blank spacer Text carries a
            # bare "" style, which is not a Style object and is skipped.
            style = node.style
            if isinstance(style, Style):
                out.append((node.plain, style))

    walk(block.renderable)
    return out


def test_agent_and_team_listing_headers_outrank_their_entries() -> None:
    """D1: the section header must read as a header, not as one more entry.

    Indentation alone did not separate them — header and entry names shared the
    one muted style. The header now takes a brighter, bold weight while entries
    keep the muted one, and both listings get the SAME treatment so they stay
    consistent. Asserted on the rendered style, not the pixels, because "the
    header is heavier than its entries" is exactly a style-attribute claim.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))

    from local_operator.tui import theme as theme_mod

    fg = theme_mod.semantic_color("fg")
    muted = theme_mod.semantic_color("muted")

    def _colour(style: Style) -> str:
        colour = style.color
        assert colour is not None, style
        return colour.name

    def _assert_header_outranks(block, header_text: str, entry_text: str) -> None:
        pairs = _first_text_styles(block)
        header_style = next(s for t, s in pairs if t == header_text)
        entry_style = next(s for t, s in pairs if t == entry_text)
        assert header_style.bold, f"{header_text!r} header must be bold"
        assert _colour(header_style) == fg, header_style
        assert not entry_style.bold, f"{entry_text!r} entry must not be bold"
        assert _colour(entry_style) == muted, entry_style

    _assert_header_outranks(
        app._agent_list_block([("auditor", "role", "Audit changes")]), "agents", "auditor"
    )

    # /team gets the identical treatment (needs a real team object).
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    registry = TeamRegistry(Path(tempfile.mkdtemp()))
    team = registry.create_team(
        TeamEditFields(
            name="feature-release", manager="manager", members=[TeamMember(role="coder")]
        )
    )
    _assert_header_outranks(app._team_list_block([team]), "teams", "feature-release")


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
    assert "goal set — applies from the next step" in painted, painted


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


def _dock_geometry(app: OperatorApp, editor: Editor) -> dict[str, int]:
    """ABSOLUTE dock/composer/picker/status coordinates (R7-3).

    Deliberately absolute rather than dock-relative. The reflow this guards
    kept every relative offset intact — composer, picker and status band all
    moved TOGETHER — while the whole dock floated four rows off the bottom of
    the screen with a visible gap under it. The existing catch-up tests assert
    relative geometry and single-match queries, and neither could see it.
    """
    dock = app.query_one("#input-dock", Container)
    return {
        "dock_y": dock.region.y,
        "dock_h": dock.region.height,
        "pad_b": int(dock.styles.padding.bottom),
        "composer_y": editor.region.y,
        "picker_y": editor.picker.region.y,
        "picker_h": editor.picker.region.height,
        "status_y": app.query_one("#status-band").region.y,
    }


async def _team_picker_geometry(match_count: int, *, delayed: bool) -> dict[str, int]:
    """Settle `/team lop` against ``match_count`` matching rows and measure.

    ``delayed`` selects the arm: False adopts the session BEFORE the query is
    typed (the ordinary path), True types the query while the factory is still
    blocked and releases it afterwards (the catch-up path this PR added).
    """
    from local_operator.teams import TeamEditFields, TeamRegistry

    names = ["lopdev", "lopsec", "lopops", "lopqa"][:match_count]
    session = FakeSession()
    registry = TeamRegistry(Path(tempfile.mkdtemp()))
    for name in (*names, "other"):  # `other` never matches `lop`
        registry.create_team(TeamEditFields(name=name, manager="manager"))
    session.team_registry = registry
    release = asyncio.Event()

    async def factory() -> FakeSession:
        if delayed:
            await release.wait()
        return session

    app = OperatorApp(factory)
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        if not delayed:
            await _boot(pilot, app)
        for char in "/team lop":
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()
        if delayed:
            release.set()
            await _boot(pilot, app)
        for _ in range(6):
            await pilot.pause()
        assert editor.picker.region.height == match_count, "query did not match as intended"
        settled = _dock_geometry(app, editor)
        # A settled frame must not move again: a second pass that differs is a
        # reflow the user sees as motion (AGENTS.md "Animation and multi-frame").
        await pilot.pause()
        assert _dock_geometry(app, editor) == settled
        assert app.screen.size == app.screen.virtual_size
        assert app.screen.show_vertical_scrollbar is False
        return settled


@pytest.mark.asyncio
@pytest.mark.parametrize("match_count", [1, 2, 3])
async def test_delayed_and_direct_team_picker_settle_at_identical_geometry(
    match_count: int,
) -> None:
    """R7-3: catch-up must land the dock exactly where the direct path does.

    Parametrised past ONE match on purpose. With a single match the delayed
    reserve and the real list are both one row, so the composition measured
    against the reserve happens to stay correct and the bug is invisible; at
    two rows the dock kept the shorter list's lift and floated `-5` rows with
    an empty band under the status line.
    """
    direct = await _team_picker_geometry(match_count, delayed=False)
    delayed = await _team_picker_geometry(match_count, delayed=True)
    assert delayed == direct, f"delayed arm reflowed: {delayed} != {direct}"
    # Pin the ABSOLUTE frame, not just the agreement between the two arms: two
    # arms that agree on a WRONG geometry is exactly the regression this
    # guards, and the numbers below are the ones the at-rest composition
    # produces for this 100x30 host.
    assert direct["picker_h"] == match_count
    assert direct["composer_y"] == direct["dock_y"] + 1
    assert direct["picker_y"] == direct["composer_y"] + 1
    assert direct["status_y"] == direct["picker_y"] + match_count
    # The lift is the ONE quantity that went wrong: it is the composition's
    # reserve BELOW the dock, and the bug wrote a lift measured against the
    # 1-row reserve while a taller list was showing. Whatever it is, both arms
    # must have computed it from the same list.
    assert delayed["pad_b"] == direct["pad_b"]


@pytest.mark.asyncio
async def test_team_picker_absolute_geometry_matches_at_rest_composition() -> None:
    """R7-3: the picker must not shift the dock away from where boot put it.

    The reflow was only visible against an ABSOLUTE reference, so this test
    supplies one that does not come from the picker at all: the same app with
    no picker open. Opening a list may change the dock's HEIGHT (it gains
    rows), but the bottom of the composition — the status band — must stay put.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        for _ in range(4):
            await pilot.pause()
        at_rest_status_y = app.query_one("#status-band").region.y

    for match_count in (1, 2, 3):
        for delayed in (False, True):
            geometry = await _team_picker_geometry(match_count, delayed=delayed)
            # One match still fits under the at-rest splash budget; more rows
            # legitimately buy space from the composition's lift. Either way the
            # band never falls BELOW where boot placed it, which is what a dock
            # floating above the bottom of the screen looks like numerically.
            assert geometry["status_y"] <= at_rest_status_y, (
                f"{match_count} matches, delayed={delayed}: status band moved "
                f"to {geometry['status_y']} against at-rest {at_rest_status_y}"
            )


@pytest.mark.asyncio
async def test_failed_boot_retires_the_team_reserve_and_restores_enter() -> None:
    """U7-1: after boot FAILS, `/team` must stop promising rows and take Enter.

    The reserve is keyed to "no session adopted yet", which a failed boot makes
    permanently true: the row read `loading teams…` forever and `is_loading()`
    swallowed every Enter, so the user pressed into silence. A boot failure is
    as authoritative as an empty roster — nothing is arriving.
    """

    async def failing_factory() -> FakeSession:
        raise RuntimeError("registry exploded")

    app = OperatorApp(failing_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for _ in range(40):
            await pilot.pause()
            if app._boot_failed:
                break
        assert app._boot_failed is True
        assert app._session is None

        for char in "/team lop":
            await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
        await pilot.pause()
        await pilot.pause()
        # No eternal placeholder, and no loading latch to eat accept keys.
        assert editor.picker._notice == ""
        assert editor.picker.is_loading() is False

        await pilot.press("enter")
        await pilot.pause()
        # Enter reached the ordinary submit path: the draft was consumed and the
        # app reported the session state, instead of the keypress vanishing.
        assert editor.text == ""
        rendered = " ".join(
            str(getattr(block, "_text", "") or "") for block in app._transcript_view().children
        )
        # U8-2: that report names the FAILURE. It used to say "still starting"
        # one row under "✗ session failed to start", which contradicted it; the
        # property this test guards is that Enter is answered at all, and the
        # wording assertion moved with the fix rather than pinning the old lie.
        assert "session failed to start — /login" in rendered


@pytest.mark.asyncio
async def test_boot_failure_flag_clears_when_a_new_session_transition_starts() -> None:
    """U7-1: a later `/resume`/`/login` re-opens the arriving window."""

    async def failing_factory() -> FakeSession:
        raise RuntimeError("registry exploded")

    app = OperatorApp(failing_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._boot_failed:
                break
        assert app._boot_failed is True

        settled = asyncio.Event()

        async def transition() -> None:
            await settled.wait()

        app._run_session_transition(transition())
        await pilot.pause()
        # A session is arriving again, so the reserve is allowed to promise
        # rows: the failure no longer describes the app's state.
        assert app._boot_failed is False
        assert app._name_list_pending_notice("team") == "loading teams…"
        settled.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_pending_notice_reads_the_same_for_canonical_and_alias_spellings() -> None:
    """U8-1: `/agents` must not render "loading agents roster…".

    The notice interpolated the typed command word, so the plural alias leaked
    into copy the product would never write, in the one row whose job is to say
    the roster is coming. Both spellings of each family open the SAME list, so
    they must read identically — one constant per family, the way the team
    branch already did it. No test asserted either ALIAS, which is how this
    survived seven rounds.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Hold the window the notice describes open, so the branch is reached.
        app._session_transition_pending = True
        assert app._name_list_pending_notice("team") == "loading teams…"
        assert app._name_list_pending_notice("teams") == "loading teams…"
        assert app._name_list_pending_notice("agent") == "loading agent roster…"
        assert app._name_list_pending_notice("agents") == "loading agent roster…"
        app._session_transition_pending = False


@pytest.mark.asyncio
async def test_alias_spellings_paint_the_same_pending_row_in_the_picker() -> None:
    """U8-1 through the real surface: type the alias and read the rendered row."""
    for typed, expected in (("/teams x", "loading teams…"), ("/agents a", "loading agent roster…")):
        release = asyncio.Event()

        async def delayed_factory() -> FakeSession:
            await release.wait()
            return FakeSession()

        app = OperatorApp(delayed_factory)
        async with app.run_test(size=(100, 30)) as pilot:
            editor = app.query_one(Editor)
            editor.focus()
            for char in typed:
                await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
            await pilot.pause()
            assert editor.picker._notice == expected, typed
            release.set()
            await pilot.pause()


@pytest.mark.asyncio
async def test_failed_boot_enter_reports_the_failure_not_a_pending_start() -> None:
    """U8-2: Enter after a DEFINITIVE boot failure must not say "still starting".

    U7-1 routes `/team` back into the ordinary submit path, whose no-session
    guard answered "session is still starting…" — directly beneath the app's own
    `✗ session failed to start: …`. The two cannot both be true, and the second
    is the one that tells the user what to do next, so it sent them off to wait
    for something that was never arriving.
    """

    async def failing_factory() -> FakeSession:
        raise RuntimeError("registry exploded")

    app = OperatorApp(failing_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        for _ in range(40):
            await pilot.pause()
            if app._boot_failed:
                break
        assert app._boot_failed is True

        for probe in ("/team lop", "/agent aud"):
            for char in probe:
                await pilot.press("slash" if char == "/" else ("space" if char == " " else char))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

        rendered = " ".join(
            str(getattr(block, "_text", "") or "") for block in app._transcript_view().children
        )
        # The draft is still consumed by the ordinary path (U7-1 holds)...
        assert editor.text == ""
        # ...but nothing tells the user to wait for a session that failed.
        assert "still starting" not in rendered, rendered
        assert rendered.count("session failed to start — /login") == 2, rendered


@pytest.mark.asyncio
async def test_genuine_still_starting_keeps_its_wording() -> None:
    """U8-2 must not regress the case the string was written for.

    While the boot worker is genuinely running, "still starting" is true and
    telling the user to wait is the right advice; only the FAILED state gets the
    other answer.
    """
    release = asyncio.Event()

    async def delayed_factory() -> FakeSession:
        await release.wait()
        return FakeSession()

    app = OperatorApp(delayed_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app._boot_failed is False
        assert app._no_session_notice() == ("session is still starting…", "warning")
        release.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_stopped_session_shows_real_rows_not_an_eternal_placeholder() -> None:
    """`/stop` is TERMINAL: nothing is arriving, so the list must answer.

    The reserve was keyed to ``self._session is None`` with a single
    subtraction for the failed boot, so every LATER way to sit without a
    session inherited "loading teams…" forever. ``/stop`` detaches the session
    on purpose (``_stop_local_session``), which is exactly that shape: the user
    ended the session, no roster is on its way, and the picker promised one
    anyway. Worse than a blank row, because ``is_loading()`` gates Tab/Enter.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Exactly what `_stop_local_session` leaves behind.
        app._session = None
        app._stopped_session_id = "sess-abc123"

        assert app._name_list_pending_notice("team") == ""
        assert app._name_list_pending_notice("agent") == ""


@pytest.mark.asyncio
async def test_the_setup_state_shows_real_rows_not_an_eternal_placeholder() -> None:
    """First-run setup is TERMINAL too, and is NOT covered by `_boot_failed`.

    ``_enter_setup_state`` returns from ``_on_boot_failed`` BEFORE the
    ``_boot_failed = True`` assignment — deliberately, because "no hosting
    configured" is guidance rather than a crash. So the one subtraction the old
    gate made did not apply here, and a user who opened `lop` with nothing
    configured got a permanent "loading teams…" with Tab and Enter swallowed.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._session = None
        app._setup_state = True
        # The flag the old gate keyed on is explicitly NOT set in this state.
        assert app._boot_failed is False

        assert app._name_list_pending_notice("team") == ""
        assert app._name_list_pending_notice("agent") == ""


@pytest.mark.asyncio
async def test_a_stopped_session_does_not_swallow_tab_and_enter() -> None:
    """The other half of the reported breakage, through the real surface.

    ``is_loading()`` gates Tab/Enter, so a stranded reserve does not merely
    withhold rows — it eats the keys that would dismiss the list or submit the
    line. That is what makes an empty list read as "the command is broken".
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._session = None
        app._stopped_session_id = "sess-abc123"

        editor = app._editor()
        editor.focus()
        await pilot.pause()
        for ch in "/team ":
            await pilot.press(ch if ch != " " else "space")
        await pilot.pause()

        # No latch, so accept keys reach their ordinary handlers.
        assert editor.picker.is_loading() is False

        await pilot.press("enter")
        await pilot.pause()
        assert editor.text == "", "Enter was swallowed by a stranded loading reserve"


@pytest.mark.asyncio
async def test_a_genuinely_arriving_roster_still_reserves_its_row() -> None:
    """The reserve must survive for the window it was written for (D1).

    Both real windows: the boot worker still constructing, and a transition the
    user just asked for. Removing the placeholder from these would bring back
    the one-row dock jump the reserve exists to prevent.
    """
    release = asyncio.Event()

    async def delayed_factory() -> FakeSession:
        await release.wait()
        return FakeSession()

    app = OperatorApp(delayed_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Boot worker still running: no session, and no terminal state set.
        assert app._session is None
        assert app._boot_failed is False
        assert app._setup_state is False
        assert app._stopped_session_id == ""
        assert app._name_list_pending_notice("team") == "loading teams…"
        assert app._name_list_pending_notice("agent") == "loading agent roster…"

        # A transition promises rows even from a terminal state.
        app._stopped_session_id = "sess-abc123"
        app._session_transition_pending = True
        assert app._name_list_pending_notice("team") == "loading teams…"

        app._session_transition_pending = False
        app._stopped_session_id = ""
        release.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_team_launch_refuses_instead_of_sending_without_the_briefs(tmp_path) -> None:
    """A session that cannot attach a team must not run the request anyway.

    The guard was ``if callable(attach):`` with NO else, so a session whose
    implementation has no ``attach_team`` fell through to the prompt: the
    receipt said "<manager> is coordinating" while the turn ran with no roster
    and no briefs. A confidently wrong persona is worse than a refusal, because
    nothing on screen distinguishes the two.

    Reachable on the viewer (`RemoteSession`), which lists teams from local
    config but has no seam to stamp an attachment onto the runtime that builds
    the turn.
    """
    from local_operator.teams import TeamEditFields, TeamRegistry

    registry = TeamRegistry(tmp_path)
    registry.create_team(TeamEditFields(name="alpha", description="d", manager="manager"))

    # Exactly the viewer's shape: the LISTING resolves, the ATTACH does not.
    # A subclass rather than `del type(session).attach_team`, which would strip
    # the method from the SHARED FakeSession class for every later test.
    class NoAttachSession(FakeSession):
        attach_team = None  # type: ignore[assignment]

    session = NoAttachSession()
    session.team_registry = registry

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        assert app._session is session
        assert not callable(getattr(app._session, "attach_team", None))

        app._cmd_team("alpha do the thing", app._notice, None)
        await pilot.pause()

        # The request was NOT sent: no briefs, no turn.
        assert session.prompts == [], "the request ran without the team briefs"
        rendered = " ".join(
            str(getattr(block, "_text", "") or "") for block in app._transcript_view().children
        )
        assert "teams cannot be attached in this session" in rendered, rendered
        # The old wording would now be a lie: `/team` lists them one line up.
        assert "teams are unavailable" not in rendered, rendered
