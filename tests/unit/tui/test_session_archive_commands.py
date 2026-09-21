"""``/archive``, ``/unarchive`` and ``/delete`` through the REAL composer.

Every assertion here goes through the editor and the submit handler — the pair a
reported bug lives in — rather than calling a handler directly, so the registry
entry, the primary-name resolution, the argument picker and the dispatch chain
are all exercised by the same keystrokes a user makes.

The four properties that are load-bearing:

* **The receipt names the way back.** An archive is only different from a delete
  because the conversation is still there, and a user who has just watched one
  leave every list needs to be told how to reach it.
* **``/unarchive`` is OFFERED only while the session is archived, and ANSWERS
  when typed anyway.** A registry is static, so the seam is the list handed to
  the composer; the sentence on the other branch is what stops a user who
  learned the word elsewhere from reading a no-op as a broken command.
* **``/delete`` bare is a REHEARSAL.** It reports exactly what the real one would
  remove — including any refusal the guards would give — and removes nothing.
* **A successful delete lands the app on a FRESH conversation.** Not on a dead
  one: the session it was standing in no longer exists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.session.archived import ARCHIVED_LIMIT, read_archived
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.command_picker import ArgumentChoice, PickerMode
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import (
    FakeSession,
    _await_session,
    _factory,
    _resume_factory,
    _StoppedFollowerSession,
    _transcript_text,
    _unwrapped,
)
from tests.unit.tui.test_slash_echo import _submit

SESSION = "sess"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated config root whose CURRENT session is resumable.

    ``_resumable_session_id`` is gated on the transcript EXISTING, and every
    command here acts on the current session — so a fixture without this file
    would test the "nothing saved yet" branch while reading like it tested the
    command.
    """
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    session_dir = tmp_path / "sessions" / SESSION
    session_dir.mkdir(parents=True)
    # The store MARKER is what ``remove_session_dir`` requires, and a real config
    # root always has it: session construction marks the store it creates. A
    # fixture that skipped it would test the "unmarked store" refusal and read
    # like it tested the delete.
    from local_operator.session.cleanup import mark_store

    mark_store(tmp_path / "sessions")
    (session_dir / "transcript.jsonl").write_text(
        json.dumps({"type": "message", "payload": {"role": "user", "content": "hello there"}})
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "created_at.json").write_text("1700000000", encoding="utf-8")
    return tmp_path


def _notices(app: OperatorApp) -> str:
    """Every notice body on screen, joined — the receipts a user reads."""
    return "\n".join(
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    )


def _offered(app: OperatorApp) -> set[str]:
    """The command names the composer's picker completes from.

    Read off the PICKER rather than the registry, because that list is the seam
    this feature uses for a conditional command — the registry is static and
    deliberately stays that way.
    """
    return {entry.name for entry in app.query_one(Editor)._picker._commands}


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return
    raise AssertionError("the session never bound")


# ---------------------------------------------------------------------------
# /archive and /unarchive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archive_hides_the_current_session_and_says_how_to_get_it_back(
    root: Path,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/archive")

        assert read_archived(root) == [SESSION]
        receipt = _notices(app)
        assert "archived" in receipt
        assert SESSION in receipt, "the id is the spelling that still resolves it"
        assert "/unarchive" in receipt, "the way back is the whole difference from /delete"
        # And it is gone from the listing a user browses.
        from local_operator.resume import recent_sessions

        assert recent_sessions(root, limit=None) == []


@pytest.mark.asyncio
async def test_unarchive_puts_it_back(root: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/archive")
        await _submit(pilot, app, "/unarchive")

        assert read_archived(root) == []
        assert "listed again" in _notices(app)


@pytest.mark.asyncio
async def test_unarchive_typed_when_it_does_not_apply_answers_with_a_sentence(
    root: Path,
) -> None:
    """Not silence: a user who learned the word in another window must not read
    the no-op as a broken command."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/unarchive")

        assert read_archived(root) == []
        assert "not archived" in _notices(app)


@pytest.mark.asyncio
async def test_unarchive_is_offered_only_while_the_session_is_archived(root: Path) -> None:
    """The one conditional command, asserted on the list the composer carries.

    A static registry plus a filtered list is the mechanism; the editor's own
    ``set_commands`` is what the picker completes from, so the assertion is made
    against that object rather than against the registry.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        names = _offered(app)
        assert "archive" in names and "delete" in names
        assert "unarchive" not in names

        await _submit(pilot, app, "/archive")
        names = _offered(app)
        assert "unarchive" in names, "archiving the current session offers the way back"
        assert "archive" in names, "and does not remove the command that is a no-op here"

        await _submit(pilot, app, "/unarchive")
        assert "unarchive" not in _offered(app)


# ---------------------------------------------------------------------------
# /delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_delete_is_a_rehearsal_that_removes_nothing(root: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/delete")

        receipt = _notices(app)
        assert "cannot be undone" in receipt
        assert "/delete yes" in receipt
        assert (root / "sessions" / SESSION).is_dir(), "a rehearsal removes nothing"


@pytest.mark.asyncio
async def test_delete_yes_removes_the_session_and_lands_on_a_fresh_one(root: Path) -> None:
    boots: list[str | None] = []
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory(boots))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/delete yes")
        for _ in range(40):
            await pilot.pause()
            if not (root / "sessions" / SESSION).exists():
                break

        assert not (root / "sessions" / SESSION).exists()
        assert "deleted" in _notices(app)
        # ``/new`` asks the resume factory for ``None`` — a fresh conversation
        # through the identical path a cold launch takes. Without this the app
        # would be standing on a conversation whose directory is gone.
        assert boots == [None], boots


@pytest.mark.asyncio
async def test_a_running_session_is_refused_and_the_refusal_names_the_guard(root: Path) -> None:
    """The current session IS live in a real app; here it is made live explicitly.

    The claim marker is the same one the runtime writes, so the guard is
    exercised through its real reader rather than through a stub.
    """
    from local_operator.session.retention import LIVE_MARKER_NAME

    (root / "sessions" / SESSION / LIVE_MARKER_NAME).write_text(str(os.getpid()), encoding="utf-8")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/delete yes")

        assert (root / "sessions" / SESSION).is_dir()
        receipt = _notices(app)
        assert "running session" in receipt
        assert "Stop it" in receipt, "the remedy is the point of naming the guard"


@pytest.mark.asyncio
async def test_the_delete_row_is_painted_dangerous_and_one_enter_only_fills_it(
    root: Path,
) -> None:
    """The picker's half of the confirmation, through the real argument list.

    ``/delete `` opens a one-row list whose row carries ``alert`` — the flag the
    picker paints danger colours from — and ``delete`` is in
    ``Editor.DESTRUCTIVE_COMMANDS``, so Enter on that row FILLS the word rather
    than running it. One keystroke on a highlighted row must not be able to
    remove a conversation.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        # TYPED, and with the caret placed at the end: the value list opens on
        # the keystroke that follows the command word, so a test that merely
        # assigned ``text`` would leave the cursor where the setter left it and
        # press space into the wrong end of the line.
        editor.text = "/delete"
        editor.cursor_location = (0, len("/delete"))
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()

        picker = editor._picker
        assert picker.is_open()
        assert picker.mode is PickerMode.ARGUMENT
        [row] = picker.suggestions()
        assert row[0] == "yes"
        assert getattr(row[1], "alert", False) is True, "the picker paints it as dangerous"
        assert editor._argument_is_destructive() is True

        await pilot.press("enter")
        await pilot.pause()
        assert (root / "sessions" / SESSION).is_dir(), "one Enter must not delete"

        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if not (root / "sessions" / SESSION).exists():
                break
        assert not (root / "sessions" / SESSION).exists(), "the second Enter is the confirmation"


# ---------------------------------------------------------------------------
# The state a user is ACTUALLY in: an attached viewer, and a stopped one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_works_on_a_stopped_owner_that_still_advertises_the_slash(
    root: Path,
) -> None:
    """``/delete yes`` removes the conversation in the state ``/stop`` leaves behind.

    THE DEFECT THIS PINS (review round 1, MAJOR-1) was not in the delete itself
    but in where the command was answered. ``/delete`` was classified
    ``authoritative_session``, so it routed to the runtime owner — and the
    conversation a viewer is standing in is by construction the one with a live
    owner, which holds both ``.session.pid`` and ``.execution-lease``, so the
    owner refused with "open in a running session. Stop it before deleting it."
    Stopping did not help: a stopped facade's pre-route answer ("this session was
    stopped; /resume …") intercepts every ROUTED command, so the remedy the
    sentence named could not be carried out in the terminal that printed it.

    Two claims are therefore asserted together, and either one alone would pass
    on the broken build:

    * **NOTHING IS ROUTED** — ``session.routed`` stays empty, which is what the
      frontend-local classification buys;
    * **THE CONVERSATION IS GONE** — the directory is removed and the app booted
      a fresh conversation, which is what the pre-route exemption buys (the
      fixture advertises ``delete`` as authoritative from the STALE capability
      snapshot an owner leaves behind, so a pre-route that trusted the
      advertisement alone would answer "/resume" here and delete nothing).
    """
    boots: list[str | None] = []
    session = _StoppedFollowerSession(cold=True, commands=("model", "delete", "archive"))
    app = OperatorApp(lambda: _factory(session), resume_factory=_resume_factory(boots))
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_session(app, pilot)
        # The viewer is a follower whose owner was stopped: the id is recorded
        # before the socket closes, and the facade is cold.
        app._stopped_session_id = SESSION
        app._run_slash_command("/delete yes")
        # The receipt is published ACROSS the transition to the fresh
        # conversation, so the screen is mid-rebuild for a tick: wait for the
        # directory to be gone AND the transcript to be back before reading it.
        for _ in range(60):
            await pilot.pause()
            if not (root / "sessions" / SESSION).exists() and app.query(TranscriptView):
                break
        receipt = _unwrapped(_transcript_text(app))

    assert session.routed == [], "the command must be answered by this frontend"
    assert not (root / "sessions" / SESSION).exists()
    assert "deleted" in receipt, receipt
    assert boots == [None], boots


@pytest.mark.asyncio
async def test_archive_on_a_stopped_owner_writes_this_terminals_store(root: Path) -> None:
    """``/archive`` hides the row THIS sidebar paints, even on a stopped follower.

    The wrong-machine half of MAJOR-1: the store is ``config_dir()/
    archived-sessions.json`` and the sidebar and picker read THAT file, so
    routing the write to a runtime would hide the conversation on the runtime's
    host while the receipt promised the sidebar in front of the user.
    """
    session = _StoppedFollowerSession(cold=True, commands=("model", "archive"))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_session(app, pilot)
        app._stopped_session_id = SESSION
        app._run_slash_command("/archive")
        await pilot.pause()
        await pilot.pause()

    assert session.routed == []
    assert read_archived(root) == [SESSION], "the local root's store is the one written"


# ---------------------------------------------------------------------------
# The conditional offer survives every path that re-hands the composer a list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closing_an_aside_does_not_re_offer_unarchive(root: Path) -> None:
    """The aside-close restore path hands over the INVOKER's list (MINOR-1).

    ``_close_aside`` used to hand the composer the raw registry while its sibling
    restore path had already been moved to ``_offered_commands()``, so opening
    and closing an aside on any session re-offered ``/unarchive`` for a
    conversation that is not archived — the one command whose contract is that it
    appears only in the state it can act on.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        before = _offered(app)
        assert "unarchive" not in before

        app._open_aside()
        await pilot.pause()
        assert app._close_aside() is True
        await pilot.pause()

        assert _offered(app) == before, _offered(app) - before
        assert "unarchive" not in _offered(app)


# ---------------------------------------------------------------------------
# The cap's consequence is said out loud when it happens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaching_the_cap_names_the_conversation_put_back_in_the_lists(
    root: Path,
) -> None:
    """Archiving the 201st conversation revives the oldest, and the receipt says so.

    ``ARCHIVED_LIMIT`` bounds the FILE by dropping its oldest entry, which puts a
    conversation the user deliberately hid back into the picker, the sidebar, the
    desktop catalogue and search. Silence there reads as the archive forgetting
    (review round 1, MINOR-2), so the evicted id is named in the receipt.
    """
    filled = [f"{index:012x}" for index in range(ARCHIVED_LIMIT)]
    for session_id in filled:
        (root / "sessions" / session_id).mkdir(parents=True)
    (root / "archived-sessions.json").write_text(json.dumps(filled), encoding="utf-8")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/archive")

        receipt = _notices(app)
        assert SESSION in receipt
        assert filled[-1] in receipt, "the evicted conversation is named"
        assert f"at most {ARCHIVED_LIMIT}" in receipt, receipt
        assert read_archived(root)[0] == SESSION
        assert filled[-1] not in read_archived(root)


# ---------------------------------------------------------------------------
# What the sentences say (design round 1 D2/D4/D6, UX round 1 U3/U5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_rehearsal_names_the_conversation_by_title(root: Path) -> None:
    """Design round 1 (D2): the id is not the handle that is on the screen.

    The rehearsal is typed into a screen whose status band carries the model and
    the cwd; the id appears only as a dim right-hand column inside ``/resume``. So
    a rehearsal that named only the id asked the user to confirm the destruction of
    something they could not see — and in the round-1 frame the fixture's id
    rendered as the word ``sess``, reading like a truncation. The title leads, the
    id stays in parentheses because it is what resolves.
    """
    from local_operator.resume import write_session_title

    write_session_title(
        root / "sessions" / SESSION,
        "Parser crash on nested frontmatter",
        user_set=True,
        past_names=[],
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/delete")

        rehearsal = _notices(app)
        assert "Parser crash on nested frontmatter" in rehearsal, rehearsal
        assert f"({SESSION})" in rehearsal, rehearsal
        assert (root / "sessions" / SESSION).is_dir(), "still a rehearsal"


@pytest.mark.asyncio
async def test_the_archive_receipt_is_plain_text_and_names_the_chord(root: Path) -> None:
    """UX U3 / design D4 (backticks) and UX U5 (the chord).

    The receipt was the only sentence in the family that printed markdown — two
    cells of punctuation that mean nothing in a terminal — and it is the one a user
    reads when a conversation leaves every list, so it is the last place to spend
    them. It also names the picker's control by a chord the picker's own first line
    has always shown, so a keyboard user does not have to open the picker to learn
    the key.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/archive")

        receipt = _notices(app)
        assert "`" not in receipt, receipt
        assert "(ctrl+a)" in receipt, receipt


@pytest.mark.asyncio
async def test_the_delete_confirmation_row_fits_its_warning_whole(root: Path) -> None:
    """Design round 1 (D6): the row cut the operative noun mid-word.

    ``yes | delete this conversation and its tran… | cannot be undone`` truncates
    the word that names what is destroyed — the transcript — on the row whose whole
    job is to slow a finger down. The replacement is short enough to paint whole at
    the standard width, which is what this asserts rather than its spelling.
    """
    from tests.unit.tui.test_command_picker import _argument_picker

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.text = "/delete"
        editor.cursor_location = (0, len("/delete"))
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()

        [suggestion] = editor._picker.suggestions()
        choice = suggestion[1]
        assert isinstance(choice, ArgumentChoice), choice
        painted = _argument_picker([choice]).render_rows(100)[0].plain
        assert "…" not in painted, painted
        assert choice.description in painted, painted
