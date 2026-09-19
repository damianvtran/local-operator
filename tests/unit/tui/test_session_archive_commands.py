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

from local_operator.session.archived import read_archived
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.command_picker import PickerMode
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _resume_factory
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
