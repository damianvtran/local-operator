"""`/move` driven through the REAL app: the editor, the submit handler, the band.

Calling ``_cmd_move`` directly would skip the editor and the submit handler,
which is the pair a user actually goes through — so everything here types into
the real composer and presses Enter.

The band assertions are the load-bearing ones. The failure this feature must
not have is the one AGENTS.md names for `/reload`: the screen showing one
directory while the session works in another. So every path that changes the
directory is checked against what the band says, and every path that refuses is
checked against the band NOT having changed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.move_picker import MovePickerScreen
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class MovableSession(FakeSession):
    """A session that can actually be moved, recording what it was asked.

    ``FakeSession`` deliberately has no ``set_working_directory``: the app must
    refuse a facade that cannot be moved rather than reporting a move that did
    not happen, and that refusal is asserted below too.
    """

    def __init__(self, cwd: str = "/tmp", outcome: str = "cold") -> None:
        super().__init__()
        self._cwd = cwd
        self._outcome = outcome
        self.moves: list[str] = []
        self.error: Exception | None = None

    async def set_working_directory(self, cwd: str) -> str:
        self.moves.append(cwd)
        if self.error is not None:
            raise self.error
        self._cwd = cwd
        return self._outcome


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _band(app: OperatorApp) -> str:
    status = app._status
    assert status is not None
    return status.render_text(200).plain


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    for _ in range(6):
        await pilot.pause()


@pytest.mark.asyncio
async def test_moving_a_cold_session_updates_the_band_and_says_so(tmp_path: Path) -> None:
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/move {destination}")

        assert session.moves == [str(destination)]
        assert str(destination) in _band(app)
        assert any("moved to" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_rebound_session_says_its_runtime_restarted(tmp_path: Path) -> None:
    """The user must be told the runtime was replaced — it is a visible pause
    and a real event, even though the conversation is untouched."""
    session = MovableSession(cwd=str(tmp_path), outcome="rebound")
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/move {destination}")
        assert any("runtime restarted" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_tilde_path_is_expanded(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "project").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move ~/project")
        assert session.moves == [str(home / "project")]


@pytest.mark.asyncio
async def test_a_relative_path_resolves_against_the_SESSIONS_directory(tmp_path: Path) -> None:
    (tmp_path / "child").mkdir()
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move child")
        assert session.moves == [str(tmp_path / "child")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix,expected",
    [("nope", "no such directory"), ("file.txt", "not a directory")],
)
async def test_an_invalid_target_is_refused_without_moving(
    tmp_path: Path, suffix: str, expected: str
) -> None:
    """A clear notice, never a traceback and never a half-applied state."""
    (tmp_path / "file.txt").write_text("x")
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        before = _band(app)
        await _submit(pilot, app, f"/move {tmp_path / suffix}")

        assert session.moves == []
        assert any(expected in text for text in _notices(app))
        assert _band(app) == before


@pytest.mark.asyncio
async def test_moving_to_the_directory_youre_already_in_is_a_no_op(tmp_path: Path) -> None:
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/move {tmp_path}")
        assert session.moves == []
        assert any("already in" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_refusal_from_the_session_reaches_the_user_verbatim(tmp_path: Path) -> None:
    """A busy session's refusal is the receipt; the band must not move."""
    session = MovableSession(cwd=str(tmp_path))
    session.error = RuntimeError("this session is working right now")
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        before = _band(app)
        await _submit(pilot, app, f"/move {destination}")

        assert any("working right now" in text for text in _notices(app))
        assert _band(app) == before


@pytest.mark.asyncio
async def test_a_session_that_cannot_be_moved_is_refused_not_silently_ignored() -> None:
    """``FakeSession`` has no ``set_working_directory``; reporting success
    would tell the user a move happened that did not."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move /tmp")
        assert any("cannot be moved" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_bare_move_opens_the_picker_on_the_current_directory(tmp_path: Path) -> None:
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move")
        for _ in range(4):
            await pilot.pause()

        screen = app.screen
        assert isinstance(screen, MovePickerScreen)
        assert screen.visible_rows
        assert screen.visible_rows[0].path == str(tmp_path)
        assert screen.visible_rows[0].kind == "current"


@pytest.mark.asyncio
async def test_escaping_the_picker_leaves_the_session_where_it_was(tmp_path: Path) -> None:
    """A cancelled picker is not an event worth a transcript line."""
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        before = _band(app)
        await _submit(pilot, app, "/move")
        for _ in range(4):
            await pilot.pause()
        await pilot.press("escape")
        for _ in range(4):
            await pilot.pause()

        assert session.moves == []
        assert _band(app) == before


@pytest.mark.asyncio
async def test_choosing_a_row_in_the_picker_moves_there(tmp_path: Path) -> None:
    """The whole point of the card, driven the way a user drives it."""
    child = tmp_path / "child"
    child.mkdir()
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move")
        for _ in range(4):
            await pilot.pause()

        screen = app.screen
        assert isinstance(screen, MovePickerScreen)
        # Row 0 is the current directory, so move down to a real destination.
        index = next(i for i, row in enumerate(screen.visible_rows) if row.path != str(tmp_path))
        for _ in range(index):
            await pilot.press("down")
            await pilot.pause()
        chosen = screen.selected_path()
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()

        assert session.moves == [chosen]
        assert str(chosen) in _band(app)
