"""Real OperatorApp/CSS tests for read-only page routing and lifecycle."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from textual.containers import ScrollableContainer
from textual.widgets import Static

from local_operator.tui.app import OperatorApp
from local_operator.tui.file_viewer import file_identity
from local_operator.tui.widgets.file_viewer import FileViewer
from local_operator.tui.widgets.supplement_block import SupplementBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


async def boot(pilot, app):
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return
    pytest.fail("session did not boot")


class _ViewerSession(FakeSession):
    def __init__(self, cwd: Path) -> None:
        super().__init__()
        # Both public callouts and the session's local completion contract
        # refer to this synthetic root, never the operator's live session cwd.
        self.cwd = str(cwd)
        self._cwd = str(cwd)


def make_app(tmp_path: Path) -> OperatorApp:
    session = _ViewerSession(tmp_path)
    return OperatorApp(lambda: _factory(session))


async def settled(pilot, app):
    await pilot.pause()
    page = app._file_viewer
    assert isinstance(page, FileViewer)
    if page._read_worker:
        await page._read_worker.wait()
    await pilot.pause()
    return page


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (60, 24)])
async def test_open_wrap_scroll_reload_copy_and_close(tmp_path, monkeypatch, size):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    path = tmp_path / "source.py"
    path.write_text("# " + "long line " * 30 + "\n" + "print('ok')\n" * 70)
    app = make_app(tmp_path)
    async with app.run_test(size=size) as pilot:
        await boot(pilot, app)
        editor = app._editor()
        editor.load_text("draft to preserve")
        app._run_slash_command('/view "source.py"')
        page = await settled(pilot, app)
        assert page.document and page.document.kind == "code"
        assert not page.wrapped
        assert not app._transcript_view().display
        assert page.region.width <= size[0]
        assert app.screen.virtual_size.height == app.screen.size.height
        body = page.query_one(".viewer-body", ScrollableContainer)
        await pilot.press("end", "down", "right")
        await pilot.pause()
        assert body.scroll_y == body.max_scroll_y
        assert body.scroll_x == body.max_scroll_x
        await pilot.press("home", "up", "left")
        await pilot.pause()
        assert body.scroll_y == body.scroll_x == 0
        await pilot.press("w")
        assert page.wrapped
        assert page.query_one(".viewer-legend").region.bottom <= size[1]
        path.write_text("print('reload')")
        await pilot.press("r")
        await settled(pilot, app)
        assert page.document.text == "print('reload')"
        assert page.wrapped  # reload does not discard the reader's preference
        await pilot.press("y")
        assert app.clipboard == str(path)
        assert "Clipboard unavailable" in str(page.query_one(".viewer-status", Static).render())
        await pilot.press("escape")
        await pilot.pause()
        assert app._file_viewer is None
        assert app._transcript_view().display
        assert editor.text == "draft to preserve"
        assert not app.query(FileViewer)


@pytest.mark.asyncio
@pytest.mark.parametrize("name,preview", [("readme.md", True), ("report.pdf", False)])
async def test_callout_click_routes_supported_and_unsupported(tmp_path, monkeypatch, name, preview):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    path = tmp_path / name
    path.write_text("# A document")
    app = make_app(tmp_path)
    async with app.run_test(size=(100, 30)) as pilot:
        await boot(pilot, app)
        # Three entries exercise the existing featured-file callout renderer;
        # the <=2 compact summary is intentionally not a hyperlink surface.
        for extra in ("extra1.txt", "extra2.txt"):
            (tmp_path / extra).write_text("neighbor")
        block = SupplementBlock(
            {"state": "done", "files": [{"path": p, "name": p}
             for p in (name, "extra1.txt", "extra2.txt")], "components": []},
            cwd=str(tmp_path), hyperlinks=True,
        )
        app._append_block(block)
        await pilot.pause()
        hits = [(x, y) for y in range(block.size.height) for x in range(block.size.width)
                if block.link_at(x, y) == path.as_uri()]
        assert hits, "the real compositor did not paint the file hyperlink"
        revealed = AsyncMock()
        monkeypatch.setattr(app, "_reveal_file", revealed)
        await pilot.click(block, offset=hits[0])
        await pilot.pause()
        if preview:
            page = await settled(pilot, app)
            assert page.document is not None and page.document.kind == "markdown"
            revealed.assert_not_awaited()
        else:
            assert app._file_viewer is None
            revealed.assert_awaited_once()
            assert revealed.call_args.args[0] == str(path)


@pytest.mark.asyncio
async def test_snapshot_refusal_and_external_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    path = tmp_path / "a.txt"
    path.write_text("first")
    expected = file_identity(path)
    path.write_text("replacement")
    app = make_app(tmp_path)
    async with app.run_test(size=(60, 24)) as pilot:
        await boot(pilot, app)
        app.open_transcript_file(path.as_uri(), source_path="a.txt", expected=expected)
        await pilot.pause()
        assert app._file_viewer is None
        app._run_slash_command("/view a.txt")
        page = await settled(pilot, app)
        opened = AsyncMock(return_value=False)
        monkeypatch.setattr("local_operator.tui.widgets.file_viewer.open_path_quietly", opened)
        await pilot.press("o")
        await pilot.pause()
        opened.assert_awaited_once_with(str(path))
        assert "Could not open" in str(page.query_one(".viewer-status", Static).render())
        opened.return_value = True
        await pilot.press("o")
        await pilot.pause()
        assert opened.await_count == 2


@pytest.mark.asyncio
async def test_close_cancels_the_reader_and_reload_cancels_previous(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    starts = 0
    exits = 0

    async def blocked(*args, **kwargs):
        nonlocal starts, exits
        starts += 1
        try:
            await asyncio.Event().wait()
        finally:
            exits += 1
    monkeypatch.setattr("local_operator.tui.widgets.file_viewer.read_document_async", blocked)
    app = make_app(tmp_path)
    async with app.run_test(size=(60, 24)) as pilot:
        await boot(pilot, app)
        app._run_slash_command("/view a.txt")
        await pilot.pause()
        page = app._file_viewer
        assert isinstance(page, FileViewer)
        first = page._read_worker
        assert first is not None
        await pilot.press("r")
        await pilot.pause()
        assert first.is_cancelled and exits == 1
        second = page._read_worker
        assert second is not None
        await pilot.press("escape")
        await pilot.pause()
        assert starts == exits == 2
        assert second.is_cancelled and app._file_viewer is None


@pytest.mark.asyncio
async def test_path_completion_preserves_command_and_nested_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a file.txt").write_text("hello")
    app = make_app(tmp_path)
    async with app.run_test(size=(100, 30)) as pilot:
        await boot(pilot, app)
        editor = app._editor()
        editor.load_text("/view sub/a")
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == '/view "sub/a file.txt"'
        await pilot.press("enter")
        page = await settled(pilot, app)
        assert page.document is not None and page.document.text == "hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("name,data,message", [
    ("binary.txt", b"bad\x00payload", "Cannot preview a binary file"),
    ("missing.txt", None, "Cannot read this file"),
    ("report.pdf", b"%PDF", "no inline preview"),
])
async def test_refusals_remain_visible_on_wrap_key(tmp_path, monkeypatch, name, data, message):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    if data is not None:
        (tmp_path / name).write_bytes(data)
    app = make_app(tmp_path)
    async with app.run_test(size=(60, 24)) as pilot:
        await boot(pilot, app)
        app._run_slash_command(f"/view {name}")
        page = await settled(pilot, app)
        assert page.document is None
        before = str(page.query_one(".viewer-status", Static).render())
        assert message in before
        await pilot.press("w")
        assert str(page.query_one(".viewer-status", Static).render()) == before


@pytest.mark.asyncio
async def test_settings_transition_cancels_and_replaces_viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "a.txt").write_text("hello")
    app = make_app(tmp_path)
    async with app.run_test(size=(60, 24)) as pilot:
        await boot(pilot, app)
        app._run_slash_command("/view a.txt")
        await settled(pilot, app)
        app._run_slash_command("/settings")
        await pilot.pause()
        assert app._file_viewer is None and app._settings_view is not None
        assert not app.query(FileViewer)
        app._run_slash_command("/view a.txt")
        page = await settled(pilot, app)
        assert page.document is not None and page.document.text == "hello"
        assert app._settings_view is None
