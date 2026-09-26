"""The ``@project:`` arm in the composer: rows, completion, and the ink.

Slice 2's TUI half (§6.3). The picker composes project rows AHEAD of the
directory listing; each row carries the namespaced token as its VALUE
(``name``) while the name column paints the bare project name (``display``),
so every existing FILE completion path — Tab, the ghost, the "already in the
buffer" rule — writes ``@project:<name>`` through the one function that owns
"what does accepting this row put in the buffer". The reference ink paints a
project token through ``references.reference_resolves``, the same gate the
resolver itself asks, which is why the ink needs no editor change of its own.

Driven over ``FakeSession`` + the assembled app, like ``test_at_picker.py``.
The resolver-side half (classification, element shape, idempotence) lives in
``tests/unit/test_project_references.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.projects import ProjectEdit, ProjectRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor

from .test_app_pilot import FakeSession, _factory


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small tree as the process cwd, with two files the directory arm lists."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("app\n")
    (tmp_path / "README.md").write_text("readme\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _session_with(registry: ProjectRegistry | None) -> FakeSession:
    session = FakeSession()
    session.project_registry = registry
    return session


async def _draft(app: OperatorApp, pilot, text: str) -> Editor:
    """Put ``text`` in the composer with the caret at the end, picker settled."""
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.load_text(text)
    editor.move_cursor(editor._end_of_buffer())
    await pilot.pause()
    # The app answers `FileQueryOpened` one message-loop tick later.
    await pilot.pause()
    return editor


def _rows(editor: Editor) -> list[str]:
    return [name for name, _ in editor.picker.suggestions()]


@pytest.mark.asyncio
async def test_project_rows_are_prepended_to_the_directory_listing(workspace) -> None:
    registry = ProjectRegistry(workspace / "cfg")
    registry.create_project(ProjectEdit(name="alpha", description="Alpha workstream"))
    registry.create_project(ProjectEdit(name="beta", description="Beta workstream"))
    app = OperatorApp(lambda: _factory(_session_with(registry)))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")

        rows = _rows(editor)
        assert rows[:2] == ["project:alpha", "project:beta"], rows
        assert "README.md" in rows
        assert "src/" in rows  # the directory arm composed, not replaced


@pytest.mark.asyncio
async def test_typing_a_fragment_after_the_colon_narrows_to_project_rows(workspace) -> None:
    """``@project:al`` is a prefix of the row's VALUE, so the picker's own filter
    keeps the row — the composer needs no second eligibility rule."""
    registry = ProjectRegistry(workspace / "cfg")
    registry.create_project(ProjectEdit(name="alpha"))
    registry.create_project(ProjectEdit(name="beta"))
    app = OperatorApp(lambda: _factory(_session_with(registry)))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@project:al")
        assert _rows(editor) == ["project:alpha"]


@pytest.mark.asyncio
async def test_accepting_a_project_row_writes_the_namespaced_token(workspace) -> None:
    """Tab on a project row inserts ``@project:<name>`` — the row's VALUE, and
    exactly what `reference_resolves` reads back on the next keystroke."""
    registry = ProjectRegistry(workspace / "cfg")
    registry.create_project(ProjectEdit(name="alpha"))
    app = OperatorApp(lambda: _factory(_session_with(registry)))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "status of @pro")
        await pilot.press("tab")
        await pilot.pause()

        assert editor.text == "status of @project:alpha"


@pytest.mark.asyncio
async def test_the_name_column_paints_the_bare_project_name(workspace) -> None:
    """The row VALUE is the token; the row PAINTS the project name (the
    ``/new`` device picker's split — a value repeating a keyword must not be
    what the name column shows)."""
    registry = ProjectRegistry(workspace / "cfg")
    registry.create_project(ProjectEdit(name="alpha", description="Alpha workstream"))
    app = OperatorApp(lambda: _factory(_session_with(registry)))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")
        rendered = editor.picker.render_text(80).plain.split("\n")
        first = rendered[0]
        assert "alpha" in first
        assert "project:alpha" not in first
        assert "Alpha workstream" in first


@pytest.mark.asyncio
async def test_the_project_rows_are_capped_at_eight(workspace) -> None:
    registry = ProjectRegistry(workspace / "cfg")
    for index in range(10):
        registry.create_project(ProjectEdit(name=f"project-{index:02d}"))
    app = OperatorApp(lambda: _factory(_session_with(registry)))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")

        projects = [name for name in _rows(editor) if name.startswith("project:")]
        assert len(projects) == 8
        # The first eight by name — the store's own listing order.
        assert projects == [f"project:project-{index:02d}" for index in range(8)]


@pytest.mark.asyncio
async def test_no_registry_means_no_project_rows_and_no_error(workspace) -> None:
    """A session without a registry (an older runtime, a store that failed to
    open) degrades to the directory list alone — the picker still works."""
    app = OperatorApp(lambda: _factory(_session_with(None)))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")

        rows = _rows(editor)
        assert "README.md" in rows
        assert not [name for name in rows if name.startswith("project:")]


@pytest.mark.asyncio
async def test_the_reference_ink_paints_a_project_token(workspace, monkeypatch) -> None:
    """The editor's ink gate is `reference_resolves`, so the project arm lands
    in the ink with no editor-side rule: a token naming a row is painted, one
    naming nothing is not."""
    root = workspace / "cfg"
    registry = ProjectRegistry(root)
    registry.create_project(ProjectEdit(name="alpha"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    app = OperatorApp(lambda: _factory(_session_with(registry)))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "see @project:alpha")
        assert editor._reference_runs() == {0: [(4, 18)]}

        editor.load_text("see @project:ghost")
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        assert editor._reference_runs() == {}
