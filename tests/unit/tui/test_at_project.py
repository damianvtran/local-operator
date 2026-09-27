"""The ``@project:`` arm in the composer: rows, completion, and the ink.

Slice 2's TUI half (§6.3). The picker composes project rows AHEAD of the
directory listing; each row carries the namespaced token as its VALUE
(``name``) while the name column paints the bare project name (``display``),
so every existing FILE completion path — Tab, the ghost, the "already in the
buffer" rule — writes ``@project:<name>`` through the one function that owns
"what does accepting this row put in the buffer". The reference ink paints a
project token through ``references.reference_resolves``, the same gate the
resolver itself asks, which is why the ink needs no editor change of its own.

THE STORE IS REACHED THE WAY PRODUCTION REACHES IT: the app and the ink both
read the process's one registry through ``local_operator.references``
(``project_picker_rows`` on a daemon thread; ``_project_names_cache`` for the
ink), rooted at ``LOCAL_OPERATOR_CONFIG_DIR``. These tests therefore seed row
files on a throwaway config root and point that env var at it — a
session-level fake registry would prove nothing about either path, which is
why the earlier wiring's fake was retired in review round 1 (M-2/M-4).

Driven over ``FakeSession`` + the assembled app, like ``test_at_picker.py``.
The resolver-side half (classification, element shape, idempotence) lives in
``tests/unit/test_project_references.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import local_operator.references as references
from local_operator.projects import ProjectEdit, ProjectRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor

from .test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _cold_project_caches():
    """Both module caches start — and end — cold, per test.

    The registry and names caches are process-wide by design (one store read
    per keystroke budget); a test that primed them would leak a snapshot into
    the next test's assertions. Cheap to clear, so cleared around every test.
    """
    references._project_registry_cache = None
    references._project_names_cache = None
    yield
    references._project_registry_cache = None
    references._project_names_cache = None


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small tree as the process cwd, with two files the directory arm lists."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("app\n")
    (tmp_path / "README.md").write_text("readme\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectRegistry:
    """A project store on a throwaway config root — the resolver's own lookup env."""
    root = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return ProjectRegistry(root)


async def _settle(editor: Editor, pilot, predicate, tries: int = 400) -> None:
    """Poll until ``predicate`` holds, bounded, yielding to the event loop.

    The project half of ``FileQueryOpened`` is answered on a daemon thread
    (``references.project_picker_rows``), so its rows land when they land;
    counting message-loop ticks loses that race under load. A bound keeps a
    genuinely stuck fill a loud failure rather than a hang.
    """
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the picker never settled")


async def _draft(app: OperatorApp, pilot, text: str) -> Editor:
    """Put ``text`` in the composer with the caret at the end, picker settled."""
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.load_text(text)
    editor.move_cursor(editor._end_of_buffer())
    # Rows (or the no-rows notice) land one message loop plus one daemon-thread
    # hop later; poll on the visible effect rather than counting pauses.
    await _settle(editor, pilot, lambda: bool(editor.picker.display))
    return editor


def _rows(editor: Editor) -> list[str]:
    return [name for name, _ in editor.picker.suggestions()]


def _project_rows(editor: Editor) -> list[str]:
    return [name for name in _rows(editor) if name.startswith("project:")]


@pytest.mark.asyncio
async def test_project_rows_are_prepended_to_the_directory_listing(workspace, store) -> None:
    store.create_project(ProjectEdit(name="alpha", description="Alpha workstream"))
    store.create_project(ProjectEdit(name="beta", description="Beta workstream"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")

        rows = _rows(editor)
        assert rows[:2] == ["project:alpha", "project:beta"], rows
        assert "README.md" in rows
        assert "src/" in rows  # the directory arm composed, not replaced


@pytest.mark.asyncio
async def test_typing_a_fragment_after_the_colon_narrows_to_project_rows(workspace, store) -> None:
    """``@project:al`` is a prefix of the row's VALUE, so the picker's own filter
    keeps the row — the composer needs no second eligibility rule."""
    store.create_project(ProjectEdit(name="alpha"))
    store.create_project(ProjectEdit(name="beta"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@project:al")
        assert _rows(editor) == ["project:alpha"]


@pytest.mark.asyncio
async def test_an_exact_name_beyond_the_eighth_project_still_surfaces_its_row(
    workspace, store
) -> None:
    """The shortcut CAPS MATCHES, never the vocabulary (review round 1, M-2).

    With twelve projects the ninth and beyond used to be unreachable — even by
    their exact name — while the ink painted the same token as a valid
    reference, and the picker said ``nothing here matches`` about a live row.
    The error state is a highlight that lies, so the test asserts BOTH halves:
    the row surfaces (no notice) and the bare ``@`` still shows exactly eight
    shortcut rows.
    """
    for index in range(12):
        store.create_project(ProjectEdit(name=f"sigma-{index:02d}"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@project:sigma-11")

        assert _rows(editor) == ["project:sigma-11"]
        assert "nothing here matches" not in editor.picker.render_text(120).plain

        # Back to a bare `@`: the shortcut is capped ON SCREEN at eight rows,
        # and it still leaves the directory listing behind it.
        editor.load_text("@")
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        assert len(_project_rows(editor)) == 8
        assert "README.md" in _rows(editor)


@pytest.mark.asyncio
async def test_the_painted_name_is_also_a_way_to_find_the_row(workspace, store) -> None:
    """``@al`` reaches the row the screen spells ``alpha`` (review round 1, F9).

    The row's VALUE is the namespaced token, so the bare name only matches
    through the row's alias — without it a user who reads "alpha" off the
    screen and types a fragment of it gets an empty list. Completing still
    writes the namespaced token: the alias is how the row is FOUND, never what
    the buffer receives.
    """
    store.create_project(ProjectEdit(name="alpha"))
    store.create_project(ProjectEdit(name="beta"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@al")

        assert _rows(editor) == ["project:alpha"]
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == "@project:alpha"


@pytest.mark.asyncio
async def test_accepting_a_project_row_writes_the_namespaced_token(workspace, store) -> None:
    """Tab on a project row inserts ``@project:<name>`` — the row's VALUE, and
    exactly what `reference_resolves` reads back on the next keystroke."""
    store.create_project(ProjectEdit(name="alpha"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "status of @pro")
        await pilot.press("tab")
        await pilot.pause()

        assert editor.text == "status of @project:alpha"


@pytest.mark.asyncio
async def test_project_rows_are_not_offered_under_a_directory_token(workspace, store) -> None:
    """``@src/`` is not "typing toward ``project:``" (review round 1, M-3).

    Project rows used to be prepended there too, and FILE completion then
    accepted one into ``@src/project:<name>`` — a token the resolver reads as
    prose, with no ink and no expansion. The directory gate is
    ``message.directory``, which already carries exactly this fact.
    """
    store.create_project(ProjectEdit(name="alpha"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@src/")

        assert _project_rows(editor) == []
        assert "app.py" in _rows(editor)

        # And the completion cannot produce the dead form either: Tab here
        # writes the directory entry, never a project token.
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == "@src/app.py"


@pytest.mark.asyncio
async def test_a_bare_at_does_not_seed_the_highlight_onto_a_project_row(workspace, store) -> None:
    """Enter on a bare ``@`` completes what it completed before projects existed.

    Project rows lead the list, but seeding the highlight onto one meant the
    first Enter silently rewrote the draft into ``@project:<first>`` (review
    round 1, F7 / QA Q-3). The highlight starts on the first NON-project row,
    so the completion stays the pre-slice one.
    """
    store.create_project(ProjectEdit(name="alpha"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")
        rows = _rows(editor)
        assert rows and rows[0].startswith("project:"), "premise: projects lead the list"
        first_file = next(name for name in rows if not name.startswith("project:"))

        await pilot.press("enter")
        await pilot.pause()

        assert editor.text == f"@{first_file}"
        assert not editor.text.startswith("@project:")


@pytest.mark.asyncio
async def test_the_overflow_row_counts_the_directory_not_the_shortcut(workspace, store) -> None:
    """The ``… N more`` row describes the DIRECTORY (review round 1, F11).

    Project rows are prepended, and counting them made a twelve-project store
    read a number that moved with the shortcut's own cap — a fact about the
    shortcut wearing the directory's label. The row now counts only rows that
    are not the shortcut (plus the scan cap's ``unlisted``, unchanged).
    """
    for index in range(12):
        store.create_project(ProjectEdit(name=f"sigma-{index:02d}"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")
        picker = editor.picker
        start, end, total = picker.visible_window()
        shown = [name for name, _ in picker.suggestions()][start:end]
        all_files = [name for name in _rows(editor) if not name.startswith("project:")]
        hidden_files = len(all_files) - len(
            [name for name in shown if not name.startswith("project:")]
        )

        text = picker.render_text(80).plain
        if hidden_files > 0:
            assert f"… {hidden_files} more" in text
        else:
            assert "more" not in text


@pytest.mark.asyncio
async def test_the_name_column_paints_the_bare_project_name(workspace, store) -> None:
    """The row VALUE is the token; the row PAINTS the project name (the
    ``/new`` device picker's split — a value repeating a keyword must not be
    what the name column shows)."""
    store.create_project(ProjectEdit(name="alpha", description="Alpha workstream"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")
        rendered = editor.picker.render_text(80).plain.split("\n")
        first = rendered[0]
        assert "alpha" in first
        assert "project:alpha" not in first
        assert "Alpha workstream" in first


@pytest.mark.asyncio
async def test_a_project_row_says_it_is_a_project(workspace, store) -> None:
    """The state column names the kind (review round 1, F10).

    A project row otherwise paints exactly like a file row — bare name plus a
    right-hand column — and the only tell was the ghost, visible only while
    the row is highlighted. ``detail="project"`` fills the same state column
    argument rows already use (skills spell ``hidden`` there).
    """
    store.create_project(ProjectEdit(name="alpha", description="Alpha workstream"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")
        rendered = editor.picker.render_text(80).plain.split("\n")
        file_row = next(row for row in rendered if "README.md" in row)

        assert rendered[0].rstrip().endswith("project")
        assert not file_row.rstrip().endswith("project")


@pytest.mark.asyncio
async def test_the_project_rows_are_capped_at_eight(workspace, store) -> None:
    for index in range(10):
        store.create_project(ProjectEdit(name=f"project-{index:02d}"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")

        projects = _project_rows(editor)
        assert len(projects) == 8
        # The first eight by name — the store's own listing order.
        assert projects == [f"project:project-{index:02d}" for index in range(8)]


@pytest.mark.asyncio
async def test_an_empty_store_means_no_project_rows_and_no_error(
    workspace, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No store at the config root degrades to the directory list alone —
    the picker still works (the degrade-one-feature rule)."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "empty-cfg"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")

        rows = _rows(editor)
        assert "README.md" in rows
        assert _project_rows(editor) == []


@pytest.mark.asyncio
async def test_an_unreadable_store_degrades_to_the_directory_listing(
    workspace, store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store read that RAISES is "no projects", never a broken picker."""
    store.create_project(ProjectEdit(name="alpha"))
    from local_operator.projects import ProjectRegistry

    def _boom(self) -> list[object]:
        raise OSError("store gone")

    monkeypatch.setattr(ProjectRegistry, "list_projects", _boom)
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")

        rows = _rows(editor)
        assert "README.md" in rows
        assert _project_rows(editor) == []


@pytest.mark.asyncio
async def test_the_reference_ink_paints_a_project_token(workspace, store) -> None:
    """The editor's ink gate is `reference_resolves`, so the project arm lands
    in the ink with no editor-side rule: a token naming a row is painted, one
    naming nothing is not — and the ink reads the names snapshot the picker's
    own store read published, never a store read of its own."""
    store.create_project(ProjectEdit(name="alpha"))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "see @project:alpha")
        await _settle(editor, pilot, lambda: references._project_names_cache is not None)
        await _settle(editor, pilot, lambda: bool(editor._reference_runs()))
        assert editor._reference_runs() == {0: [(4, 18)]}

        editor.load_text("see @project:ghost")
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        assert editor._reference_runs() == {}
