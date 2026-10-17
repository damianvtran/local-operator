"""The ``/import`` full-page mode: gating, real service flow and i18n."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from rich.text import Text

from local_operator.harness_import.jobs import JOBS
from local_operator.i18n import catalogues
from local_operator.slash_commands import SLASH_COMMANDS, slash_command_for
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.harness_import_view import HarnessImportView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

FIXTURES = Path(__file__).resolve().parents[1] / "harness_import" / "fixtures"


def _plain(static) -> str:
    content = static.content
    assert isinstance(content, Text)
    return content.plain


def _seed_omp() -> Path:
    root = Path.home() / ".omp" / "agent"
    root.mkdir(parents=True)
    shutil.copytree(FIXTURES / "omp" / "sessions", root / "sessions")
    return root


async def _settle(pilot, predicate, *, turns: int = 200) -> None:
    for _ in range(turns):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("condition did not settle")


async def _open_import(pilot) -> HarnessImportView:
    app = pilot.app
    app._run_slash_command("/import")
    await pilot.pause()
    view = app.query_one(HarnessImportView)
    await _settle(pilot, lambda: view._state == "sources")
    return view


async def _open_omp(pilot) -> HarnessImportView:
    view = await _open_import(pilot)
    for _ in range(len(view._sources)):
        if view._sources[view._source_index].source == "omp":
            break
        await pilot.press("down")
    await pilot.press("enter")
    await _settle(pilot, lambda: view._state == "sessions")
    return view


@pytest.mark.asyncio
async def test_import_command_opens_the_page_and_esc_restores_the_conversation() -> None:
    _seed_omp()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app._editor()
        editor.load_text("half-typed prompt")
        await pilot.pause()
        view = await _open_import(pilot)
        assert app.screen.has_class("settings")
        assert not app._transcript_view().display
        assert "omp" in _plain(view._body_text)
        assert "1 session" in _plain(view._body_text)
        await pilot.press("escape")
        await pilot.pause()
        assert not app.screen.has_class("settings")
        assert app._transcript_view().display
        assert app._editor().text == "half-typed prompt"


@pytest.mark.asyncio
async def test_sessions_require_selection_preview_and_confirmation() -> None:
    _seed_omp()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        view = await _open_omp(pilot)
        # No selection: Enter is refused, not a silent import.
        await pilot.press("enter")
        await _settle(pilot, lambda: view._state == "error")
        assert "Select at least one session" in _plain(view._body_text)
        await pilot.press("escape")
        await _settle(pilot, lambda: view._state == "sessions")
        await pilot.press("space")
        await pilot.press("enter")
        await _settle(pilot, lambda: view._state == "preview")
        assert "Nothing has been imported" in _plain(view._body_text)
        await pilot.press("enter")
        await _settle(pilot, lambda: view._state in ("progress", "result"))
        await _settle(pilot, lambda: view._state == "result")
        result_text = _plain(view._body_text)
        assert "Result" in result_text
        assert "imported" in result_text
        assert "→" in result_text
        assert (Path.home() / ".local-operator" / "sessions").is_dir()


@pytest.mark.asyncio
async def test_import_status_and_cancel_use_the_shared_job_registry() -> None:
    _seed_omp()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        view = await _open_omp(pilot)
        await pilot.press("space")
        await pilot.press("enter")
        await _settle(pilot, lambda: view._state == "preview")
        await pilot.press("enter")
        # The fixture is tiny, so the import can finish before the next pause.
        # Cancel only if the page is still in flight; either way it must settle
        # through the shared registry rather than crash.
        await _settle(pilot, lambda: view._state in ("progress", "result"))
        if view._state == "progress":
            await pilot.press("c")
        await _settle(pilot, lambda: view._state == "result")
        assert view._job is not None


def test_import_has_a_slash_alias_and_i18n_copy() -> None:
    import_command = slash_command_for("/import")
    migrate_command = slash_command_for("/migrate")
    assert import_command is not None and import_command.name == "import"
    assert migrate_command is not None and migrate_command.name == "import"
    entry = next(command for command in SLASH_COMMANDS if command.name == "import")
    assert entry.aliases == ("migrate",)
    assert entry.description
    messages = catalogues.load_catalogue("en", "tui.import")
    assert "tui.import.title" in messages
    assert "tui.import.hints_sessions" in messages
    assert "tui.import.review_instructions" in messages


@pytest.mark.asyncio
async def test_import_subcommands_parse_into_a_launch() -> None:
    _seed_omp()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/import preview omp")
        await pilot.pause()
        view = app.query_one(HarnessImportView)
        assert view.launch.action == "preview"
        assert view.launch.source == "omp"
        app._close_import_view()
        await pilot.pause()
        app._run_slash_command("/import run omp all")
        await pilot.pause()
        view = app.query_one(HarnessImportView)
        assert view.launch.action == "run"
        assert view.launch.all is True
        assert view.launch.session_id == ""
        app._close_import_view()
        await pilot.pause()
        app._run_slash_command("/import nonsense")
        await pilot.pause()
        assert not app.query(HarnessImportView)


@pytest.mark.asyncio
async def test_cancel_in_the_queued_window_ends_cancelled_with_no_rows() -> None:
    _seed_omp()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        view = await _open_omp(pilot)
        await pilot.press("space")
        await pilot.press("enter")
        await _settle(pilot, lambda: view._state == "preview")
        # ``_start`` schedules the service task; cancel before the event loop
        # gives it a turn, which is the queued window a route/tool cancel must
        # preserve.
        view._start()
        view._cancel()
        await _settle(pilot, lambda: view._state == "result")
        assert view._job is not None
        assert view._job.progress.state == "cancelled"
        assert view._job.result is not None
        assert view._job.result.state == "cancelled"
        assert view._job.result.messages == 0


@pytest.mark.asyncio
async def test_status_without_a_job_is_an_explicit_error(monkeypatch) -> None:
    monkeypatch.setattr(JOBS, "latest", lambda **kwargs: None)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/import status")
        await pilot.pause()
        view = app.query_one(HarnessImportView)
        await _settle(pilot, lambda: view._state == "error")
        assert "No import job found" in _plain(view._body_text)


def test_import_is_frontend_local() -> None:
    """The page reads source stores and writes config on THIS machine."""
    from local_operator.session.frontend_state import _FRONTEND_LOCAL_SLASHES

    assert "import" in _FRONTEND_LOCAL_SLASHES


def test_result_paints_instruction_and_memory_review_paths() -> None:
    """Both review artifacts are user-visible when the core reports them."""
    from types import SimpleNamespace

    from local_operator.harness_import.types import ImportProgress, ImportResult

    view = HarnessImportView()
    result = ImportResult(
        job_id="job",
        state="done",
        instruction_review_path="/tmp/config/imports/instructions/omp.md",
        memory_review_path="/tmp/config/imports/memory-omp.json",
    )
    view._job = SimpleNamespace(
        result=result,
        progress=ImportProgress(job_id="job", state="done"),
        job_id="job",
    )
    view._state = "result"
    body = view._body_rich().plain
    assert "Instructions saved for review at /tmp/config/imports/instructions/omp.md" in body
    assert "Untethered memory saved for review at /tmp/config/imports/memory-omp.json" in body
