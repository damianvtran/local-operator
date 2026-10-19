"""The ``/import`` full-page mode: gating, real service flow and i18n."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from rich.cells import cell_len
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
    await pilot.pause()
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
        await _settle(pilot, lambda: view._state == "empty")
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
    assert "tui.import.hint_preview" in messages
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
    assert "Instructions saved for review at" in body
    assert "/tmp/config/imports/instructions/omp.md" in body
    assert "Untethered memory saved for review at" in body
    assert "/tmp/config/imports/memory-omp.json" in body


def _seed_many_omp(count: int = 12) -> None:
    source = FIXTURES / "omp" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    root = Path.home() / ".omp" / "agent" / "sessions" / "--many--"
    root.mkdir(parents=True)
    for index in range(count):
        copy = [dict(record) for record in records]
        copy[0]["title"] = f"Synthetic omp session {index:02d}"
        copy[1]["id"] = f"omp-synthetic-{index:02d}"
        with (root / f"omp-synthetic-{index:02d}.jsonl").open("w", encoding="utf-8") as handle:
            for record in copy:
                handle.write(json.dumps(record) + "\n")


def _seed_long_title() -> None:
    path = Path.home() / ".omp" / "agent" / "sessions" / "--long--" / "omp-long.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        json.loads(line)
        for line in (FIXTURES / "omp" / "sessions" / "--tmp--" / "omp-synthetic.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    records[0]["title"] = "A very long synthetic title " * 8
    records[1]["id"] = "omp-long"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


@pytest.mark.asyncio
async def test_cancel_command_reattaches_and_arms_the_poll(monkeypatch) -> None:
    """Q-1: a fresh `/import cancel` page must leave Importing and show cancelled."""
    from local_operator.harness_import.service import HarnessImportService

    original = HarnessImportService._import_one

    def slow_import(self, *args, **kwargs):
        # Hold the job until the REAL cancel signal, not a 0.6s head start:
        # under load the close/reopen can take longer and truthfully show done.
        assert self._cancel.wait(5), "the reattached page never requested cancellation"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HarnessImportService, "_import_one", slow_import)
    _seed_omp()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        view = await _open_omp(pilot)
        await pilot.press("space")
        await pilot.press("enter")
        await _settle(pilot, lambda: view._state == "preview")
        await pilot.press("enter")
        await _settle(pilot, lambda: view._state == "progress")
        app._close_import_view()
        await pilot.pause()
        app._run_slash_command("/import cancel")
        await pilot.pause()
        view = app.query_one(HarnessImportView)
        await _settle(pilot, lambda: view._state == "result")
        assert view._job is not None
        assert view._job.progress.state == "cancelled"
        assert "cancelled" in _plain(view._body_text)
        assert "Nothing running" not in _plain(view._body_text)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [41, 1203])
async def test_end_and_arrow_show_the_full_last_row(count) -> None:
    _seed_many_omp(count)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        view = await _open_omp(pilot)
        view.action_jump(1)
        await pilot.pause()
        index = count - 1
        assert view._session_index == index
        offsets = [2 + (1 if row.cwd else 0) for row in view._sessions]
        top = view._body_leading_lines() + sum(offsets[:index])
        y = view._body.scroll_offset.y
        height = view._body.size.height
        assert top + offsets[index] <= y + height
        view.action_move(1)  # clamped at the last row
        await pilot.pause()
        assert view._session_index == index


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (80, 24), (60, 20)])
async def test_long_titles_are_one_clipped_line_at_every_width(size) -> None:
    _seed_long_title()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        view = await _open_omp(pilot)
        await pilot.pause()
        body = _plain(view._body_text)
        lines = body.splitlines()
        title_lines = [line for line in lines if "long synthetic title" in line or "…" in line]
        assert len(title_lines) >= 1
        assert all(cell_len(line) <= view._line_width() for line in lines)
        # The fixture's timestamp crosses a UTC date boundary; assert the
        # rendered date SHAPE rather than one timezone's calendar date (Q-8).
        assert any(
            "…" in line and re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", line)
            for line in title_lines
        )
        assert view._pane.display is (size[0] >= 100)


@pytest.mark.asyncio
async def test_source_and_session_rows_are_clickable() -> None:
    from types import SimpleNamespace

    _seed_many_omp()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        view = await _open_import(pilot)
        body = view._body
        leading = view._body_leading_lines()

        def click_line(line: int) -> None:
            view.on_click(
                SimpleNamespace(
                    button=1,
                    screen_x=body.region.x + 2,
                    screen_y=body.region.y + line - body.scroll_offset.y,
                    stop=lambda: None,
                )
            )

        # Every source card, including the last one, opens its own source.
        for index, source in enumerate(("claude", "codex", "omp")):
            click_line(leading + index * 2)
            await pilot.pause()
            await _settle(
                pilot,
                lambda: view._state in ("sessions", "empty"),
            )
            assert view._source == source
            if source != "omp":
                await pilot.press("escape")
                await _settle(pilot, lambda: view._state == "sources")

        # Session title, meta and cwd lines all resolve to their own row.
        for line, expected in ((leading, 0), (leading + 4, 1), (leading + 5, 1)):
            click_line(line)
            await pilot.pause()
            assert view._session_index == expected


def test_selection_count_is_outside_the_body_and_a_toggles_both_ways() -> None:
    from types import SimpleNamespace

    view = HarnessImportView()
    view._state = "sessions"
    view._sessions = [
        SimpleNamespace(source_id="a", cwd=""),
        SimpleNamespace(source_id="b", cwd=""),
    ]
    view._selected = {"a"}
    assert "1 selected" in view._detail_text()
    view._paint = lambda: None  # type: ignore[method-assign]
    view.action_all()
    assert view._selected == {"a", "b"}
    view.action_all()
    assert view._selected == set()


def test_enum_values_are_mapped_to_user_copy() -> None:
    from types import SimpleNamespace

    from local_operator.tui.widgets.harness_import_view import (
        _availability_label,
        _outcome_label,
        _state_label,
    )

    assert _availability_label(SimpleNamespace(availability="metadata_only")) == "index only"
    assert _outcome_label("skipped_already_imported") == "already imported"
    assert "skipped_already_imported" not in _outcome_label("skipped_already_imported")
    assert _state_label("cancelled") == "cancelled"


def test_status_paints_request_rows_and_result_warning_messages() -> None:
    from types import SimpleNamespace

    from local_operator.harness_import.types import (
        ImportProgress,
        ImportRequest,
        ImportResult,
        ImportWarning,
    )

    view = HarnessImportView()
    request = ImportRequest(sources=["omp"], session_ids={"omp": ["s1", "s2"]})
    view._job = SimpleNamespace(
        job_id="job",
        request=request,
        progress=ImportProgress(
            job_id="job",
            state="running",
            phase="session",
            completed=1,
            total=2,
            session_id="s2",
            results=[{"source_id": "s1", "outcome": "imported", "messages": 3}],
        ),
        result=None,
    )
    view._state = "progress"
    body = view._body_rich().plain
    assert "s1" in body and "s2" in body
    assert "imported" in body
    assert "pending" not in body.lower() or "s2" in body

    result = ImportResult(
        job_id="job",
        state="done",
        warnings=[ImportWarning(source="omp", session_id="s", code="test", message="a warning")],
        sessions=[],
    )
    view._job.result = result
    view._state = "result"
    assert "a warning" in view._body_rich().plain


def test_cancel_while_loading_is_a_noop() -> None:
    view = HarnessImportView()
    view._state = "loading"
    view.action_cancel()
    assert view._state == "loading"


def test_preview_paints_memory_counts() -> None:
    from types import SimpleNamespace

    from local_operator.harness_import.types import ImportRequest

    view = HarnessImportView()
    view._request = ImportRequest(sources=["omp"], session_ids={"omp": ["s"]})
    view._preview = SimpleNamespace(
        total_sessions=1,
        total_messages=8,
        total_compactions=1,
        total_memory_records=2,
        projected_memory_tokens=1234,
        projected_global_tokens=12,
        warnings=[],
    )
    view._selected = {"s"}
    view._state = "preview"
    body = view._body_rich().plain
    assert "2 memory records" in body
    assert "1,234 tokens" in body


def test_preview_warning_total_and_wrapped_tail() -> None:
    from types import SimpleNamespace

    from local_operator.harness_import.types import ImportRequest

    view = HarnessImportView()
    view._request = ImportRequest(sources=["omp"], session_ids={"omp": ["s"]})
    view._selected = {"s"}
    view._preview = SimpleNamespace(
        total_sessions=1,
        total_messages=8,
        total_compactions=1,
        total_memory_records=0,
        projected_memory_tokens=0,
        projected_global_tokens=0,
        warnings=[
            SimpleNamespace(message=f"warning {index} with an actionable tail")
            for index in range(10)
        ],
    )
    view._state = "preview"
    body = view._body_rich().plain
    assert "10 warnings" in body
    assert "warning 0 with an actionable tail" in body
    assert "warning 3" in body
    assert "warning 4" not in body


def test_one_failure_is_rendered_and_counted_once() -> None:
    from types import SimpleNamespace

    from local_operator.harness_import.types import (
        ImportProgress,
        ImportRequest,
        ImportResult,
        ImportWarning,
        SessionResult,
    )

    view = HarnessImportView()
    request = ImportRequest(sources=["omp"], session_ids={"omp": ["s1"]})
    result = ImportResult(
        job_id="job",
        state="failed",
        warnings=[
            ImportWarning(
                source="omp",
                session_id="s1",
                code="import_failed",
                message="OSError: source disappeared",
            )
        ],
        sessions=[
            SessionResult(
                source="omp",
                source_id="s1",
                outcome="failed",
                warnings=["OSError: source disappeared"],
            )
        ],
    )
    view._job = SimpleNamespace(
        result=result,
        progress=ImportProgress(job_id="job", state="failed"),
        job_id="job",
        request=request,
    )
    view._state = "result"
    body = view._body_rich().plain
    assert body.count("OSError: source disappeared") == 1
    assert "1 warning" in body
    assert "2 warnings" not in body


def test_result_row_is_two_lines_and_keeps_outcome_and_count() -> None:
    view = HarnessImportView()
    row = {
        "source_id": "omp-synthetic-with-a-long-identifier",
        "target_session_id": "e0bb9751a4e6",
        "outcome": "skipped_already_imported",
        "messages": 7,
    }
    lines = view._result_row(row).splitlines()
    assert len(lines) == 2
    assert "already imported" in lines[1]
    assert "7 messages" in lines[1]
    assert "…" not in lines[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("size,visible", [((80, 24), True), ((60, 20), False)])
async def test_move_hint_uses_the_width_budget(size, visible) -> None:
    _seed_omp()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        view = await _open_omp(pilot)
        await pilot.pause()
        assert view._hint_move.display is visible
